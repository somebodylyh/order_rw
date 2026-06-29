"""Run frozen 9-variant block-level diagnostic on a given checkpoint.

Parameterized version of diagnose_10k_patch8x8.py for overnight automation.

Usage:
    python block_lo_arm_order_network/run_frozen_diagnostic.py \
        --ckpt path/to/ckpt.pt \
        --data path/to/train.bin \
        --step 30000 \
        --out-dir probe_results_image_large/diagnostics_patch8x8/step_30000 \
        --device cuda:1
"""

import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_NANO = _ROOT.parent / "nanogpt-learned-order"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_NANO))

from AOGPT import AOGPTConfig, AOGPT
from extract_image_attention import load_model, extract_A_global

T = 256
BLOCK_LEN = 4
NUM_BLOCKS = 64
GRID = 8
N_IMAGES = 500
K_SAMPLE = 100
SEED = 42


def aggregate_to_blocks(A_token):
    A_block = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            block = A_token[i*BLOCK_LEN:(i+1)*BLOCK_LEN, j*BLOCK_LEN:(j+1)*BLOCK_LEN]
            A_block[i, j] = block.mean()
    return A_block


def build_B_and_signal(A_block):
    from directed_graph_policy import build_directed_graph, compute_source
    B = build_directed_graph(A_block)
    source, out_deg, in_deg = compute_source(B, alpha_dep=0.5)
    s = float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))
    return B, s


def sample_order_safe(B, policy, params, seed):
    from directed_graph_policy import sample_order
    return sample_order(B, policy, params, seed=seed)


def run_variants(B_block):
    from directed_graph_policy import sample_order
    N = B_block.shape[0]
    variants = {}

    # random
    orders_rand = np.zeros((K_SAMPLE, N), dtype=np.int64)
    rng = np.random.RandomState(SEED)
    for k in range(K_SAMPLE):
        orders_rand[k] = rng.permutation(N)
    variants["random"] = orders_rand

    # raster
    raster_order = np.arange(N)
    variants["raster"] = np.tile(raster_order, (K_SAMPLE, 1))

    # full_v3
    params_v3 = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                 "alpha_dep": 0.5, "lam": 0.75, "rho": 0.5, "top_k": 4, "epsilon_uniform": 0.0}
    orders_v3 = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_v3[k], _ = sample_order(B_block, params_v3["policy"], params_v3, seed=SEED*1000+k)
    variants["full_v3"] = orders_v3

    # local_only
    params_loc = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                  "alpha_dep": 0.5, "lam": 1.0, "rho": 0.1, "top_k": 4, "epsilon_uniform": 0.0}
    orders_loc = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_loc[k], _ = sample_order(B_block, params_loc["policy"], params_loc, seed=SEED*1000+k)
    variants["local_only"] = orders_loc

    # no_readiness
    params_nr = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                 "alpha_dep": 0.5, "lam": 0.75, "rho": 0.0, "top_k": 4, "epsilon_uniform": 0.0}
    orders_nr = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_nr[k], _ = sample_order(B_block, params_nr["policy"], params_nr, seed=SEED*1000+k)
    variants["no_readiness"] = orders_nr

    # readiness_only
    params_ro = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                 "alpha_dep": 0.5, "lam": 0.0, "rho": 0.5, "top_k": 4, "epsilon_uniform": 0.0}
    orders_ro = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_ro[k], _ = sample_order(B_block, params_ro["policy"], params_ro, seed=SEED*1000+k)
    variants["readiness_only"] = orders_ro

    # shuffled_B
    B_shuf = B_block.copy()
    rng = np.random.RandomState(SEED)
    rng.shuffle(B_shuf)
    orders_shuf = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_shuf[k], _ = sample_order(B_shuf, params_v3["policy"], params_v3, seed=SEED*1000+k)
    variants["shuffled_B"] = orders_shuf

    # random_B
    B_rand = np.random.RandomState(SEED).uniform(0.01, 0.05, size=(N, N)).astype(np.float64)
    np.fill_diagonal(B_rand, 0)
    orders_rb = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_rb[k], _ = sample_order(B_rand, params_v3["policy"], params_v3, seed=SEED*1000+k)
    variants["random_B"] = orders_rb

    # graph_guided (same as full_v3 but explicit label)
    orders_gg = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_gg[k], _ = sample_order(B_block, params_v3["policy"], params_v3, seed=SEED*1000+k)
    variants["graph_guided"] = orders_gg

    return variants


def compute_diagnostics(variants):
    from scipy.stats import kendalltau
    N = NUM_BLOCKS
    rows = np.arange(N) // GRID
    cols = np.arange(N) % GRID
    quad = (rows // 4) * 2 + (cols // 4)
    L2R = np.arange(N)

    results = {}
    for name, orders in variants.items():
        K = orders.shape[0]

        step_manh = []
        for k in range(K):
            for t in range(N - 1):
                u, v = orders[k, t], orders[k, t + 1]
                step_manh.append(abs(int(rows[u]) - int(rows[v])) + abs(int(cols[u]) - int(cols[v])))
        step_manh = np.array(step_manh)
        mean_step_manh = float(step_manh.mean())
        p_d1 = float((step_manh <= 1).mean())
        p_d2 = float((step_manh <= 2).mean())

        same_q = []
        same_s = []
        for k in range(K):
            for t in range(N - 1):
                u, v = orders[k, t], orders[k, t + 1]
                same_q.append(int(quad[u] == quad[v]))
                dr = abs(int(rows[u]) - int(rows[v]))
                dc = abs(int(cols[u]) - int(cols[v]))
                same_s.append(int(dr <= 1 and dc <= 1))
        p_same_q = float(np.mean(same_q))
        p_same_s = float(np.mean(same_s))

        # Run length
        run_lens = []
        for k in range(K):
            run = 1
            for t in range(N - 1):
                u, v = orders[k, t], orders[k, t + 1]
                if rows[u] == rows[v] or cols[u] == cols[v]:
                    run += 1
                else:
                    if run > 1:
                        run_lens.append(run)
                    run = 1
            if run > 1:
                run_lens.append(run)
        mean_run = float(np.mean(run_lens)) if run_lens else 1.0

        # τ vs L2R
        taus = [kendalltau(orders[k], L2R)[0] for k in range(K)]
        mean_tau = float(np.mean(taus))

        # Pairwise τ
        n_ck = min(K, 50)
        pair_taus = []
        for i in range(n_ck):
            for j in range(i + 1, n_ck):
                pair_taus.append(kendalltau(orders[i], orders[j])[0])
        pair_tau = float(np.mean(pair_taus)) if pair_taus else 0.0

        # First-node entropy
        first_nodes = orders[:, 0]
        counts = np.bincount(first_nodes, minlength=N).astype(float)
        probs = counts / K
        probs_nz = probs[probs > 0]
        first_H = float(-np.sum(probs_nz * np.log(probs_nz)))

        results[name] = {
            "mean_step_manhattan": mean_step_manh,
            "P(d<=1)": p_d1,
            "P(d<=2)": p_d2,
            "P(same_quadrant)": p_same_q,
            "P(same_subgrid)": p_same_s,
            "mean_run_length": mean_run,
            "tau_vs_L2R": mean_tau,
            "pairwise_tau": pair_tau,
            "first_node_entropy": first_H,
        }
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--n-images", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # Step 1: Extract A_global
    print(f"\n{'='*60}")
    print(f"Frozen diagnostic: step {args.step}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  device: {args.device}")
    print(f"{'='*60}")

    device = torch.device(args.device)
    model, model_args = load_model(args.ckpt, args.device)
    print(f"  n_layer={model_args['n_layer']}, n_head={model_args['n_head']}")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    print(f"  Data: {len(data)//T} images available")

    A_token = extract_A_global(model, data, args.n_images, args.device)
    np.save(out_dir / "A_token_256x256.npy", A_token)

    # Step 2: Aggregate
    A_block = aggregate_to_blocks(A_token)
    np.save(out_dir / "A_block_64x64.npy", A_block.astype(np.float32))
    print(f"  A_block: mean={A_block.mean():.6f}, std={A_block.std():.6f}")

    # Step 3: Build B
    B_block, s = build_B_and_signal(A_block)
    np.save(out_dir / "B_block_64x64.npy", B_block.astype(np.float32))
    print(f"  Readiness signal s = {s:.4f}")

    # Step 4: Run variants
    print(f"\n  Running 9 variants (K={K_SAMPLE})...")
    variants = run_variants(B_block)

    # Step 5: Diagnostics
    print(f"  Computing diagnostics...")
    results = compute_diagnostics(variants)

    # Print table
    metrics = ["mean_step_manhattan", "P(d<=1)", "P(d<=2)", "P(same_quadrant)",
               "P(same_subgrid)", "mean_run_length", "tau_vs_L2R", "pairwise_tau",
               "first_node_entropy"]

    print(f"\n{'='*80}")
    print(f"SUMMARY: step {args.step} patch8x8 block-level (64×64) diagnostic")
    print(f"  Readiness signal s = {s:.4f}")
    print(f"{'='*80}")
    print(f"{'variant':<22s}", end="")
    for m in metrics:
        print(f"  {m:>14s}", end="")
    print()
    for name in ["random", "raster", "full_v3", "local_only", "no_readiness",
                 "readiness_only", "shuffled_B", "random_B", "graph_guided"]:
        r = results[name]
        print(f"  {name:<20s}", end="")
        for m in metrics:
            print(f"  {r[m]:14.4f}", end="")
        print()

    # Save
    out_json = {
        "checkpoint": args.ckpt,
        "step": args.step,
        "data": args.data,
        "N_images": args.n_images,
        "K_sample": K_SAMPLE,
        "readiness_signal_s": s,
        "results": results,
    }
    with open(out_dir / "diagnostic_results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Output: {out_dir}/")
    print("DONE")


if __name__ == "__main__":
    main()
