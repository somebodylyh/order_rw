"""Extract attention from 10k ckpt → 64x64 block aggregation → 9-variant diagnostic.

Variants:
  random       — pure random permutation (lower bound)
  raster       — 8×8 raster scan (reference)
  full_v3      — adaptive progressive_rw_v3 (readiness params)
  local_only   — high locality, no readiness (lam=1.0, rho=0.1)
  no_readiness — no readiness term (rho=0.0)
  readiness_only — no locality (lam=0.0)
  shuffled_B   — shuffle rows of B (destroy column structure)
  random_B     — replace B with uniform random (negative control)
  graph_guided — B directly as transition weights
"""

import os, sys, time, json
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_NANO = _ROOT.parent / "nanogpt-learned-order"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_NANO))

# --- Paths ---
CKPT_10K = _NANO / "out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline/checkpoints/ckpt_iter0010000.pt"
DATA = _NANO / "data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin"
OUT_DIR = _ROOT / "probe_results_image_large" / "diagnostic_10k"
DEVICE = "cuda:1"  # use GPU 1 to avoid interfering with training on GPU 0
N_IMAGES = 500
M_ORDERS = 3
K_SAMPLE = 100
SEED = 42
BLOCK_SIZE = 4  # tokens per 2x2 patch
NUM_BLOCKS = 64  # 8x8 blocks


# ===================================================================
# Step 1: Extract token-level A
# ===================================================================
def extract_attention():
    from extract_image_attention import load_model, extract_A_global

    out = OUT_DIR / "attention"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"\n{'='*60}")
    print(f"Step 1: Extract A_global ({N_IMAGES} imgs × {M_ORDERS} orders)")
    print(f"  ckpt: {CKPT_10K}")
    print(f"  device: {DEVICE}")
    print(f"{'='*60}")

    model, model_args = load_model(str(CKPT_10K), DEVICE)
    print(f"  n_layer={model_args['n_layer']}, n_head={model_args['n_head']}, "
          f"n_embd={model_args['n_embd']}, block_size={model_args['block_size']}, "
          f"bol={model_args.get('block_order_block_len', 'N/A')}")

    data = np.memmap(str(DATA), dtype=np.uint16, mode="r")
    print(f"  Data: {len(data)//256} images available")

    t0 = time.time()
    A = extract_A_global(model, data, N_IMAGES, DEVICE)
    elapsed = time.time() - t0

    np.save(out / "A_token_256x256.npy", A)
    print(f"  Saved A_token: shape={A.shape}, mean={A.mean():.6f}, max={A.max():.6f}, time={elapsed:.1f}s")
    return A, model_args


# ===================================================================
# Step 2: Aggregate to block-level 64×64
# ===================================================================
def aggregate_to_blocks(A_token):
    """Mean-pool 4 consecutive tokens → 64×64 block matrix."""
    N = A_token.shape[0]  # 256
    assert N == 256 and NUM_BLOCKS == 64 and BLOCK_SIZE == 4

    A_block = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            block = A_token[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
            A_block[i, j] = block.mean()

    print(f"\n  A_block: shape={A_block.shape}, mean={A_block.mean():.6f}, "
          f"max={A_block.max():.6f}")

    out = OUT_DIR / "attention"
    np.save(out / "A_block_64x64.npy", A_block)
    return A_block


# ===================================================================
# Step 3: Build B and compute signal
# ===================================================================
def build_B_and_signal(A_block):
    from directed_graph_policy import build_directed_graph, compute_source

    B = build_directed_graph(A_block)
    N = B.shape[0]
    out = OUT_DIR / "attention"
    np.save(out / "B_block_64x64.npy", B)

    source, out_deg, in_deg = compute_source(B, alpha_dep=0.5)
    s = float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))

    # Manhattan on 8×8 grid
    grid = 8
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid

    B_flat = B.copy()
    np.fill_diagonal(B_flat, 0)
    top_k = 1000
    flat_idx = np.argpartition(B_flat.ravel(), -top_k)[-top_k:]
    src = flat_idx // N
    dst = flat_idx % N
    manh = np.abs(rows[src] - rows[dst]) + np.abs(cols[src] - cols[dst])
    mean_manh = float(manh.mean())
    median_manh = float(np.median(manh))

    print(f"\n{'='*60}")
    print(f"Step 3: B block 64×64 signal")
    print(f"{'='*60}")
    print(f"  B: mean={B.mean():.6f}, std={B.std():.6f}")
    print(f"  Readiness signal s = {s:.4f}")
    print(f"  Mean manhattan (top B pairs): {mean_manh:.3f}")
    print(f"  Median manhattan: {median_manh:.3f}")

    # Histogram of manhattan distances
    all_manh = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])
    # Weight by B (excluding diagonal)
    B_no_diag = B.copy(); np.fill_diagonal(B_no_diag, 0)
    for d in range(15):
        mask_d = (all_manh == d)
        weight = B_no_diag[mask_d].sum()
        if weight > 0:
            print(f"    d={d:2d}: weight={weight:.6f}")

    return B, s, mean_manh


# ===================================================================
# Step 4: Run 9 variants
# ===================================================================
def run_variants(B_block):
    from directed_graph_policy import sample_order, build_directed_graph, compute_source

    N = B_block.shape[0]
    grid = int(np.sqrt(N))  # 8
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid

    variants = {}

    # ---- random ----
    print("\n  [random] pure random permutations")
    orders_rand = np.zeros((K_SAMPLE, N), dtype=np.int64)
    rng = np.random.RandomState(SEED)
    for k in range(K_SAMPLE):
        orders_rand[k] = rng.permutation(N)
    variants["random"] = orders_rand

    # ---- raster ----
    print("  [raster] 8x8 raster scan")
    raster_order = np.arange(N).reshape(grid, grid).ravel()
    orders_rast = np.tile(raster_order, (K_SAMPLE, 1))
    variants["raster"] = orders_rast

    # ---- full_v3 (readiness_dominant params) ----
    print("  [full_v3] progressive_rw_v3 readiness_dominant")
    params_v3 = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                 "alpha_dep": 0.5, "lam": 0.75, "rho": 0.5, "top_k": 4, "epsilon_uniform": 0.0}
    orders_v3 = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_v3[k], _ = sample_order(B_block, params_v3["policy"], params_v3, seed=SEED*1000+k)
    variants["full_v3"] = orders_v3

    # ---- local_only ----
    print("  [local_only] lam=1.0 rho=0.1")
    params_loc = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                  "alpha_dep": 0.5, "lam": 1.0, "rho": 0.1, "top_k": 4, "epsilon_uniform": 0.0}
    orders_loc = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_loc[k], _ = sample_order(B_block, params_loc["policy"], params_loc, seed=SEED*1000+k)
    variants["local_only"] = orders_loc

    # ---- no_readiness ----
    print("  [no_readiness] rho=0.0")
    params_nr = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                 "alpha_dep": 0.5, "lam": 0.75, "rho": 0.0, "top_k": 4, "epsilon_uniform": 0.0}
    orders_nr = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_nr[k], _ = sample_order(B_block, params_nr["policy"], params_nr, seed=SEED*1000+k)
    variants["no_readiness"] = orders_nr

    # ---- readiness_only ----
    print("  [readiness_only] lam=0.0")
    params_ro = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                 "alpha_dep": 0.5, "lam": 0.0, "rho": 0.5, "top_k": 4, "epsilon_uniform": 0.0}
    orders_ro = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_ro[k], _ = sample_order(B_block, params_ro["policy"], params_ro, seed=SEED*1000+k)
    variants["readiness_only"] = orders_ro

    # ---- shuffled_B ----
    print("  [shuffled_B] row-shuffled B")
    B_shuf = B_block.copy()
    rng = np.random.RandomState(SEED)
    rng.shuffle(B_shuf)  # shuffle rows
    orders_shuf = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_shuf[k], _ = sample_order(B_shuf, params_v3["policy"], params_v3, seed=SEED*1000+k)
    variants["shuffled_B"] = orders_shuf

    # ---- random_B ----
    print("  [random_B] uniform random B")
    B_rand = np.random.RandomState(SEED).uniform(0.01, 0.05, size=(N, N)).astype(np.float64)
    np.fill_diagonal(B_rand, 0)
    orders_rb = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_rb[k], _ = sample_order(B_rand, params_v3["policy"], params_v3, seed=SEED*1000+k)
    variants["random_B"] = orders_rb

    # ---- graph_guided ----
    print("  [graph_guided] raw B as transition")
    params_gg = {"policy": "progressive_rw_v3", "tau_start": 0.10, "tau_step": 0.10,
                 "alpha_dep": 0.5, "lam": 0.75, "rho": 0.5, "top_k": 4, "epsilon_uniform": 0.0}
    orders_gg = np.zeros((K_SAMPLE, N), dtype=np.int64)
    for k in range(K_SAMPLE):
        orders_gg[k], _ = sample_order(B_block, params_gg["policy"], params_gg, seed=SEED*1000+k)
    variants["graph_guided"] = orders_gg

    return variants


# ===================================================================
# Step 5: Compute diagnostics for each variant
# ===================================================================
def compute_diagnostics(variants, B_block):
    from scipy.stats import kendalltau

    N = B_block.shape[0]
    grid = 8
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid

    # Patch quadrature: which quadrant of 8x8 grid
    quad_rows = np.where(rows < grid//2, 0, 1)  # 0=top, 1=bottom
    quad_cols = np.where(cols < grid//2, 0, 1)  # 0=left, 1=right

    L2R = np.arange(N)

    results = {}
    for name, orders in variants.items():
        K = orders.shape[0]

        # Step manhattan
        step_manh = []
        for k in range(K):
            for t in range(N - 1):
                u, v = orders[k, t], orders[k, t + 1]
                step_manh.append(abs(rows[u] - rows[v]) + abs(cols[u] - cols[v]))
        mean_step_manh = float(np.mean(step_manh))

        # P(d<=1), P(d<=2)
        p_d1 = np.mean(np.array(step_manh) <= 1)
        p_d2 = np.mean(np.array(step_manh) <= 2)

        # Same quadrant probability
        same_q = []
        for k in range(K):
            for t in range(N - 1):
                u, v = orders[k, t], orders[k, t + 1]
                same_q.append(int(quad_rows[u] == quad_rows[v] and quad_cols[u] == quad_cols[v]))
        p_same_q = float(np.mean(same_q))

        # Same sub-grid (2x2 block within 8x8 → immediate neighbor)
        same_s = []
        for k in range(K):
            for t in range(N - 1):
                u, v = orders[k, t], orders[k, t + 1]
                dr = abs(rows[u] - rows[v])
                dc = abs(cols[u] - cols[v])
                same_s.append(int(dr <= 1 and dc <= 1))
        p_same_s = float(np.mean(same_s))

        # Run length (consecutive steps staying within same row or col)
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
        pair_tau = float(np.mean(pair_taus))

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


# ===================================================================
# Step 6: Transition heatmap (full_v3 only)
# ===================================================================
def compute_heatmaps(orders_full_v3):
    N = orders_full_v3.shape[1]
    grid = 8
    trans = np.zeros((N, N), dtype=np.float64)
    K = orders_full_v3.shape[0]
    for k in range(K):
        for t in range(N - 1):
            u, v = orders_full_v3[k, t], orders_full_v3[k, t + 1]
            trans[u, v] += 1
    trans /= (K * (N - 1))

    # Average step heatmap (by row, col of source)
    avg_step = np.zeros((grid, grid), dtype=np.float64)
    count_step = np.zeros((grid, grid), dtype=np.float64)
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid
    for k in range(K):
        for t in range(N - 1):
            u, v = orders_full_v3[k, t], orders_full_v3[k, t + 1]
            dr = abs(rows[u] - rows[v])
            dc = abs(cols[u] - cols[v])
            avg_step[rows[u], cols[u]] += dr + dc
            count_step[rows[u], cols[u]] += 1
    avg_step /= np.maximum(count_step, 1)

    np.save(OUT_DIR / "transition_heatmap_64x64.npy", trans)
    np.save(OUT_DIR / "avg_step_heatmap_8x8.npy", avg_step)
    return trans, avg_step


# ===================================================================
# Main
# ===================================================================
def main():
    t_total = time.time()

    # Step 1
    A_token, model_args = extract_attention()

    # Step 2
    A_block = aggregate_to_blocks(A_token)

    # Step 3
    B_block, s, manh = build_B_and_signal(A_block)

    # Step 4
    print(f"\n{'='*60}")
    print(f"Step 4: Run 9 variants (K={K_SAMPLE})")
    print(f"{'='*60}")
    variants = run_variants(B_block)

    # Step 5
    print(f"\n{'='*60}")
    print(f"Step 5: Diagnostics")
    print(f"{'='*60}")
    results = compute_diagnostics(variants, B_block)

    # Step 6
    print(f"\n{'='*60}")
    print(f"Step 6: Heatmaps (full_v3)")
    print(f"{'='*60}")
    trans, avg_step = compute_heatmaps(variants["full_v3"])

    # ---- Print table ----
    metrics = ["mean_step_manhattan", "P(d<=1)", "P(d<=2)", "P(same_quadrant)",
               "P(same_subgrid)", "mean_run_length", "tau_vs_L2R", "pairwise_tau",
               "first_node_entropy"]

    print(f"\n{'='*80}")
    print(f"SUMMARY: 10k patch8x8 block-level (64×64) diagnostic")
    print(f"  Readiness signal s = {s:.4f}")
    print(f"  Mean manhattan (top B pairs) = {manh:.3f}")
    print(f"{'='*80}")
    print(f"{'variant':<22s}", end="")
    for m in metrics:
        print(f"  {m:>18s}", end="")
    print()

    for name in ["random", "raster", "full_v3", "local_only", "no_readiness",
                 "readiness_only", "shuffled_B", "random_B", "graph_guided"]:
        r = results[name]
        print(f"  {name:<20s}", end="")
        for m in metrics:
            print(f"  {r[m]:18.4f}", end="")
        print()

    # Save
    out_json = {
        "checkpoint": str(CKPT_10K),
        "data": str(DATA),
        "N_images": N_IMAGES,
        "K_sample": K_SAMPLE,
        "block_size": BLOCK_SIZE,
        "num_blocks": NUM_BLOCKS,
        "readiness_signal_s": s,
        "mean_manhattan_topB": manh,
        "results": results,
    }
    with open(OUT_DIR / "diagnostic_results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # Save orders for full_v3
    np.save(OUT_DIR / "orders_full_v3_K100.npy", variants["full_v3"])

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Output: {OUT_DIR}/")
    print("DONE")


if __name__ == "__main__":
    main()
