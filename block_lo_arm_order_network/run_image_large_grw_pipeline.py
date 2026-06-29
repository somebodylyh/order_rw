"""Image-large Graph-RW pipeline: extract A → compute hyperparams → sample & diagnose.

One-shot script:
  1. Extract A_global (256×256) from patch2x2 baseline 20k checkpoint
  2. Build B = A^T, compute readiness signal s = std(r)/mean(|B|)
  3. Determine optimal hyperparams (adaptive rule from text/image ablation)
  4. Sample K=100 orders with progressive_rw_v3, compute full diagnostics
"""

import os, sys, time
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_NANO = _ROOT.parent / "nanogpt-learned-order"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_NANO))

CKPT = _NANO / "out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_random_baseline/ckpt.pt"
DATA = _NANO / "data/Imagenet64VQ_f4_800k_patch2x2/train.bin"
OUT_DIR = _ROOT / "probe_results_image_large"
DEVICE = "cuda:0"
N_IMAGES = 500
M_ORDERS_PER_IMAGE = 3
K_SAMPLE = 100


def extract_attention():
    """Step 1: extract A_global from checkpoint."""
    from extract_image_attention import load_model, extract_A_global

    out_dir = OUT_DIR / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"\n{'='*60}")
    print(f"Step 1: Extract A_global ({N_IMAGES} images × {M_ORDERS_PER_IMAGE} orders)")
    print(f"  ckpt: {CKPT}")
    print(f"  data: {DATA}")
    print(f"{'='*60}\n")

    model, model_args = load_model(str(CKPT), DEVICE)
    print(f"  Model: n_layer={model_args['n_layer']}, n_head={model_args['n_head']}, "
          f"n_embd={model_args['n_embd']}, block_size={model_args['block_size']}, "
          f"block_order_block_len={model_args.get('block_order_block_len', 'N/A')}")

    data = np.memmap(str(DATA), dtype=np.uint16, mode="r")
    n_total = len(data) // 256
    print(f"  Data: {n_total} images available")

    t0 = time.time()
    A = extract_A_global(model, data, N_IMAGES, DEVICE)
    elapsed = time.time() - t0

    np.save(out_dir / "A_global.npy", A)
    print(f"\n  Saved A_global.npy: shape={A.shape}, mean={A.mean():.6f}, "
          f"max={A.max():.6f}, time={elapsed:.1f}s")

    return A


def compute_hyperparams(A):
    """Step 2: Build B, compute readiness signal, determine optimal hyperparams."""
    from directed_graph_policy import build_directed_graph, compute_source

    print(f"\n{'='*60}")
    print(f"Step 2: Compute Graph-RW hyperparameters")
    print(f"{'='*60}\n")

    B = build_directed_graph(A)
    N = B.shape[0]

    # Save B
    out_dir = OUT_DIR / "attention"
    np.save(out_dir / "B_global.npy", B)

    # Readiness signal
    source, out_deg, in_deg = compute_source(B, alpha_dep=0.5)
    s = float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))

    print(f"  B shape: {B.shape}")
    print(f"  B mean: {B.mean():.6f}, B std: {B.std():.6f}")
    print(f"  out_deg: mean={out_deg.mean():.4f}, std={out_deg.std():.4f}")
    print(f"  in_deg:  mean={in_deg.mean():.4f}, std={in_deg.std():.4f}")
    print(f"  source:  mean={source.mean():.4f}, std={source.std():.4f}")
    print(f"  Readiness signal s = std(r)/mean(|B|) = {s:.4f}")

    # Spatial locality analysis (manhattan distance of top attention pairs)
    grid_size = 16  # 16×16 grid for 256 tokens
    rows = np.arange(N) // grid_size
    cols = np.arange(N) % grid_size

    # For patch2x2: tokens are rearranged. block_order_block_len=4 means
    # tokens [4k, 4k+1, 4k+2, 4k+3] form a 2×2 patch.
    # patch_idx in 8×8 grid, then 4 tokens within each patch.
    patch_grid = 8
    patch_rows = np.zeros(N, dtype=int)
    patch_cols = np.zeros(N, dtype=int)
    for i in range(N):
        patch_idx = i // 4
        patch_rows[i] = patch_idx // patch_grid
        patch_cols[i] = patch_idx % patch_grid
    # sub-patch position (0-3): 2×2 within patch
    sub_pos = np.arange(N) % 4
    # actual pixel-level position
    pix_rows = patch_rows * 2 + sub_pos // 2
    pix_cols = patch_cols * 2 + sub_pos % 2

    # Top-k attention pairs manhattan distance
    A_flat = A.copy()
    np.fill_diagonal(A_flat, 0)
    top_k_pairs = 1000
    flat_idx = np.argpartition(A_flat.ravel(), -top_k_pairs)[-top_k_pairs:]
    src_idx = flat_idx // N
    dst_idx = flat_idx % N
    manh_dists = np.abs(pix_rows[src_idx] - pix_rows[dst_idx]) + \
                 np.abs(pix_cols[src_idx] - pix_cols[dst_idx])
    mean_manh = float(manh_dists.mean())
    median_manh = float(np.median(manh_dists))

    print(f"\n  Spatial analysis (top-{top_k_pairs} attention pairs):")
    print(f"    mean manhattan dist: {mean_manh:.3f}")
    print(f"    median manhattan dist: {median_manh:.3f}")

    # Adaptive rule (from ablation results):
    # text: s > 1.0 → readiness_only (high rho, low lam)
    # image: s < 1.0 AND manh < 4.0 → local_only (high loc, low readiness)
    print(f"\n  Adaptive rule classification:")
    if s > 1.0:
        mode = "readiness_dominant"
        params = {
            "policy": "progressive_rw_v3",
            "tau_start": 0.10,
            "tau_step": 0.10,
            "alpha_dep": 0.5,
            "lam": 0.75,
            "rho": 0.5,
            "top_k": 4,
            "epsilon_uniform": 0.0,
        }
        print(f"    → readiness_dominant (s={s:.4f} > 1.0)")
    else:
        mode = "local_dominant"
        params = {
            "policy": "progressive_rw_v3",
            "tau_start": 0.10,
            "tau_step": 0.10,
            "alpha_dep": 0.5,
            "lam": 1.0,
            "rho": 0.1,
            "top_k": 4,
            "epsilon_uniform": 0.0,
        }
        print(f"    → local_dominant (s={s:.4f} <= 1.0, manh={mean_manh:.3f})")

    print(f"\n  Optimal Graph-RW params:")
    for k, v in params.items():
        print(f"    {k}: {v}")

    # Save params
    import json
    with open(OUT_DIR / "grw_params.json", "w") as f:
        json.dump({
            "mode": mode,
            "readiness_signal_s": s,
            "mean_manhattan": mean_manh,
            "median_manhattan": median_manh,
            "params": params,
        }, f, indent=2)

    return B, params, {"s": s, "mean_manh": mean_manh, "mode": mode}


def sample_and_diagnose(B, params):
    """Step 3: Sample K orders, compute diagnostics."""
    from directed_graph_policy import sample_order, compute_source, build_directed_graph

    print(f"\n{'='*60}")
    print(f"Step 3: Sample {K_SAMPLE} Graph-RW orders & diagnostics")
    print(f"{'='*60}\n")

    N = B.shape[0]
    policy = params.pop("policy", "progressive_rw_v3")

    t0 = time.time()
    orders = np.zeros((K_SAMPLE, N), dtype=np.int64)
    logprobs = np.zeros(K_SAMPLE)

    for k in range(K_SAMPLE):
        orders[k], logprobs[k] = sample_order(B, policy, params, seed=42*10000+k)
        if (k+1) % 20 == 0:
            print(f"  Sampled {k+1}/{K_SAMPLE}", flush=True)

    elapsed = time.time() - t0
    print(f"  Sampling done in {elapsed:.1f}s")

    # Legality check
    valid = sum(1 for k in range(K_SAMPLE)
                if sorted(orders[k].tolist()) == list(range(N)))
    print(f"\n  Legality: {valid}/{K_SAMPLE}")

    # τ vs L2R
    from scipy.stats import kendalltau
    L2R = np.arange(N)
    taus_l2r = np.array([kendalltau(orders[k], L2R)[0] for k in range(K_SAMPLE)])
    print(f"  τ vs L2R: mean={taus_l2r.mean():.4f}, std={taus_l2r.std():.4f}")

    # Pairwise τ (first 50)
    n_check = min(50, K_SAMPLE)
    pair_taus = []
    for i in range(n_check):
        for j in range(i+1, n_check):
            pair_taus.append(kendalltau(orders[i], orders[j])[0])
    pairwise_tau = float(np.mean(pair_taus))
    print(f"  Pairwise τ (first {n_check}): mean={pairwise_tau:.4f}")

    # First-node entropy
    first_nodes = orders[:, 0]
    counts = np.bincount(first_nodes, minlength=N).astype(float)
    probs = counts / K_SAMPLE
    probs_nz = probs[probs > 0]
    first_H = float(-np.sum(probs_nz * np.log(probs_nz)))
    print(f"  First-node entropy: {first_H:.4f}")

    # Spatial locality of generated orders
    patch_grid = 8
    patch_rows = np.zeros(N, dtype=int)
    patch_cols = np.zeros(N, dtype=int)
    for i in range(N):
        patch_idx = i // 4
        patch_rows[i] = patch_idx // patch_grid
        patch_cols[i] = patch_idx % patch_grid
    sub_pos = np.arange(N) % 4
    pix_rows = patch_rows * 2 + sub_pos // 2
    pix_cols = patch_cols * 2 + sub_pos % 2

    step_manh = []
    for k in range(K_SAMPLE):
        for t in range(N-1):
            u, v = orders[k, t], orders[k, t+1]
            step_manh.append(abs(pix_rows[u]-pix_rows[v]) + abs(pix_cols[u]-pix_cols[v]))
    mean_step_manh = float(np.mean(step_manh))
    print(f"  Mean step manhattan distance: {mean_step_manh:.3f}")

    # Logprob stats
    print(f"  Logprob: mean={logprobs.mean():.2f}, std={logprobs.std():.2f}")

    # Save
    out_dir = OUT_DIR / "grw_orders"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "orders_K100.npy", orders)
    np.save(out_dir / "logprobs_K100.npy", logprobs)

    import json
    diag = {
        "K": K_SAMPLE,
        "legality": f"{valid}/{K_SAMPLE}",
        "tau_vs_l2r_mean": float(taus_l2r.mean()),
        "tau_vs_l2r_std": float(taus_l2r.std()),
        "pairwise_tau_mean": pairwise_tau,
        "first_node_entropy": first_H,
        "mean_step_manhattan": mean_step_manh,
        "logprob_mean": float(logprobs.mean()),
        "logprob_std": float(logprobs.std()),
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n  Saved to {out_dir}/")

    return diag


if __name__ == "__main__":
    A = extract_attention()
    B, params, signal_info = compute_hyperparams(A)
    diag = sample_and_diagnose(B, params)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Readiness signal s: {signal_info['s']:.4f}")
    print(f"  Mode: {signal_info['mode']}")
    print(f"  Mean manhattan (top attn): {signal_info['mean_manh']:.3f}")
    print(f"  τ vs L2R: {diag['tau_vs_l2r_mean']:.4f}")
    print(f"  Pairwise τ: {diag['pairwise_tau_mean']:.4f}")
    print(f"  Step manhattan: {diag['mean_step_manhattan']:.3f}")
    print(f"  All outputs in: {OUT_DIR}/")
