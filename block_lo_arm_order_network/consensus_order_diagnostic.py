"""Consensus order diagnostic.

For a given policy, sample K orders, compute average rank per block,
sort by average rank → consensus order.

Reports:
  - tau_vs_L2R (consensus, single-sample mean/std)
  - pairwise_tau among single-sample orders
  - first-node entropy
  - optional frozen CE/MSE evaluation
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from directed_graph_policy import (
    build_directed_graph,
    compute_source,
    sample_order,
    _softmax,
)
from order_diagnostics import _kendall_tau


def consensus_order(orders: np.ndarray) -> np.ndarray:
    """Compute consensus order from K sampled orders.

    average_rank[v] = mean position of block v across K orders.
    consensus = sort blocks by average_rank ascending.
    """
    K, N = orders.shape
    ranks = np.argsort(orders, axis=1).astype(np.float64)  # (K, N): rank of each block
    # Compute position of each block in each order
    positions = np.zeros((K, N), dtype=np.float64)
    for k in range(K):
        positions[k, orders[k]] = np.arange(N, dtype=np.float64)
    avg_pos = positions.mean(axis=0)  # (N,)
    consensus = np.argsort(avg_pos).astype(np.int64)
    return consensus, avg_pos


def run_text_diagnostic(
    B: np.ndarray,
    policy: str,
    params: dict,
    K: int = 100,
    seed_base: int = 20260513,
):
    """Run consensus diagnostic for text-side policy."""
    N = B.shape[0]
    l2r = np.arange(N, dtype=np.int64)

    # Sample K orders
    print(f"Sampling K={K} orders with policy={policy}...")
    t0 = time.time()
    orders = np.zeros((K, N), dtype=np.int64)
    for k in range(K):
        seed = seed_base * 10000 + k
        orders[k], _ = sample_order(B, policy, params, seed=seed)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/K*1000:.1f}ms per order)")

    # Compute consensus
    consensus, avg_pos = consensus_order(orders)
    consensus_tau_l2r = _kendall_tau(consensus, l2r)

    # Single-sample stats
    taus_vs_l2r = np.array([_kendall_tau(orders[k], l2r) for k in range(K)])
    taus_vs_consensus = np.array([_kendall_tau(orders[k], consensus) for k in range(K)])

    # Pairwise tau among samples
    n_pairs = min(500, K * (K - 1) // 2)
    pair_rng = np.random.default_rng(seed_base)
    all_i, all_j = np.triu_indices(K, k=1)
    if len(all_i) > n_pairs:
        idx = pair_rng.choice(len(all_i), size=n_pairs, replace=False)
        all_i = all_i[idx]
        all_j = all_j[idx]
    pairwise_taus = np.array([_kendall_tau(orders[i], orders[j]) for i, j in zip(all_i, all_j)])

    # First-node entropy
    first_nodes = orders[:, 0]
    counts = np.bincount(first_nodes, minlength=N).astype(np.float64)
    probs = counts / K
    probs_nz = probs[probs > 0]
    first_node_entropy = float(-np.sum(probs_nz * np.log(probs_nz)))
    consensus_first = int(consensus[0])

    # Entropy of per-step distributions (replay first few orders)
    tau_step = float(params.get("tau_step", 0.1))
    top_k = int(params.get("top_k", 0) or 0)
    alpha_dep = float(params.get("alpha_dep", 0.5))
    source, _, _ = compute_source(B, alpha_dep)

    print(f"\n{'='*60}")
    print(f"Consensus Order Diagnostic — Text (policy={policy})")
    print(f"{'='*60}")
    print(f"  K                              = {K}")
    print(f"  N                              = {N}")
    print(f"  consensus tau vs L2R           = {consensus_tau_l2r:.4f}")
    print(f"  single-sample tau vs L2R       = {taus_vs_l2r.mean():.4f} ± {taus_vs_l2r.std():.4f}")
    print(f"  single-sample tau vs consensus = {taus_vs_consensus.mean():.4f} ± {taus_vs_consensus.std():.4f}")
    print(f"  pairwise tau (single-sample)   = {pairwise_taus.mean():.4f} ± {pairwise_taus.std():.4f}")
    print(f"  first-node entropy             = {first_node_entropy:.4f} (max ln(N)={np.log(N):.4f})")
    print(f"  consensus first block          = {consensus_first}")
    print(f"  L2R first block                = 0")
    print(f"  consensus top-8                 = {consensus[:8].tolist()}")
    print(f"  single-sample top-8 (first 3)   =")
    for k in range(min(3, K)):
        print(f"    sample[{k}]: {orders[k, :8].tolist()}")

    return {
        "policy": policy,
        "K": K, "N": N,
        "consensus_tau_vs_l2r": float(consensus_tau_l2r),
        "single_tau_vs_l2r_mean": float(taus_vs_l2r.mean()),
        "single_tau_vs_l2r_std": float(taus_vs_l2r.std()),
        "pairwise_tau_mean": float(pairwise_taus.mean()),
        "pairwise_tau_std": float(pairwise_taus.std()),
        "first_node_entropy": float(first_node_entropy),
        "consensus_first_block": int(consensus_first),
        "consensus_top8": consensus[:8].tolist(),
    }


def run_image_diagnostic(
    A_global: np.ndarray,
    K: int = 100,
    seed_base: int = 20260513,
):
    """Run consensus diagnostic for image top4 policy."""
    N = A_global.shape[0]
    A_sym = 0.5 * (A_global + A_global.T)
    np.fill_diagonal(A_sym, 0.0)
    B = build_directed_graph(A_sym)

    # Image uses top-4 self-avoiding RW
    params = {
        "tau_start": 0.1,
        "tau_step": 0.1,
        "alpha_dep": 0.5,
        "alpha_pr": 0.85,
        "top_k": 4,
        "epsilon_uniform": 0.0,
    }
    policy = "self_avoiding_rw"

    print(f"Sampling K={K} image orders with top_k=4 self_avoiding_rw...")
    t0 = time.time()
    orders = np.zeros((K, N), dtype=np.int64)
    for k in range(K):
        seed = seed_base * 10000 + k
        orders[k], _ = sample_order(B, policy, params, seed=seed)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/K*1000:.1f}ms per order)")

    # Consensus
    consensus, avg_pos = consensus_order(orders)
    raster = np.arange(N, dtype=np.int64)
    consensus_tau_raster = _kendall_tau(consensus, raster)

    # Single-sample stats
    taus_vs_raster = np.array([_kendall_tau(orders[k], raster) for k in range(K)])
    taus_vs_consensus = np.array([_kendall_tau(orders[k], consensus) for k in range(K)])

    # Pairwise
    n_pairs = min(500, K * (K - 1) // 2)
    pair_rng = np.random.default_rng(seed_base)
    all_i, all_j = np.triu_indices(K, k=1)
    if len(all_i) > n_pairs:
        idx = pair_rng.choice(len(all_i), size=n_pairs, replace=False)
        all_i = all_i[idx]
        all_j = all_j[idx]
    pairwise_taus = np.array([_kendall_tau(orders[i], orders[j]) for i, j in zip(all_i, all_j)])

    # First-node entropy
    first_nodes = orders[:, 0]
    counts = np.bincount(first_nodes, minlength=N).astype(np.float64)
    probs = counts / K
    probs_nz = probs[probs > 0]
    first_node_entropy = float(-np.sum(probs_nz * np.log(probs_nz)))

    # Mean Manhattan distance among singles
    manh_distances = []
    for i in range(min(500, K)):
        for j in range(i + 1, min(500, K)):
            manh = float(np.abs(orders[i] - orders[j]).sum() / N)
            manh_distances.append(manh)
    mean_manh = float(np.mean(manh_distances)) if manh_distances else 0.0

    print(f"\n{'='*60}")
    print(f"Consensus Order Diagnostic — Image (top_k=4)")
    print(f"{'='*60}")
    print(f"  K                              = {K}")
    print(f"  N                              = {N}")
    print(f"  consensus tau vs raster        = {consensus_tau_raster:.4f}")
    print(f"  single-sample tau vs raster    = {taus_vs_raster.mean():.4f} ± {taus_vs_raster.std():.4f}")
    print(f"  single-sample tau vs consensus = {taus_vs_consensus.mean():.4f} ± {taus_vs_consensus.std():.4f}")
    print(f"  pairwise tau (single-sample)   = {pairwise_taus.mean():.4f} ± {pairwise_taus.std():.4f}")
    print(f"  mean Manhattan dist (single)   = {mean_manh:.4f}")
    print(f"  first-node entropy             = {first_node_entropy:.4f} (max ln(N)={np.log(N):.4f})")
    print(f"  consensus top-8                 = {consensus[:8].tolist()}")
    print(f"  raster top-8                    = {raster[:8].tolist()}")

    return {
        "policy": "image_top4",
        "K": K, "N": N,
        "consensus_tau_vs_raster": float(consensus_tau_raster),
        "single_tau_vs_raster_mean": float(taus_vs_raster.mean()),
        "single_tau_vs_raster_std": float(taus_vs_raster.std()),
        "pairwise_tau_mean": float(pairwise_taus.mean()),
        "pairwise_tau_std": float(pairwise_taus.std()),
        "mean_manhattan_single": float(mean_manh),
        "first_node_entropy": float(first_node_entropy),
        "consensus_top8": consensus[:8].tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-a-path", default="probe_results/A_train_n64_10k.npy")
    p.add_argument("--image-a-path", default="probe_results/A_global_from_ckpt20000_image.npy")
    p.add_argument("--policy", default="progressive_rw_v3")
    p.add_argument("--lam", type=float, default=0.75)
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--tau-start", type=float, default=0.1)
    p.add_argument("--tau-step", type=float, default=0.1)
    p.add_argument("--K", type=int, default=100, help="Number of order samples")
    p.add_argument("--output", default=None)
    p.add_argument("--skip-image", action="store_true")
    args = p.parse_args()

    results = {}

    # ---- Text ----
    a_path = Path(args.text_a_path)
    if a_path.exists():
        A_all = np.load(a_path)
        A_global = A_all.mean(axis=0).astype(np.float32)
        np.fill_diagonal(A_global, 0.0)
        B = build_directed_graph(A_global)
        print(f"Loaded text A: {A_all.shape} → global ({B.shape[0]}×{B.shape[1]})")
    else:
        print(f"ERROR: text A path not found: {a_path}")
        sys.exit(1)

    params = {
        "tau_start": args.tau_start,
        "tau_step": args.tau_step,
        "alpha_dep": 0.5,
        "alpha_pr": 0.85,
        "top_k": args.top_k,
        "epsilon_uniform": 0.0,
        "lam": args.lam,
        "rho": args.rho,
    }

    results["text"] = run_text_diagnostic(B, args.policy, params, K=args.K)

    # ---- Image ----
    img_path = Path(args.image_a_path)
    if not args.skip_image and img_path.exists():
        A_img = np.load(img_path)
        print(f"\nLoaded image A: {A_img.shape}")
        results["image"] = run_image_diagnostic(A_img, K=args.K)
    else:
        print(f"\nImage A not found at {img_path}, skipping image diagnostic.")

    # Write output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
