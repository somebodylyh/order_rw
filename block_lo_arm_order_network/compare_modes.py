"""Compare all structural greedy baselines: edge, prefix_mean, prefix_max, hybrid.

Also runs grid search over λ_last, λ_mean, λ_max for hybrid mode.
No training, no AO-GPT — pure attention-structural evaluation.
"""

import argparse
import os
import sys
import time
import itertools
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reranker import (
    load_old_on,
    OldONPrior,
    generate_order_A_greedy,
    compute_path_weight,
    check_order_valid,
)


def grid_search_hybrid(
    A_batch: torch.Tensor,
    old_on_prior: OldONPrior,
    lambda_grid: list = [0.0, 0.5, 1.0, 2.0],
    mode: str = "edge",
    num_seqs: int = 50,
):
    """Grid search λ_last, λ_mean, λ_max for hybrid prefix scoring.

    Evaluates all combos on num_seqs, returns best lambdas and results.
    """
    A_sub = A_batch[:num_seqs]
    A_np = A_sub.cpu().numpy()
    device = A_batch.device

    combos = list(itertools.product(lambda_grid, lambda_grid))
    # Only test non-zero combos (skip 0,0 which is degenerate)
    combos = [c for c in combos if c[0] + c[1] > 0]

    print(f"Grid search over λ_last × λ_mean: {len(combos)} combos on {num_seqs} seqs")

    # For hybrid, keep λ_max = 0.5 * λ_mean initially (quick scan)
    best_weight = -np.inf
    best_lambdas = None
    all_combos = []

    for lam_last, lam_mean in tqdm(combos, desc="Grid search"):
        lam_max = 0.5 * lam_mean  # heuristic: max has smaller weight
        lambdas = (lam_last, lam_mean, lam_max)

        orders = generate_order_A_greedy(
            A_sub, mode="hybrid", hybrid_lambdas=lambdas
        )  # (num_seqs, N)

        weights = []
        for b in range(num_seqs):
            w = compute_path_weight(A_np[b], orders[b].cpu().numpy(), mode=mode)
            weights.append(w)
        avg_weight = np.mean(weights)

        all_combos.append({
            "lam_last": lam_last,
            "lam_mean": lam_mean,
            "lam_max": lam_max,
            "avg_weight": avg_weight,
            "std_weight": float(np.std(weights)),
        })

        if avg_weight > best_weight:
            best_weight = avg_weight
            best_lambdas = lambdas

    # Sort by weight descending
    all_combos.sort(key=lambda x: x["avg_weight"], reverse=True)

    # Also compute baseline single-mode weights
    baselines = {}
    for m in ["edge", "prefix_mean", "prefix_max"]:
        orders = generate_order_A_greedy(A_sub, mode=m)
        weights = [compute_path_weight(A_np[b], orders[b].cpu().numpy(), mode=mode)
                   for b in range(num_seqs)]
        baselines[m] = {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
        }

    # Old ON
    if old_on_prior is not None:
        old_on_weights = []
        for b in range(num_seqs):
            order = old_on_prior.get_full_order(A_sub[b:b+1])[0].cpu().numpy()
            old_on_weights.append(compute_path_weight(A_np[b], order, mode=mode))
        baselines["old_on"] = {
            "mean": float(np.mean(old_on_weights)),
            "std": float(np.std(old_on_weights)),
        }

    # Random
    rand_weights = []
    for b in range(num_seqs):
        rand_ws = [compute_path_weight(A_np[b], np.random.permutation(16), mode=mode)
                   for _ in range(20)]
        rand_weights.append(np.mean(rand_ws))
    baselines["random"] = {
        "mean": float(np.mean(rand_weights)),
        "std": float(np.std(rand_weights)),
    }

    return {
        "best_lambdas": best_lambdas,
        "best_weight": best_weight,
        "top5_combos": all_combos[:5],
        "baselines": baselines,
    }


def compare_all_modes(
    A_batch: torch.Tensor,
    old_on_prior: OldONPrior = None,
    eval_mode: str = "edge",
    hybrid_lambdas: tuple = None,
    num_seqs: int = None,
):
    """Compare all greedy modes + old ON + random on path weight."""
    if num_seqs is not None:
        A_batch = A_batch[:num_seqs]
    B = len(A_batch)
    A_np = A_batch.cpu().numpy()

    modes_to_test = {
        "edge": ("edge", None),
        "prefix_mean": ("prefix_mean", None),
        "prefix_max": ("prefix_max", None),
    }
    if hybrid_lambdas is not None:
        modes_to_test["hybrid"] = ("hybrid", hybrid_lambdas)

    results = {}
    for name, (mode, lam) in modes_to_test.items():
        orders = generate_order_A_greedy(A_batch, mode=mode, hybrid_lambdas=lam or (1.0, 0.5, 0.3))
        weights = []
        for b in range(B):
            w = compute_path_weight(A_np[b], orders[b].cpu().numpy(), mode=eval_mode)
            weights.append(w)
        results[name] = {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
        }

    # Old ON
    if old_on_prior is not None:
        old_on_weights = []
        for b in range(B):
            order = old_on_prior.get_full_order(A_batch[b:b+1])[0].cpu().numpy()
            old_on_weights.append(compute_path_weight(A_np[b], order, mode=eval_mode))
        results["old_on"] = {
            "mean": float(np.mean(old_on_weights)),
            "std": float(np.std(old_on_weights)),
        }

    # Random
    rand_weights = []
    for b in range(B):
        rand_ws = [compute_path_weight(A_np[b], np.random.permutation(16), mode=eval_mode)
                   for _ in range(20)]
        rand_weights.append(np.mean(rand_ws))
    results["random"] = {
        "mean": float(np.mean(rand_weights)),
        "std": float(np.std(rand_weights)),
    }

    # Print table
    print(f"\n{'='*60}")
    print(f"Mode Comparison (eval_mode={eval_mode}, {B} seqs)")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Path Weight':>12} {'Δ vs Random':>12} {'Δ vs Edge':>12}")
    print(f"{'-'*60}")
    rand_mean = results["random"]["mean"]
    edge_mean = results.get("edge", {}).get("mean", rand_mean)
    for name in ["edge", "prefix_mean", "prefix_max", "hybrid", "old_on", "random"]:
        if name in results:
            r = results[name]
            delta_rand = r["mean"] - rand_mean
            delta_edge = r["mean"] - edge_mean
            print(f"{name:<20} {r['mean']:>12.4f} {delta_rand:>+12.4f} {delta_edge:>+12.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-path", type=str,
                        default="probe_results/A_train_10k.npy")
    parser.add_argument("--old-on-ckpt", type=str,
                        default="probe_results/crossattn_on_best.pt")
    parser.add_argument("--num-seqs", type=int, default=200)
    parser.add_argument("--grid-seqs", type=int, default=50,
                        help="Seqs for grid search (0 to skip)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print("Loading A matrices...")
    A_all = np.load(args.data_path, mmap_mode="r")
    total = min(A_all.shape[0], args.num_seqs)
    A_batch = torch.from_numpy(A_all[:total].copy()).float().to(device)

    print("Loading Old ON...")
    old_on = load_old_on(args.old_on_ckpt, device)
    old_on_prior = OldONPrior(old_on, num_blocks=16)

    # ── Grid search for hybrid lambdas ──
    hybrid_lambdas = None
    if args.grid_seqs > 0:
        print("\n=== Grid Search: Hybrid λ_last × λ_mean ===")
        grid_result = grid_search_hybrid(
            A_batch, old_on_prior,
            lambda_grid=[0.0, 0.5, 1.0, 2.0],
            mode="edge",
            num_seqs=args.grid_seqs,
        )
        hybrid_lambdas = grid_result["best_lambdas"]
        print(f"\nBest hybrid lambdas: λ_last={hybrid_lambdas[0]}, "
              f"λ_mean={hybrid_lambdas[1]}, λ_max={hybrid_lambdas[2]}")
        print(f"Best weight: {grid_result['best_weight']:.4f}")
        print(f"\nTop 5:")
        for c in grid_result["top5_combos"]:
            print(f"  λ=({c['lam_last']}, {c['lam_mean']}, {c['lam_max']}): "
                  f"W={c['avg_weight']:.4f}")
        print(f"\nBaselines:")
        for k, v in grid_result["baselines"].items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # ── Full comparison ──
    print("\n=== Full Comparison ===")
    results = compare_all_modes(
        A_batch, old_on_prior,
        eval_mode="edge",
        hybrid_lambdas=hybrid_lambdas,
    )

    # Save
    out_dir = "probe_results/reranker"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(results, os.path.join(out_dir, "mode_comparison.pt"))
    print(f"\nSaved to {out_dir}/mode_comparison.pt")


if __name__ == "__main__":
    main()
