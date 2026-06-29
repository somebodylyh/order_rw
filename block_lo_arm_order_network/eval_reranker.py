"""Evaluate Frozen Old ON Prior + Reranker against all baselines.

Flow:
1. Stage 1: Grid search alpha, beta, gamma for greedy linear reranker
2. Stage 2: Full comparison (L2R, random_N16, old_ON, old_ON+A-edge, old_ON+MLP, A-DP teacher)
3. Report primary metrics (AO-GPT NLL) + secondary (tau, swap) + tertiary (diversity)

Usage:
    python -u eval_reranker.py --device cuda:0
"""

import argparse
import os
import sys
import time
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RerankerConfig
from reranker import (
    load_old_on,
    load_aogpt,
    OldONPrior,
    build_reranker_features,
    StepWiseGreedyReranker,
    StepWiseMLPReranker,
    sequential_generate,
    compute_batch_full_order_nll,
    n16_batch_to_token_orders,
    n16_order_to_token_order,
    masks_to_revealed_bool,
    compute_kendall_tau,
    compute_swap_metrics,
    check_order_valid,
)
from dp_solver import solve_dp_batch
from train_reranker import load_token_chunks, phys_to_model_idx


# ── Stage 1: Grid search ──────────────────────────────────────────────────────


@torch.no_grad()
def grid_search_greedy(
    aogpt, old_on_prior, idx_batch, block_perm, inv_perm, A_batch, cfg
):
    """Grid search alpha, beta, gamma on tune_val_set. Minimize AOGPT NLL."""
    alphas = cfg.grid_alphas
    betas = cfg.grid_betas
    gammas = cfg.grid_gammas

    device = A_batch.device
    num_seqs = A_batch.shape[0]
    N = cfg.num_blocks

    best_nll = float("inf")
    best_params = None
    results_grid = []

    total_combos = len(alphas) * len(betas) * len(gammas)
    print(f"Grid search: {total_combos} combinations on {num_seqs} val seqs")

    for alpha, beta, gamma in tqdm(
        itertools.product(alphas, betas, gammas),
        total=total_combos,
        desc="Grid search",
    ):
        reranker = StepWiseGreedyReranker(alpha=alpha, beta=beta, gamma=gamma)
        nlls = []

        for s in range(num_seqs):
            A_s = A_batch[s:s + 1]
            sigma_old = old_on_prior.get_full_order(A_s)[0]

            order = sequential_generate(
                reranker, A_s, old_on_prior,
                sigma_old=sigma_old.unsqueeze(0),
                temperature=0.0,
                use_greedy_reranker=True,
            )[0]

            if not check_order_valid(order, N):
                nlls.append(float("inf"))
                continue

            idx_s = idx_batch[s:s + 1]
            nll = compute_batch_full_order_nll(
                aogpt, idx_s.expand(1, -1), block_perm, order.unsqueeze(0),
            ).item()
            nlls.append(nll)

        mean_nll = np.mean([n for n in nlls if np.isfinite(n)])
        results_grid.append({
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "nll": mean_nll,
        })

        if mean_nll < best_nll:
            best_nll = mean_nll
            best_params = (alpha, beta, gamma)

    print(f"Best grid params: α={best_params[0]}, β={best_params[1]}, γ={best_params[2]}, NLL={best_nll:.4f}")
    return best_params, results_grid


# ── Stage 2: Full comparison ──────────────────────────────────────────────────


@torch.no_grad()
def full_comparison(
    aogpt, old_on_prior, mlp_reranker, greedy_params,
    idx_batch, block_perm, inv_perm, A_batch, cfg,
):
    """Compare all methods on test set.

    Methods: L2R, random_N16 (MC), old_ON, old_ON+A-edge (greedy), old_ON+MLP, A-DP teacher
    """
    device = A_batch.device
    num_seqs = A_batch.shape[0]
    N = cfg.num_blocks

    # Results accumulators
    methods = ["L2R", "random_N16", "old_ON", "old_ON+A-edge", "old_ON+MLP", "A-DP_teacher"]
    results = {m: {"nlls": [], "taus": [], "swap": [], "invalid": 0} for m in methods}

    # Setup rerankers
    greedy = StepWiseGreedyReranker(*greedy_params)
    mlp_reranker.eval()

    # Pre-compute A-DP teacher orders
    print("Computing A-DP teacher orders...")
    A_np = A_batch.cpu().numpy()
    dp_results = solve_dp_batch(A_np, num_blocks=N, show_progress=False)
    dp_orders = torch.tensor([r["optimal_path"] for r in dp_results], dtype=torch.long, device=device)

    l2r_order = torch.arange(N, device=device).unsqueeze(0).expand(num_seqs, -1)

    # L2R NLL
    print("Evaluating L2R...")
    l2r_nlls = compute_batch_full_order_nll(
        aogpt, idx_batch, block_perm, l2r_order,
    )
    results["L2R"]["nlls"] = l2r_nlls.tolist()

    # Random N16 MC
    print(f"Evaluating random_N16 (MC={cfg.num_random_mc})...")
    for s in tqdm(range(num_seqs), desc="random_N16"):
        idx_s = idx_batch[s:s + 1]
        rand_orders = torch.stack([
            torch.randperm(N, device=device) for _ in range(cfg.num_random_mc)
        ])
        rand_nlls = compute_batch_full_order_nll(
            aogpt, idx_s.expand(cfg.num_random_mc, -1), block_perm, rand_orders,
        )
        results["random_N16"]["nlls"].append(rand_nlls.mean().item())

    # Old ON
    print("Evaluating old_ON...")
    for s in tqdm(range(num_seqs), desc="old_ON"):
        A_s = A_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]
        idx_s = idx_batch[s:s + 1]
        nll = compute_batch_full_order_nll(
            aogpt, idx_s.expand(1, -1), block_perm, sigma_old.unsqueeze(0),
        ).item()
        results["old_ON"]["nlls"].append(nll)

    # Old ON + A-edge (greedy linear reranker)
    print("Evaluating old_ON+A-edge...")
    for s in tqdm(range(num_seqs), desc="old_ON+A-edge"):
        A_s = A_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]

        order = sequential_generate(
            greedy, A_s, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0),
            temperature=0.0,
            use_greedy_reranker=True,
        )[0]

        if not check_order_valid(order, N):
            results["old_ON+A-edge"]["invalid"] += 1
            results["old_ON+A-edge"]["nlls"].append(float("nan"))
            continue

        idx_s = idx_batch[s:s + 1]
        nll = compute_batch_full_order_nll(
            aogpt, idx_s.expand(1, -1), block_perm, order.unsqueeze(0),
        ).item()
        results["old_ON+A-edge"]["nlls"].append(nll)

    # Old ON + MLP
    print("Evaluating old_ON+MLP...")
    for s in tqdm(range(num_seqs), desc="old_ON+MLP"):
        A_s = A_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]

        order = sequential_generate(
            mlp_reranker, A_s, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0),
            temperature=0.0,
            use_greedy_reranker=False,
        )[0]

        if not check_order_valid(order, N):
            results["old_ON+MLP"]["invalid"] += 1
            results["old_ON+MLP"]["nlls"].append(float("nan"))
            continue

        idx_s = idx_batch[s:s + 1]
        nll = compute_batch_full_order_nll(
            aogpt, idx_s.expand(1, -1), block_perm, order.unsqueeze(0),
        ).item()
        results["old_ON+MLP"]["nlls"].append(nll)

    # A-DP teacher
    print("Evaluating A-DP teacher...")
    dp_nlls = compute_batch_full_order_nll(
        aogpt, idx_batch, block_perm, dp_orders,
    )
    results["A-DP_teacher"]["nlls"] = dp_nlls.tolist()

    # Compute order quality metrics vs A-DP teacher
    print("Computing order quality metrics...")
    for s in tqdm(range(num_seqs), desc="Order quality"):
        A_s = A_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]
        dp_ref = dp_orders[s]

        # Old ON tau
        results["old_ON"]["taus"].append(compute_kendall_tau(sigma_old, dp_ref))
        results["old_ON"]["swap"].append(compute_swap_metrics(sigma_old, dp_ref))

        # Greedy + edge
        greedy_order = sequential_generate(
            greedy, A_s, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0),
            temperature=0.0,
            use_greedy_reranker=True,
        )[0]
        if check_order_valid(greedy_order, N):
            results["old_ON+A-edge"]["taus"].append(compute_kendall_tau(greedy_order, dp_ref))
            results["old_ON+A-edge"]["swap"].append(compute_swap_metrics(greedy_order, dp_ref))
        else:
            results["old_ON+A-edge"]["taus"].append(float("nan"))
            results["old_ON+A-edge"]["swap"].append({"pairwise_acc": float("nan"), "adjacent_repair": float("nan")})

        # MLP
        mlp_order = sequential_generate(
            mlp_reranker, A_s, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0),
            temperature=0.0,
            use_greedy_reranker=False,
        )[0]
        if check_order_valid(mlp_order, N):
            results["old_ON+MLP"]["taus"].append(compute_kendall_tau(mlp_order, dp_ref))
            results["old_ON+MLP"]["swap"].append(compute_swap_metrics(mlp_order, dp_ref))
        else:
            results["old_ON+MLP"]["taus"].append(float("nan"))
            results["old_ON+MLP"]["swap"].append({"pairwise_acc": float("nan"), "adjacent_repair": float("nan")})

        # L2R tau
        results["L2R"]["taus"].append(compute_kendall_tau(l2r_order[s], dp_ref))

    return results, dp_orders


# ── Report ─────────────────────────────────────────────────────────────────────

def print_comparison_table(results: Dict, cfg):
    """Print primary + secondary metrics comparison table."""
    methods = ["L2R", "random_N16", "old_ON", "old_ON+A-edge", "old_ON+MLP", "A-DP_teacher"]
    method_labels = ["L2R", "random_N16", "old_ON", "old_ON+A-edge", "old_ON+MLP", "A-DP_teacher"]

    # Primary: AO-GPT NLL
    print("\n" + "=" * 80)
    print("Primary: AO-GPT Utility (NLL)")
    print("-" * 80)
    print(f"{'Method':<20} {'NLL':>8} {'Δ vs random_N16':>16} {'Δ vs old_ON':>14}")
    print("-" * 80)

    old_on_nll = np.nanmean(results["old_ON"]["nlls"])
    random_nll = np.nanmean(results["random_N16"]["nlls"])

    for m, label in zip(methods, method_labels):
        nlls = [n for n in results[m]["nlls"] if np.isfinite(n)]
        if nlls:
            mean_nll = np.mean(nlls)
            delta_random = random_nll - mean_nll
            delta_old_on = old_on_nll - mean_nll
            print(f"{label:<20} {mean_nll:>8.4f} {delta_random:>+16.4f} {delta_old_on:>+14.4f}")
        else:
            print(f"{label:<20} {'N/A':>8}")

    # Secondary: Order Quality
    print("\n" + "=" * 80)
    print("Secondary: Order Quality vs A-DP teacher (diagnostic only)")
    print("-" * 80)
    print(f"{'Method':<20} {'Kendall τ':>10} {'Pairwise Acc':>13} {'Adj Repair':>11} {'Invalid':>8}")
    print("-" * 80)

    for m, label in zip(methods, method_labels):
        taus = [t for t in results[m].get("taus", []) if np.isfinite(t)]
        swaps = results[m].get("swap", [])
        pw_accs = [s["pairwise_acc"] for s in swaps if np.isfinite(s["pairwise_acc"])]
        repairs = [s["adjacent_repair"] for s in swaps if np.isfinite(s["adjacent_repair"])]
        invalid = results[m].get("invalid", 0)

        tau_str = f"{np.mean(taus):.4f}" if taus else "N/A"
        pw_str = f"{np.mean(pw_accs):.4f}" if pw_accs else "N/A"
        rep_str = f"{np.mean(repairs):.4f}" if repairs else "N/A"
        print(f"{label:<20} {tau_str:>10} {pw_str:>13} {rep_str:>11} {invalid:>8}")

    # Success criteria
    print("\n" + "=" * 80)
    print("Success Criteria")
    print("-" * 80)
    mlp_nll = np.nanmean(results["old_ON+MLP"]["nlls"])
    mlp_vs_old = old_on_nll - mlp_nll
    mlp_vs_random = random_nll - mlp_nll
    mlp_invalid = results["old_ON+MLP"].get("invalid", 0)

    print(f"  1. MLP NLL < old_ON NLL:         {mlp_vs_old > 0}  (Δ={mlp_vs_old:+.4f})")
    print(f"  2. MLP NLL < random_N16 NLL:      {mlp_vs_random > 0}  (Δ={mlp_vs_random:+.4f})")
    print(f"  3. Invalid rate = 0:              {mlp_invalid == 0}  (count={mlp_invalid})")
    print(f"  All pass: {mlp_vs_old > 0 and mlp_vs_random > 0 and mlp_invalid == 0}")

    # Diversity (tertiary)
    print("\n" + "=" * 80)
    print("Tertiary: Order Diversity (old_ON+MLP at different temperatures)")
    print("-" * 80)
    # Computed separately in diversity_eval if needed
    print("  (see diversity_eval output)")


@torch.no_grad()
def diversity_eval(mlp_reranker, old_on_prior, A_batch, cfg, num_seqs=50):
    """Evaluate order diversity at different temperatures."""
    device = A_batch.device
    N = cfg.num_blocks

    for temp in cfg.eval_temperatures:
        unique_orders = set()
        for s in range(min(num_seqs, A_batch.shape[0])):
            A_s = A_batch[s:s + 1]
            sigma_old = old_on_prior.get_full_order(A_s)[0]
            for _ in range(10):
                order = sequential_generate(
                    mlp_reranker, A_s, old_on_prior,
                    sigma_old=sigma_old.unsqueeze(0),
                    temperature=temp,
                    use_greedy_reranker=False,
                )[0]
                if check_order_valid(order, N):
                    unique_orders.add(tuple(order.tolist()))

        print(f"  T={temp:.1f}: {len(unique_orders)} unique orders from {min(num_seqs, A_batch.shape[0]) * 10} samples")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mlp-ckpt", type=str, default="probe_results/reranker/mlp_reranker.pt")
    parser.add_argument("--output-dir", type=str, default="probe_results/reranker")
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()

    cfg = RerankerConfig()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Load models ──
    print("Loading models...")
    old_on = load_old_on(cfg.old_on_ckpt, device)
    old_on_prior = OldONPrior(old_on, num_blocks=cfg.num_blocks)

    aogpt_ckpt = os.path.expanduser(cfg.aogpt_ckpt)
    aogpt, block_perm, inv_perm = load_aogpt(aogpt_ckpt, device)
    block_perm = block_perm.to(device)
    inv_perm = inv_perm.to(device)

    # Load MLP reranker
    mlp_ckpt = torch.load(args.mlp_ckpt, map_location=device, weights_only=False)
    mlp_reranker = StepWiseMLPReranker(
        feature_dim=7,
        hidden_dim=cfg.adapter_hidden_dim,
        num_layers=cfg.adapter_num_layers,
        dropout=cfg.adapter_dropout,
    ).to(device)
    mlp_reranker.load_state_dict(mlp_ckpt["model_state_dict"])
    mlp_reranker.eval()
    print(f"Loaded MLP reranker from {args.mlp_ckpt}")

    # ── Load data ──
    print("Loading data...")
    tokenizer_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--gpt2/snapshots/"
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    wikitext_dir = os.path.expanduser(
        "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3"
    )

    A_all = np.load(cfg.data_path, mmap_mode="r")
    total_needed = cfg.num_train_seqs + cfg.num_tune_seqs + cfg.num_test_seqs
    total_seqs = min(A_all.shape[0], total_needed)
    A_all = torch.from_numpy(A_all[:total_seqs].copy()).float().to(device)
    idx_all = load_token_chunks(total_seqs, tokenizer_dir, wikitext_dir).to(device)

    n_train = cfg.num_train_seqs
    n_tune = cfg.num_tune_seqs
    n_test = cfg.num_test_seqs

    A_tune = A_all[n_train:n_train + n_tune]
    A_test = A_all[n_train + n_tune:n_train + n_tune + n_test]

    idx_tune = idx_all[n_train:n_train + n_tune]
    idx_test = idx_all[n_train + n_tune:n_train + n_tune + n_test]

    print(f"Tune: {n_tune} seqs, Test: {n_test} seqs")

    # ── Stage 1: Grid search on tune_val_set ──
    if not args.skip_grid:
        print("\n=== Stage 1: Grid Search (Greedy Reranker) ===")
        t0 = time.time()
        best_params, grid_results = grid_search_greedy(
            aogpt, old_on_prior, idx_tune, block_perm, inv_perm, A_tune, cfg
        )
        print(f"Grid search took {time.time() - t0:.1f}s")

        # Save grid results
        grid_path = os.path.join(args.output_dir, "grid_results.pt")
        torch.save({"best_params": best_params, "grid": grid_results}, grid_path)
        print(f"Saved grid results to {grid_path}")
    else:
        # Use defaults
        best_params = (0.5, 0.25, 0.25)
        print(f"Using default grid params: {best_params}")

    # ── Stage 2: Full comparison on test set ──
    print("\n=== Stage 2: Full Comparison ===")
    t0 = time.time()
    results, dp_orders = full_comparison(
        aogpt, old_on_prior, mlp_reranker, best_params,
        idx_test, block_perm, inv_perm, A_test, cfg,
    )
    print(f"Full comparison took {time.time() - t0:.1f}s")

    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.pt")
    torch.save({"results": results, "dp_orders": dp_orders, "cfg": cfg}, results_path)
    print(f"Saved eval results to {results_path}")

    # ── Report ──
    print_comparison_table(results, cfg)

    # ── Diversity ──
    print("\n=== Order Diversity ===")
    diversity_eval(mlp_reranker, old_on_prior, A_test, cfg)


if __name__ == "__main__":
    main()
