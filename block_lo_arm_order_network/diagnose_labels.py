"""Label quality diagnostics for Frozen ON + AO-GPT Reranker.

Runs 5 analyses:
1. Candidate NLL gap distribution (best vs median, 2nd-best vs best)
2. Soft label entropy / effective candidates
3. Top-1 oracle upper bound (per-step best candidate → full-order NLL)
4. Stage 1 A-edge grid search (small subset)
5. Bootstrap CI for ΔNLL(MLP - old_ON) and ΔNLL(MLP - random)

Usage:
    python -u diagnose_labels.py --device cuda:0
"""

import argparse
import os
import sys
import time
import math
import itertools
import numpy as np
import torch
import torch.nn.functional as F
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
    n16_order_to_token_order,
    masks_to_revealed_bool,
    check_order_valid,
    compute_kendall_tau,
)
from dp_solver import solve_dp_batch
from train_reranker import load_token_chunks


# ── Helper: compute per-sequence ΔNLLs for bootstrap ──────────────────────────

@torch.no_grad()
def compute_per_seq_nlls(
    aogpt, old_on_prior, mlp_reranker,
    idx_batch, block_perm, A_batch, cfg,
):
    """Compute per-sequence NLL for old_ON, MLP, and random_N16 (MC avg)."""
    device = A_batch.device
    N = cfg.num_blocks
    num_seqs = A_batch.shape[0]

    old_on_nlls = []
    mlp_nlls = []
    random_nlls = []

    for s in tqdm(range(num_seqs), desc="Per-seq NLLs"):
        A_s = A_batch[s:s + 1]
        idx_s = idx_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]

        # Old ON
        old_nll = compute_batch_full_order_nll(
            aogpt, idx_s, block_perm, sigma_old.unsqueeze(0),
        ).item()
        old_on_nlls.append(old_nll)

        # MLP
        mlp_order = sequential_generate(
            mlp_reranker, A_s, old_on_prior,
            sigma_old=sigma_old.unsqueeze(0),
            temperature=0.0,
            use_greedy_reranker=False,
        )[0]
        if check_order_valid(mlp_order, N):
            mlp_nll = compute_batch_full_order_nll(
                aogpt, idx_s, block_perm, mlp_order.unsqueeze(0),
            ).item()
        else:
            mlp_nll = float("nan")
        mlp_nlls.append(mlp_nll)

        # Random N16 MC
        rand_orders = torch.stack([
            torch.randperm(N, device=device) for _ in range(cfg.num_random_mc)
        ])
        rand_batch = compute_batch_full_order_nll(
            aogpt, idx_s.expand(cfg.num_random_mc, -1), block_perm, rand_orders,
        )
        random_nlls.append(rand_batch.mean().item())

    return {
        "old_on": np.array(old_on_nlls),
        "mlp": np.array(mlp_nlls),
        "random": np.array(random_nlls),
    }


def bootstrap_ci(per_seq, key_a, key_b, n_bootstrap=10000, seed=42):
    """Bootstrap CI for mean ΔNLL(key_a - key_b)."""
    rng = np.random.RandomState(seed)
    a = per_seq[key_a]
    b = per_seq[key_b]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    deltas = a - b
    n = len(deltas)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means.append(deltas[idx].mean())
    means = np.sort(means)
    return {
        "mean": deltas.mean(),
        "ci_lower": means[int(0.025 * n_bootstrap)],
        "ci_upper": means[int(0.975 * n_bootstrap)],
        "n_valid": n,
        "n_total": len(per_seq[key_a]),
    }


# ── 1. Candidate NLL gap distribution ─────────────────────────────────────────

def analyze_label_gaps(labels, cfg):
    """Compute per-step NLL gap statistics."""
    gaps_best_vs_median = []
    gaps_2nd_vs_best = []
    nll_ranges = []

    for lab in labels:
        nlls = lab["nll_targets"].numpy()
        K = len(nlls)
        if K < 2:
            continue
        sorted_nlls = np.sort(nlls)
        best = sorted_nlls[0]
        median = sorted_nlls[K // 2]
        second_best = sorted_nlls[1] if K >= 2 else None

        gaps_best_vs_median.append(best - median)
        if second_best is not None:
            gaps_2nd_vs_best.append(second_best - best)
        nll_ranges.append(sorted_nlls[-1] - sorted_nlls[0])

    gaps_bvm = np.array(gaps_best_vs_median)
    gaps_2vb = np.array(gaps_2nd_vs_best)
    nll_ranges = np.array(nll_ranges)

    print("\n" + "=" * 60)
    print("1. Candidate NLL Gap Distribution")
    print("-" * 60)
    print(f"  Queries analyzed: {len(gaps_bvm)}")
    print()
    print(f"  Best - Median NLL:")
    print(f"    mean:  {gaps_bvm.mean():.6f}")
    print(f"    std:   {gaps_bvm.std():.6f}")
    print(f"    min:   {gaps_bvm.min():.6f}")
    print(f"    p25:   {np.percentile(gaps_bvm, 25):.6f}")
    print(f"    p50:   {np.percentile(gaps_bvm, 50):.6f}")
    print(f"    p75:   {np.percentile(gaps_bvm, 75):.6f}")
    print(f"    max:   {gaps_bvm.max():.6f}")
    frac_positive = (gaps_bvm < -0.001).mean()
    print(f"    fraction < -0.001: {frac_positive:.3f} (best clearly better)")
    print()
    print(f"  2nd Best - Best NLL:")
    print(f"    mean:  {gaps_2vb.mean():.6f}")
    print(f"    std:   {gaps_2vb.std():.6f}")
    print(f"    p50:   {np.percentile(gaps_2vb, 50):.6f}")
    print(f"    p90:   {np.percentile(gaps_2vb, 90):.6f}")
    print()
    print(f"  NLL range (max - min per query):")
    print(f"    mean:  {nll_ranges.mean():.6f}")
    print(f"    p50:   {np.percentile(nll_ranges, 50):.6f}")
    print(f"    p90:   {np.percentile(nll_ranges, 90):.6f}")

    # Signal-to-noise ratio: gap / range
    snr = np.abs(gaps_bvm) / (nll_ranges + 1e-8)
    print()
    print(f"  |best-median| / range SNR:")
    print(f"    mean:  {snr.mean():.6f}")
    print(f"    p50:   {np.percentile(snr, 50):.6f}")

    return {
        "gaps_best_vs_median": gaps_bvm,
        "gaps_2nd_vs_best": gaps_2vb,
        "nll_ranges": nll_ranges,
    }


# ── 2. Soft label entropy ─────────────────────────────────────────────────────

def analyze_label_entropy(labels, cfg):
    """Compute soft label entropy and effective candidates per step."""
    entropies = []
    effective_candidates = []
    max_entropy_frac = []  # H / H_max
    n_candidates_list = []

    for lab in labels:
        nlls = lab["nll_targets"]
        K = len(nlls)
        if K < 2:
            continue

        q_raw = F.softmax(-nlls / cfg.label_tau, dim=0)
        eps = 1e-6
        q = (1 - eps) * q_raw + eps / K

        H = -(q * torch.log(q + 1e-12)).sum().item()
        H_max = math.log(K)
        n_eff = math.exp(H)

        entropies.append(H)
        effective_candidates.append(n_eff)
        max_entropy_frac.append(H / H_max)
        n_candidates_list.append(K)

    ents = np.array(entropies)
    effs = np.array(effective_candidates)
    n_cands = np.array(n_candidates_list, dtype=float)

    print("\n" + "=" * 60)
    print("2. Soft Label Entropy (τ_label=%.2f)" % cfg.label_tau)
    print("-" * 60)
    print(f"  Queries: {len(ents)}")
    print()
    print(f"  Entropy H(q):")
    print(f"    mean:  {ents.mean():.4f}")
    print(f"    std:   {ents.std():.4f}")
    print(f"    p50:   {np.percentile(ents, 50):.4f}")
    print()
    print(f"  Effective candidates exp(H):")
    print(f"    mean:  {effs.mean():.4f}")
    print(f"    std:   {effs.std():.4f}")
    print(f"    p50:   {np.percentile(effs, 50):.4f}")
    print(f"    p90:   {np.percentile(effs, 90):.4f}")
    print()
    print(f"  H / H_max:")
    print(f"    mean:  {np.array(max_entropy_frac).mean():.4f}")
    print()
    print(f"  Average candidates (N - t): {n_cands.mean():.1f}")
    conc_ratio = effs.mean() / n_cands.mean()
    print(f"  Concentration ratio (eff/N): {conc_ratio:.4f}")
    if conc_ratio > 0.8:
        print(f"  WARNING: labels are nearly uniform → AOGPT can't distinguish candidates at τ={cfg.label_tau}")

    # Show distribution of effective candidates by step
    by_step = defaultdict(list)
    for lab in labels:
        by_step[lab["step"]].append(len(lab["nll_targets"]))
    print()
    print(f"  Candidates per step (avg):")
    for step in sorted(by_step.keys()):
        vals = by_step[step]
        print(f"    step {step}: avg K={np.mean(vals):.1f}")

    return {
        "entropies": ents,
        "effective_candidates": effs,
        "concentration_ratio": conc_ratio,
    }


# ── 3. Oracle local upper bound ───────────────────────────────────────────────

@torch.no_grad()
def evaluate_oracle_local(
    aogpt, old_on_prior, idx_batch, block_perm, inv_perm, A_batch, cfg, max_seqs=30,
):
    """Generate full orders by picking per-step best candidate (by AOGPT NLL).

    Uses old ON greedy prefix at each step, evaluates all candidates with AOGPT,
    picks the one with lowest first-k NLL.
    """
    device = A_batch.device
    num_seqs = min(A_batch.shape[0], max_seqs)
    N = cfg.num_blocks
    all_blocks = torch.arange(N, device=device)

    results = []
    for s in range(num_seqs):
        A_s = A_batch[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]

        idx_s = phys_to_model_idx(idx_batch[s:s + 1], inv_perm, blk_size=cfg.block_len)

        oracle_order = torch.zeros(N, dtype=torch.long, device=device)

        for t in range(N):
            prefix = oracle_order[:t]
            candidate_set = all_blocks[~torch.isin(all_blocks, prefix)]

            if len(candidate_set) == 0:
                break

            on_prefix = sigma_old[:t]
            # rest_base must equal candidate_set so each candidate is in rest
            rest_base = candidate_set.clone()

            from reranker import compute_batched_candidate_nlls

            nlls = compute_batched_candidate_nlls(
                aogpt, idx_s, block_perm,
                prefix_order=on_prefix,
                candidates=candidate_set,
                rest_order=rest_base,
                chunk_size=cfg.label_chunk_size,
                sub_blocks=cfg.sub_blocks,
                block_len=cfg.block_len,
                k=cfg.label_k,
            )

            best = candidate_set[nlls.argmin().item()]
            oracle_order[t] = best

        nll = compute_batch_full_order_nll(
            aogpt, idx_s, block_perm, oracle_order.unsqueeze(0),
        ).item()

        results.append({
            "seq_idx": s,
            "oracle_order": oracle_order.cpu(),
            "nll": nll,
        })

    return results


# ── 4. Stage 1 grid search (small subset) ─────────────────────────────────────

@torch.no_grad()
def grid_search_quick(
    aogpt, old_on_prior, idx_batch, block_perm, A_batch, cfg, n_seqs=50,
):
    """Quick grid search on n_seqs subset."""
    alphas = cfg.grid_alphas
    betas = cfg.grid_betas
    gammas = cfg.grid_gammas

    device = A_batch.device
    N = cfg.num_blocks
    A_sub = A_batch[:n_seqs]
    idx_sub = idx_batch[:n_seqs]

    best_nll = float("inf")
    best_params = None
    results_grid = []

    total_combos = len(alphas) * len(betas) * len(gammas)
    print(f"\nGrid search: {total_combos} combos on {n_seqs} seqs")

    for alpha, beta, gamma in tqdm(
        itertools.product(alphas, betas, gammas),
        total=total_combos,
        desc="Grid search",
    ):
        reranker = StepWiseGreedyReranker(alpha=alpha, beta=beta, gamma=gamma)
        nlls = []

        for s in range(A_sub.shape[0]):
            A_s = A_sub[s:s + 1]
            sigma_old = old_on_prior.get_full_order(A_s)[0]

            order = sequential_generate(
                reranker, A_s, old_on_prior,
                sigma_old=sigma_old.unsqueeze(0),
                temperature=0.0,
                use_greedy_reranker=True,
            )[0]

            if not check_order_valid(order, N):
                nlls.append(float("nan"))
                continue

            idx_s = idx_sub[s:s + 1]
            nll = compute_batch_full_order_nll(
                aogpt, idx_s, block_perm, order.unsqueeze(0),
            ).item()
            nlls.append(nll)

        valid_nlls = [n for n in nlls if np.isfinite(n)]
        if valid_nlls:
            mean_nll = np.mean(valid_nlls)
        else:
            mean_nll = float("inf")
        results_grid.append({
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "nll": mean_nll,
        })

        if mean_nll < best_nll:
            best_nll = mean_nll
            best_params = (alpha, beta, gamma)

    # Sort by NLL
    results_grid.sort(key=lambda x: x["nll"])

    print(f"\n  Top 5 configurations:")
    for r in results_grid[:5]:
        marker = " *** BEST" if (r["alpha"], r["beta"], r["gamma"]) == best_params else ""
        print(f"    α={r['alpha']:.2f}, β={r['beta']:.2f}, γ={r['gamma']:.2f} → NLL={r['nll']:.4f}{marker}")

    # Compute baselines for reference
    old_on_nlls = []
    random_nlls = []
    for s in range(A_sub.shape[0]):
        A_s = A_sub[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]
        idx_s = idx_sub[s:s + 1]
        old_nll = compute_batch_full_order_nll(
            aogpt, idx_s, block_perm, sigma_old.unsqueeze(0),
        ).item()
        old_on_nlls.append(old_nll)

        rand_orders = torch.stack([
            torch.randperm(N, device=device) for _ in range(cfg.num_random_mc)
        ])
        rand_batch = compute_batch_full_order_nll(
            aogpt, idx_s.expand(cfg.num_random_mc, -1), block_perm, rand_orders,
        )
        random_nlls.append(rand_batch.mean().item())

    print(f"\n  Baselines on this subset:")
    print(f"    old_ON NLL:     {np.mean(old_on_nlls):.4f}")
    print(f"    random_N16 NLL: {np.mean(random_nlls):.4f}")
    print(f"    best grid NLL:  {best_nll:.4f}")
    print(f"    Δ vs old_ON:    {np.mean(old_on_nlls) - best_nll:+.4f}")
    print(f"    Δ vs random:    {np.mean(random_nlls) - best_nll:+.4f}")

    return best_params, results_grid


def phys_to_model_idx(idx_phys, inv_perm, blk_size=4):
    """Convert token sequences from physical to model coordinates."""
    B, T = idx_phys.shape
    device = idx_phys.device
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--labels-file", type=str,
                        default="probe_results/reranker/labels_train.pt")
    parser.add_argument("--mlp-ckpt", type=str,
                        default="probe_results/reranker/mlp_reranker.pt")
    parser.add_argument("--n-oracle-seqs", type=int, default=30,
                        help="Number of sequences for oracle local eval (expensive)")
    parser.add_argument("--n-grid-seqs", type=int, default=50,
                        help="Number of sequences for quick grid search")
    parser.add_argument("--n-bootstrap-seqs", type=int, default=200,
                        help="Number of sequences for bootstrap CI")
    args = parser.parse_args()

    cfg = RerankerConfig()
    device = args.device

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Load labels ──
    print(f"Loading labels from {args.labels_file}")
    labels = torch.load(args.labels_file, weights_only=False)
    print(f"  {len(labels)} per-step queries from {len(set(l['seq_idx'] for l in labels))} seqs")

    # ── Diagnostics 1 & 2 run on labels only (no model needed) ──
    gap_results = analyze_label_gaps(labels, cfg)
    entropy_results = analyze_label_entropy(labels, cfg)

    # ── Load models and data for 3, 4, 5 ──
    print("\nLoading models and data for oracle/grid/bootstrap...")

    aogpt_ckpt = os.path.expanduser(cfg.aogpt_ckpt)
    tokenizer_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--gpt2/snapshots/"
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    wikitext_dir = os.path.expanduser(
        "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3"
    )

    old_on = load_old_on(cfg.old_on_ckpt, device)
    old_on_prior = OldONPrior(old_on, num_blocks=cfg.num_blocks)
    aogpt, block_perm, inv_perm = load_aogpt(aogpt_ckpt, device)
    block_perm = block_perm.to(device)
    inv_perm = inv_perm.to(device)

    A_all = np.load(cfg.data_path, mmap_mode="r")
    total_needed = cfg.num_train_seqs + cfg.num_tune_seqs + cfg.num_test_seqs
    total_seqs = min(A_all.shape[0], total_needed)
    A_all = torch.from_numpy(A_all[:total_seqs].copy()).float().to(device)
    idx_all = load_token_chunks(total_seqs, tokenizer_dir, wikitext_dir).to(device)

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

    n_train, n_tune = cfg.num_train_seqs, cfg.num_tune_seqs
    idx_tune = idx_all[n_train:n_train + n_tune]
    A_tune = A_all[n_train:n_train + n_tune]

    # ── 3. Oracle local upper bound ──
    print("\n" + "=" * 60)
    print("3. Oracle Local Upper Bound (per-step best candidate)")
    print("-" * 60)
    print(f"  Evaluating on {args.n_oracle_seqs} sequences (this is expensive)...")
    t0 = time.time()

    oracle_results = evaluate_oracle_local(
        aogpt, old_on_prior, idx_tune, block_perm, inv_perm, A_tune, cfg,
        max_seqs=args.n_oracle_seqs,
    )

    oracle_nlls = [r["nll"] for r in oracle_results]

    # Compare with old_ON and random on same seqs
    old_on_nlls_oracle = []
    random_nlls_oracle = []
    N = cfg.num_blocks
    for r in oracle_results:
        s = r["seq_idx"]
        A_s = A_tune[s:s + 1]
        idx_s = idx_tune[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]
        old_nll = compute_batch_full_order_nll(
            aogpt, idx_s, block_perm, sigma_old.unsqueeze(0),
        ).item()
        old_on_nlls_oracle.append(old_nll)

        rand_orders = torch.stack([
            torch.randperm(N, device=device) for _ in range(cfg.num_random_mc)
        ])
        rand_batch = compute_batch_full_order_nll(
            aogpt, idx_s.expand(cfg.num_random_mc, -1), block_perm, rand_orders,
        )
        random_nlls_oracle.append(rand_batch.mean().item())

    oracle_mean = np.mean(oracle_nlls)
    old_on_mean_oracle = np.mean(old_on_nlls_oracle)
    random_mean_oracle = np.mean(random_nlls_oracle)

    print(f"\n  Oracle local NLL:     {oracle_mean:.4f}")
    print(f"  old_ON NLL:           {old_on_mean_oracle:.4f}")
    print(f"  random_N16 NLL:       {random_mean_oracle:.4f}")
    print(f"  Δ oracle vs old_ON:   {old_on_mean_oracle - oracle_mean:+.4f}")
    print(f"  Δ oracle vs random:   {random_mean_oracle - oracle_mean:+.4f}")

    # Show per-step accuracy of argmax NLL vs old ON's choice
    # For each seq, compare at each step: does old ON pick the oracle candidate?
    oracle_match_rate = []
    for r_idx, r in enumerate(oracle_results):
        s = r["seq_idx"]
        A_s = A_tune[s:s + 1]
        sigma_old = old_on_prior.get_full_order(A_s)[0]
        oracle_order = r["oracle_order"]
        # At each step t, old ON would pick sigma_old[t] (in the context of its own prefix)
        # But the oracle builds its own prefix... so this isn't directly comparable.
        # Instead, compute kendall tau between oracle and old_ON
        tau = compute_kendall_tau(oracle_order, sigma_old)
        oracle_match_rate.append(tau)

    print(f"\n  Kendall τ (oracle_local vs old_ON): {np.mean(oracle_match_rate):.4f}")

    oracle_delta = old_on_mean_oracle - oracle_mean
    if oracle_delta > 0.02:
        print(f"\n  VERDICT: Oracle-local notably better than old_ON. Per-step label is a useful signal.")
        print(f"  → MLP underfitting relative to oracle. Consider better features/model.")
    elif oracle_delta > 0.005:
        print(f"\n  VERDICT: Oracle-local marginally better. Signal exists but weak.")
        print(f"  → May need more data, lower τ, or better features to amplify.")
    else:
        print(f"\n  VERDICT: Oracle-local NOT clearly better than old_ON (Δ={oracle_delta:.4f}).")
        print(f"  → AOGPT per-step first-k NLL is not the right signal for re-ranking.")

    print(f"  Time: {time.time() - t0:.1f}s")

    # ── 4. Stage 1 grid search (quick) ──
    print("\n" + "=" * 60)
    print("4. Stage 1 A-edge Grid Search (quick)")
    print("-" * 60)
    t0 = time.time()
    best_params, grid_results = grid_search_quick(
        aogpt, old_on_prior, idx_tune, block_perm, A_tune, cfg,
        n_seqs=args.n_grid_seqs,
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── 5. Bootstrap CI ──
    print("\n" + "=" * 60)
    print("5. Bootstrap CI for ΔNLL")
    print("-" * 60)
    n_bootstrap_seqs = min(args.n_bootstrap_seqs, A_tune.shape[0])
    print(f"  Computing per-sequence NLLs on {n_bootstrap_seqs} seqs...")
    t0 = time.time()

    per_seq = compute_per_seq_nlls(
        aogpt, old_on_prior, mlp_reranker,
        idx_tune[:n_bootstrap_seqs], block_perm, A_tune[:n_bootstrap_seqs], cfg,
    )

    ci_old_on = bootstrap_ci(per_seq, "mlp", "old_on")
    ci_random = bootstrap_ci(per_seq, "mlp", "random")

    print(f"\n  ΔNLL(MLP - old_ON):")
    print(f"    mean:    {ci_old_on['mean']:+.6f}")
    print(f"    95% CI:  [{ci_old_on['ci_lower']:+.6f}, {ci_old_on['ci_upper']:+.6f}]")
    print(f"    n valid: {ci_old_on['n_valid']}/{ci_old_on['n_total']}")
    if ci_old_on['ci_lower'] > 0:
        print(f"    → Statistically significant improvement over old_ON")
    elif ci_old_on['ci_upper'] < 0:
        print(f"    → Statistically significantly WORSE than old_ON")
    else:
        print(f"    → NOT statistically significant (CI crosses zero)")

    print(f"\n  ΔNLL(MLP - random_N16):")
    print(f"    mean:    {ci_random['mean']:+.6f}")
    print(f"    95% CI:  [{ci_random['ci_lower']:+.6f}, {ci_random['ci_upper']:+.6f}]")
    print(f"    n valid: {ci_random['n_valid']}/{ci_random['n_total']}")
    if ci_random['ci_lower'] > 0:
        print(f"    → Statistically significant improvement over random_N16")
    elif ci_random['ci_upper'] < 0:
        print(f"    → Statistically significantly WORSE than random_N16")
    else:
        print(f"    → NOT statistically significant (CI crosses zero)")

    print(f"\n  Per-seq NLL summary:")
    print(f"    old_ON mean:  {per_seq['old_on'].mean():.4f} ± {per_seq['old_on'].std():.4f}")
    print(f"    MLP mean:     {per_seq['mlp'][np.isfinite(per_seq['mlp'])].mean():.4f}")
    print(f"    random mean:  {per_seq['random'].mean():.4f} ± {per_seq['random'].std():.4f}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)

    issues = []
    if gap_results["gaps_best_vs_median"].mean() > -0.005:
        issues.append(f"Best-vs-median NLL gap near zero ({gap_results['gaps_best_vs_median'].mean():.5f})")
    if entropy_results["concentration_ratio"] > 0.8:
        issues.append(f"Labels nearly uniform (concentration={entropy_results['concentration_ratio']:.2f})")
    if oracle_delta < 0.005:
        issues.append(f"Oracle-local Δ vs old_ON < 0.005 ({oracle_delta:.4f})")
    ci_passes = ci_old_on.get('ci_lower', -1) > 0 or ci_random.get('ci_lower', -1) > 0

    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Recommendation: DO NOT expand to 500.")
        print("  AOGPT per-step first-k NLL has insufficient signal for reranking.")
        if oracle_delta < 0.005:
            print("  The oracle itself can't beat old_ON → label target is fundamentally limited.")
        else:
            print("  Oracle signal exists but MLP can't capture it with current features.")
    else:
        print("\n  All diagnostics pass. Consider expanding to 500 with caveats.")


if __name__ == "__main__":
    main()
