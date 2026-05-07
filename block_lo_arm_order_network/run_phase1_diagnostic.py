#!/usr/bin/env python3
"""
Phase 1 Offline Diagnostic Script
==================================
Runs 7 policies x K orders on the round-0 attention graph and reports diagnostics.
Pure NumPy, no AOGPT training.

Policies evaluated:
  1. progressive_rw        — directed progressive random walk
  2. self_avoiding_rw      — self-avoiding random walk
  3. pagerank_source_det   — PageRank with source-based q, deterministic argsort
  4. pagerank_source_stoch — PageRank with source-based q, stochastic walk
  5. pagerank_uniform_det  — PageRank with uniform q, deterministic argsort
  6. pagerank_uniform_stoch— PageRank with uniform q, stochastic walk
  7. random_permutation    — uniform random permutations (baseline)
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

# -- Resolve script directory for local imports and default paths --
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from directed_graph_policy import (  # noqa: E402
    build_directed_graph,
    compute_source,
    _softmax,
    progressive_rw_step,
    self_avoiding_rw_step,
    pagerank,
    sample_order,
    sample_orders,
)
from order_diagnostics import _kendall_tau  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    'beta_sup': 1.0,
    'beta_fut': 0.5,
    'beta_src': 0.2,
    'beta_loc': 0.5,
    'tau_start': 1.0,
    'tau_step': 1.0,
    'alpha_dep': 0.5,
    'alpha_pr': 0.85,
}

# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _to_json_safe(obj):
    """Recursively convert numpy scalars/arrays to native Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Entropy replay (adapted from _policy_step_entropy_replay with start_source)
# ---------------------------------------------------------------------------


def _replay_step_entropy(B, policy, params, order, seed, start_source=None):
    """Replay one order step-by-step; return per-step entropies H_t (N,) float64.

    Mirrors ``_policy_step_entropy_replay`` from ``directed_graph_policy`` but
    accepts an explicit ``start_source`` so that uniform-pagerank policies
    compute the correct p0 entropy.
    """
    rng = np.random.default_rng(seed)
    B = np.asarray(B, dtype=np.float64)
    order = np.asarray(order, dtype=np.int64)
    N = B.shape[0]

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))
    alpha_pr = float(params.get('alpha_pr', 0.85))

    # Resolve source for p0
    if start_source is None:
        source, _, _ = compute_source(B, alpha_dep)
    elif isinstance(start_source, str) and start_source == 'uniform':
        source = np.ones(N, dtype=np.float64)
    else:
        source = np.asarray(start_source, dtype=np.float64)

    # pagerank_det is fully deterministic – no per-step distribution
    if policy == 'pagerank_det':
        return np.zeros(N, dtype=np.float64)

    betas = {
        'sup': float(params.get('beta_sup', 1.0)),
        'fut': float(params.get('beta_fut', 0.5)),
        'src': float(params.get('beta_src', 0.2)),
        'loc': float(params.get('beta_loc', 0.5)),
    }

    H = np.zeros(N, dtype=np.float64)

    # --- Step 0 ---
    if policy in ('progressive_rw', 'self_avoiding_rw'):
        p0 = _softmax(source, tau_start, rng)
    elif policy == 'pagerank_stoch':
        q = _softmax(source, tau_start, rng)
        r = pagerank(B, q, alpha_pr)
        p0 = _softmax(r, tau_start, rng)
    else:
        raise ValueError(f"Unknown policy for entropy replay: {policy}")

    eps = 1e-300
    H[0] = float(-np.sum(p0 * np.log(np.maximum(p0, eps))))

    # --- Steps 1 .. N-1 ---
    first = int(order[0])
    S_set = np.array([first], dtype=np.int64)
    U = np.setdiff1d(np.arange(N, dtype=np.int64), S_set)
    last = first

    for t in range(1, N):
        if policy == 'progressive_rw':
            p_t, _ = progressive_rw_step(B, S_set, U, last, betas, tau_step, source, rng)
        elif policy in ('self_avoiding_rw', 'pagerank_stoch'):
            p_t, _ = self_avoiding_rw_step(B, U, last, tau_step, rng)
        else:
            p_t = np.ones(len(U)) / len(U)  # fallback

        H[t] = float(-np.sum(p_t * np.log(np.maximum(p_t, eps))))

        chosen = int(order[t])
        if chosen not in U:
            break  # safety: should not happen for valid permutations
        S_set = np.append(S_set, chosen)
        U = U[U != chosen]
        last = chosen

    return H


# ---------------------------------------------------------------------------
# Manual sampling (for policies that need start_source != None)
# ---------------------------------------------------------------------------


def _sample_orders_manual(B, policy, params, K, seed_base, start_source=None):
    """Sample K orders with explicit *start_source* and compute diagnostics.

    This is a drop-in replacement for ``sample_orders`` when the caller needs
    to pass a non-default ``start_source`` (e.g. ``'uniform'``).
    """
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]

    # -- Sample --------------------------------------------------------------
    orders = np.zeros((K, N), dtype=np.int64)
    logprobs = np.zeros(K, dtype=np.float64)

    for k in range(K):
        seed = seed_base * 10000 + k
        orders[k], logprobs[k] = sample_order(
            B, policy, params, seed, start_source=start_source,
        )

    # -- Legality ------------------------------------------------------------
    valid = sum(1 for k in range(K) if sorted(orders[k].tolist()) == list(range(N)))
    legal_rate = float(valid / K)

    # -- First-node entropy --------------------------------------------------
    first_nodes = orders[:, 0]
    counts = np.bincount(first_nodes, minlength=N).astype(np.float64)
    probs = counts / K
    probs_nz = probs[probs > 0]
    first_node_entropy = float(-np.sum(probs_nz * np.log(probs_nz)))

    # -- Pairwise tau (deterministic subsample) ------------------------------
    n_pairs = min(1000, K * (K - 1) // 2)
    pair_rng = np.random.default_rng(seed_base * 10000 + 99999)
    if K >= 2:
        all_i, all_j = np.triu_indices(K, k=1)
        if len(all_i) > n_pairs:
            idx = pair_rng.choice(len(all_i), size=n_pairs, replace=False)
            all_i = all_i[idx]
            all_j = all_j[idx]
        pair_taus = [
            _kendall_tau(orders[i], orders[j]) for i, j in zip(all_i, all_j)
        ]
        pairwise_tau_mean = float(np.mean(pair_taus)) if pair_taus else 0.0
    else:
        pairwise_tau_mean = 0.0

    # -- Directed score (mean B[sigma_t, sigma_{t+1}]) -----------------------
    directed_scores = [
        float(B[orders[k, t], orders[k, t + 1]])
        for k in range(K)
        for t in range(N - 1)
    ]
    mean_directed_score = float(np.mean(directed_scores)) if directed_scores else 0.0

    # -- Progressive support -------------------------------------------------
    supports = []
    for k in range(K):
        S_set = set()
        for t in range(N):
            v = int(orders[k, t])
            if t > 0:
                sup = sum(float(B[u, v]) for u in S_set)
                supports.append(sup / float(t))
            S_set.add(v)
    mean_progressive_support = float(np.mean(supports)) if supports else 0.0

    # -- Policy-step entropy (replay with correct start_source) --------------
    n_replay = min(100, K)
    all_H = np.zeros((n_replay, N), dtype=np.float64)
    for i in range(n_replay):
        replay_seed = seed_base * 10000 + i
        all_H[i] = _replay_step_entropy(
            B, policy, params, orders[i], replay_seed, start_source=start_source,
        )

    mean_per_step = all_H.mean(axis=0)
    third = max(1, N // 3)
    early = float(mean_per_step[:third].mean())
    mid = float(mean_per_step[third:2 * third].mean())
    late = float(mean_per_step[2 * third:].mean())
    overall = float(mean_per_step.mean())

    # -- tau vs L2R ----------------------------------------------------------
    L2R = np.arange(N, dtype=np.int64)
    taus_l2r = np.array(
        [_kendall_tau(orders[i], L2R) for i in range(K)], dtype=np.float64,
    )
    tau_vs_l2r_mean = float(taus_l2r.mean())
    tau_vs_l2r_std = float(taus_l2r.std())

    # -- Logprob stats -------------------------------------------------------
    logprob_mean = float(logprobs.mean())
    logprob_std = float(logprobs.std())
    logprob_p10 = float(np.percentile(logprobs, 10))
    logprob_p90 = float(np.percentile(logprobs, 90))

    return {
        'orders': orders,
        'logprobs': logprobs,
        'legal_rate': legal_rate,
        'tau_vs_l2r_mean': tau_vs_l2r_mean,
        'tau_vs_l2r_std': tau_vs_l2r_std,
        'first_node_entropy': first_node_entropy,
        'pairwise_tau_mean': pairwise_tau_mean,
        'mean_directed_score': mean_directed_score,
        'mean_progressive_support': mean_progressive_support,
        'policy_step_entropy': {
            'mean': overall,
            'early': early,
            'mid': mid,
            'late': late,
        },
        'cross_logprob': None,  # filled later by cross-policy metric
        'logprob_mean': logprob_mean,
        'logprob_std': logprob_std,
        'logprob_p10': logprob_p10,
        'logprob_p90': logprob_p90,
    }


# ---------------------------------------------------------------------------
# Random-permutation baseline
# ---------------------------------------------------------------------------


def _generate_random_orders(B, K, seed_base):
    """Generate K uniform random permutations and compute diagnostic metrics."""
    N = B.shape[0]
    seed = seed_base * 10000 + 77777
    rng = np.random.default_rng(seed)

    orders = np.zeros((K, N), dtype=np.int64)
    for k in range(K):
        orders[k] = rng.permutation(N)

    logprobs = np.zeros(K, dtype=np.float64)

    # Random permutations are always legal
    legal_rate = 1.0

    # First-node entropy
    first_nodes = orders[:, 0]
    counts = np.bincount(first_nodes, minlength=N).astype(np.float64)
    probs = counts / K
    probs_nz = probs[probs > 0]
    first_node_entropy = float(-np.sum(probs_nz * np.log(probs_nz)))

    # Pairwise tau (deterministic subsample)
    n_pairs = min(1000, K * (K - 1) // 2)
    pair_rng = np.random.default_rng(seed_base * 10000 + 99999)
    if K >= 2:
        all_i, all_j = np.triu_indices(K, k=1)
        if len(all_i) > n_pairs:
            idx = pair_rng.choice(len(all_i), size=n_pairs, replace=False)
            all_i = all_i[idx]
            all_j = all_j[idx]
        pair_taus = [
            _kendall_tau(orders[i], orders[j]) for i, j in zip(all_i, all_j)
        ]
        pairwise_tau_mean = float(np.mean(pair_taus)) if pair_taus else 0.0
    else:
        pairwise_tau_mean = 0.0

    # Directed score
    directed_scores = [
        float(B[orders[k, t], orders[k, t + 1]])
        for k in range(K)
        for t in range(N - 1)
    ]
    mean_directed_score = float(np.mean(directed_scores)) if directed_scores else 0.0

    # Progressive support
    supports = []
    for k in range(K):
        S_set = set()
        for t in range(N):
            v = int(orders[k, t])
            if t > 0:
                sup = sum(float(B[u, v]) for u in S_set)
                supports.append(sup / float(t))
            S_set.add(v)
    mean_progressive_support = float(np.mean(supports)) if supports else 0.0

    # tau vs L2R
    L2R = np.arange(N, dtype=np.int64)
    taus_l2r = np.array(
        [_kendall_tau(orders[i], L2R) for i in range(K)], dtype=np.float64,
    )
    tau_vs_l2r_mean = float(taus_l2r.mean())
    tau_vs_l2r_std = float(taus_l2r.std())

    return {
        'orders': orders,
        'logprobs': logprobs,
        'legal_rate': legal_rate,
        'tau_vs_l2r_mean': tau_vs_l2r_mean,
        'tau_vs_l2r_std': tau_vs_l2r_std,
        'first_node_entropy': first_node_entropy,
        'pairwise_tau_mean': pairwise_tau_mean,
        'mean_directed_score': mean_directed_score,
        'mean_progressive_support': mean_progressive_support,
        'policy_step_entropy': {},
        'cross_logprob': None,
        'logprob_mean': 0.0,
        'logprob_std': 0.0,
        'logprob_p10': 0.0,
        'logprob_p90': 0.0,
    }


# ---------------------------------------------------------------------------
# Cross-policy logprob (progressive_rw <-> self_avoiding_rw)
# ---------------------------------------------------------------------------


def _compute_cross_logprob(B, orders, target_policy, params, seed_base):
    """Compute mean logprob of *orders* under *target_policy*'s step distribution.

    Parameters
    ----------
    B : (N, N) float64
        Directed graph.
    orders : (M, N) int64
        Orders generated by the *source* policy.
    target_policy : str
        Either ``'progressive_rw'`` or ``'self_avoiding_rw'``.
    params : dict
        Standard parameter dict.
    seed_base : int
        Seeding base for RNG.

    Returns
    -------
    float
        Mean logprob per order.
    """
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    orders = np.asarray(orders, dtype=np.int64)
    M = len(orders)

    tau_start = float(params.get('tau_start', 1.0))
    tau_step = float(params.get('tau_step', 1.0))
    alpha_dep = float(params.get('alpha_dep', 0.5))

    source, _, _ = compute_source(B, alpha_dep)

    total_logprob = 0.0

    for i in range(M):
        seed = seed_base * 10000 + i
        rng = np.random.default_rng(seed)
        order = orders[i]
        order_logprob = 0.0

        # --- Step 0: both policies share the same initial distribution -------
        p0 = _softmax(source, tau_start, rng)
        first = int(order[0])
        order_logprob += float(np.log(max(float(p0[first]), 1e-300)))

        S_set = np.array([first], dtype=np.int64)
        U = np.setdiff1d(np.arange(N, dtype=np.int64), S_set)
        last = first

        # --- Steps 1 .. N-1 -------------------------------------------------
        betas = {
            'sup': float(params.get('beta_sup', 1.0)),
            'fut': float(params.get('beta_fut', 0.5)),
            'src': float(params.get('beta_src', 0.2)),
            'loc': float(params.get('beta_loc', 0.5)),
        }

        for t in range(1, N):
            if target_policy == 'progressive_rw':
                p_t, _ = progressive_rw_step(
                    B, S_set, U, last, betas, tau_step, source, rng,
                )
            elif target_policy == 'self_avoiding_rw':
                p_t, _ = self_avoiding_rw_step(B, U, last, tau_step, rng)
            else:
                raise ValueError(f"Unknown target_policy: {target_policy}")

            chosen = int(order[t])
            if chosen not in U:
                break  # safety: should not happen for valid permutations
            idx = int(np.where(U == chosen)[0][0])
            order_logprob += float(np.log(max(float(p_t[idx]), 1e-300)))

            S_set = np.append(S_set, chosen)
            U = U[U != chosen]
            last = chosen

        total_logprob += order_logprob

    return float(total_logprob / M)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 offline diagnostic: 7 policies x K orders on attention graph',
    )
    parser.add_argument('--K', type=int, default=5000,
                        help='Number of orders per policy (default: 5000)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: probe_results/phase1_diagnostic)')
    parser.add_argument('--A-path', type=str, default=None,
                        help='Path to A_train_n64_10k.npy (default: probe_results/A_train_n64_10k.npy)')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base seed for reproducibility (default: 42)')
    args = parser.parse_args()

    K = args.K
    seed_base = args.seed_base

    # Resolve paths relative to the script directory
    output_dir = args.output_dir or os.path.join(
        _SCRIPT_DIR, 'probe_results', 'phase1_diagnostic',
    )
    A_path = args.A_path or os.path.join(
        _SCRIPT_DIR, 'probe_results', 'A_train_n64_10k.npy',
    )

    # -------------------------------------------------------------------
    # 1. Create output directory
    # -------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 2. Load data
    # -------------------------------------------------------------------
    print(f"Loading A from: {A_path}", flush=True)
    if not os.path.isfile(A_path):
        print(f"ERROR: A file not found: {A_path}", flush=True)
        sys.exit(1)

    A_all = np.load(A_path)  # (20000, 64, 64)
    print(f"  A_all shape: {A_all.shape}, dtype: {A_all.dtype}", flush=True)

    A_global = A_all.mean(axis=0).astype(np.float64)
    np.fill_diagonal(A_global, 0.0)
    B = build_directed_graph(A_global)  # B = A_global.T with zero diagonal
    print(f"  A_global shape: {A_global.shape}, B shape: {B.shape}", flush=True)
    print(f"  A_global mean: {float(A_global.mean()):.6f}, "
          f"B mean: {float(B.mean()):.6f}", flush=True)
    print(flush=True)

    # -------------------------------------------------------------------
    # 3. Define the 7 policies
    # -------------------------------------------------------------------
    policies = OrderedDict([
        # (dict_key, policy_arg, start_source, use_manual_wrapper)
        ('progressive_rw',      {'policy': 'progressive_rw',  'start_source': None,      'manual': False}),
        ('self_avoiding_rw',    {'policy': 'self_avoiding_rw','start_source': None,      'manual': False}),
        ('pagerank_source_det', {'policy': 'pagerank_det',     'start_source': None,      'manual': False}),
        ('pagerank_source_stoch',{'policy': 'pagerank_stoch',  'start_source': None,      'manual': False}),
        ('pagerank_uniform_det', {'policy': 'pagerank_det',    'start_source': 'uniform', 'manual': True}),
        ('pagerank_uniform_stoch',{'policy': 'pagerank_stoch', 'start_source': 'uniform', 'manual': True}),
        ('random_permutation',  None),  # special handling below
    ])

    # -------------------------------------------------------------------
    # 4. Run each policy
    # -------------------------------------------------------------------
    results = {}
    sampled_orders = {}

    for name, cfg in policies.items():
        if name == 'random_permutation':
            print(f"Sampling random_permutation (K={K}) ... ", end='', flush=True)
            t0 = time.time()
            result = _generate_random_orders(B, K, seed_base)
            elapsed = time.time() - t0
            print(f"done in {elapsed:.1f}s", flush=True)
        elif cfg['manual']:
            print(f"Sampling {name} (K={K}, start_source={cfg['start_source']}) ... ",
                  end='', flush=True)
            t0 = time.time()
            result = _sample_orders_manual(
                B, cfg['policy'], BASE_PARAMS, K, seed_base,
                start_source=cfg['start_source'],
            )
            elapsed = time.time() - t0
            print(f"done in {elapsed:.1f}s", flush=True)
        else:
            print(f"Sampling {name} (K={K}) ... ", end='', flush=True)
            t0 = time.time()
            result = sample_orders(B, cfg['policy'], BASE_PARAMS, K, seed_base)
            elapsed = time.time() - t0
            print(f"done in {elapsed:.1f}s", flush=True)

        result['time_s'] = elapsed
        results[name] = result
        sampled_orders[name] = result['orders']

        # Quick per-policy summary
        pse = result.get('policy_step_entropy', {})
        print(f"  legal_rate={result['legal_rate']:.4f}, "
              f"tau_vs_l2r={result['tau_vs_l2r_mean']:.4f}+-{result['tau_vs_l2r_std']:.4f}, "
              f"H0={result['first_node_entropy']:.4f}, "
              f"pairwise_tau={result['pairwise_tau_mean']:.4f}, "
              f"directed_score={result['mean_directed_score']:.6f}, "
              f"prog_support={result['mean_progressive_support']:.6f}, "
              f"H_step=({pse.get('early', 0):.2f}/{pse.get('mid', 0):.2f}/{pse.get('late', 0):.2f})",
              flush=True)
        print(flush=True)

    # -------------------------------------------------------------------
    # 5. Cross-policy logprob (progressive_rw <-> self_avoiding_rw)
    # -------------------------------------------------------------------
    print("Computing cross-policy logprob (progressive_rw <-> self_avoiding_rw) ...",
          flush=True)
    N_CROSS = min(100, K)

    rw_orders = results['progressive_rw']['orders'][:N_CROSS]
    sa_orders = results['self_avoiding_rw']['orders'][:N_CROSS]

    # progressive_rw orders scored under self_avoiding_rw step distribution
    rw_sa = _compute_cross_logprob(
        B, rw_orders, 'self_avoiding_rw', BASE_PARAMS, seed_base,
    )
    print(f"  progressive_rw under self_avoiding_rw: {rw_sa:.4f}", flush=True)

    # self_avoiding_rw orders scored under progressive_rw step distribution
    sa_rw = _compute_cross_logprob(
        B, sa_orders, 'progressive_rw', BASE_PARAMS, seed_base,
    )
    print(f"  self_avoiding_rw under progressive_rw: {sa_rw:.4f}", flush=True)

    cross_logprob = {'rw_sa': rw_sa, 'sa_rw': sa_rw}
    results['progressive_rw']['cross_logprob'] = cross_logprob
    results['self_avoiding_rw']['cross_logprob'] = cross_logprob
    print(flush=True)

    # -------------------------------------------------------------------
    # 6. Write outputs
    # -------------------------------------------------------------------

    # 6a. Per-policy JSON (exclude 'orders' array to keep files small)
    for name, result in results.items():
        json_path = os.path.join(output_dir, f'{name}.json')
        json_data = {k: v for k, v in result.items() if k != 'orders'}
        json_data = _to_json_safe(json_data)
        # Add a convenience alias
        if 'tau_vs_l2r_mean' in json_data:
            json_data['tau_vs_l2r'] = json_data['tau_vs_l2r_mean']
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Wrote {json_path}", flush=True)

    # 6b. Summary TSV
    tsv_path = os.path.join(output_dir, 'summary.tsv')
    columns = [
        'policy', 'legal_rate', 'tau_vs_l2r', 'first_node_entropy',
        'pairwise_tau', 'directed_score', 'progressive_support',
        'cross_logprob', 'time_s',
    ]
    with open(tsv_path, 'w') as f:
        f.write('\t'.join(columns) + '\n')
        for name, result in results.items():
            clp = result.get('cross_logprob')
            if isinstance(clp, dict):
                clp_str = f"rw_sa={clp['rw_sa']:.4f},sa_rw={clp['sa_rw']:.4f}"
            elif clp is not None:
                clp_str = f"{float(clp):.4f}"
            else:
                clp_str = 'N/A'

            row = [
                name,
                f"{result['legal_rate']:.4f}",
                f"{result['tau_vs_l2r_mean']:.4f}",
                f"{result['first_node_entropy']:.4f}",
                f"{result['pairwise_tau_mean']:.4f}",
                f"{result['mean_directed_score']:.6f}",
                f"{result['mean_progressive_support']:.6f}",
                clp_str,
                f"{result.get('time_s', 0):.1f}",
            ]
            f.write('\t'.join(row) + '\n')
    print(f"Wrote {tsv_path}", flush=True)

    # 6c. Sampled orders NPZ
    npz_path = os.path.join(output_dir, 'sampled_orders.npz')
    np.savez_compressed(npz_path, **sampled_orders)
    print(f"Wrote {npz_path}", flush=True)
    print(flush=True)

    # -------------------------------------------------------------------
    # 7. Go / no-go check (spec section 1.9)
    # -------------------------------------------------------------------
    pr = results['progressive_rw']
    sa = results['self_avoiding_rw']
    rp = results['random_permutation']

    print("=" * 60, flush=True)
    print("GO / NO-GO CHECK", flush=True)
    print("=" * 60, flush=True)

    checks = []

    # 1. legal_rate == 1.0 for progressive_rw (hard requirement)
    c1 = pr['legal_rate'] == 1.0
    checks.append(('legal_rate(progressive_rw) == 1.0', c1, pr['legal_rate']))

    # 2. first_node_entropy > 0 for progressive_rw (no collapse)
    c2 = pr['first_node_entropy'] > 0.0
    checks.append(('first_node_entropy(progressive_rw) > 0', c2,
                   pr['first_node_entropy']))

    # 3. pairwise_tau_mean < 0.95 for progressive_rw (real order family)
    c3 = pr['pairwise_tau_mean'] < 0.95
    checks.append(('pairwise_tau(progressive_rw) < 0.95', c3,
                   pr['pairwise_tau_mean']))

    # 4. mean_directed_score for progressive_rw > random
    c4 = pr['mean_directed_score'] > rp['mean_directed_score']
    checks.append(('directed_score(progressive_rw) > random', c4,
                   (pr['mean_directed_score'], rp['mean_directed_score'])))

    # 5. mean_progressive_support for progressive_rw > max(sa, random)
    threshold = max(sa['mean_progressive_support'], rp['mean_progressive_support'])
    c5 = pr['mean_progressive_support'] > threshold
    checks.append(('prog_support(progressive_rw) > max(sa, random)', c5,
                   (pr['mean_progressive_support'], threshold)))

    for desc, passed, val in checks:
        status = "[PASS]" if passed else "[FAIL]"
        if isinstance(val, tuple):
            print(f"  {status} {desc}: {val[0]:.6f} vs {val[1]:.6f}", flush=True)
        else:
            print(f"  {status} {desc}: {val}", flush=True)

    all_pass = all(c[1] for c in checks)
    print(flush=True)
    if all_pass:
        print(">>> OVERALL: GO <<<", flush=True)
    else:
        print(">>> OVERALL: NO-GO <<<", flush=True)
    print("=" * 60, flush=True)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
