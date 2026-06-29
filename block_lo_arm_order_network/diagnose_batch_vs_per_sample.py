"""Diagnostic: per-sample A noise vs batch-aggregated attention signal.

Hypothesis (from ych comparison): attention signal exists but is buried in per-sample
noise. Batch averaging should reveal it. If A_avg NN path has tau >> per-sample tau (~0.60),
the signal granularity is the root cause — not signal absence.
"""

import numpy as np
import argparse


def compute_nn_path(A, start=None):
    """NN greedy Hamiltonian path starting from `start` (low-degree endpoint if None)."""
    n = A.shape[0]
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0)

    if start is None:
        start = int(np.argmin(W.sum(axis=1)))  # low-degree endpoint

    visited = {start}
    path = [start]
    current = start
    for _ in range(n - 1):
        remaining = [j for j in range(n) if j not in visited]
        next_node = max(remaining, key=lambda j: W[current, j])
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path


def kendall_tau(path, ref=None):
    """Kendall tau of path vs L2R [0,1,...,n-1]."""
    n = len(path)
    if ref is None:
        ref = list(range(n))
    pos = {v: i for i, v in enumerate(path)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pos[ref[i]] < pos[ref[j]]):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="probe_results/A32_from_N64_10k.npy")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--subsample-sizes", nargs="+", type=int,
                        default=[1, 4, 16, 64, 256, 1024, 4096])
    args = parser.parse_args()

    A_all = np.load(args.input)  # (N, 32, 32)
    N = min(A_all.shape[0], args.max_samples)
    A_all = A_all[:N]
    print(f"Loaded {N} A matrices, shape {A_all.shape}")

    # 1. Per-sample tau distribution
    per_sample_taus = []
    for i in range(N):
        path = compute_nn_path(A_all[i])
        tau = kendall_tau(path)
        per_sample_taus.append(tau)
    per_sample_taus = np.array(per_sample_taus)
    print(f"\n=== Per-sample tau (N={N}) ===")
    print(f"  Mean:  {per_sample_taus.mean():.4f}")
    print(f"  Median: {np.median(per_sample_taus):.4f}")
    print(f"  Std:   {per_sample_taus.std():.4f}")
    print(f"  Min:   {per_sample_taus.min():.4f}")
    print(f"  Max:   {per_sample_taus.max():.4f}")
    print(f"  Frac tau>0.5: {(per_sample_taus > 0.5).mean():.3f}")
    print(f"  Frac tau>0.7: {(per_sample_taus > 0.7).mean():.3f}")

    # 2. Batch-aggregated A_avg
    A_avg = A_all.mean(axis=0)
    path_avg = compute_nn_path(A_avg)
    tau_avg = kendall_tau(path_avg)
    print(f"\n=== Global A_avg (N={N} samples) ===")
    print(f"  Path: {path_avg}")
    print(f"  Tau vs L2R: {tau_avg:.4f}")
    print(f"  is L2R: {path_avg == list(range(32))}")

    # 3. Subsample scaling: how does tau change with aggregation size?
    print(f"\n=== Tau vs aggregation size (mean ± std over 20 trials) ===")
    rng = np.random.default_rng(42)
    for k in args.subsample_sizes:
        if k > N:
            continue
        taus = []
        n_trials = max(1, min(20, N // k))
        for _ in range(n_trials):
            idx = rng.choice(N, size=k, replace=False)
            A_sub = A_all[idx].mean(axis=0)
            path = compute_nn_path(A_sub)
            taus.append(kendall_tau(path))
        taus = np.array(taus)
        print(f"  k={k:5d}: tau={taus.mean():.4f} ± {taus.std():.4f}  [{n_trials} trials]")

    # 4. Per-sample NN paths vs aggregated path: overlap
    print(f"\n=== Per-sample path similarity to batch path ===")
    batch_path = path_avg
    same_first = 0
    overlap_scores = []
    for i in range(min(N, 500)):
        pp = compute_nn_path(A_all[i])
        if pp[0] == batch_path[0]:
            same_first += 1
        overlap = len(set(pp[:8]) & set(batch_path[:8])) / 8
        overlap_scores.append(overlap)
    overlap_scores = np.array(overlap_scores)
    print(f"  Same first node as batch: {same_first}/{min(N, 500)} = {same_first/min(N, 500):.3f}")
    print(f"  Top-8 overlap with batch path: {overlap_scores.mean():.3f} ± {overlap_scores.std():.3f}")


if __name__ == "__main__":
    main()
