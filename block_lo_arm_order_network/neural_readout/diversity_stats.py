"""NR-1 Task 5: teacher order diversity diagnostic (spec §5.1b).

These four stats must accompany every student tau report so reviewers can
distinguish "student learned a per-sample mapping" from "student learned a
global order prior":

  - unique_sigma_count: how many distinct teacher orders over the dataset
  - first_node_entropy: nats of the first-revealed-node distribution
  - distinct_first3_prefix_count: how many distinct 3-prefix patterns
  - mean_pairwise_tau: mean Kendall tau between random pairs of sigma_T

Interpretation guide (§5.1b):
  Low diversity + high student tau -> student learned a global prior
  High diversity + high student tau -> student learned a per-sample mapping

This module never imports or computes NLL — see spec §1 supervision boundary.
"""
import numpy as np
from scipy.stats import kendalltau


def _first_node_entropy(sigma):
    """Entropy (nats) of the empirical first-node distribution over the dataset."""
    first = sigma[:, 0]
    _, counts = np.unique(first, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def teacher_diversity(sigma, max_pairs=100, seed=0):
    """Compute the four diversity diagnostic stats over a batch of teacher orders.

    Args:
        sigma: (M, N) int, M teacher orders of length N.
        max_pairs: subsample at most this many random pairs for mean Kendall tau,
                   to keep the computation O(max_pairs * N) instead of O(M^2 * N).
                   100 is enough for a stable estimate at M >= 100.
        seed: RNG seed for the pair subsampling.

    Returns:
        dict with keys unique_sigma_count, first_node_entropy,
        distinct_first3_prefix_count, mean_pairwise_tau.
    """
    sigma = np.asarray(sigma, dtype=np.int64)
    if sigma.ndim != 2:
        raise ValueError(f"sigma must be (M, N), got shape {sigma.shape}")
    M, _N = sigma.shape

    unique_sigma = np.unique(sigma, axis=0)
    first3 = np.unique(sigma[:, :3], axis=0) if sigma.shape[1] >= 3 else np.unique(sigma, axis=0)

    if M <= 1:
        mean_tau = 1.0
    else:
        rng = np.random.default_rng(seed)
        n_pairs = min(max_pairs, M * (M - 1) // 2)
        taus = []
        for _ in range(n_pairs):
            i, j = rng.choice(M, size=2, replace=False)
            tau, _ = kendalltau(sigma[i], sigma[j])
            # kendalltau returns nan when one input is constant; skip those.
            if np.isfinite(tau):
                taus.append(tau)
        mean_tau = float(np.mean(taus)) if taus else 1.0

    return {
        "unique_sigma_count": int(unique_sigma.shape[0]),
        "first_node_entropy": _first_node_entropy(sigma),
        "distinct_first3_prefix_count": int(first3.shape[0]),
        "mean_pairwise_tau": mean_tau,
    }
