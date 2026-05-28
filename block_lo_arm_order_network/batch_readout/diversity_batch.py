"""BR-1 Task 4: teacher-label diversity statistics across M batch-mean graphs.

Records three numbers needed to interpret BR-1 Phase-1 results:

  - mean_pairwise_tau : average Kendall tau between every pair (sigma_i, sigma_j).
                       Near 1.0 means teachers collapsed (batch-mean smoothed
                       the per-batch structure away). Near 0.0 means teachers
                       are essentially random across batches.
  - unique_sigma_ratio: |{sigma_m}| / M.
  - first_step_entropy: Shannon entropy of the sigma[:, 0] distribution.
"""
from __future__ import annotations
import math

import numpy as np
from scipy.stats import kendalltau


def teacher_diversity_stats(sigma: np.ndarray) -> dict:
    """sigma: (M, N) int orders. Returns dict of summary statistics.

    For M=1 the pairwise tau is NaN (no pair) but the other two are well-defined.
    """
    sigma = np.asarray(sigma)
    if sigma.ndim != 2:
        raise ValueError(f"sigma must be 2D (M, N); got {sigma.shape}")
    M, N = sigma.shape

    if M < 2:
        mean_tau = float("nan")
    else:
        taus = []
        for i in range(M):
            for j in range(i + 1, M):
                t, _ = kendalltau(sigma[i], sigma[j])
                taus.append(t)
        mean_tau = float(np.mean(taus))

    unique_n = len({tuple(s.tolist()) for s in sigma})

    first_counts = np.bincount(sigma[:, 0], minlength=N)
    total = first_counts.sum()
    if total == 0:
        H = 0.0
    else:
        p = first_counts / total
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log(p[mask])))

    return {
        "mean_pairwise_tau": mean_tau,
        "unique_sigma_ratio": unique_n / M,
        "first_step_entropy": H,
        "first_step_entropy_max": math.log(N),
    }
