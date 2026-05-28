"""BR-1 Task 8: matching metrics for sigma_pred vs sigma_T.

NLL lives elsewhere (neural_readout.eval_frozen_nll / batch_readout.eval_frozen_phase2)
and MUST NOT be imported here — the selection-policy guard in BR-1 Task 9
forbids any AR-NLL signal from leaking into the model-selection path.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


def kendall_tau_batch(sigma_pred, sigma_true) -> float:
    """Mean Kendall tau over a (M, N) batch of orders."""
    sigma_pred = np.asarray(sigma_pred)
    sigma_true = np.asarray(sigma_true)
    vals = [kendalltau(sigma_pred[i], sigma_true[i])[0] for i in range(len(sigma_pred))]
    return float(np.mean(vals))


def spearman_rho_batch(sigma_pred, sigma_true) -> float:
    """Mean Spearman rho over a (M, N) batch of orders."""
    sigma_pred = np.asarray(sigma_pred)
    sigma_true = np.asarray(sigma_true)
    vals = [spearmanr(sigma_pred[i], sigma_true[i])[0] for i in range(len(sigma_pred))]
    return float(np.mean(vals))


def pairwise_acc(logits: torch.Tensor, rank: torch.Tensor) -> float:
    """Fraction of (i, j) with rank_i < rank_j where z_i > z_j."""
    zi = logits.unsqueeze(2)
    zj = logits.unsqueeze(1)
    mask = (rank.unsqueeze(2) < rank.unsqueeze(1))
    correct = (zi > zj) & mask
    return (correct.sum().float() / mask.sum().clamp_min(1).float()).item()


def top1_acc(sigma_pred: torch.Tensor, sigma_true: torch.Tensor) -> float:
    """Fraction of rows where sigma_pred[0] == sigma_true[0]."""
    return (sigma_pred[:, 0] == sigma_true[:, 0]).float().mean().item()


def first_k_overlap(sigma_pred: torch.Tensor, sigma_true: torch.Tensor, k: int = 3) -> float:
    """Mean |first-k(pred) ∩ first-k(true)| / k across rows."""
    out = []
    for p, t in zip(sigma_pred.tolist(), sigma_true.tolist()):
        out.append(len(set(p[:k]) & set(t[:k])) / k)
    return float(np.mean(out))
