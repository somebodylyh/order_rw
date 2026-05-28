"""BR-1 Task 6: pairwise logistic + Plackett-Luce listwise NLL.

Convention (load-bearing; asserted by tests):
  - rank-0 = earliest reveal
  - earliest reveal node should get the HIGHEST logit
  - pairwise loss penalises pairs (i, j) where rank_i < rank_j but z_i <= z_j
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_logistic_loss(logits: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Mean over (i, j) of -log sigmoid(z_i - z_j) for pairs with rank_i < rank_j.

    Args:
        logits: (B, N) float scores; higher = earlier reveal.
        rank:   (B, N) int with rank-0 = earliest.
    Returns:
        scalar loss.
    """
    zi = logits.unsqueeze(2)  # (B, N, 1)
    zj = logits.unsqueeze(1)  # (B, 1, N)
    mask = (rank.unsqueeze(2) < rank.unsqueeze(1)).float()  # 1 where i earlier than j
    diff = zi - zj
    loss_mat = -F.logsigmoid(diff) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_mat.sum() / denom


def plackett_luce_nll(logits: torch.Tensor, sigma_T: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Listwise Plackett-Luce NLL:

        - sum_{t=0..N-2} log P(sigma_T[t] | remaining at step t)
        = sum_{t=0..N-2} (logsumexp(z[t:N]/tau) - z[sigma_T[t]]/tau)

    Args:
        logits:  (B, N) float scores.
        sigma_T: (B, N) int order with sigma_T[:, 0] earliest.
        tau:     temperature; smaller -> sharper, lower NLL when sigma matches argsort.
    Returns:
        scalar mean NLL over the batch.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    B, N = logits.shape
    # z_perm[t] = logits[b, sigma_T[b, t]] / tau
    z_perm = logits.gather(1, sigma_T) / tau                 # (B, N)
    # Backward cumulative logsumexp: cum_lse_t = logsumexp(z_perm[t:N])
    nll = z_perm.new_zeros(B)
    cum_lse = torch.full((B,), float("-inf"), device=z_perm.device, dtype=z_perm.dtype)
    for t in range(N - 1, -1, -1):
        cum_lse = torch.logaddexp(cum_lse, z_perm[:, t])
        if t < N - 1:
            nll = nll + (cum_lse - z_perm[:, t])
    # Last term (t=N-1) is logsumexp([z[N-1]]) - z[N-1] = 0; intentionally skipped.
    return nll.mean()
