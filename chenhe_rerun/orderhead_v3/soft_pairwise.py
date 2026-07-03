"""Soft pairwise BCE loss, accuracy, and regularisation for g_beta pretraining (ported).

All losses operate on the strict upper-triangle of the 64×64 pairwise matrix.
Teacher Y of exactly 0.5 off-diagonal are ties (excluded from accuracy).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_mask(N: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean (N, N): True for i < j (strict upper triangle)."""
    return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)


def non_tie_mask(Y_pair: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    return ~torch.isclose(Y_pair, torch.tensor(0.5, device=Y_pair.device,
                                                dtype=Y_pair.dtype), atol=atol)


def soft_pairwise_bce_loss(scores: torch.Tensor, Y_pair: torch.Tensor,
                           pair_mask: torch.Tensor | None = None) -> torch.Tensor:
    """BCE on score-difference logits vs soft teacher (mean over included pairs)."""
    B, N = scores.shape
    if Y_pair.shape != (B, N, N):
        raise ValueError(f"Y_pair must be ({B}, {N}, {N}), got {Y_pair.shape}")
    if pair_mask is None:
        pair_mask = pairwise_mask(N, device=scores.device)
    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    flat_diff = diff[:, pair_mask]
    flat_target = Y_pair[:, pair_mask]
    return F.binary_cross_entropy_with_logits(flat_diff, flat_target)


def pairwise_accuracy(scores: torch.Tensor, Y_pair: torch.Tensor) -> float:
    """Fraction of off-diagonal, non-tie pairs where sign(score_i-score_j) matches Y>0.5."""
    B, N = scores.shape
    device = scores.device
    p_mask = pairwise_mask(N, device=device)
    nt_mask = non_tie_mask(Y_pair)
    eval_mask = p_mask.unsqueeze(0) & nt_mask
    total = eval_mask.sum()
    if total == 0:
        return 1.0
    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    pred_correct = (diff > 0) == (Y_pair > 0.5)
    correct = (pred_correct & eval_mask).sum()
    return float((correct / total).item())


def per_head_aux_loss(scores_per_head: torch.Tensor, Y_pair: torch.Tensor) -> torch.Tensor:
    """Mean over heads of BCE(scores_h, Y) — per-head alignment regulariser."""
    B, H, N = scores_per_head.shape
    losses = []
    p_mask = pairwise_mask(N, device=scores_per_head.device)
    for h in range(H):
        losses.append(soft_pairwise_bce_loss(scores_per_head[:, h, :], Y_pair, pair_mask=p_mask))
    return torch.stack(losses).mean()


def gate_entropy(alpha: torch.Tensor) -> torch.Tensor:
    return -(alpha * torch.log(alpha.clamp_min(1e-9))).sum(dim=1)


def entropy_floor_loss(alpha: torch.Tensor, min_entropy: float = 1.5) -> torch.Tensor:
    H = gate_entropy(alpha)
    return F.relu(min_entropy - H).mean()


def total_pretrain_loss(outputs: dict, Y_pair: torch.Tensor, lambda_aux: float = 0.05,
                        lambda_ent: float = 0.001, min_gate_entropy: float = 1.5) -> dict:
    """Combined loss: L_final + λ_aux·L_aux + λ_ent·L_ent, plus pairwise_acc."""
    scores = outputs["scores"]
    scores_per_head = outputs["scores_per_head"]
    alpha = outputs["alpha"]
    p_mask = pairwise_mask(64, device=scores.device)
    loss_final = soft_pairwise_bce_loss(scores, Y_pair, pair_mask=p_mask)
    loss_aux = per_head_aux_loss(scores_per_head, Y_pair)
    loss_ent = entropy_floor_loss(alpha, min_entropy=min_gate_entropy)
    loss = loss_final + lambda_aux * loss_aux + lambda_ent * loss_ent
    acc = pairwise_accuracy(scores, Y_pair)
    return {"loss": loss, "loss_final": loss_final, "loss_aux": loss_aux,
            "loss_ent": loss_ent, "pairwise_acc": acc}
