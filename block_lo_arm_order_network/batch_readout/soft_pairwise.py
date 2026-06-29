"""Soft pairwise BCE loss, accuracy, and regularisation for g_beta pretraining.

All losses operate on the upper-triangle of the 64×64 pairwise matrix.
The diagonal (always 0.5) is excluded — it carries no ordering signal.
Teacher Y values of exactly 0.5 on off-diagonal entries are treated as
ties and excluded from accuracy (but still contribute to BCE loss).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pairwise mask (strict upper triangle)
# ---------------------------------------------------------------------------

def pairwise_mask(N: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean mask of shape (N, N): True for i < j (strict upper triangle)."""
    return torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1,
    )


def non_tie_mask(Y_pair: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """Boolean mask: True where Y_pair is not (close to) 0.5."""
    return ~torch.isclose(Y_pair, torch.tensor(0.5, device=Y_pair.device,
                                                dtype=Y_pair.dtype), atol=atol)


# ---------------------------------------------------------------------------
# Soft pairwise BCE loss
# ---------------------------------------------------------------------------

def soft_pairwise_bce_loss(
    scores: torch.Tensor,
    Y_pair: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy on score-difference logits versus soft teacher.

    For each ordered pair (i, j):
        logit  = score_i - score_j
        target = Y_pair[i, j]  ∈ [0, 1]

    Intuitively: if block i should come before block j (Y > 0.5),
    then score_i should be larger than score_j.

    Args:
        scores: (B, 64) per-block reveal-priority logits.
        Y_pair: (B, 64, 64) soft pairwise teacher.
        pair_mask: (64, 64) bool — which entries to include.
                   Defaults to strict upper triangle.

    Returns:
        Scalar loss (mean over included pairs).
    """
    B, N = scores.shape
    if Y_pair.shape != (B, N, N):
        raise ValueError(
            f"Y_pair must be ({B}, {N}, {N}), got {Y_pair.shape}"
        )

    if pair_mask is None:
        pair_mask = pairwise_mask(N, device=scores.device)

    # diff[b, i, j] = score[b, i] - score[b, j]
    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # (B, N, N)

    flat_diff = diff[:, pair_mask]    # (B, num_pairs)
    flat_target = Y_pair[:, pair_mask]  # (B, num_pairs)

    return F.binary_cross_entropy_with_logits(flat_diff, flat_target)


# ---------------------------------------------------------------------------
# Pairwise accuracy (off-diagonal, non-tie)
# ---------------------------------------------------------------------------

def pairwise_accuracy(
    scores: torch.Tensor,
    Y_pair: torch.Tensor,
) -> float:
    """Fraction of off-diagonal, non-tie pairs where the sign of the score
    difference matches whether the teacher prefers i before j (Y > 0.5).

    Diagonal entries and entries where Y ≈ 0.5 (ties) are excluded.

    Args:
        scores: (B, 64).
        Y_pair: (B, 64, 64).

    Returns:
        Accuracy ∈ [0, 1] as a Python float.
    """
    B, N = scores.shape
    device = scores.device

    p_mask = pairwise_mask(N, device=device)        # (N, N) upper triangle
    nt_mask = non_tie_mask(Y_pair)                    # (B, N, N)
    eval_mask = p_mask.unsqueeze(0) & nt_mask         # (B, N, N)

    total = eval_mask.sum()
    if total == 0:
        return 1.0  # no evaluable pairs → perfect by vacuity

    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # (B, N, N)
    # sign(diff) > 0  ⇔  Y > 0.5  means correct
    pred_correct = (diff > 0) == (Y_pair > 0.5)  # (B, N, N)
    correct = (pred_correct & eval_mask).sum()

    return float((correct / total).item())


# ---------------------------------------------------------------------------
# Per-head auxiliary loss
# ---------------------------------------------------------------------------

def per_head_aux_loss(
    scores_per_head: torch.Tensor,
    Y_pair: torch.Tensor,
) -> torch.Tensor:
    """Average of soft pairwise BCE computed independently for each head.

    This is a per-head regulariser — it encourages each individual head's
    block scores to align with the consensus teacher, preventing any single
    head from drifting into degenerate solutions.

    Args:
        scores_per_head: (B, H, 64).
        Y_pair: (B, 64, 64) — shared across heads.

    Returns:
        Scalar loss = mean over heads of BCE(scores_h, Y).
    """
    B, H, N = scores_per_head.shape
    losses = []
    p_mask = pairwise_mask(N, device=scores_per_head.device)
    for h in range(H):
        lh = soft_pairwise_bce_loss(scores_per_head[:, h, :], Y_pair, pair_mask=p_mask)
        losses.append(lh)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Entropy regularisation
# ---------------------------------------------------------------------------

def gate_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of the student gate weights α.

    Args:
        alpha: (B, H), rows sum to 1.

    Returns:
        (B,) entropy per sample.  Max = log(H) for uniform; min ≈ 0 for
        one-hot (collapsed to a single head).
    """
    # H = -(α · log(α)), with 0·log(0) = 0.
    return -(alpha * torch.log(alpha.clamp_min(1e-9))).sum(dim=1)


def entropy_floor_loss(
    alpha: torch.Tensor,
    min_entropy: float = 1.5,
) -> torch.Tensor:
    """Penalise gate entropy below a floor: ReLU(min_entropy - H(α)).

    For H=8 heads, max entropy = log(8) ≈ 2.08.  A floor of log(4) ≈ 1.39
    or 1.5 ensures the gate uses at least ~4 effective heads.

    Args:
        alpha: (B, H).
        min_entropy: minimum allowed entropy (nats).

    Returns:
        Scalar loss (mean over batch).
    """
    H = gate_entropy(alpha)  # (B,)
    return F.relu(min_entropy - H).mean()


# ---------------------------------------------------------------------------
# Combined pretraining loss
# ---------------------------------------------------------------------------

def total_pretrain_loss(
    outputs: dict,
    Y_pair: torch.Tensor,
    lambda_aux: float = 0.05,
    lambda_ent: float = 0.001,
    min_gate_entropy: float = 1.5,
) -> dict:
    """Combined label-free g_beta pretraining loss.

        L = L_final + λ_aux · L_aux + λ_ent · L_ent

    where:
      L_final = soft_pairwise_bce(scores, Y)      — imitation of teacher
      L_aux   = per_head_aux_loss(scores_h, Y)    — per-head alignment
      L_ent   = entropy_floor_loss(α, H_min)       — anti-collapse

    Args:
        outputs: dict from L0DynamicGBeta.forward():
            scores, scores_per_head, alpha, gate_logits, ...
        Y_pair: (B, 64, 64) soft pairwise teacher.
        lambda_aux: weight on per-head auxiliary loss.
        lambda_ent: weight on entropy floor penalty.
        min_gate_entropy: floor for gate entropy (nats).

    Returns:
        dict with keys: loss, loss_final, loss_aux, loss_ent, and
        pairwise_acc (float) for logging.
    """
    scores = outputs["scores"]
    scores_per_head = outputs["scores_per_head"]
    alpha = outputs["alpha"]

    B = scores.shape[0]
    p_mask = pairwise_mask(64, device=scores.device)

    loss_final = soft_pairwise_bce_loss(scores, Y_pair, pair_mask=p_mask)
    loss_aux = per_head_aux_loss(scores_per_head, Y_pair)
    loss_ent = entropy_floor_loss(alpha, min_entropy=min_gate_entropy)

    loss = loss_final + lambda_aux * loss_aux + lambda_ent * loss_ent
    acc = pairwise_accuracy(scores, Y_pair)

    return {
        "loss": loss,
        "loss_final": loss_final,
        "loss_aux": loss_aux,
        "loss_ent": loss_ent,
        "pairwise_acc": acc,
    }
