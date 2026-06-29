"""Objective-agnostic fidelity metrics for full CDL order distillation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau

from batch_readout.order_distillation_losses import validate_teacher_order


def _validate(scores: torch.Tensor, teacher_order: torch.Tensor) -> None:
    if scores.shape != teacher_order.shape:
        raise ValueError(
            "scores and teacher_order must have the same shape, "
            f"got {scores.shape} vs {teacher_order.shape}"
        )
    validate_teacher_order(scores, teacher_order)


def predicted_order(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores must be [B, N], got {scores.shape}")
    return scores.argsort(dim=1, descending=True)


def _order_to_rank(order: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty_like(order)
    positions = torch.arange(
        order.shape[1], device=order.device, dtype=order.dtype,
    ).expand_as(order)
    ranks.scatter_(1, order, positions)
    return ranks


def hard_pairwise_accuracy(
    scores: torch.Tensor,
    teacher_order: torch.Tensor,
) -> float:
    _validate(scores, teacher_order)
    teacher_ranks = _order_to_rank(teacher_order)
    target = teacher_ranks.unsqueeze(-1) < teacher_ranks.unsqueeze(-2)
    predicted = scores.unsqueeze(-1) > scores.unsqueeze(-2)
    n_items = scores.shape[1]
    mask = torch.triu(
        torch.ones(
            n_items, n_items, dtype=torch.bool, device=scores.device,
        ),
        diagonal=1,
    ).unsqueeze(0).expand(scores.shape[0], -1, -1)
    return float((predicted[mask] == target[mask]).float().mean().item())


def prefix_overlap(
    scores: torch.Tensor,
    teacher_order: torch.Tensor,
    k: int,
) -> float:
    _validate(scores, teacher_order)
    n_items = scores.shape[1]
    if not 0 < k <= n_items:
        raise ValueError(f"k must be in [1, {n_items}], got {k}")
    pred = predicted_order(scores)[:, :k]
    target = teacher_order[:, :k]
    pred_mask = torch.zeros(
        scores.shape[0], n_items, dtype=torch.bool, device=scores.device,
    )
    target_mask = torch.zeros_like(pred_mask)
    pred_mask.scatter_(1, pred, True)
    target_mask.scatter_(1, target, True)
    return float((pred_mask & target_mask).sum(dim=1).float().mean().item() / k)


def kendall_tau_batch(
    scores: torch.Tensor,
    teacher_order: torch.Tensor,
) -> float:
    _validate(scores, teacher_order)
    pred = predicted_order(scores).detach().cpu()
    target = teacher_order.detach().cpu()
    pred_ranks = _order_to_rank(pred).numpy()
    target_ranks = _order_to_rank(target).numpy()
    values = []
    for pred_rank, target_rank in zip(pred_ranks, target_ranks):
        tau, _ = kendalltau(pred_rank, target_rank)
        if not np.isnan(tau):
            values.append(float(tau))
    return float(np.mean(values)) if values else float("nan")


def order_metrics(
    scores: torch.Tensor,
    teacher_order: torch.Tensor,
) -> dict:
    return {
        "kendall_tau": kendall_tau_batch(scores, teacher_order),
        "pairwise_acc": hard_pairwise_accuracy(scores, teacher_order),
        "prefix8": prefix_overlap(scores, teacher_order, min(8, scores.shape[1])),
        "prefix16": prefix_overlap(
            scores, teacher_order, min(16, scores.shape[1]),
        ),
    }
