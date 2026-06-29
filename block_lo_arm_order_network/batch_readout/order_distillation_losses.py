"""Listwise and distributional losses for full CDL order distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def validate_teacher_order(
    student_logits: torch.Tensor,
    teacher_order: torch.Tensor,
) -> None:
    if student_logits.ndim != 2:
        raise ValueError(
            f"student_logits must be 2D [B, N], got {student_logits.shape}"
        )
    if teacher_order.shape != student_logits.shape:
        raise ValueError(
            "teacher_order must have same shape as student_logits, "
            f"got {teacher_order.shape} vs {student_logits.shape}"
        )
    if teacher_order.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"teacher_order must have integer dtype, got {teacher_order.dtype}"
        )

    _, n_items = student_logits.shape
    if teacher_order.numel() and (
        teacher_order.min().item() < 0
        or teacher_order.max().item() >= n_items
    ):
        raise ValueError(
            f"teacher_order indices must be in range [0, {n_items - 1}]"
        )

    expected = torch.arange(
        n_items, device=teacher_order.device, dtype=teacher_order.dtype,
    ).expand_as(teacher_order)
    if not torch.equal(torch.sort(teacher_order, dim=1).values, expected):
        raise ValueError("each teacher_order row must be a permutation")


def listmle_loss(
    student_logits: torch.Tensor,
    teacher_order: torch.Tensor,
    position_weights: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Plackett-Luce negative log likelihood for a full teacher permutation."""
    validate_teacher_order(student_logits, teacher_order)
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"unknown reduction {reduction!r}")

    _, n_items = student_logits.shape
    ordered_logits = student_logits.gather(1, teacher_order.long())
    suffix_lse = torch.logcumsumexp(
        ordered_logits.flip(dims=[1]), dim=1,
    ).flip(dims=[1])
    per_position = suffix_lse - ordered_logits

    if position_weights is not None:
        if position_weights.shape != (n_items,):
            raise ValueError(
                f"position_weights must be [{n_items}], "
                f"got {position_weights.shape}"
            )
        weights = position_weights.to(
            device=student_logits.device, dtype=student_logits.dtype,
        )
        per_position = per_position * weights.unsqueeze(0)

    per_sample = per_position.sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def order_to_rank_tensor(teacher_order: torch.Tensor) -> torch.Tensor:
    if teacher_order.ndim != 2:
        raise ValueError(
            f"teacher_order must be 2D [B, N], got {teacher_order.shape}"
        )
    dummy_logits = torch.empty(
        teacher_order.shape, device=teacher_order.device, dtype=torch.float32,
    )
    validate_teacher_order(dummy_logits, teacher_order)
    ranks = torch.empty_like(teacher_order)
    positions = torch.arange(
        teacher_order.shape[1],
        device=teacher_order.device,
        dtype=teacher_order.dtype,
    ).expand_as(teacher_order)
    ranks.scatter_(1, teacher_order.long(), positions)
    return ranks


def rank_kl_target(
    teacher_order: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    ranks = order_to_rank_tensor(teacher_order).float()
    return F.softmax(-ranks / float(temperature), dim=1)


def rank_kl_loss(
    student_logits: torch.Tensor,
    teacher_order: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    validate_teacher_order(student_logits, teacher_order)
    target = rank_kl_target(teacher_order, temperature).to(
        dtype=student_logits.dtype,
    )
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        target,
        reduction="batchmean",
    )
