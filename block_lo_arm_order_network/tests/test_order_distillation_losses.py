"""Tests for direct full-order distillation losses."""

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.order_distillation_losses import (  # noqa: E402
    listmle_loss,
    rank_kl_loss,
    rank_kl_target,
)


def test_listmle_prefers_teacher_order():
    order = torch.tensor([[0, 1, 2, 3]])
    perfect = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    reversed_logits = perfect.flip(1)
    assert listmle_loss(perfect, order) < listmle_loss(
        reversed_logits, order,
    )


def test_listmle_supports_batched_64_block_inputs():
    logits = torch.randn(3, 64)
    order = torch.stack([torch.randperm(64) for _ in range(3)])
    assert listmle_loss(logits, order).ndim == 0
    assert listmle_loss(logits, order, reduction="none").shape == (3,)


def test_listmle_backward_is_finite():
    logits = torch.randn(3, 64, requires_grad=True)
    order = torch.stack([torch.randperm(64) for _ in range(3)])
    loss = listmle_loss(logits, order)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "logits,order,match",
    [
        (torch.randn(64), torch.arange(64).unsqueeze(0), "2D"),
        (torch.randn(2, 64), torch.arange(64).unsqueeze(0), "same shape"),
        (torch.randn(1, 4), torch.tensor([[0.0, 1.0, 2.0, 3.0]]), "integer"),
        (torch.randn(1, 4), torch.tensor([[0, 1, 2, 4]]), "range"),
        (torch.randn(1, 4), torch.tensor([[0, 1, 1, 3]]), "permutation"),
    ],
)
def test_listmle_validates_teacher_permutation(logits, order, match):
    with pytest.raises((TypeError, ValueError), match=match):
        listmle_loss(logits, order)


def test_listmle_validates_weights_and_reduction():
    logits = torch.randn(1, 4)
    order = torch.tensor([[0, 1, 2, 3]])
    with pytest.raises(ValueError, match="position_weights"):
        listmle_loss(logits, order, position_weights=torch.ones(3))
    with pytest.raises(ValueError, match="reduction"):
        listmle_loss(logits, order, reduction="median")


def test_rank_kl_target_is_normalized_and_respects_order():
    order = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    target = rank_kl_target(order, temperature=4.0)
    torch.testing.assert_close(target.sum(dim=1), torch.ones(2))
    assert target[0, 2] > target[0, 0] > target[0, 3] > target[0, 1]


def test_rank_kl_prefers_logits_matching_teacher_priority():
    order = torch.tensor([[0, 1, 2, 3]])
    perfect = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    reversed_logits = perfect.flip(1)
    assert rank_kl_loss(perfect, order, temperature=1.0) < rank_kl_loss(
        reversed_logits, order, temperature=1.0,
    )


def test_rank_kl_backward_is_finite():
    logits = torch.randn(2, 64, requires_grad=True)
    order = torch.stack([torch.randperm(64) for _ in range(2)])
    rank_kl_loss(logits, order, temperature=4.0).backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_rank_kl_rejects_nonpositive_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        rank_kl_target(torch.tensor([[0, 1, 2, 3]]), temperature)
