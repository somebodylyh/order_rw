"""Tests for objective-agnostic CDL order fidelity metrics."""

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.order_distillation_metrics import (  # noqa: E402
    hard_pairwise_accuracy,
    kendall_tau_batch,
    order_metrics,
    prefix_overlap,
    predicted_order,
)


def _scores_for(order):
    order = torch.as_tensor(order, dtype=torch.long)
    scores = torch.empty(order.shape, dtype=torch.float32)
    values = torch.arange(
        order.shape[1], 0, -1, dtype=torch.float32,
    ).expand_as(scores)
    scores.scatter_(1, order, values)
    return scores


def test_perfect_order_has_perfect_metrics():
    order = torch.stack([torch.arange(64), torch.randperm(64)])
    metrics = order_metrics(_scores_for(order), order)
    assert metrics["kendall_tau"] == pytest.approx(1.0)
    assert metrics["pairwise_acc"] == pytest.approx(1.0)
    assert metrics["prefix8"] == pytest.approx(1.0)
    assert metrics["prefix16"] == pytest.approx(1.0)


def test_reversed_order_has_negative_tau_and_zero_pairwise_accuracy():
    order = torch.arange(64).unsqueeze(0)
    scores = _scores_for(order.flip(1))
    assert kendall_tau_batch(scores, order) == pytest.approx(-1.0)
    assert hard_pairwise_accuracy(scores, order) == pytest.approx(0.0)


def test_prefix_overlap_is_normalized():
    teacher = torch.arange(64).unsqueeze(0)
    predicted = torch.cat([
        torch.arange(4),
        torch.arange(8, 12),
        torch.arange(4, 8),
        torch.arange(12, 64),
    ]).unsqueeze(0)
    scores = _scores_for(predicted)
    assert prefix_overlap(scores, teacher, 8) == pytest.approx(0.5)


def test_predicted_order_sorts_larger_scores_first():
    scores = torch.tensor([[0.1, 2.0, -1.0, 1.0]])
    torch.testing.assert_close(
        predicted_order(scores), torch.tensor([[1, 3, 0, 2]]),
    )


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        order_metrics(torch.randn(2, 64), torch.arange(64).unsqueeze(0))
