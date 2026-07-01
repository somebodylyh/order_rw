import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analyses.order_head_module import OrderHeadModule
from analyses.p7_gbeta_policy import GBETA_CKPT
from analyses.v3_group_credit import (
    GroupEMA,
    group_ids_for,
    group_rewards,
    per_sample_loss,
)


def test_per_sample_loss_returns_mean_for_each_sample():
    token_losses = torch.tensor([[1.0, 3.0], [2.0, 4.0]])

    result = per_sample_loss(token_losses)

    assert result.shape == (2,)
    assert torch.equal(result, torch.tensor([2.0, 3.0]))


def test_group_ids_and_rewards_are_contiguous_group_means():
    groups = group_ids_for(8, 2)

    assert all(isinstance(group, np.ndarray) for group in groups)
    assert [group.tolist() for group in groups] == [[0, 1], [2, 3], [4, 5], [6, 7]]

    per_sample = torch.arange(8, dtype=torch.float64)
    rewards = group_rewards(per_sample, groups)
    assert rewards.shape == (4,)
    assert rewards.device == per_sample.device
    assert rewards.dtype == per_sample.dtype
    assert torch.equal(rewards, torch.tensor([0.5, 2.5, 4.5, 6.5], dtype=torch.float64))


@pytest.mark.parametrize(
    ("batch_size", "m", "message"),
    [(0, 1, "batch_size"), (4, 0, "m"), (4, -1, "m"), (5, 2, "divisible")],
)
def test_group_ids_rejects_invalid_grouping(batch_size, m, message):
    with pytest.raises(ValueError, match=message):
        group_ids_for(batch_size, m)


@pytest.mark.parametrize(
    "invalid_group",
    [np.array([0.5, 1.0]), np.array([True, False])],
)
def test_group_rewards_rejects_non_integer_indices(invalid_group):
    with pytest.raises(ValueError, match="integer"):
        group_rewards(torch.arange(4.0), [invalid_group])


@pytest.mark.parametrize(
    "invalid_group",
    [np.array([[0, 1]]), np.array([-1, 0]), np.array([0, 4])],
)
def test_group_rewards_rejects_malformed_or_out_of_range_indices(invalid_group):
    with pytest.raises(ValueError):
        group_rewards(torch.arange(4.0), [invalid_group])


def test_group_ema_returns_pre_update_baseline_and_moves_state():
    ema = GroupEMA(2, alpha=0.5)

    first = ema.update(torch.tensor([2.0, 4.0]))
    assert torch.equal(first, torch.tensor([2.0, 4.0]))
    assert first.requires_grad is False

    second = ema.update(torch.tensor([0.0, 2.0], requires_grad=True))
    assert torch.equal(second, torch.tensor([2.0, 4.0]))
    assert torch.equal(ema.b, torch.tensor([1.0, 3.0]))
    assert ema.b.requires_grad is False

    third = ema.update(torch.tensor([0.0, 0.0]))
    assert torch.equal(third, torch.tensor([1.0, 3.0]))


@pytest.mark.parametrize("G", [0, -1, 1.5])
def test_group_ema_rejects_invalid_group_count(G):
    with pytest.raises(ValueError, match="G"):
        GroupEMA(G)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_group_ema_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        GroupEMA(2, alpha=alpha)


def test_group_ema_rejects_wrong_loss_shape():
    with pytest.raises(ValueError, match="shape"):
        GroupEMA(2).update(torch.ones(3))


def test_group_ema_state_follows_input_dtype():
    ema = GroupEMA(2)
    ema.update(torch.ones(2, dtype=torch.float32))

    baseline = ema.update(torch.zeros(2, dtype=torch.float64))

    assert baseline.dtype == torch.float64
    assert ema.b.dtype == torch.float64


def test_orderhead_device_follows_to():
    order_head = OrderHeadModule(GBETA_CKPT, device="cpu")
    assert order_head.device == next(order_head.gbeta.parameters()).device

    order_head.to("meta")
    assert order_head.device.type == "meta"
    assert order_head.device == next(order_head.gbeta.parameters()).device
