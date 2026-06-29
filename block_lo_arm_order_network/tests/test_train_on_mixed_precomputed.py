"""Tests for precomputed-order sampling in train_on_mixed."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_on_mixed import NUM_BLOCKS, get_checkpoint_metric, sample_mixed_block_orders


def test_precomputed_order_source_uses_sample_indices():
    A = torch.zeros(2, NUM_BLOCKS, NUM_BLOCKS)
    sample_indices = torch.tensor([2, 0], dtype=torch.long)
    precomputed_orders = torch.stack([
        torch.arange(NUM_BLOCKS - 1, -1, -1),
        torch.arange(NUM_BLOCKS),
        torch.roll(torch.arange(NUM_BLOCKS), shifts=3),
    ])

    orders = sample_mixed_block_orders(
        A,
        alpha=1.0,
        order_source="precomputed",
        on_model=None,
        temperature=1.0,
        device=torch.device("cpu"),
        precomputed_orders=precomputed_orders,
        sample_indices=sample_indices,
    )

    assert torch.equal(orders[0], precomputed_orders[2])
    assert torch.equal(orders[1], precomputed_orders[0])


def test_precomputed_order_source_honors_alpha_zero_random_branch():
    A = torch.zeros(1, NUM_BLOCKS, NUM_BLOCKS)
    precomputed_orders = torch.arange(NUM_BLOCKS).unsqueeze(0)

    orders = sample_mixed_block_orders(
        A,
        alpha=0.0,
        order_source="precomputed",
        on_model=None,
        temperature=1.0,
        device=torch.device("cpu"),
        precomputed_orders=precomputed_orders,
        sample_indices=torch.tensor([0], dtype=torch.long),
    )

    assert sorted(orders[0].tolist()) == list(range(NUM_BLOCKS))


def test_precomputed_order_source_reuses_order_pool_for_global_indices():
    precomputed_orders = torch.stack([
        torch.arange(NUM_BLOCKS),
        torch.roll(torch.arange(NUM_BLOCKS), shifts=1),
        torch.roll(torch.arange(NUM_BLOCKS), shifts=2),
    ])

    orders = sample_mixed_block_orders(
        None,
        alpha=1.0,
        order_source="precomputed",
        on_model=None,
        temperature=1.0,
        device=torch.device("cpu"),
        precomputed_orders=precomputed_orders,
        sample_indices=torch.tensor([10001, 10002], dtype=torch.long),
    )

    assert torch.equal(orders[0], precomputed_orders[2])
    assert torch.equal(orders[1], precomputed_orders[0])


def test_checkpoint_metric_can_use_val_ar():
    val_losses = {"train_mode": 4.2, "ar": 4.0}

    assert get_checkpoint_metric(val_losses, "ar") == 4.0
