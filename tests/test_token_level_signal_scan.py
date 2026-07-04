from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import kendalltau


def test_random_reveal_orders_are_independent_and_deterministic():
    from analyses.token_level_signal_scan import random_reveal_orders

    first = random_reveal_orders(3, seed=17, num_blocks=4, block_len=2)
    second = random_reveal_orders(3, seed=17, num_blocks=4, block_len=2)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 8)
    assert len({tuple(row) for row in first}) == 3

    for sample_idx, row in enumerate(first):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17 + sample_idx)
        blocks = torch.randperm(4, generator=generator).numpy()
        expected = (blocks[:, None] * 2 + np.arange(2)[None, :]).reshape(-1)
        np.testing.assert_array_equal(row, expected)


def test_order_kendall_tau_uses_item_rank_vectors():
    from analyses.token_level_signal_scan import order_kendall_tau, order_to_rank

    left = np.array([1, 2, 0])
    right = np.array([1, 0, 2])
    expected = kendalltau(order_to_rank(left), order_to_rank(right))[0]
    direct = kendalltau(left, right)[0]

    assert order_kendall_tau(left, right) == pytest.approx(expected)
    assert expected == pytest.approx(1 / 3)
    assert direct == pytest.approx(-1.0)


def test_model_orders_map_to_physical_blocks_and_token_offsets():
    from analyses.token_level_signal_scan import (
        model_block_order_to_physical,
        model_token_order_to_physical,
    )

    inv_perm_model_to_phys = np.array([2, 0, 1])
    block_model = np.array([1, 2, 0])
    token_model = np.array([0, 1, 4, 5, 2, 3])

    np.testing.assert_array_equal(
        model_block_order_to_physical(block_model, inv_perm_model_to_phys),
        np.array([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        model_token_order_to_physical(token_model, inv_perm_model_to_phys, block_len=2),
        np.array([4, 5, 2, 3, 0, 1]),
    )


def test_coarsegrain_token_B_recovers_block_B_off_diagonal():
    from analyses.token_level_signal_scan import coarsegrain_token_B

    token_B = np.arange(64, dtype=np.float64).reshape(8, 8)
    np.fill_diagonal(token_B, 0.0)
    missing_source_token = 5

    actual = coarsegrain_token_B(
        token_B,
        block_len=2,
        missing_source_token=missing_source_token,
    )

    expected = np.zeros((4, 4), dtype=np.float64)
    for source_block in range(4):
        source_tokens = [2 * source_block, 2 * source_block + 1]
        source_tokens = [t for t in source_tokens if t != missing_source_token]
        for target_block in range(4):
            if source_block == target_block:
                continue
            target_tokens = [2 * target_block, 2 * target_block + 1]
            expected[source_block, target_block] = token_B[
                np.ix_(source_tokens, target_tokens)
            ].mean()

    np.testing.assert_allclose(actual, expected)


def test_canonical_sample_count_is_M_times_batch_size():
    from analyses.token_level_signal_scan import canonical_sample_count

    assert canonical_sample_count(M=40, batch_size=4) == 160
    assert canonical_sample_count(M=20, batch_size=8) == 160
    with pytest.raises(ValueError, match="positive"):
        canonical_sample_count(M=0, batch_size=4)
