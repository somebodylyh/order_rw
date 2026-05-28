"""BR-1 Task 4: tests for teacher-label diversity stats."""
import sys
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_identical_labels_yield_degenerate_diversity():
    from batch_readout.diversity_batch import teacher_diversity_stats
    sigma = np.tile(np.arange(64), (50, 1))
    stats = teacher_diversity_stats(sigma)
    assert stats["mean_pairwise_tau"] == pytest.approx(1.0, abs=1e-6)
    assert stats["unique_sigma_ratio"] == pytest.approx(1 / 50)
    assert stats["first_step_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert stats["first_step_entropy_max"] == pytest.approx(np.log(64))


def test_random_labels_yield_high_diversity():
    from batch_readout.diversity_batch import teacher_diversity_stats
    rng = np.random.default_rng(0)
    sigma = np.stack([rng.permutation(64) for _ in range(200)])
    stats = teacher_diversity_stats(sigma)
    assert -0.05 < stats["mean_pairwise_tau"] < 0.05, stats["mean_pairwise_tau"]
    assert stats["unique_sigma_ratio"] == 1.0
    # 64 buckets, ~uniform first step -> entropy near log(64) ≈ 4.16
    assert stats["first_step_entropy"] > 3.5, stats["first_step_entropy"]


def test_M1_returns_nan_pairwise():
    from batch_readout.diversity_batch import teacher_diversity_stats
    sigma = np.arange(8)[None, :]
    stats = teacher_diversity_stats(sigma)
    assert np.isnan(stats["mean_pairwise_tau"])
    assert stats["unique_sigma_ratio"] == 1.0


def test_invalid_shape_rejected():
    from batch_readout.diversity_batch import teacher_diversity_stats
    with pytest.raises(ValueError, match="2D"):
        teacher_diversity_stats(np.arange(8))
