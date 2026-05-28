"""NR-1 Task 5: tests for teacher order diversity diagnostic.

Pins two corner cases that anchor the two interpretation regimes in spec §5.1b:
  - degenerate dataset (all sigmas identical) -> unique=1, entropy=0, mean_tau=1
  - diverse dataset (M random permutations) -> unique≈M, entropy high, mean_tau≈0
"""
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_degenerate_dataset():
    """All-identical sigmas -> unique=1, first-node entropy=0, mean pairwise tau=1."""
    from neural_readout.diversity_stats import teacher_diversity
    sigma = np.tile(np.arange(64), (8, 1))  # 8 identical permutations
    stats = teacher_diversity(sigma)
    assert stats["unique_sigma_count"] == 1
    assert abs(stats["first_node_entropy"]) < 1e-9
    assert stats["distinct_first3_prefix_count"] == 1
    assert abs(stats["mean_pairwise_tau"] - 1.0) < 1e-9


def test_diverse_dataset():
    """M independent random permutations -> high diversity, low pairwise tau."""
    from neural_readout.diversity_stats import teacher_diversity
    rng = np.random.default_rng(0)
    sigma = np.stack([rng.permutation(64) for _ in range(64)]).astype(np.int64)
    stats = teacher_diversity(sigma)
    assert stats["unique_sigma_count"] >= 60  # almost all distinct
    assert stats["first_node_entropy"] > 3.0  # high entropy (max log(64) ≈ 4.16)
    assert abs(stats["mean_pairwise_tau"]) < 0.1  # uncorrelated random perms


def test_single_sample_does_not_crash():
    """M=1 edge case: nothing to compare against, mean_tau defined as 1.0."""
    from neural_readout.diversity_stats import teacher_diversity
    sigma = np.arange(64)[None, :]
    stats = teacher_diversity(sigma)
    assert stats["unique_sigma_count"] == 1
    assert stats["mean_pairwise_tau"] == 1.0


def test_rejects_non_2d_input():
    from neural_readout.diversity_stats import teacher_diversity
    import pytest
    with pytest.raises(ValueError, match=r"\(M, N\)"):
        teacher_diversity(np.arange(64))  # 1D
