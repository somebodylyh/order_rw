"""NR-1 Task 8: tests for attention-order matching metrics.

Pins extremes (identity / reversed / random) for the three hard-gate metrics
and the two diagnostic metrics.
"""
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_identity_metrics():
    from neural_readout.eval_metrics import compute_matching_metrics
    N = 64
    rank = np.arange(N)[None, :]
    s = -rank.astype(np.float32)  # earliest gets highest score
    out = compute_matching_metrics(scores=s, rank=rank)
    assert out["kendall_tau"] > 0.999
    assert out["pairwise_precedence_acc"] > 0.999
    assert out["spearman_rho"] > 0.999
    assert out["top1_first_node_match"] == 1.0
    assert out["first3_set_match"] == 1.0


def test_reversed_metrics():
    from neural_readout.eval_metrics import compute_matching_metrics
    N = 64
    rank = np.arange(N)[None, :]
    s = rank.astype(np.float32)  # earliest gets LOWEST score; fully inverted
    out = compute_matching_metrics(scores=s, rank=rank)
    assert out["kendall_tau"] < -0.999
    assert out["pairwise_precedence_acc"] < 0.001
    assert out["spearman_rho"] < -0.999
    assert out["top1_first_node_match"] == 0.0
    assert out["first3_set_match"] == 0.0


def test_random_metrics_near_zero():
    from neural_readout.eval_metrics import compute_matching_metrics
    rng = np.random.default_rng(0)
    M, N = 100, 64
    rank = np.stack([rng.permutation(N) for _ in range(M)])
    s = rng.standard_normal((M, N)).astype(np.float32)
    out = compute_matching_metrics(scores=s, rank=rank)
    assert abs(out["kendall_tau"]) < 0.05
    assert abs(out["pairwise_precedence_acc"] - 0.5) < 0.02
    # top-1 chance baseline ~ 1/64
    assert out["top1_first_node_match"] < 0.05
    # first-3 chance baseline ~ 3 hits over 64 spots when sampling 3
    assert out["first3_set_match"] < 0.10


def test_shape_mismatch_raises():
    from neural_readout.eval_metrics import compute_matching_metrics
    import pytest
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_matching_metrics(scores=np.zeros((2, 4)), rank=np.zeros((2, 5)))
