import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_build_none_separated_B_keeps_none_as_node0_and_content_blocks_physical():
    from none_separated_block_graph import build_none_separated_B

    A = np.zeros((3, 4), dtype=np.float32)
    A[:, 0] = [10, 20, 30]       # None -> block i
    A[:, 1:] = np.arange(9).reshape(3, 3)

    B = build_none_separated_B(A)

    assert B.shape == (4, 4)
    np.testing.assert_allclose(B[0, 1:], [10, 20, 30])
    expected_content = A[:, 1:].T.copy()
    np.fill_diagonal(expected_content, 0.0)
    np.testing.assert_allclose(B[1:, 1:], expected_content)
    assert np.allclose(B[:, 0], 0.0)
    assert np.allclose(np.diag(B), 0.0)


def test_rollout_from_none_outputs_only_content_blocks_and_recovers_chain():
    from none_separated_block_graph import build_none_separated_B, rollout_from_none

    N = 5
    A = np.zeros((N, N + 1), dtype=np.float32)
    A[0, 0] = 5.0  # None -> block0
    for i in range(1, N):
        A[i, i] = 5.0  # source block i-1 -> target block i

    B = build_none_separated_B(A)
    sigma = rollout_from_none(B)

    np.testing.assert_array_equal(sigma, np.arange(N))
    assert 0 not in (sigma + 1)[1:]  # None node is not in the content order.


def test_discovery_metrics_reports_phys0_rank_and_prefixes():
    from none_separated_block_graph import discovery_metrics

    metrics = discovery_metrics(np.array([1, 0, 2, 3, 4, 5, 7, 6]))

    assert metrics["first_block"] == 1
    assert metrics["first_is_phys0"] is False
    assert metrics["phys0_rank"] == 1
    assert metrics["prefix4_exact"] is False
    assert metrics["prefix8_exact"] is False
    assert metrics["prefix4_overlap"] == 4
    assert metrics["prefix8_overlap"] == 8


def test_destroyed_controls_preserve_none_node_shape_and_zero_diag():
    from none_separated_block_graph import (
        build_none_separated_B,
        content_label_permutation_control,
        entry_shuffled_control,
    )

    rng = np.random.default_rng(0)
    A = rng.random((6, 7)).astype(np.float32)
    B = build_none_separated_B(A)

    for control in (entry_shuffled_control(B, seed=1), content_label_permutation_control(B, seed=1)):
        assert control.shape == B.shape
        assert np.allclose(control[:, 0], 0.0)
        assert np.allclose(np.diag(control), 0.0)


def test_gate_status_strong_weak_fail():
    from none_separated_block_graph import classify_gate_status

    strong = {
        "first_is_phys0": True,
        "phys0_rank": 0,
        "tau_vs_l2r": 0.72,
        "prefix4_overlap": 3,
        "prefix8_overlap": 5,
    }
    weak = {
        "first_is_phys0": False,
        "phys0_rank": 2,
        "tau_vs_l2r": 0.55,
        "prefix4_overlap": 2,
        "prefix8_overlap": 4,
    }
    fail = {
        "first_is_phys0": False,
        "phys0_rank": 20,
        "tau_vs_l2r": 0.29,
        "prefix4_overlap": 0,
        "prefix8_overlap": 0,
    }

    assert classify_gate_status(strong, destroyed_abs_tau_mean=0.08) == "strong_pass"
    assert classify_gate_status(weak, destroyed_abs_tau_mean=0.08) == "weak_pass"
    assert classify_gate_status(strong, destroyed_abs_tau_mean=0.12) == "fail"
    assert classify_gate_status(fail, destroyed_abs_tau_mean=0.08) == "fail"


def test_combined_discovery_score_prefers_better_start_tau_and_prefix():
    from none_separated_block_graph import combined_discovery_score

    weak = {
        "first_is_phys0": False,
        "phys0_rank": 3,
        "tau_vs_l2r": 0.50,
        "prefix4_overlap": 2,
        "prefix8_overlap": 3,
    }
    better = {
        "first_is_phys0": True,
        "phys0_rank": 0,
        "tau_vs_l2r": 0.60,
        "prefix4_overlap": 3,
        "prefix8_overlap": 5,
    }

    assert combined_discovery_score(better, 0.05) > combined_discovery_score(weak, 0.05)


def test_rollout_by_method_supports_cdl_and_static_none_edge():
    from none_separated_block_graph import build_none_separated_B, rollout_by_method

    N = 5
    A = np.zeros((N, N + 1), dtype=np.float32)
    A[0, 0] = 5.0
    for i in range(1, N):
        A[i, i] = 5.0
    B = build_none_separated_B(A)
    np.testing.assert_array_equal(rollout_by_method(B, "C-D+L"), np.arange(N))

    A2 = np.zeros((N, N + 1), dtype=np.float32)
    A2[:, 0] = [0.2, 0.9, 0.1, 0.4, 0.3]
    B2 = build_none_separated_B(A2)
    np.testing.assert_array_equal(rollout_by_method(B2, "none_edge"), [1, 3, 4, 0, 2])
