"""Tests for P1 DP validation summary."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p1_validate_dp import resolve_worker_count, summarize_paths


def test_summarize_paths_counts_monotonic_and_adjacent_steps():
    results = [
        {"optimal_path": list(range(4)), "max_weight": 1.25},
        {"optimal_path": [3, 2, 1, 0], "max_weight": 2.5},
        {"optimal_path": [0, 2, 1, 3], "max_weight": 3.75},
    ]

    summary = summarize_paths(results, num_blocks=4)

    assert summary["num_sequences"] == 3
    assert summary["l2r_count"] == 1
    assert summary["r2l_count"] == 1
    assert summary["adj_counts"] == [3, 3, 1]
    assert summary["avg_adj"] == 7 / 3
    assert summary["passes_route_a_acceptance"] is False


def test_resolve_worker_count_caps_to_sequence_count_and_cpu_count():
    assert resolve_worker_count(requested=100, num_sequences=7, cpu_count=32) == 7
    assert resolve_worker_count(requested=0, num_sequences=7, cpu_count=32) == 1
    assert resolve_worker_count(requested=-3, num_sequences=7, cpu_count=32) == 1
    assert resolve_worker_count(requested=None, num_sequences=7, cpu_count=4) == 4
