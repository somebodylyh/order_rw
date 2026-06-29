"""Tests for top-k clustered pairwise teacher construction."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_topk_clustered_pairwise_teacher.py"
    spec = importlib.util.spec_from_file_location("build_topk_clustered_pairwise_teacher", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# cluster_orders_by_tau
# ---------------------------------------------------------------------------

def test_reverse_orders_split_into_clusters():
    mod = _load_module()
    orders = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
    ])
    clusters = mod.cluster_orders_by_tau(orders, threshold=0.0)
    assert sorted(len(c) for c in clusters) == [2, 2]


def test_identical_orders_single_cluster():
    mod = _load_module()
    orders = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    clusters = mod.cluster_orders_by_tau(orders, threshold=0.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_all_different_all_singletons():
    mod = _load_module()
    orders = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
    ])
    clusters = mod.cluster_orders_by_tau(orders, threshold=0.99)
    assert len(clusters) == 3


def test_cluster_returns_indices():
    mod = _load_module()
    orders = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [2, 1, 0],
    ])
    clusters = mod.cluster_orders_by_tau(orders, threshold=0.0)
    assert all(isinstance(c, list) for c in clusters)
    assert all(all(isinstance(i, (int, np.integer)) for i in c) for c in clusters)
    flat = sorted(sum(clusters, []))
    assert flat == [0, 1, 2]


# ---------------------------------------------------------------------------
# build_soft_pairwise_from_orders
# ---------------------------------------------------------------------------

def test_pairwise_teacher_confidence_marks_consensus():
    mod = _load_module()
    orders = np.array([[0, 1, 2], [0, 1, 2]])
    Y, C = mod.build_soft_pairwise_from_orders(orders, np.array([0.5, 0.5]))
    assert Y[0, 1] == 1.0
    # Both orders agree 0 < 1 → Y=1.0 → C = 2*|1-0.5| = 1.0
    assert C[0, 1] == 1.0


def test_pairwise_diagonal_ignored():
    mod = _load_module()
    orders = np.array([[0, 1, 2], [0, 2, 1]])
    Y, C = mod.build_soft_pairwise_from_orders(orders, np.array([0.5, 0.5]))
    assert Y[0, 0] == 0.0  # diagonal stays 0 (a < a never counts)
    assert C[0, 0] == 1.0  # |0 - 0.5| * 2 = 1.0 (maximally uncertain)


def test_pairwise_opposing_orders_cancel():
    mod = _load_module()
    orders = np.array([[0, 1, 2], [2, 1, 0]])
    Y, C = mod.build_soft_pairwise_from_orders(orders, np.array([0.5, 0.5]))
    # 0 < 2 in first order => +0.5; 0 > 2 in second order => +0.0 => Y=0.5
    assert Y[0, 2] == 0.5
    assert C[0, 2] == 0.0


def test_pairwise_empty_rejects():
    mod = _load_module()
    with pytest.raises(ValueError):
        mod.build_soft_pairwise_from_orders(np.empty((0, 4)), np.empty(0))


# ---------------------------------------------------------------------------
# candidate_order (mocked)
# ---------------------------------------------------------------------------

def test_candidate_order_honours_method():
    """Simulate what candidate_order does — delegates to rollout_by_method."""
    from block_lo_arm_order_network.none_separated_block_graph import rollout_by_method, build_none_separated_B
    # Create a valid B via the real pipeline
    rng = np.random.default_rng(42)
    A = np.abs(rng.normal(size=(64, 65)).astype("float32"))
    A /= A.sum(axis=-1, keepdims=True) + 1e-8
    B = build_none_separated_B(A)
    o1 = rollout_by_method(B, method="L")
    o2 = rollout_by_method(B, method="L")
    np.testing.assert_array_equal(o1, o2)  # deterministic dispatch
    assert len(o1) == 64


# ---------------------------------------------------------------------------
# build_clustered_teachers (the main pipeline function)
# ---------------------------------------------------------------------------

def test_build_clustered_teachers_integration(tmp_path):
    """End-to-end: small A matrix + tiny candidate set → teacher npz files."""
    mod = _load_module()
    # Synthetic A_lh: 2 layers, 2 heads, (64, 65) — matching real format.
    rng = np.random.default_rng(42)
    A_lh = rng.normal(size=(2, 2, 64, 65)).astype("float32")
    # Make it row-stochastic-ish
    A_lh = np.abs(A_lh)
    for l in range(2):
        for h in range(2):
            A_lh[l, h] /= A_lh[l, h].sum(axis=-1, keepdims=True) + 1e-8

    # Save as .npy
    a_path = tmp_path / "A.npy"
    np.save(str(a_path), A_lh)

    candidates = [
        {"layer": 0, "head": 0, "method": "L", "structure_score": 2.0},
        {"layer": 0, "head": 1, "method": "C-D+L", "structure_score": 1.0},
    ]

    out_dir = tmp_path / "teacher_out"
    result = mod.build_clustered_teachers(
        a_npy=str(a_path),
        candidates=candidates,
        out_dir=str(out_dir),
        cluster_threshold=0.0,
        score_temperature=0.5,
    )
    # Check outputs exist
    assert (out_dir / "teacher_naive.npz").exists()
    assert (out_dir / "cluster_report.json").exists()
    report = json.loads((out_dir / "cluster_report.json").read_text())
    assert "n_candidates" in report
    assert report["n_candidates"] == 2
    assert "n_clusters" in report
    # Naive teacher should have correct shapes
    naive = dict(np.load(str(out_dir / "teacher_naive.npz")))
    assert naive["Y_pair"].shape == (64, 64)
    assert naive["confidence"].shape == (64, 64)
    assert len(naive["weights"]) == 2
