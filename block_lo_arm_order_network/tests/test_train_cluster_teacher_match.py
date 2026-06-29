"""Tests for cluster teacher-match training."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_cluster_teacher_match.py"
    spec = importlib.util.spec_from_file_location("train_cluster_teacher_match", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tiny overfit test
# ---------------------------------------------------------------------------

def test_tiny_pairwise_model_overfits_synthetic_teacher(tmp_path):
    """A minimal pairwise readout must overfit a clean L2R teacher."""
    mod = _load_module()
    rng = np.random.default_rng(0)
    # 4 heads worth of B data
    B_heads = rng.normal(size=(4, 65, 65)).astype("float32")
    # Clean L2R soft teacher: Y[i,j] = 1 if i<j else 0
    N = 64
    Y = np.zeros((N, N), dtype="float32")
    for i in range(N):
        for j in range(N):
            if i == j:
                Y[i, j] = 0.5
            elif i < j:
                Y[i, j] = 1.0
            else:
                Y[i, j] = 0.0
    C = 2.0 * np.abs(Y - 0.5)  # confidence (0 for diagonal)
    C = C.astype("float32")

    result = mod.train_teacher_match(B_heads, Y, C, epochs=30, hidden=64, lr=1e-2, seed=0)
    assert result["val_pairwise_acc"] > 0.9, f"acc={result['val_pairwise_acc']}"
    assert result["val_bce"] < 0.5, f"bce={result['val_bce']}"


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_train_rejects_mismatched_shapes():
    mod = _load_module()
    B = np.random.default_rng(0).normal(size=(4, 65, 65)).astype("float32")
    Y_bad = np.zeros((32, 32), dtype="float32")
    with pytest.raises(ValueError):
        mod.train_teacher_match(B, Y_bad, np.ones((32, 32), dtype="float32"), epochs=1)


# ---------------------------------------------------------------------------
# deterministic reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_result():
    mod = _load_module()
    B = np.random.default_rng(0).normal(size=(2, 65, 65)).astype("float32")
    N = 64
    Y = (np.arange(N)[:, None] < np.arange(N)[None, :]).astype("float32")
    C = 2.0 * np.abs(Y - 0.5).astype("float32")
    r1 = mod.train_teacher_match(B, Y, C, epochs=5, hidden=32, lr=1e-2, seed=42)
    r2 = mod.train_teacher_match(B, Y, C, epochs=5, hidden=32, lr=1e-2, seed=42)
    assert r1["val_bce"] == r2["val_bce"]
    assert r1["val_pairwise_acc"] == r2["val_pairwise_acc"]


# ---------------------------------------------------------------------------
# ClusterTeacherReadout module
# ---------------------------------------------------------------------------

def test_model_forward_shapes():
    mod = _load_module()
    model = mod.ClusterTeacherReadout(K=4, N=64, hidden=64)
    import torch
    B = torch.randn(2, 4, 65, 65)
    scores, meta = model(B)
    assert scores.shape == (2, 64)
    assert "alpha" in meta
    assert meta["alpha"].shape == (2, 4)


def test_alpha_sums_to_one():
    mod = _load_module()
    model = mod.ClusterTeacherReadout(K=4, N=64, hidden=64)
    import torch
    B = torch.randn(3, 4, 65, 65)
    _, meta = model(B)
    alpha = meta["alpha"]
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(3), atol=1e-5)
