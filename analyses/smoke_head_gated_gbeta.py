#!/usr/bin/env python3
"""Smoke test for head-gated g_beta variants.

Verifies:
  - All three variants produce correct output shapes.
  - Gate weights sum to 1.
  - single_head and mean_head give different results (non-trivial).
  - head_gated gate produces non-uniform weights (soft_all mode).
  - All variants run without NaN.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from batch_readout.head_gated_gbeta import (
    SingleHeadGBeta,
    MeanHeadGBeta,
    HeadGatedGBeta,
    build_head_gated_gbeta,
)

N, H = 64, 8
B = 4


def _random_B_heads(batch=B, heads=H, n=N):
    """Synthetic B_heads with non-trivial per-head structure."""
    rng = np.random.default_rng(42)
    B_h = np.zeros((batch, heads, n, n), dtype=np.float32)
    for b in range(batch):
        for h in range(heads):
            # Each head has different structured pattern
            B_h[b, h] = rng.normal(0, 1, (n, n)).astype(np.float32)
            # Zero diagonal
            np.fill_diagonal(B_h[b, h], 0.0)
    return torch.from_numpy(B_h)


def test_single_head():
    B_heads = _random_B_heads()
    model = SingleHeadGBeta(N=N, H=H, head_idx=2)
    model.eval()
    with torch.no_grad():
        scores, aux = model(B_heads)
    assert scores.shape == (B, N), f"expected ({B},{N}), got {scores.shape}"
    assert aux["gate_weights"].shape == (B, H)
    assert torch.allclose(aux["gate_weights"].sum(dim=1), torch.ones(B))
    assert torch.all(aux["gate_weights"][:, 2] == 1.0), "head_idx=2 should have weight 1"
    assert not torch.any(torch.isnan(scores))
    print("  single_head_gbeta ok")


def test_mean_head():
    B_heads = _random_B_heads()
    model = MeanHeadGBeta(N=N, H=H)
    model.eval()
    with torch.no_grad():
        scores, aux = model(B_heads)
    assert scores.shape == (B, N)
    assert aux["gate_weights"].shape == (B, H)
    assert torch.allclose(aux["gate_weights"].sum(dim=1), torch.ones(B))
    assert torch.allclose(aux["gate_weights"], torch.full((B, H), 1.0 / H))
    assert not torch.any(torch.isnan(scores))
    print("  mean_head_gbeta ok")


def test_head_gated_soft_all():
    B_heads = _random_B_heads()
    model = HeadGatedGBeta(N=N, H=H, gate_mode="soft_all")
    model.eval()
    with torch.no_grad():
        scores, aux = model(B_heads)
    assert scores.shape == (B, N)
    gw = aux["gate_weights"]
    assert gw.shape == (B, H)
    assert torch.allclose(gw.sum(dim=1), torch.ones(B)), "gate weights must sum to 1"
    assert not torch.any(torch.isnan(scores))
    # With random init, weights should be non-uniform
    assert not torch.allclose(gw, torch.full((B, H), 1.0 / H)), \
        "soft_all gate should produce non-uniform weights"
    print("  sparse_head_gated_gbeta (soft_all) ok")


def test_head_gated_topk():
    B_heads = _random_B_heads()
    model = HeadGatedGBeta(N=N, H=H, gate_mode="topk", topk=2)
    model.eval()
    with torch.no_grad():
        scores, aux = model(B_heads)
    assert scores.shape == (B, N)
    gw = aux["gate_weights"]
    assert gw.shape == (B, H)
    assert torch.allclose(gw.sum(dim=1), torch.ones(B))
    # Exactly 2 non-zero weights per sample
    nonzero = (gw > 0).sum(dim=1)
    assert torch.all(nonzero == 2), f"topk=2 should have 2 nonzero weights, got {nonzero}"
    assert not torch.any(torch.isnan(scores))
    print("  sparse_head_gated_gbeta (topk=2) ok")


def test_single_vs_mean_different():
    """Single-head and mean-head must produce different scores."""
    B_heads = _random_B_heads()
    m1 = SingleHeadGBeta(N=N, H=H, head_idx=0)
    m2 = MeanHeadGBeta(N=N, H=H)
    m1.eval(); m2.eval()
    with torch.no_grad():
        s1, _ = m1(B_heads)
        s2, _ = m2(B_heads)
    assert not torch.allclose(s1, s2, atol=1e-4), \
        "single and mean should differ on structured input"
    print("  single != mean (non-trivial) ok")


def test_factory():
    for variant in ("single_head", "mean_head", "head_gated"):
        m = build_head_gated_gbeta(variant, N=N, H=H)
        B_heads = _random_B_heads(batch=2)
        m.eval()
        with torch.no_grad():
            scores, aux = m(B_heads)
        assert scores.ndim == 2
        assert "gate_weights" in aux
    print("  factory ok")


def test_gate_weights_sum_to_1():
    """Cross-check: all gate-weight vectors sum to 1."""
    B_heads = _random_B_heads()
    for variant in ("single_head", "mean_head", "head_gated"):
        m = build_head_gated_gbeta(variant, N=N, H=H)
        m.eval()
        with torch.no_grad():
            _, aux = m(B_heads)
        gw = aux["gate_weights"]
        assert torch.allclose(gw.sum(dim=1), torch.ones(B), atol=1e-5), \
            f"{variant}: gate weights don't sum to 1"
    print("  gate_weights sum to 1 ok")


if __name__ == "__main__":
    print("Running head-gated g_beta smoke tests...")
    test_single_head()
    test_mean_head()
    test_head_gated_soft_all()
    test_head_gated_topk()
    test_single_vs_mean_different()
    test_factory()
    test_gate_weights_sum_to_1()
    print("All smoke tests PASSED")
