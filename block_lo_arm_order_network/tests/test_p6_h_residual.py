import pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p6_h_residual_modulation import (   # noqa: E402
    _corr, position_profile, h_residual, HResidualController, _h_variant,
)
from analyses.p5_utility_controller import BOnlyController, N


def test_residual_removes_position():
    # H[t] = position_profile + per-text noise -> residual mean over texts ~ 0
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((N, 8)).astype(np.float32)
    samples = [{"H": pos + 0.1 * rng.standard_normal((N, 8)).astype(np.float32)}
               for _ in range(40)]
    Hp = position_profile(samples)
    np.testing.assert_allclose(Hp, pos, atol=0.05)
    res = np.mean([h_residual(s, Hp) for s in samples], axis=0)
    assert np.abs(res).mean() < 0.02          # residual has ~zero across-text mean


def test_corr_self_is_one():
    x = torch.randn(N)
    assert abs(float(_corr(x, x)) - 1.0) < 1e-5
    assert abs(float(_corr(x, -x)) + 1.0) < 1e-5


def test_lambda_zero_is_b_only():
    g_B = BOnlyController(b_dim=10)
    sc = HResidualController(g_B, h_dim=8, lambda_init=0.0)
    B = np.random.randn(N, 10).astype(np.float32)
    H = np.random.randn(N, 8).astype(np.float32)
    z = sc(B, H).detach()
    zB = g_B(torch.tensor(B)).detach()
    torch.testing.assert_close(z, zB)         # lambda=0 -> exactly B-only


def test_gB_frozen_in_controller():
    g_B = BOnlyController(b_dim=10)
    sc = HResidualController(g_B, h_dim=8)
    assert all(not p.requires_grad for p in sc.g_B.parameters())
    assert sc.lam.requires_grad
    assert any(p.requires_grad for p in sc.g_H.parameters())


def test_h_variant_shuffle_and_zero():
    Hs = [np.full((N, 3), i, dtype=np.float32) for i in range(4)]
    assert all(np.all(h == 0) for h in _h_variant(Hs, "zero"))
    sh = _h_variant(Hs, "shuffle")
    assert all(not np.allclose(sh[i], Hs[i]) for i in range(4))
