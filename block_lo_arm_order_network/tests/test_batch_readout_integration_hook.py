"""BR-1 Task 14: tests for the in-loop FrozenBetaHook.

Uses an untrained FlattenReadout written to tmp_path as the g_beta ckpt
so the tests are pure-CPU and don't require the real Phase-1 winner.
"""
import sys
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


@pytest.fixture
def fake_g_beta_ckpt(tmp_path):
    from batch_readout.model import FlattenReadout
    m = FlattenReadout(N=64, hidden=(64,))
    with torch.no_grad():
        for param in m.parameters():
            param.zero_()
        m.net[-1].bias.copy_(torch.arange(63, -1, -1, dtype=torch.float32))
    p = tmp_path / "fake_gbeta.pt"
    torch.save({"model": m.state_dict(), "config": {"model_name": "flatten", "N": 64, "hidden": (64,)}}, p)
    return str(p)


def test_argsort_returns_valid_permutation(fake_g_beta_ckpt):
    from batch_readout.integration_hook import FrozenBetaHook
    hook = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="argsort", device="cpu")
    A = torch.randn(8, 64, 64)
    sig = hook.step(A)
    assert sig.shape == (64,)
    assert sig.dtype == torch.int64
    assert sorted(sig.tolist()) == list(range(64))


def test_sample_seed_determinism(fake_g_beta_ckpt):
    from batch_readout.integration_hook import FrozenBetaHook
    A = torch.randn(8, 64, 64)
    h1 = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="sample", tau=0.5, seed=0, device="cpu")
    h2 = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="sample", tau=0.5, seed=0, device="cpu")
    assert torch.equal(h1.step(A), h2.step(A))


def test_argsort_is_deterministic_across_calls(fake_g_beta_ckpt):
    from batch_readout.integration_hook import FrozenBetaHook
    A = torch.randn(8, 64, 64)
    hook = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="argsort", device="cpu")
    a = hook.step(A)
    b = hook.step(A)
    assert torch.equal(a, b)


def test_reverse_flips_argsort_order(fake_g_beta_ckpt):
    from batch_readout.integration_hook import FrozenBetaHook
    A = torch.randn(8, 64, 64)
    base = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="argsort", device="cpu")
    rev = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="argsort", reverse=True, device="cpu")
    assert torch.equal(rev.step(A), torch.flip(base.step(A), dims=[0]))


def test_rejects_bad_attention_shape(fake_g_beta_ckpt):
    from batch_readout.integration_hook import FrozenBetaHook
    hook = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="argsort", device="cpu")
    with pytest.raises(ValueError, match=r"attention must be \(batch, N, N\)"):
        hook.step(torch.randn(8, 64))


def test_rejects_invalid_mode(fake_g_beta_ckpt):
    from batch_readout.integration_hook import FrozenBetaHook
    with pytest.raises(ValueError, match="mode must be"):
        FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="greedy", device="cpu")


def test_sample_under_cold_temperature_approaches_argsort(fake_g_beta_ckpt):
    """Very cold tau in the sample hook should match argsort hook output."""
    from batch_readout.integration_hook import FrozenBetaHook
    A = torch.randn(8, 64, 64)
    h_argsort = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="argsort", device="cpu")
    h_cold = FrozenBetaHook(g_beta_ckpt=fake_g_beta_ckpt, mode="sample", tau=1e-5, seed=0, device="cpu")
    assert torch.equal(h_argsort.step(A), h_cold.step(A))
