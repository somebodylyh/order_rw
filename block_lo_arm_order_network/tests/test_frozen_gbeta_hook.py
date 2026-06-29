"""Tests for model-frame frozen g_beta order provider."""

import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from batch_readout.frozen_gbeta_hook import (  # noqa: E402
    FrozenGBetaModelFrameBlockProvider,
    FrozenGBetaModelFrameProvider,
    extract_probe_averaged_model_frame_strict65,
    l2r_model_frame_token_order,
    layout_path_model_frame_token_order,
    random_model_frame_token_order,
)
from clean_training_protocol import CleanPermutation  # noqa: E402
from training_utils import SEQ_LEN, N, BLOCK_LEN  # noqa: E402


class _FakeAOGPT:
    def __init__(self, heads=2):
        self.config = type("Config", (), {"n_head": heads})()
        self.calls = 0
        self.heads = heads

    def eval(self):
        return self

    def forward_fn(self, idx_batch, probe, return_attentions=False):
        self.calls += 1
        batch = idx_batch.shape[0]
        attn = torch.zeros(batch, self.heads, SEQ_LEN + 1, SEQ_LEN + 1)
        attn[:, :, :SEQ_LEN, 0] = 1.0
        return None, None, [attn]


# ---------------------------------------------------------------------------
# random_model_frame_token_order
# ---------------------------------------------------------------------------

def test_random_model_frame_order_is_valid_permutation():
    order = random_model_frame_token_order(
        batch_size=1, seed=0, global_step=0, device=torch.device("cpu"),
    )
    assert order.shape == (1, SEQ_LEN)
    tokens = set(order[0].tolist())
    assert tokens == set(range(SEQ_LEN))


def test_random_model_frame_order_is_deterministic():
    a = random_model_frame_token_order(1, seed=7, global_step=0, device="cpu")
    b = random_model_frame_token_order(1, seed=7, global_step=0, device="cpu")
    torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Shared model-frame strict65 extraction
# ---------------------------------------------------------------------------

def test_probe_averaged_extraction_runs_requested_probes_and_returns_strict65():
    model = _FakeAOGPT(heads=2)
    idx = torch.zeros(3, SEQ_LEN, dtype=torch.long)

    B = extract_probe_averaged_model_frame_strict65(
        model,
        idx,
        global_step=11,
        seed=7,
        batch_mean_probes=4,
        device=torch.device("cpu"),
    )

    assert model.calls == 4
    assert B.shape == (3, 2, 65, 65)
    assert B.dtype == torch.float32


def test_probe_averaged_extraction_is_deterministic():
    idx = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    first = extract_probe_averaged_model_frame_strict65(
        _FakeAOGPT(),
        idx,
        global_step=5,
        seed=9,
        batch_mean_probes=2,
        device=torch.device("cpu"),
    )
    second = extract_probe_averaged_model_frame_strict65(
        _FakeAOGPT(),
        idx,
        global_step=5,
        seed=9,
        batch_mean_probes=2,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first, second)


def test_probe_averaged_extraction_signature_is_label_free():
    import inspect

    params = set(inspect.signature(
        extract_probe_averaged_model_frame_strict65
    ).parameters)
    forbidden = {"inv_perm", "clean_perm", "block_perm", "physical_order"}
    assert not params & forbidden


def test_model_frame_block_provider_honors_refresh_interval(monkeypatch):
    calls = []

    class FakeInnerProvider:
        def __init__(self, **kwargs):
            pass

        def model_frame_token_order(self, model, idx_batch, global_step):
            calls.append(global_step)
            blocks = torch.arange(N).unsqueeze(0)
            return (
                blocks.unsqueeze(-1) * BLOCK_LEN
                + torch.arange(BLOCK_LEN).view(1, 1, -1)
            ).reshape(1, SEQ_LEN)

    monkeypatch.setattr(
        "batch_readout.frozen_gbeta_hook.FrozenGBetaModelFrameProvider",
        FakeInnerProvider,
    )
    provider = FrozenGBetaModelFrameBlockProvider(
        g_beta_ckpt="unused.pt",
        batch_mean_probes=4,
        refresh_every=5,
        seed=0,
        device="cpu",
    )
    idx = torch.zeros(1, SEQ_LEN, dtype=torch.long)

    provider.physical_order(object(), idx, 10)
    provider.physical_order(object(), idx, 14)
    provider.physical_order(object(), idx, 15)

    assert calls == [10, 15]


# ---------------------------------------------------------------------------
# Oracle baselines
# ---------------------------------------------------------------------------

def test_l2r_baseline_uses_clean_perm():
    cp = CleanPermutation.from_model_to_phys(torch.randperm(N))
    order = l2r_model_frame_token_order(cp, device="cpu")
    assert order.shape == (1, SEQ_LEN)
    # First 4 tokens should come from the same model block.
    first_block = order[0, 0] // BLOCK_LEN
    assert (order[0, :4] // BLOCK_LEN == first_block).all()


def test_layout_path_baseline_matches_block_perm():
    cp = CleanPermutation.from_model_to_phys(torch.randperm(N))
    order = layout_path_model_frame_token_order(cp, device="cpu")
    assert order.shape == (1, SEQ_LEN)
    # The first model block should be block_perm[0].
    first_model_block = int(order[0, 0] // BLOCK_LEN)
    assert first_model_block == int(cp.block_perm_phys_to_model[0])


# ---------------------------------------------------------------------------
# Label-free audit
# ---------------------------------------------------------------------------

def test_frozen_gbeta_provider_class_has_no_physical_terms():
    """FrozenGBetaModelFrameProvider source must not contain physical-coord terms."""
    import inspect
    src = inspect.getsource(FrozenGBetaModelFrameProvider)
    forbidden = ("inv_perm", "clean_perm", "block_perm")
    for term in forbidden:
        assert term not in src, (
            f"FrozenGBetaModelFrameProvider source contains forbidden term: {term}"
        )


def test_frozen_gbeta_provider_init_has_no_physical_params():
    import inspect
    sig = inspect.signature(FrozenGBetaModelFrameProvider.__init__)
    forbidden = {"inv_perm", "clean_perm", "block_perm", "phys_to_model"}
    overlap = set(sig.parameters.keys()) & forbidden
    assert not overlap, f"Provider __init__ has forbidden params: {overlap}"
