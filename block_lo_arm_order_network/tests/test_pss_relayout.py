import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import analyses.physical_signal_source as pss


class _FakePermutation:
    def __init__(self, layout_id):
        self.layout_id = layout_id
        self.inv_perm_model_to_phys = torch.tensor([layout_id])


class _FakeModel:
    def __init__(self):
        self.reveal_calls = []

    def forward_fn(self, chunks, reveal, return_attentions):
        assert return_attentions is True
        self.reveal_calls.append(tuple(reveal.flatten().tolist()))
        signal = float(chunks[0, 0])
        return None, None, [torch.full((1, 1, 1, 1), signal)]


def test_relayout_orchestrates_anchor_then_k_minus_one_layouts(monkeypatch):
    """K counts total layouts: layout 0 anchor plus K-1 relayouts."""
    model = _FakeModel()
    training_perm = _FakePermutation(0)
    chunks = torch.ones((2, 1))
    layouts_seen = []

    monkeypatch.setattr(
        pss, "_load_model_and_chunks",
        lambda *args, **kwargs: (model, chunks, training_perm, torch.device("cpu"), None))
    monkeypatch.setattr(
        pss, "random_reveal_orders",
        lambda n, seed: np.asarray([[10], [20]], dtype=np.int64))
    monkeypatch.setattr(
        pss, "_attn_to_A_block_loss_aligned_with_none_vec",
        lambda attn, reveal, inv: np.full((1, 1, 1, 1), float(attn[0, 0, 0, 0])))
    monkeypatch.setattr(pss, "build_none_separated_B", lambda A: A)
    monkeypatch.setattr(pss, "rollout_by_method", lambda B, method: B)
    monkeypatch.setattr(
        pss, "discovery_metrics", lambda result: {"tau_vs_l2r": float(result.item())})

    import analyses.position_prior_decomp as ppd
    monkeypatch.setattr(
        ppd, "make_layouts",
        lambda perm, K: [{"layout_id": i} for i in range(K)])
    monkeypatch.setattr(
        ppd, "clean_perm_from_layout",
        lambda layout: _FakePermutation(layout["layout_id"]))

    def fake_relayout(input_chunks, source_perm, target_perm):
        layouts_seen.append(target_perm.layout_id)
        return input_chunks - 0.3 * target_perm.layout_id

    monkeypatch.setattr(ppd, "relayout_chunks", fake_relayout)

    out = pss.relayout_diagnostic(
        "unused.pt", layer=0, carrier_heads=[0], K=3, M=2, n_reveals=2)

    assert layouts_seen == [1, 2]
    assert out == pytest.approx({
        "anchor_tau": 1.0,
        "relayout_mean_tau": 0.55,
        "drop": 0.45,
    })
    assert model.reveal_calls == [(10,), (20,)] * 2 * 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"K": 1}, "K must be at least 2"),
        ({"M": 0}, "M must be positive"),
        ({"n_reveals": 0}, "n_reveals must be positive"),
        ({"carrier_heads": []}, "carrier_heads must not be empty"),
    ],
)
def test_relayout_rejects_invalid_inputs(kwargs, message):
    base = dict(layer=0, carrier_heads=[0], K=2, M=1, n_reveals=1)
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        pss.relayout_diagnostic("unused.pt", **base)


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_relayout_reports_anchor_and_drop():
    out = pss.relayout_diagnostic(
        str(CKPT), layer=0, carrier_heads=[2, 3, 4, 5], K=2, M=8)
    assert set(out) == {"anchor_tau", "relayout_mean_tau", "drop"}
    assert out["anchor_tau"] > 0.8
    assert out["drop"] == pytest.approx(
        out["anchor_tau"] - out["relayout_mean_tau"])
