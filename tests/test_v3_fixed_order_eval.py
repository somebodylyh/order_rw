import math
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"
_N = 64


# ── mock helpers ──────────────────────────────────────────────────────────────

class _MockModel(torch.nn.Module):
    """Backbone that returns a predictable loss."""
    def __init__(self):
        super().__init__()
        self.forward_calls = []

    def forward_fn(self, idx, token_order):
        self.forward_calls.append((idx.clone(), token_order.clone()))
        loss = torch.tensor(3.5, device=idx.device)
        return None, loss


class _MockPerm:
    block_perm_phys_to_model = torch.arange(_N)
    inv_perm_model_to_phys = torch.arange(_N)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_fixed_order_val_loss_mock(monkeypatch):
    """With identity permutations, val loss is finite and PPL computable."""
    import analyses.v3_fixed_order_eval as fe

    monkeypatch.setattr(fe, "N", _N)
    monkeypatch.setattr(fe, "L2R", np.arange(_N, dtype=np.int64))

    model = _MockModel()
    chunks = [torch.zeros(_N, dtype=torch.long) for _ in range(8)]

    loss = fe.fixed_order_val_loss(model, chunks, _MockPerm(), "cpu")
    assert math.isfinite(loss)
    assert loss > 0
    ppl = math.exp(loss)
    assert math.isfinite(ppl)
    assert ppl > 1.0
    # Each chunk → one forward_fn call
    assert len(model.forward_calls) == 8


def test_eval_curve_produces_sorted_keys(monkeypatch, tmp_path):
    """eval_curve returns loss/ppl keyed by step, sorted."""
    import analyses.v3_fixed_order_eval as fe

    monkeypatch.setattr(fe, "N", _N)
    monkeypatch.setattr(fe, "L2R", np.arange(_N, dtype=np.int64))

    # eval_curve does a local `from analyses.p5_utility_controller import load_p5_ckpt`
    import analyses.p5_utility_controller as p5

    def fake_load(ckpt_path, M, device="cpu"):
        return _MockModel(), None, _MockPerm(), torch.device("cpu")

    monkeypatch.setattr(p5, "load_p5_ckpt", fake_load)

    paths = {10000: "ckpt_10k.pt", 15000: "ckpt_15k.pt", 20000: "ckpt_20k.pt"}
    chunks = [torch.zeros(_N, dtype=torch.long) for _ in range(4)]

    curve = fe.eval_curve(paths, chunks, _MockPerm(), "cpu")
    assert list(curve.keys()) == [10000, 15000, 20000]
    for step, entry in curve.items():
        assert "loss" in entry and "ppl" in entry
        assert math.isfinite(entry["loss"])
        assert math.isfinite(entry["ppl"])
        assert math.isclose(math.exp(entry["loss"]), entry["ppl"], rel_tol=1e-9)


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="10k parent ckpt absent")
def test_fixed_order_val_loss_with_real_ckpt():
    """Smoke test against the actual 10k checkpoint."""
    from analyses.v3_fixed_order_eval import fixed_order_val_loss
    from analyses.p5_utility_controller import load_p5_ckpt

    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, 16, device="cpu")
    loss = fixed_order_val_loss(
        model, [chunks[i] for i in range(16)], clean_perm, "cpu"
    )
    assert math.isfinite(loss)
    assert loss > 0
    # PPL should be computable and reasonable (>1, <1000 for a 47M model at 10k)
    ppl = math.exp(loss)
    assert 1.0 < ppl < 1000.0
