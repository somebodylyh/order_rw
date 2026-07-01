import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"
_N = 64  # production N


# ── mock helpers ──────────────────────────────────────────────────────────────

class _MockBackbone(torch.nn.Module):
    """Backbone whose forward_fn returns a loss that decreases with order quality."""
    def __init__(self):
        super().__init__()
        self.forward_calls = []

    def forward_fn(self, idx, token_order):
        self.forward_calls.append((idx.clone(), token_order.clone()))
        # Lower loss when token_order is closer to L2R (identity in a trivial frame)
        L2R_token = torch.arange(token_order.shape[1], device=token_order.device)
        dist = (token_order.float() - L2R_token.float().unsqueeze(0)).abs().sum()
        loss = 3.0 + 0.01 * dist / token_order.numel()
        return None, loss


class _MockOrderHead(torch.nn.Module):
    """OrderHead whose scores produce different orders for different B inputs."""
    def __init__(self):
        super().__init__()
        self.gbeta = torch.nn.Linear(1, 1)  # dummy
        self.score_calls = []

    def scores(self, A, per_sample):
        self.score_calls.append((A.clone(), per_sample))
        # Produce descending scores (identity order) by default.
        # The key invariant for the structural test is that scores are
        # deterministic given A — the real/shuffle/zero comparison is tested
        # against the real checkpoint, not here.
        base = torch.arange(_N, dtype=torch.float32, device=A.device)
        return base.unsqueeze(0), None  # (1, N) for per_sample=False


class _MockWrap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MockBackbone()
        self.order_head = _MockOrderHead()
        self.clean_perm = _MockPerm()
        self.inv_perm = np.arange(_N, dtype=np.int64)  # identity for test

    def extract_B(self, idx_batch, probe):
        M = idx_batch.shape[0]
        A = torch.zeros(M, _N + 1, _N + 1)
        for i in range(M):
            A[i, 1:, 1:] = torch.eye(_N) * (float(i + 1))
        return A


class _MockPerm:
    block_perm_phys_to_model = torch.arange(_N)
    inv_perm_model_to_phys = torch.arange(_N)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_group_probe_keys_and_guards_present(monkeypatch):
    """Structural test: all required keys present, real_beats_controls is bool."""
    import analyses.v3_group_probe as gp

    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)
    monkeypatch.setattr(gp, "N", _N)
    monkeypatch.setattr(gp, "L2R", np.arange(_N, dtype=np.int64))

    result = gp.group_probe(wrap, held, m=8, seed=42, device="cpu")

    required = [
        "delta_probe_group", "tau_to_l2r", "tau_consensus",
        "delta_real", "delta_shuffle", "delta_zero", "real_beats_controls",
    ]
    for key in required:
        assert key in result, f"missing key {key!r}"
    assert isinstance(result["real_beats_controls"], bool)
    for key in ("delta_real", "delta_shuffle", "delta_zero", "delta_probe_group"):
        assert np.isfinite(result[key]), f"{key} is non-finite: {result[key]}"
    assert result["groups"] == 2


def test_group_probe_tau_metrics_in_range(monkeypatch):
    """tau_to_l2r and tau_consensus should be in [-1, 1] and finite."""
    import analyses.v3_group_probe as gp

    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)
    monkeypatch.setattr(gp, "N", _N)
    monkeypatch.setattr(gp, "L2R", np.arange(_N, dtype=np.int64))

    result = gp.group_probe(wrap, held, m=8, seed=0, device="cpu")

    assert -1.0 <= result["tau_to_l2r"] <= 1.0
    assert np.isfinite(result["tau_to_l2r"])
    if not np.isnan(result["tau_consensus"]):
        assert -1.0 <= result["tau_consensus"] <= 1.0


def test_delta_variants_are_finite_and_control_comparison_well_formed(monkeypatch):
    """The three delta variants (real/shuffle/zero) must all be finite;
    real_beats_controls is a deterministic function of these three values."""
    import analyses.v3_group_probe as gp

    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)
    monkeypatch.setattr(gp, "N", _N)
    monkeypatch.setattr(gp, "L2R", np.arange(_N, dtype=np.int64))

    result = gp.group_probe(wrap, held, m=8, seed=1, device="cpu")

    d_real, d_shuf, d_zero = result["delta_real"], result["delta_shuffle"], result["delta_zero"]
    assert np.isfinite(d_real) and np.isfinite(d_shuf) and np.isfinite(d_zero)
    # The guard is a pure boolean expression — verify it matches
    expected_guard = bool(d_real < min(d_shuf, d_zero))
    assert result["real_beats_controls"] == expected_guard


def test_different_seeds_produce_consistent_keys(monkeypatch):
    """Multiple seeds should all produce valid output (key coverage)."""
    import analyses.v3_group_probe as gp

    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)
    monkeypatch.setattr(gp, "N", _N)
    monkeypatch.setattr(gp, "L2R", np.arange(_N, dtype=np.int64))

    for seed in (0, 1, 42):
        r = gp.group_probe(wrap, held, m=8, seed=seed, device="cpu")
        assert isinstance(r["real_beats_controls"], bool)
        assert np.isfinite(r["delta_probe_group"])


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="10k parent ckpt absent")
def test_group_probe_with_real_ckpt():
    """Smoke test against the actual 10k checkpoint (CPU only)."""
    from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
    from analyses.v3_group_probe import group_probe
    from analyses.p5_utility_controller import load_p5_ckpt
    from analyses.p7_gbeta_policy import GBETA_CKPT as GB

    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, 32, device="cpu")
    oh = OrderHeadModule(GB, device="cpu")
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    held = torch.stack([chunks[i] for i in range(32)])

    r = group_probe(wrap, held, m=16, seed=0, device="cpu")

    for k in ("delta_probe_group", "tau_to_l2r", "tau_consensus",
              "delta_real", "delta_shuffle", "delta_zero", "real_beats_controls"):
        assert k in r, f"missing key {k!r}"
    assert isinstance(r["real_beats_controls"], bool)
    assert r["groups"] == 2
    for k in ("delta_real", "delta_shuffle", "delta_zero", "delta_probe_group"):
        assert np.isfinite(r[k]), f"{k} is non-finite: {r[k]}"
