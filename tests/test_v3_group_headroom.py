"""TDD for v3_group_headroom: does a SINGLE shared reveal order have room to beat
L2R at group granularity m? (P7's +0.19 headroom is per-sample; cross-sample
transfer is -0.13, so group-level room may collapse — this verifies it BEFORE the
m=16 sweep.)"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"
_N = 64


class _MockBackbone(torch.nn.Module):
    """loss minimised at L2R token order (so L2R is optimal -> headroom 0)."""
    def forward_fn(self, idx, token_order):
        L2R_token = torch.arange(token_order.shape[1], device=token_order.device)
        dist = (token_order.float() - L2R_token.float().unsqueeze(0)).abs().sum()
        return None, 3.0 + 0.01 * dist / token_order.numel()


class _MockPerm:
    block_perm_phys_to_model = torch.arange(_N)
    inv_perm_model_to_phys = torch.arange(_N)


def _group(m):
    return torch.zeros(m, _N, dtype=torch.long)


def test_group_order_nll_is_mean_of_per_row(monkeypatch):
    import analyses.v3_group_headroom as gh
    from analyses.p5_utility_controller import order_nll
    monkeypatch.setattr(gh, "N", _N, raising=False)

    model = _MockBackbone()
    rows = _group(4)
    sigma = np.arange(_N, dtype=np.int64)
    got = gh.group_order_nll(model, rows, sigma, _MockPerm(), torch.device("cpu"))

    expected = float(np.mean([
        order_nll(model, rows[i:i + 1], sigma, _MockPerm(), torch.device("cpu"))
        for i in range(rows.shape[0])
    ]))
    assert abs(got - expected) < 1e-9


def test_group_hill_climb_valid_perm_and_not_worse_than_l2r(monkeypatch):
    import analyses.v3_group_headroom as gh
    monkeypatch.setattr(gh, "N", _N, raising=False)

    model = _MockBackbone()
    rows = _group(4)
    best, best_nll, l2r_nll = gh.group_hill_climb(
        model, rows, _MockPerm(), torch.device("cpu"), n_steps=50, seed=0
    )
    assert sorted(best.tolist()) == list(range(_N)), "not a valid permutation"
    assert best_nll <= l2r_nll + 1e-9, "hill climb worse than its L2R start"
    # l2r_nll must equal the group-mean under arange
    l2r_check = gh.group_order_nll(model, rows, np.arange(_N, dtype=np.int64),
                                   _MockPerm(), torch.device("cpu"))
    assert abs(l2r_nll - l2r_check) < 1e-9


def test_group_headroom_sweep_structure(monkeypatch):
    import analyses.v3_group_headroom as gh
    monkeypatch.setattr(gh, "N", _N, raising=False)

    model = _MockBackbone()
    held = torch.zeros(16, _N, dtype=torch.long)
    out = gh.group_headroom_sweep(
        model, held, _MockPerm(), torch.device("cpu"),
        m_list=(4, 8), n_steps=20, seed=0,
    )
    for m in (4, 8):
        assert m in out
        row = out[m]
        for k in ("headroom", "group_oracle_val", "l2r_val", "n_groups"):
            assert k in row, f"m={m} missing {k}"
            assert np.isfinite(row[k])
        assert abs(row["headroom"] - (row["l2r_val"] - row["group_oracle_val"])) < 1e-9
        assert row["headroom"] >= -1e-9, "headroom below L2R start (hill climb bug)"
        assert row["n_groups"] == 16 // m


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="10k parent ckpt absent")
def test_group_headroom_curve_on_real_ckpt():
    """DECISIVE: on the real 10k backbone, headroom[1] should reproduce P7 (>0),
    and the curve over m tells us whether the m=16 group arm has any room."""
    from analyses.p5_utility_controller import load_p5_ckpt
    from analyses.v3_group_headroom import group_headroom_sweep

    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, 16, device="cpu")
    held = torch.stack([chunks[i] for i in range(16)])
    out = group_headroom_sweep(model, held, clean_perm, dev,
                               m_list=(1, 4, 16), n_steps=300, seed=0)
    print("\n[group headroom @10k]")
    for m in (1, 4, 16):
        r = out[m]
        print(f"  m={m:2d}: L2R={r['l2r_val']:.4f} oracle={r['group_oracle_val']:.4f} "
              f"headroom={r['headroom']:+.4f} (G={r['n_groups']})")
    assert out[1]["headroom"] > 0.0, "per-sample headroom should be positive (P7)"
