import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import _b_verdict, b_position_ablation


def test_b_verdict_logic():
    assert _b_verdict(1.0, 0.1, 0.95, 0.2) == "base_map_collapses"
    assert _b_verdict(1.0, 0.2, 0.95, 0.9) == "mixed_tau_only"
    assert _b_verdict(1.0, 0.95, 0.95, 0.2) == "mixed_r2_only"
    assert _b_verdict(1.0, 0.95, 0.95, 0.9) == "intact"


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_b_position_ablation_runs():
    res = b_position_ablation(str(CKPT), 0, [2], M=6, n_reveals=4)
    row = res["per_head"][0]
    assert set(row) == {"head", "tau_none", "tau_abl", "r2_none", "r2_abl",
                        "r2_abl_selfref", "verdict"}
    # the clean fixed-map table predicts held-out clean B better than the
    # position-ablated B: r2 drops under ablation (base-map disruption is visible).
    assert row["r2_none"] > row["r2_abl"]
