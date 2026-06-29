import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import a0_self_qk_calibration


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_a0_selfpatch_collapses_and_is_flagged_non_load_bearing():
    res = a0_self_qk_calibration(str(CKPT), 0, [2], M=4, n_reveals=4)
    assert res["load_bearing"] is False
    row = res["per_head"][0]
    assert row["tau_selfpatch"] < row["tau_clean"]
