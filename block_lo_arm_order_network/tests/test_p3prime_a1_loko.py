import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import a1_leave_k_out


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_loko_reports_full_and_subsets_no_retrain():
    res = a1_leave_k_out(str(CKPT), 0, [2, 3, 4, 5], M=3, n_reveals=4)
    assert res["single_head"] is False
    assert res["full_tau"] > 0.0
    kept_sizes = sorted(len(s["kept"]) for s in res["subset_tau"])
    assert kept_sizes == [3, 3, 3, 3]
