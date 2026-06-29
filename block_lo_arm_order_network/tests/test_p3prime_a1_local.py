import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import a1_head_local


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_a1_offdiagonal_is_null_diagonal_collapses():
    res = a1_head_local(str(CKPT), 0, [2, 3], M=4, n_reveals=4)
    d = res["delta"]
    assert abs(d[2][3]) < 0.1
    assert d[2][2] > 0.0
    assert res["parallel_independent"] is True
