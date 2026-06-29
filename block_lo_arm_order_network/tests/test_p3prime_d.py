import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import canonical_null_heads, d_locus


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_null_heads_exclude_carriers():
    nh = canonical_null_heads(str(CKPT), 0, [2, 3, 4, 5], k=2, M=2)
    assert len(nh) == 2 and not (set(nh) & {2, 3, 4, 5})


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_d_locus_ov_is_degenerate_qk_is_not():
    res = d_locus(str(CKPT), 0, 2, M=4, n_reveals=4)
    assert res["ov_degenerate"] is True
    assert "qk_changes" in res
