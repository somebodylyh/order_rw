import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import c_content_residual, make_block_swap_corrupt


def test_block_swap_corrupt_changes_tokens():
    import torch

    ch = torch.arange(256).repeat(3, 1)
    out = make_block_swap_corrupt(((1, 2),))(ch.clone())
    assert not torch.equal(out, ch)
    assert out.shape == ch.shape


CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_c_content_residual_reports_ratio():
    res = c_content_residual(str(CKPT), 0, [2], swaps=((1, 2),), M=6, n_reveals=4)
    row = res["per_head"][0]
    assert row["noise_floor"] >= 0.0 and "ratio" in row and "dtau" in row
