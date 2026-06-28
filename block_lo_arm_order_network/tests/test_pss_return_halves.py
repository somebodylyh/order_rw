# tests/test_pss_return_halves.py  (EXTRA from Task 10 note in the plan)
import pathlib, sys
import numpy as np
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import carrier_b65_per_text

CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"

@pytest.mark.skipif(not CKPT.exists(), reason="optional real-checkpoint integration test")
def test_return_halves_gives_four_lists():
    result = carrier_b65_per_text(
        str(CKPT),
        layer=0, head=2, M=4, n_reveals=4, return_halves=True)
    assert len(result) == 4, "should return (B_list, tau_list, halfA_list, halfB_list)"
    B_list, tau_list, halfA_list, halfB_list = result
    assert len(B_list) == 4 and len(tau_list) == 4
    assert len(halfA_list) == 4 and len(halfB_list) == 4
    assert halfA_list[0].shape == (65, 65)
    # halves should differ from each other
    assert not np.allclose(halfA_list[0], halfB_list[0])

@pytest.mark.skipif(not CKPT.exists(), reason="optional real-checkpoint integration test")
def test_return_halves_false_unchanged():
    result = carrier_b65_per_text(
        str(CKPT),
        layer=0, head=2, M=4, n_reveals=4, return_halves=False)
    assert len(result) == 2, "should return (B_list, tau_list)"
    B_list, tau_list = result
    assert len(B_list) == 4 and len(tau_list) == 4
