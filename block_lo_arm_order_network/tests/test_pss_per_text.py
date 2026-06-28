import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import carrier_b65_per_text

def test_per_text_b65_shapes_and_validity():
    B_list, tau_list = carrier_b65_per_text(
        str(ROOT/"runs/handoff_overnight/seed2/ckpt_step10000.pt"), layer=0, head=2, M=6, n_reveals=8)
    assert len(B_list) == 6 and B_list[0].shape == (65, 65)
    assert len(tau_list) == 6
    # carrier head: most texts read high physical tau
    assert np.mean([abs(t) > 0.9 for t in tau_list]) >= 0.5
