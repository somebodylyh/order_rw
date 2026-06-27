"""A4: winning-layer B re-extraction from ckpts reproduces the carrier tau."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.emergence_characterization import extract_carrier_B  # noqa: E402
from batch_readout.order_tau_readout import per_head_tau  # noqa: E402


def test_extracted_B_reproduces_step10000_carrier_tau():
    B = extract_carrier_B(2, 10000, winning_layer=1,
                          root=str(ROOT / "runs/handoff_overnight"))
    assert B.shape == (8, 65, 65)
    # seed2 L1 strong heads {3,5,7} roll out to ~1.0 tau (H0 pool-sensitive)
    for h in (3, 5, 7):
        info = per_head_tau(B[h], "C-D+L")
        assert info["tau_vs_l2r"] > 0.9
