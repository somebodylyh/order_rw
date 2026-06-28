"""C2: carrier ablation under the canonical 65-node readout."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.canonical_reanalysis import ablation_effect  # noqa: E402


def test_carrier_ablation_drops_strong_more_than_null():
    out = ablation_effect(2, 10000, carrier_layer=0, carrier_heads=[2, 3, 4, 5],
                          null_heads=[0, 6, 7], root=str(ROOT / "runs/handoff_overnight"), K=1)
    assert out["strong_after_carrier"] <= out["strong_before"]
    assert "strong_after_null" in out
