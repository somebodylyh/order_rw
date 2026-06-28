"""Physical-order emergence curves: per-layer max + L0 per-head."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.canonical_reanalysis import emergence_curves  # noqa: E402
from analyses.plot_canonical_reanalysis import plot_emergence_curves  # noqa: E402


def test_emergence_curves_l0_rises(tmp_path):
    s = emergence_curves(2, root=str(ROOT / "runs/handoff_overnight"),
                         steps=(0, 10000), K=1, out_dir=str(tmp_path))
    assert (tmp_path / "emergence_curves.json").exists()
    j = json.load(open(tmp_path / "emergence_curves.json"))
    assert j["by_step"]["0"]["layer_max"]["0"] < j["by_step"]["10000"]["layer_max"]["0"]
    assert j["by_step"]["10000"]["layer_max"]["0"] > 0.9   # L0 physical carrier
    plot_emergence_curves(str(tmp_path / "emergence_curves.json"), str(tmp_path))
    assert (tmp_path / "physical_emergence.png").exists()
