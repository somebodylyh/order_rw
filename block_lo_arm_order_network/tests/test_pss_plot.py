"""Physical-signal-source metric plot."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.plot_physical_signal_source import plot_source  # noqa: E402


def test_plot_source_writes_png(tmp_path):
    j = {"seed": 2, "carrier": {"layer": 0, "heads": [2]},
         "anchors": {"var_invariant": 0.0, "var_randomized": 0.2,
                     "r2_raw_null": -0.05, "r2_norm_null": 0.70},
         "per_head": [{"layer": 0, "head": 2, "content_variance": 0.001,
                       "noise_floor": 0.05, "r2_raw": 0.95, "verdict": "B"}],
         "relayout": {"anchor_tau": 1.0, "relayout_mean_tau": 0.1, "drop": -0.9},
         "seed_verdict": "B"}
    p = tmp_path / "source.json"
    json.dump(j, open(p, "w"))
    plot_source(str(p), str(tmp_path))
    assert (tmp_path / "source_metrics.png").exists()
