"""Canonical emergence curve plot."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.canonical_reanalysis import sweep_seed  # noqa: E402
from analyses.plot_canonical_reanalysis import plot_sweep  # noqa: E402


def test_plot_sweep_writes_png(tmp_path):
    sweep_seed(2, root=str(ROOT / "runs/handoff_overnight"), steps=(0, 10000), K=1,
               out_dir=str(tmp_path))
    plot_sweep(str(tmp_path / "sweep.json"), str(tmp_path))
    assert (tmp_path / "emergence.png").exists()
