"""Emergence A1/A2/A3 plotting."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.emergence_characterization import run_seed_emergence  # noqa: E402
from analyses.plot_emergence import plot_emergence  # noqa: E402


def test_plot_emergence_writes_pngs(tmp_path):
    run_seed_emergence(2, root=str(ROOT / "runs/handoff_overnight"), out_dir=str(tmp_path))
    plot_emergence(str(tmp_path / "summary.json"), str(tmp_path / "concentration.csv"),
                   str(ROOT / "runs/handoff_overnight/seed2/eval_curve.tsv"), str(tmp_path))
    assert (tmp_path / "concentration.png").exists()
    assert (tmp_path / "schedule_overlay.png").exists()
