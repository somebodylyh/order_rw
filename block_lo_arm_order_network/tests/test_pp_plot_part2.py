"""Part-2 binding plot."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import run_part2  # noqa: E402
from analyses.plot_position_prior import plot_part2  # noqa: E402


def test_plot_part2_writes_png(tmp_path):
    run_part2(2, root=str(ROOT / "runs/handoff_overnight"), out_dir=str(tmp_path),
              K=4, steps=(10000,))
    plot_part2(str(tmp_path / "part2.json"), str(tmp_path))
    assert (tmp_path / "binding.png").exists()
