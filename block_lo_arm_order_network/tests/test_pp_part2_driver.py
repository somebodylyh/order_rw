"""Part-2 driver: layouts + binding across steps."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import run_part2  # noqa: E402


def test_run_part2_smoke(tmp_path):
    run_part2(2, root=str(ROOT / "runs/handoff_overnight"), out_dir=str(tmp_path),
              K=4, steps=(10000,))
    assert (tmp_path / "layouts.json").exists()
    assert (tmp_path / "part2_binding.csv").exists()
    j = json.load(open(tmp_path / "part2.json"))
    assert "10000" in j["by_step"]
