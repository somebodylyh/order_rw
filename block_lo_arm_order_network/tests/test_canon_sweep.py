"""Per-seed ckpt sweep (emergence baseline) + TSV."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.canonical_reanalysis import sweep_seed  # noqa: E402


def test_sweep_emergence_smoke(tmp_path):
    s = sweep_seed(2, root=str(ROOT / "runs/handoff_overnight"),
                   steps=(0, 10000), K=1, out_dir=str(tmp_path))
    assert (tmp_path / "strict65_sweep.tsv").exists()
    j = json.load(open(tmp_path / "sweep.json"))
    assert j["by_step"]["0"]["best_tau"] < j["by_step"]["10000"]["best_tau"]  # emerges
    assert j["by_step"]["10000"]["best_head"][0] == 0                          # L0 carrier
