"""Part-1 summary: ablation table + floors + frame sanity."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import run_part1  # noqa: E402


def test_run_part1_writes_outputs(tmp_path):
    d = run_part1(2, root=str(ROOT / "runs/handoff_overnight"),
                  out_dir=str(tmp_path), n_batches=2)
    assert (tmp_path / "part1_ablation.csv").exists()
    assert (tmp_path / "part1.json").exists()
    j = json.load(open(tmp_path / "part1.json"))
    assert "floor" in j and "frame_sanity" in j
    assert j["frame_sanity"]["tau_model_slot"] >= j["frame_sanity"]["tau_physical"]
