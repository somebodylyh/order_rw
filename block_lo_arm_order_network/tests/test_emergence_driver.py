"""Per-seed emergence driver writes concentration.csv + summary.json."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.emergence_characterization import run_seed_emergence  # noqa: E402


def test_run_seed_emergence_smoke(tmp_path):
    s = run_seed_emergence(2, root=str(ROOT / "runs/handoff_overnight"), out_dir=str(tmp_path))
    assert (tmp_path / "concentration.csv").exists()
    assert (tmp_path / "summary.json").exists()
    j = json.load(open(tmp_path / "summary.json"))
    assert j["seed"] == 2
    assert j["winner"]["winning_layer"] == 1
    assert j["predictability"]["verdict"] in ("early-bias", "contingent")
    assert "classification" in j["schedule"]
