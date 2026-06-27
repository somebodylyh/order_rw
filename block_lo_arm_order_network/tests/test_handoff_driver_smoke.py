"""End-to-end smoke: run_seed orchestrates Stage 0/1/2 for one seed and writes
the two tables + summary.json.
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.path_patch_handoff import run_seed  # noqa: E402

SEED2_CKPT = str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt")


def test_run_seed_smoke(tmp_path):
    out = run_seed(SEED2_CKPT, str(tmp_path), n_batches=1, bs_mean=16, device="cpu")
    assert os.path.exists(tmp_path / "stage1_table.csv")
    assert os.path.exists(tmp_path / "stage2_table.csv")
    s = json.load(open(tmp_path / "summary.json"))
    assert s["seed"] == 2
    assert "path_fraction_global" in s["stage2"]
