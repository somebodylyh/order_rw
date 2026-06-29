#!/usr/bin/env bash
# run_topk_cluster_teacher_match_20260626.sh
#
# Batch-runner for first-stage teacher-match matrix.
# Usage:
#   bash scripts/run_topk_cluster_teacher_match_20260626.sh smoke   # top4 only
#   bash scripts/run_topk_cluster_teacher_match_20260626.sh full     # all 10 settings

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/block_lo_arm_order_network:${PYTHONPATH:-}"

MODE="${1:-smoke}"

python3 - "$MODE" "$REPO" <<'PYEOF'
import json, os, subprocess, sys
from pathlib import Path

MODE = sys.argv[1]
REPO = Path(sys.argv[2])
SCRIPTS = REPO / "scripts"
OUT_BASE = REPO / "reports" / "topk_cluster_teacher_match_20260626"

CONFIGS = {
    "new5k": {
        "a_npy": str(REPO / "reports/5k_signal_provenance_audit_20260626/old_diag_jun25_5k/A_with_none_lh_mean.npy"),
        "sel_json": str(REPO / "reports/label_free_selector_audit_20260626/new5k/label_free_selector_rows.json"),
    },
    "new10k": {
        "a_npy": str(REPO / "reports/10k_signal_carrier_layer_multiseed_20260626/seed123_new/A_with_none_lh_mean.npy"),
        "sel_json": str(REPO / "reports/label_free_selector_audit_20260626/new10k/label_free_selector_rows.json"),
    },
}

SMOKE_SETS = "top4:structure:4"
FULL_SETS = "top4:structure:4 top8:structure:8 random4:random:4"

ckpt_keys = ["new5k"]
set_spec = SMOKE_SETS
if MODE == "full":
    ckpt_keys = ["new5k", "new10k"]
    set_spec = FULL_SETS

def run(cmd, label=""):
    print(f"  [{label}] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

for key in ckpt_keys:
    cfg = CONFIGS[key]
    ckpt_dir = OUT_BASE / key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Step A: build candidate sets
    print(f"\n=== {key}: candidate sets ===")
    run([
        sys.executable, str(SCRIPTS / "build_label_free_candidate_sets.py"),
        "--selector-json", cfg["sel_json"],
        "--out-dir", str(ckpt_dir),
        "--sets", *set_spec.split(),
        "--seed", "0",
    ], label=f"{key}/candsets")

    # Step B+C: for each candidate set, build teacher + train match
    manifest = json.loads((ckpt_dir / "candidate_sets" / "manifest.json").read_text())
    for entry in manifest:
        set_name = entry["name"]
        teacher_dir = ckpt_dir / set_name
        teacher_dir.mkdir(parents=True, exist_ok=True)
        cand_json = entry["path"]

        # Step B: build teacher
        print(f"\n--- {key}/{set_name}: build teacher ---")
        run([
            sys.executable, str(SCRIPTS / "build_topk_clustered_pairwise_teacher.py"),
            "--a-npy", cfg["a_npy"],
            "--candidate-json", cand_json,
            "--out-dir", str(teacher_dir),
            "--cluster-threshold", "0.0",
            "--score-temperature", "0.5",
        ], label=f"{key}/{set_name}/teacher")

        # Step C: train teacher-match for each npz
        for npz in sorted(teacher_dir.glob("*.npz")):
            npz_name = npz.stem
            match_dir = teacher_dir / f"match_{npz_name}"
            print(f"  train: {match_dir}")
            run([
                sys.executable, str(SCRIPTS / "train_cluster_teacher_match.py"),
                "--teacher-npz", str(npz),
                "--out-dir", str(match_dir),
                "--epochs", "200",
                "--lr", "3e-4",
                "--seed", "0",
            ], label=f"{key}/{set_name}/match_{npz_name}")

print(f"\n=== {MODE} matrix done ===")
print(f"Now run: python3 scripts/summarize_topk_cluster_teacher_match.py")
PYEOF
