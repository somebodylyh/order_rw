#!/usr/bin/env bash
# BR-1 Phase-0 smoke: build M=100 B=32 batch-mean dataset from the
# alternating-from-0 random-warmup 5k checkpoint. Records teacher diversity
# stats so we can abort if batch-mean smoothed the labels into degenerate
# territory before paying any Phase-1 cost.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
OUT="block_lo_arm_order_network/batch_readout/data/text_5k_M100_B32_seed0.npz"
LOG_DIR="block_lo_arm_order_network/batch_readout/logs"
mkdir -p "$(dirname "$OUT")" "$LOG_DIR"

GPU=$(MIN_FREE_MB=8000 bash scripts/_br1_wait_for_gpu.sh)
echo "[BR-1 phase 0 smoke] GPU=$GPU ckpt=$CKPT out=$OUT"

CUDA_VISIBLE_DEVICES=$GPU python - <<PY 2>&1 | tee "$LOG_DIR/phase0_smoke.log"
import json
import time
from batch_readout.dataset_batch import build_dataset, load_dataset
from batch_readout.diversity_batch import teacher_diversity_stats

t0 = time.time()
info = build_dataset(
    ckpt_path="$CKPT",
    M=100, batch_size=32, seed=0, alpha_dep=0.5,
    out_path="$OUT",
    train_frac=0.8, val_frac=0.1, device="cuda:0",
)
t_build = time.time() - t0
print(f"[BR-1] build_dataset done in {t_build:.1f}s; split sizes: {info}")

ds = load_dataset("$OUT")
stats_train = teacher_diversity_stats(ds["train_sigma_T"])
print("teacher diversity (train):", json.dumps(stats_train, indent=2))
stats_all_sig = teacher_diversity_stats(
    __import__("numpy").concatenate([ds["train_sigma_T"], ds["val_sigma_T"], ds["test_sigma_T"]], axis=0)
)
print("teacher diversity (all M=100):", json.dumps(stats_all_sig, indent=2))
PY
