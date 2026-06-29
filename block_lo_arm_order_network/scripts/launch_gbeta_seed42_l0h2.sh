#!/bin/bash
# g_β pretrain: seed42 (jun05) @10k, L0H2 head
set -euo pipefail

CD="$(dirname "$0")/.."
CKPT_DIR="$CD/probe_results/random_baseline_continuous_jun05"
OUT="$CD/batch_readout/logs/phase33_gbeta_seed42_from10k_l0h2"
LOG="$CD/probe_results/phase33_gbeta_seed42_10k_l0h2.log"

echo "[$(date)] Launching g_β pretrain seed42 L0H2 @10k"
mkdir -p "$OUT"

python -u "$CD/scripts/run_phase33_gbeta.py" \
    --layer 0 --head 2 \
    --step 10000 \
    --ckpt-dir "$CKPT_DIR" \
    --device cuda:0 \
    --seed 42 \
    --M 2000 \
    --batch-size 16 \
    --epochs 40 \
    --fwd-batch 64 \
    --out "$OUT" \
    --cross-steps 20000 40000 \
    2>&1 | tee "$LOG"

echo "[$(date)] Done. g_beta_best.pt → $OUT/random_baseline_continuous_jun05/full/g_beta_best.pt"
