#!/bin/bash
# frozen_beta seed2 L0H2 from 10k → 60k
set -euo pipefail

CD="$(dirname "$0")/.."
GBETA="$CD/batch_readout/logs/phase33_gbeta_seed2_from10k_l0h2/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"
RESUME="$CD/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step10000.pt"
OUT="$CD/probe_results/frozen_beta_seed2_from10k_l0h2"
LOG="$OUT/train_log.txt"

echo "[$(date)] Launching frozen_beta seed2 L0H2 from 10k → 60k on GPU 1"
mkdir -p "$OUT"

python -u "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta \
    --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --seed 123 \
    --permute-seed 123 \
    --resume-ckpt "$RESUME" \
    --frozen-beta-ckpt "$GBETA" \
    --frozen-beta-head 0 2 \
    --frozen-beta-mode argsort \
    --frozen-beta-tau 1.0 \
    --frozen-beta-refresh 10 \
    --alpha-start 0.0 \
    --alpha-target 1.0 \
    --alpha-warmup-steps 5000 \
    --alpha-warmup-start 0 \
    --alpha-ramp-from-resume \
    --max-steps 60000 \
    --save-steps "15000,20000,25000,30000,35000,40000,45000,50000,55000,60000" \
    --output-dir "$OUT" \
    --device cuda:1 \
    --lr 0.001 \
    --min-lr 0.0001 \
    --lr-decay-steps 50000 \
    --eval-interval 500 \
    --log-interval 50 \
    --stream-eval-windows 2000 \
    2>&1 | tee "$LOG"

echo "[$(date)] Done"
