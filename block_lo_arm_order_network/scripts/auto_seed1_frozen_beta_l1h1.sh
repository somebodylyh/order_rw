#!/bin/bash
# Auto-launch: wait for seed1 g_beta pretrain (L1H1), then start frozen_beta hook to 60k.
set -euo pipefail

CD=$(dirname "$0")/..
GBETA_PT="$CD/batch_readout/logs/phase33_gbeta_seed1_from10k_l1h1/random_baseline_continuous_jun09_seed1/full/g_beta_best.pt"
PHASE33_LOG="$CD/probe_results/phase33_gbeta_seed1_10k_l1h1.log"
OUT_DIR="$CD/probe_results/frozen_beta_seed1_from10k_l1h1"
TRAIN_LOG="$OUT_DIR/train_log.txt"

echo "[$(date)] Waiting for g_beta checkpoint (L1H1): $GBETA_PT"
while [ ! -f "$GBETA_PT" ]; do
    sleep 30
    if ! pgrep -f "run_phase33_gbeta.*seed1" > /dev/null 2>&1; then
        if [ ! -f "$GBETA_PT" ]; then
            echo "[$(date)] Phase 33 process gone but g_beta not found — check $PHASE33_LOG"
            exit 1
        fi
    fi
done

echo "[$(date)] g_beta L1H1 ready. Launching frozen_beta seed1 → 60k on GPU 0"

mkdir -p "$OUT_DIR"

nohup python -u "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta \
    --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --seed 1 \
    --permute-seed 1 \
    --resume-ckpt "$CD/probe_results/random_baseline_continuous_jun09_seed1/ckpt_step10000.pt" \
    --frozen-beta-ckpt "$GBETA_PT" \
    --frozen-beta-head 1 1 \
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
    --output-dir "$OUT_DIR" \
    --device cuda:0 \
    --lr 0.001 \
    --min-lr 0.0001 \
    --lr-decay-steps 50000 \
    --eval-interval 500 \
    --log-interval 50 \
    > "$TRAIN_LOG" 2>&1 &

echo "[$(date)] frozen_beta seed1 L1H1 PID=$! launched"
