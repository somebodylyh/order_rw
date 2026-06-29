#!/usr/bin/env bash
# frozen_beta multi-start: resume from random_baseline @10k/20k/40k, all to 60k.
# Same g_β (random_baseline @5k L0H4). α 0→1 over 5k, then 1.0.
# Launches sequentially on GPU 0 (or whichever GPU is free).
set -euo pipefail

CD="$(dirname "$0")/../block_lo_arm_order_network"
GBETA="$CD/batch_readout/logs/phase33_gbeta/random_baseline_continuous_jun05/full/g_beta_best.pt"
HEAD="0 4"
GPU="${1:-0}"

run_one() {
    local FROM=$1
    local TAG=$2
    local RESUME="$CD/probe_results/random_baseline_continuous_jun05/ckpt_step${FROM}.pt"
    local OUT="$CD/probe_results/frozen_beta_random_jun05_${TAG}"

    echo "=== frozen_beta ${TAG}: ${FROM} → 60000 ==="
    mkdir -p "$OUT"

    python "$CD/train_clean_aogpt.py" \
        --run-kind frozen_beta \
        --resume-ckpt "$RESUME" \
        --frozen-beta-ckpt "$GBETA" \
        --frozen-beta-head $HEAD \
        --frozen-beta-refresh 10 \
        --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
        --alpha-ramp-from-resume \
        --data-source continuous \
        --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
        --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
        --output-dir "$OUT" \
        --max-steps 60000 \
        --save-steps 10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000 \
        --device "cuda:$GPU" \
        --seed 42 --permute-seed 42 \
        --batch-size 64 --grad-accum 2 \
        --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
        --weight-decay 0.1 --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
        --n-layer 4 --n-head 8 --n-embd 384 --dropout 0.0 \
        --vocab-size 50304 \
        --stream-eval-windows 2000 \
        --eval-interval 500 --log-interval 50

    echo "Done: $TAG"
}

echo "frozen_beta multi-start → 60k on GPU $GPU"
run_one 10000 from10k
run_one 20000 from20k
run_one 40000 from40k

echo "=== All done ==="
