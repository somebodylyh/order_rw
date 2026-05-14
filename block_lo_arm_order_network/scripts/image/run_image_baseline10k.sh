#!/usr/bin/env bash
# Image patch baseline (10 000 steps, random patch order). Produces baseline10k
# checkpoint that all Stage 2 / CI / long-run experiments fork from.

set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -u image_order/train_image_random.py \
    --output-dir probe_results_image/baseline/baseline10k \
    --max-steps 10000 \
    --batch-size 64 \
    --lr 3e-4 --min-lr 3e-5 \
    --warmup-iters 200 \
    --log-interval 200 \
    --eval-interval 500 \
    --max-eval-batches 16 \
    --val-images 1000 \
    --seed 42 \
    "$@"
