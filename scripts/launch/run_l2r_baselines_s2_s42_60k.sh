#!/bin/bash
# Chain-run: seed2 + seed42 ori-L2R baselines from scratch to 60k
# Matched protocol: continuous data, same model config as frozen_beta runs.
# GPU 1 (via CUDA_VISIBLE_DEVICES=1)
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
COMMON_ARGS="--run-kind l2r --data-source continuous \
  --n-layer 4 --n-head 8 --n-embd 384 \
  --batch-size 64 --grad-accum 2 --max-steps 60000 \
  --lr 1e-3 --min-lr 1e-4 --lr-decay-steps 60000 \
  --save-steps 0,5000,10000,20000,30000,40000,50000,60000 \
  --eval-interval 1000 --log-interval 100 \
  --device cuda:0"

echo "============================================"
echo "[$(date)] Starting seed2 ori-L2R baseline (0→60k)"
echo "============================================"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=block_lo_arm_order_network \
  python "$TRAIN" \
    --seed 2 --permute-seed 2 \
    --output-dir block_lo_arm_order_network/probe_results/l2r_continuous_seed2 \
    $COMMON_ARGS \
    2>&1 | tee block_lo_arm_order_network/probe_results/l2r_continuous_seed2/train.log

echo ""
echo "[$(date)] seed2 ori-L2R complete."

echo "============================================"
echo "[$(date)] Starting seed42 ori-L2R baseline (0→60k)"
echo "============================================"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=block_lo_arm_order_network \
  python "$TRAIN" \
    --seed 42 --permute-seed 42 \
    --output-dir block_lo_arm_order_network/probe_results/l2r_continuous_seed42 \
    $COMMON_ARGS \
    2>&1 | tee block_lo_arm_order_network/probe_results/l2r_continuous_seed42/train.log

echo ""
echo "[$(date)] seed42 ori-L2R complete."
echo ""
echo "=== Both ori-L2R baselines complete ==="
echo "Seed2: probe_results/l2r_continuous_seed2/"
echo "Seed42: probe_results/l2r_continuous_seed42/"
