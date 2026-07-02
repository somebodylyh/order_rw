#!/bin/bash
# Extend seed-123 baselines from 50k→60k (continuous data loading).
# Waits for CDL teacher chain (PID $1) to finish, then runs on GPU1.
# random_baseline_continuous_jun08_seed2: seed=123, run_kind=baseline
# l2r_continuous_seed123: seed=123, run_kind=l2r
#
# Launch: bash scripts/run_extend_baselines_50k60k.sh <CDL_TEACHER_PID>

set -euo pipefail
cd /home/admin/lyuyuhuan/order_lyu
TRAIN_SCRIPT=block_lo_arm_order_network/train_clean_aogpt.py

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin
OD=block_lo_arm_order_network/probe_results
GPU=1

CDL_PID="${1:-0}"

if [ "$CDL_PID" != "0" ] && kill -0 "$CDL_PID" 2>/dev/null; then
    echo "Waiting for CDL teacher PID $CDL_PID to finish..."
    while kill -0 "$CDL_PID" 2>/dev/null; do
        sleep 60
    done
    echo "CDL teacher done @ $(date)"
fi

echo "============================================"
echo "EXTEND BASELINES: seed-123 50k→60k"
echo "Start: $(date)"
echo "GPU: cuda:$GPU"
echo "============================================"

# 1) Random baseline: resume from 50k, run to 60k
echo ""
echo "=== [1/2] random baseline seed-123 50k→60k @ $(date) ==="
python -u "$TRAIN_SCRIPT" \
    --run-kind baseline \
    --data-source continuous \
    --train-bin "$TB" \
    --val-bin "$VB" \
    --seed 123 \
    --permute-seed 123 \
    --resume-ckpt "${OD}/random_baseline_continuous_jun08_seed2/ckpt_step50000.pt" \
    --alpha-start 0.0 \
    --alpha-target 0.0 \
    --alpha-warmup-steps 0 \
    --max-steps 60000 \
    --save-steps "55000,60000" \
    --output-dir "${OD}/random_baseline_continuous_jun08_seed2_ext60k" \
    --device "cuda:${GPU}" \
    --lr 0.001 \
    --min-lr 0.0001 \
    --lr-decay-steps 50000 \
    --eval-interval 500 \
    --log-interval 50 \
    --stream-eval-windows 2000 \
    --batch-size 64 \
    --grad-accum 2 \
    2>&1 | tee "${OD}/random_baseline_continuous_jun08_seed2_ext60k/train.log"
echo "=== [1/2] done @ $(date) ==="

# 2) L2R reference: resume from 50k, run to 60k
echo ""
echo "=== [2/2] L2R seed-123 50k→60k @ $(date) ==="
python -u "$TRAIN_SCRIPT" \
    --run-kind l2r \
    --data-source continuous \
    --train-bin "$TB" \
    --val-bin "$VB" \
    --seed 123 \
    --permute-seed 123 \
    --resume-ckpt "${OD}/l2r_continuous_seed123/ckpt_step50000.pt" \
    --alpha-start 0.0 \
    --alpha-target 0.0 \
    --alpha-warmup-steps 0 \
    --max-steps 60000 \
    --save-steps "55000,60000" \
    --output-dir "${OD}/l2r_continuous_seed123_ext60k" \
    --device "cuda:${GPU}" \
    --lr 0.001 \
    --min-lr 0.0001 \
    --lr-decay-steps 50000 \
    --eval-interval 500 \
    --log-interval 50 \
    --stream-eval-windows 2000 \
    --batch-size 64 \
    --grad-accum 2 \
    2>&1 | tee "${OD}/l2r_continuous_seed123_ext60k/train.log"
echo "=== [2/2] done @ $(date) ==="

echo ""
echo "============================================"
echo "ALL DONE @ $(date)"
echo "Results:"
echo "  ${OD}/random_baseline_continuous_jun08_seed2_ext60k/eval_curve.tsv"
echo "  ${OD}/l2r_continuous_seed123_ext60k/eval_curve.tsv"
echo "============================================"
