#!/bin/bash
# Matched CDL teacher seed-123 L0H2 multi-start (from10k, from20k, from40k)
# Seed-123 matches frozen_beta_seed2_from*_l0h2 runs exactly.
# GPU1 serial chain, ~8h total.  Launch: 2026-06-14 evening.
#
# This fills the U1 gap: CDL teacher (seed=2) was cross-seed vs frozen_beta (seed=123).
# These runs use seed=123 → direct matched-group comparison.

set -euo pipefail
cd /home/admin/lyuyuhuan/order_lyu
TRAIN_SCRIPT=block_lo_arm_order_network/train_clean_aogpt.py

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin
BD=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2
OD=block_lo_arm_order_network/probe_results

run_cdl() {
    local LABEL="$1"
    local RESUME_CKPT="$2"
    local SAVE_STEPS="$3"
    local OUTDIR="$4"
    local GPU="${5:-1}"

    echo ""
    echo "=== [${LABEL}] start @ $(date) ==="
    python -u "$TRAIN_SCRIPT" \
        --run-kind cdl_teacher \
        --data-source continuous \
        --train-bin "$TB" \
        --val-bin "$VB" \
        --seed 123 \
        --permute-seed 123 \
        --resume-ckpt "$RESUME_CKPT" \
        --cdl-teacher-head 0 2 \
        --cdl-teacher-refresh 10 \
        --cdl-teacher-tau 1.0 \
        --alpha-start 0.0 \
        --alpha-target 1.0 \
        --alpha-warmup-steps 5000 \
        --alpha-warmup-start 0 \
        --alpha-ramp-from-resume \
        --max-steps 60000 \
        --save-steps "$SAVE_STEPS" \
        --output-dir "$OUTDIR" \
        --device "cuda:${GPU}" \
        --lr 0.001 \
        --min-lr 0.0001 \
        --lr-decay-steps 50000 \
        --eval-interval 500 \
        --log-interval 50 \
        --stream-eval-windows 2000 \
        --batch-size 64 \
        --grad-accum 2 \
        2>&1 | tee "${OUTDIR}/train.log"
    echo "=== [${LABEL}] done @ $(date) ==="
}

echo "============================================"
echo "MATCHED CDL TEACHER: seed-123 L0H2 multi-start"
echo "Start: $(date)"
echo "GPU: cuda:1"
echo "Chain: from10k → from20k → from40k"
echo "Seed=123 matches frozen_beta_seed2_from*_l0h2 exactly"
echo "============================================"

# Run 1: from10k→60k (~3.8h)
run_cdl \
    "seed123_from10k_l0h2" \
    "${BD}/ckpt_step10000.pt" \
    "25000,30000,35000,40000,45000,50000,55000,60000" \
    "${OD}/cdl_teacher_seed123_from10k_l0h2"

# Run 2: from20k→60k (~3h)
run_cdl \
    "seed123_from20k_l0h2" \
    "${BD}/ckpt_step20000.pt" \
    "25000,30000,35000,40000,45000,50000,55000,60000" \
    "${OD}/cdl_teacher_seed123_from20k_l0h2"

# Run 3: from40k→60k (~1.5h)
run_cdl \
    "seed123_from40k_l0h2" \
    "${BD}/ckpt_step40000.pt" \
    "45000,50000,55000,60000" \
    "${OD}/cdl_teacher_seed123_from40k_l0h2"

echo ""
echo "============================================"
echo "ALL DONE @ $(date)"
echo "Results:"
echo "  ${OD}/cdl_teacher_seed123_from10k_l0h2/eval_curve.tsv"
echo "  ${OD}/cdl_teacher_seed123_from20k_l0h2/eval_curve.tsv"
echo "  ${OD}/cdl_teacher_seed123_from40k_l0h2/eval_curve.tsv"
echo "============================================"
