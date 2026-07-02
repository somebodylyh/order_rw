#!/bin/bash
# Overnight 2026-06-10: seed2 L0H2 multi-start (from20k, from40k)
# GPU1 (free), ~4.6h each, serial chain
# Both use SAME g_β (trained on 10k L0H2 B) → tests cross-step generalization
#
# Expected finish: ~7:30am Jun 11

set -euo pipefail
cd /home/admin/lyuyuhuan/order_lyu
TRAIN_SCRIPT=block_lo_arm_order_network/train_clean_aogpt.py

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin
GB=block_lo_arm_order_network/batch_readout/logs/phase33_gbeta_seed2_from10k_l0h2/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt
BD=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2
OD=block_lo_arm_order_network/probe_results

run_frozen() {
    local LABEL="$1"
    local RESUME_CKPT="$2"
    local SAVE_STEPS="$3"
    local OUTDIR="$4"

    echo ""
    echo "=== [${LABEL}] start @ $(date) ==="
    python -u "$TRAIN_SCRIPT" \
        --run-kind frozen_beta \
        --data-source continuous \
        --train-bin "$TB" \
        --val-bin "$VB" \
        --seed 123 \
        --permute-seed 123 \
        --resume-ckpt "$RESUME_CKPT" \
        --frozen-beta-ckpt "$GB" \
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
        --save-steps "$SAVE_STEPS" \
        --output-dir "$OUTDIR" \
        --device cuda:1 \
        --lr 0.001 \
        --min-lr 0.0001 \
        --lr-decay-steps 50000 \
        --eval-interval 500 \
        --log-interval 50 \
        --stream-eval-windows 2000 \
        --batch-size 64 \
        --grad-accum 2 \
        2>&1 | tee "probe_results/frozen_beta_seed2_${LABEL}.log"
    echo "=== [${LABEL}] done @ $(date) ==="
}

echo "============================================"
echo "OVERNIGHT: seed2 L0H2 multi-start chain"
echo "Start: $(date)"
echo "GPU1: from20k → from40k"
echo "============================================"

# Run 1: from20k→60k
run_frozen \
    "from20k_l0h2" \
    "${BD}/ckpt_step20000.pt" \
    "25000,30000,35000,40000,45000,50000,55000,60000" \
    "${OD}/frozen_beta_seed2_from20k_l0h2"

# Run 2: from40k→60k
run_frozen \
    "from40k_l0h2" \
    "${BD}/ckpt_step40000.pt" \
    "45000,50000,55000,60000" \
    "${OD}/frozen_beta_seed2_from40k_l0h2"

echo ""
echo "============================================"
echo "OVERNIGHT CHAIN DONE: $(date)"
echo "Outputs:"
echo "  ${OD}/frozen_beta_seed2_from20k_l0h2/eval_curve.tsv"
echo "  ${OD}/frozen_beta_seed2_from40k_l0h2/eval_curve.tsv"
echo "============================================"
