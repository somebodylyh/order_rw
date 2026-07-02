#!/bin/bash
# Large model (16L/16H/d=1024, ~250M params) random-order baseline training.
# Tests whether attention-order signal (τ_vs_L2R, row_conc) emerges at scale.
#
# Usage:
#   bash scripts/run_large_model_random_baseline.sh
#
# GPU: cuda:1 (24GB 4090, currently free)
# Steps: 10k (overnight)
# After training: run per_head_order_scan on ckpt_step5000 and ckpt_step10000
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

OUT_DIR=block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d
mkdir -p "$OUT_DIR"

# ── Model config ──
N_LAYER=16
N_HEAD=16
N_EMBD=1024
BATCH_SIZE=24
GRAD_ACCUM=6          # effective batch = 24*6 = 144≈128
MAX_STEPS=10000
SAVE_STEPS="0,1000,5000,10000"
EVAL_INTERVAL=500

echo "============================================"
echo "Large Model Random Baseline Training"
echo "Model: ${N_LAYER}L/${N_HEAD}H/d=${N_EMBD}"
echo "Steps: ${MAX_STEPS}"
echo "Output: ${OUT_DIR}"
echo "GPU: cuda:1"
echo "============================================"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=block_lo_arm_order_network \
  python block_lo_arm_order_network/train_clean_aogpt.py \
    --run-kind baseline \
    --data-source chunks \
    --n-layer "$N_LAYER" \
    --n-head "$N_HEAD" \
    --n-embd "$N_EMBD" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --lr-decay-steps "$MAX_STEPS" \
    --warmup-iters 200 \
    --weight-decay 0.1 \
    --beta1 0.9 \
    --beta2 0.99 \
    --grad-clip 1.0 \
    --seed 42 \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/train.log"

echo ""
echo "Training complete. Starting per_head_order_scan..."

# ── Scan step 5000 ──
for STEP in 5000 10000; do
  CKPT="${OUT_DIR}/ckpt_step${STEP}.pt"
  if [ -f "$CKPT" ]; then
    echo "Scanning ckpt_step${STEP}..."
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=block_lo_arm_order_network \
      python block_lo_arm_order_network/per_head_order_scan.py \
        --ckpt "$CKPT" \
        --M 100 --batch-size 32 --seed 42 \
        --device cuda:0 \
        --alpha-dep 0.5 \
        --none-mode b0 \
        --out "${OUT_DIR}/head_scan_step${STEP}.json" \
        2>&1 | tee -a "$OUT_DIR/scan.log"
  else
    echo "WARNING: ckpt_step${STEP}.pt not found, skipping scan"
  fi
done

echo ""
echo "Done! Results in ${OUT_DIR}/"
echo "  head_scan_step5000.json"
echo "  head_scan_step10000.json"
