#!/bin/bash
# Multi-BLOCK_LEN training + scan (Experiment 2).
# Trains 2 additional models with different block granularities:
#   - N=32, BLOCK_LEN=8 (coarse blocks)
#   - N=128, BLOCK_LEN=2 (fine blocks)
# Each: random-order baseline, 5k steps, then per_head_order_scan.
#
# NOTE: Only needed if Experiment 1 (post-hoc block granularity scan)
# shows significant τ_vs_L2R differences across granularities (>0.1 τ).
#
# Usage:
#   bash scripts/run_block_len_variation.sh
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"
TRAIN_SCRIPT=block_lo_arm_order_network/train_clean_aogpt.py
SCAN_SCRIPT=block_lo_arm_order_network/per_head_order_scan.py

# ── Common params ──
MAX_STEPS=5000
SAVE_STEPS="0,1000,5000"
BATCH_SIZE=64
GRAD_ACCUM=2
N_LAYER=4
N_HEAD=8
N_EMBD=384
EVAL_INTERVAL=500

run_one() {
  local NAME="$1"      # e.g. "blen8_n32"
  local BLOCK_LEN="$2"
  local GPU="$3"

  local OUT_DIR="block_lo_arm_order_network/probe_results/random_baseline_${NAME}"
  mkdir -p "$OUT_DIR"

  echo "============================================"
  echo "Training: ${NAME} (BLOCK_LEN=${BLOCK_LEN})"
  echo "Output: ${OUT_DIR}  GPU: ${GPU}"
  echo "============================================"

  # Patch BLOCK_LEN and N via env vars read by a wrapper
  CUDA_VISIBLE_DEVICES="${GPU#cuda:}" PYTHONPATH=block_lo_arm_order_network \
    python -c "
import sys, os
# Override constants BEFORE importing training_utils
# We modify the module-level globals after import
import training_utils
orig_N = training_utils.N
orig_BL = training_utils.BLOCK_LEN
orig_SEQ = training_utils.SEQ_LEN

BLOCK_LEN = ${BLOCK_LEN}
N = 256 // BLOCK_LEN
SEQ_LEN = 256

training_utils.N = N
training_utils.BLOCK_LEN = BLOCK_LEN
training_utils.SEQ_LEN = SEQ_LEN

# Also patch in train_clean_aogpt's namespace
import train_clean_aogpt
train_clean_aogpt.N = N
train_clean_aogpt.BLOCK_LEN = BLOCK_LEN
train_clean_aogpt.SEQ_LEN = SEQ_LEN

# Monkey-patch the imports that use these constants
import clean_training_protocol
# clean_training_protocol uses N and BLOCK_LEN from training_utils at call time, so no need to patch

print(f'[patched] N={training_utils.N} BLOCK_LEN={training_utils.BLOCK_LEN} SEQ_LEN={training_utils.SEQ_LEN}')
sys.argv = [
    sys.argv[0],
    '--run-kind', 'baseline',
    '--data-source', 'chunks',
    '--n-layer', '${N_LAYER}',
    '--n-head', '${N_HEAD}',
    '--n-embd', '${N_EMBD}',
    '--batch-size', '${BATCH_SIZE}',
    '--grad-accum', '${GRAD_ACCUM}',
    '--max-steps', '${MAX_STEPS}',
    '--save-steps', '${SAVE_STEPS}',
    '--eval-interval', '${EVAL_INTERVAL}',
    '--lr', '1e-3',
    '--min-lr', '1e-4',
    '--lr-decay-steps', '${MAX_STEPS}',
    '--warmup-iters', '0',
    '--weight-decay', '0.1',
    '--beta1', '0.9', '--beta2', '0.99',
    '--grad-clip', '1.0',
    '--seed', '42',
    '--output-dir', '${OUT_DIR}',
]
exec(open('${TRAIN_SCRIPT}').read())
" 2>&1 | tee "${OUT_DIR}/train.log"

  echo ""
  echo "Training complete. Scanning ckpt_step5000..."

  local CKPT="${OUT_DIR}/ckpt_step${MAX_STEPS}.pt"
  if [ -f "$CKPT" ]; then
    CUDA_VISIBLE_DEVICES="${GPU#cuda:}" PYTHONPATH=block_lo_arm_order_network \
      python "${SCAN_SCRIPT}" \
        --ckpt "$CKPT" \
        --M 100 --batch-size 32 --seed 42 \
        --device "cuda:${GPU#cuda:}" \
        --alpha-dep 0.5 --none-mode b0 \
        --out "${OUT_DIR}/head_scan_step${MAX_STEPS}.json" \
        2>&1 | tee -a "${OUT_DIR}/scan.log"
  fi
  echo "Done: ${NAME}"
}

# ── Launch sequentially (each uses same GPU, no parallelism needed) ──
GPU="cuda:1"

# Experiment 2a: BLOCK_LEN=8, N=32
run_one "blen8_n32" 8 "$GPU"

# Experiment 2b: BLOCK_LEN=2, N=128
run_one "blen2_n128" 2 "$GPU"

echo ""
echo "All block-len variation experiments complete."
echo "Baseline (BLOCK_LEN=4, N=64): probe_results/clean_base_random_perm/ckpt_step5000.pt"
