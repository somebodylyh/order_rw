#!/usr/bin/env bash
# Small-train audition: test gate-selected head(s) in short frozen feedback.
#
# Runs 20k→22k (or 20k→25k) for each candidate. Only candidates that show
# improvement over random continuation proceed to full 60k.
#
# Prerequisites:
#   - Task 2 dataset (headset .npz)
#   - Task 4 trained g_beta checkpoint (g_beta_best.pt)
#   - Task 7 integration (train_clean_aogpt.py --gbeta-controller support)
#
# Usage:
#   bash scripts/run_head_gated_audition.sh <gated_ckpt> <base_ckpt> <out_dir> [steps] [single_ckpt] [mean_ckpt]
#
#   gated_ckpt: path to trained head-gated g_beta_best.pt
#   base_ckpt:   path to AOGPT checkpoint to start from (e.g., step20000.pt)
#   out_dir:     output directory for audition results
#   steps:       number of training steps (default 2000 = 20k→22k)
#   single_ckpt / mean_ckpt: optional offline-distilled baseline checkpoints

set -euo pipefail

G_BETA_CKPT="${1:?usage: $0 <gated_g_beta_best.pt> <base_ckpt.pt> <out_dir> [steps] [single_ckpt] [mean_ckpt]}"
BASE_CKPT="${2:?usage: $0 <gated_g_beta_best.pt> <base_ckpt.pt> <out_dir> [steps] [single_ckpt] [mean_ckpt]}"
OUT_DIR="${3:?usage: $0 <gated_g_beta_best.pt> <base_ckpt.pt> <out_dir> [steps] [single_ckpt] [mean_ckpt]}"
AUDIT_STEPS="${4:-2000}"
SINGLE_CKPT="${5:-}"
MEAN_CKPT="${6:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CD="$ROOT/block_lo_arm_order_network"

# Extract start step from checkpoint filename
START_STEP=$(echo "$BASE_CKPT" | grep -oP 'step\K\d+' || echo "20000")
MAX_STEPS=$((START_STEP + AUDIT_STEPS))

DEVICE="${DEVICE:-cuda:0}"

echo "============================================"
echo "Small-Train Audition"
echo "g_beta:  $G_BETA_CKPT"
echo "single:  ${SINGLE_CKPT:-<skipped>}"
echo "mean:    ${MEAN_CKPT:-<skipped>}"
echo "base:    $BASE_CKPT  (start=$START_STEP)"
echo "steps:   $START_STEP → $MAX_STEPS  ($AUDIT_STEPS steps)"
echo "output:  $OUT_DIR"
echo "device:  $DEVICE"
echo "============================================"

mkdir -p "$OUT_DIR"

# --- Arm 1: Random continuation (training drift baseline) ---
echo ""
echo "[1/4] Random continuation baseline"
python3 "$CD/train_clean_aogpt.py" \
  --run-kind random_continuation \
  --data-source continuous \
  --resume-ckpt "$BASE_CKPT" \
  --max-steps "$MAX_STEPS" \
  --device "$DEVICE" --seed 2 \
  --output-dir "$OUT_DIR/random_continuation" \
  2>&1 | tail -5

if [[ -n "$SINGLE_CKPT" ]]; then
  echo ""
  echo "[2/4] Single-head audition"
  python3 "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta \
    --data-source continuous \
    --resume-ckpt "$BASE_CKPT" \
    --frozen-beta-ckpt "$SINGLE_CKPT" \
    --gbeta-input-mode layer_heads \
    --gbeta-layer 0 \
    --frozen-beta-none-mode model \
    --frozen-beta-refresh 10 \
    --max-steps "$MAX_STEPS" \
    --device "$DEVICE" --seed 2 \
    --output-dir "$OUT_DIR/single_head" \
    2>&1 | tail -5
fi

# --- Arm 3: Head-gated frozen g_beta (test candidate) ---
echo ""
echo "[3/4] Head-gated audition"
python3 "$CD/train_clean_aogpt.py" \
  --run-kind frozen_beta \
  --data-source continuous \
  --resume-ckpt "$BASE_CKPT" \
  --frozen-beta-ckpt "$G_BETA_CKPT" \
  --gbeta-input-mode layer_heads \
  --gbeta-layer 0 \
  --gbeta-topk 2 \
  --gbeta-frozen True \
  --frozen-beta-none-mode model \
  --frozen-beta-refresh 10 \
  --max-steps "$MAX_STEPS" \
  --device "$DEVICE" --seed 2 \
  --output-dir "$OUT_DIR/head_gated" \
  2>&1 | tail -5

if [[ -n "$MEAN_CKPT" ]]; then
  echo ""
  echo "[4/4] Mean-head audition"
  python3 "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta \
    --data-source continuous \
    --resume-ckpt "$BASE_CKPT" \
    --frozen-beta-ckpt "$MEAN_CKPT" \
    --gbeta-input-mode layer_heads \
    --gbeta-layer 0 \
    --gbeta-frozen True \
    --frozen-beta-none-mode model \
    --frozen-beta-refresh 10 \
    --max-steps "$MAX_STEPS" \
    --device "$DEVICE" --seed 2 \
    --output-dir "$OUT_DIR/mean_head" \
    2>&1 | tail -5
fi

echo ""
echo "============================================"
echo "Audition complete. Evaluate with:"
echo "  python3 analyses/eval_head_gated_audition.py \\"
echo "    --runs $OUT_DIR/single_head $OUT_DIR/head_gated $OUT_DIR/mean_head \\"
echo "    --random-baseline $OUT_DIR/random_continuation \\"
echo "    --out-dir $OUT_DIR/report"
echo "============================================"
