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
#   bash scripts/run_head_gated_audition.sh <g_beta_ckpt> <base_ckpt> <out_dir> [steps]
#
#   g_beta_ckpt: path to trained head-gated g_beta_best.pt
#   base_ckpt:   path to AOGPT checkpoint to start from (e.g., step20000.pt)
#   out_dir:     output directory for audition results
#   steps:       number of training steps (default 2000 = 20k→22k)

set -euo pipefail

G_BETA_CKPT="${1:?usage: $0 <g_beta_best.pt> <base_ckpt.pt> <out_dir> [steps]}"
BASE_CKPT="${2:?usage: $0 <g_beta_best.pt> <base_ckpt.pt> <out_dir> [steps]}"
OUT_DIR="${3:?usage: $0 <g_beta_best.pt> <base_ckpt.pt> <out_dir> [steps]}"
AUDIT_STEPS="${4:-2000}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CD="$ROOT/block_lo_arm_order_network"

# Extract start step from checkpoint filename
START_STEP=$(echo "$BASE_CKPT" | grep -oP 'step\K\d+' || echo "20000")
MAX_STEPS=$((START_STEP + AUDIT_STEPS))

DEVICE="${DEVICE:-cuda:0}"

echo "============================================"
echo "Small-Train Audition"
echo "g_beta:  $G_BETA_CKPT"
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
  --start-step "$START_STEP" --max-steps "$MAX_STEPS" \
  --device "$DEVICE" --seed 2 \
  --output-dir "$OUT_DIR/random_continuation" \
  2>&1 | tail -5

# --- Arm 2: Single-head frozen g_beta (canonical reference) ---
echo ""
echo "[2/4] Single-head L0H2 audition"
python3 "$CD/train_clean_aogpt.py" \
  --run-kind frozen_beta \
  --data-source continuous \
  --frozen-beta-ckpt "$G_BETA_CKPT" \
  --frozen-beta-head 0 2 \
  --frozen-beta-none-mode model \
  --frozen-beta-refresh 10 \
  --start-step "$START_STEP" --max-steps "$MAX_STEPS" \
  --device "$DEVICE" --seed 2 \
  --output-dir "$OUT_DIR/single_l0h2" \
  2>&1 | tail -5

# --- Arm 3: Head-gated frozen g_beta (test candidate) ---
echo ""
echo "[3/4] Head-gated audition"
python3 "$CD/train_clean_aogpt.py" \
  --run-kind frozen_beta \
  --data-source continuous \
  --frozen-beta-ckpt "$G_BETA_CKPT" \
  --gbeta-input-mode layer_heads \
  --gbeta-layer 0 \
  --gbeta-topk 2 \
  --gbeta-frozen True \
  --frozen-beta-none-mode model \
  --frozen-beta-refresh 10 \
  --start-step "$START_STEP" --max-steps "$MAX_STEPS" \
  --device "$DEVICE" --seed 2 \
  --output-dir "$OUT_DIR/head_gated" \
  2>&1 | tail -5

# --- Arm 4: Mean-head frozen g_beta (expected-weak baseline) ---
echo ""
echo "[4/4] Mean-head audition"
python3 "$CD/train_clean_aogpt.py" \
  --run-kind frozen_beta \
  --data-source continuous \
  --frozen-beta-ckpt "$G_BETA_CKPT" \
  --gbeta-input-mode layer_means \
  --gbeta-layer 0 \
  --gbeta-frozen True \
  --frozen-beta-none-mode model \
  --frozen-beta-refresh 10 \
  --start-step "$START_STEP" --max-steps "$MAX_STEPS" \
  --device "$DEVICE" --seed 2 \
  --output-dir "$OUT_DIR/mean_head" \
  2>&1 | tail -5

echo ""
echo "============================================"
echo "Audition complete. Evaluate with:"
echo "  python3 analyses/eval_head_gated_audition.py \\"
echo "    --runs $OUT_DIR/single_l0h2 $OUT_DIR/head_gated $OUT_DIR/mean_head \\"
echo "    --random-baseline $OUT_DIR/random_continuation \\"
echo "    --out-dir $OUT_DIR/report"
echo "============================================"
