#!/usr/bin/env bash
# Optional: End-to-end joint g_beta training (Task 9).
#
# ⚠️ HIGH RISK: g_beta gradients may reward-hack input heads.
#   - Start with --gbeta-stopgrad-head-features (stop-grad on head features)
#   - Only test full gradient after verifying stability
#   - Rerun head drift diagnostic (Task 8) after ANY full-gradient run
#
# Usage:
#   bash scripts/queue_head_gated_gbeta_joint.sh <g_beta_ckpt> <base_ckpt> <out_dir>
#
#   g_beta_ckpt: path to head-gated g_beta_best.pt (pretrained offline)
#   base_ckpt:   AOGPT checkpoint to start from (e.g., step20000.pt)
#   out_dir:     output directory

set -euo pipefail

G_BETA_CKPT="${1:?usage: $0 <g_beta_best.pt> <base_ckpt.pt> <out_dir>}"
BASE_CKPT="${2:?usage: $0 <g_beta_best.pt> <base_ckpt.pt> <out_dir>}"
OUT_DIR="${3:?usage: $0 <g_beta_best.pt> <base_ckpt.pt> <out_dir>}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CD="$ROOT/block_lo_arm_order_network"
START_STEP=$(echo "$BASE_CKPT" | grep -oP 'step\K\d+' || echo "20000")
MAX_STEPS=60000
DEVICE="${DEVICE:-cuda:0}"

echo "============================================"
echo "⚠️  Joint head-gated g_beta training"
echo "g_beta:  $G_BETA_CKPT"
echo "base:    $BASE_CKPT  (start=$START_STEP)"
echo "steps:   $START_STEP → $MAX_STEPS"
echo "output:  $OUT_DIR"
echo "device:  $DEVICE"
echo "============================================"
echo ""
echo "Required negative controls (run separately):"
echo "  1. mean-head joint"
echo "  2. reverse teacher joint"
echo "  3. identity-order joint"
echo ""

mkdir -p "$OUT_DIR"

# ── Arm A: stop-gradient (safer, first to try) ──
echo "[A] Joint with stop-gradient on head features"
python3 "$CD/train_clean_aogpt.py" \
  --run-kind frozen_beta \
  --data-source continuous \
  --frozen-beta-ckpt "$G_BETA_CKPT" \
  --gbeta-input-mode layer_heads \
  --gbeta-layer 0 \
  --gbeta-topk 2 \
  --gbeta-frozen False \
  --gbeta-lr 1e-5 \
  --gbeta-entropy-penalty 0.01 \
  --gbeta-stopgrad-head-features True \
  --frozen-beta-none-mode model \
  --frozen-beta-refresh 10 \
  --start-step "$START_STEP" --max-steps "$MAX_STEPS" \
  --device "$DEVICE" --seed 2 \
  --output-dir "$OUT_DIR/joint_stopgrad" \
  2>&1 | tail -5

echo ""
echo "[WARNING] Before running full-gradient, verify stop-gradient results:"
echo "  1. Check eval_curve.tsv in $OUT_DIR/joint_stopgrad/"
echo "  2. Check no NaN in train_log.txt"
echo "  3. Rerun Task 8 head drift diagnostic on the output checkpoints"
echo ""

# ── Arm B: full gradient (HIGH RISK, only after arm A is verified) ──
# Uncomment after arm A passes:
# echo "[B] Joint with full gradient"
# python3 "$CD/train_clean_aogpt.py" \
#   --run-kind frozen_beta \
#   --data-source continuous \
#   --frozen-beta-ckpt "$G_BETA_CKPT" \
#   --gbeta-input-mode layer_heads \
#   --gbeta-layer 0 \
#   --gbeta-topk 2 \
#   --gbeta-frozen False \
#   --gbeta-lr 1e-5 \
#   --gbeta-entropy-penalty 0.01 \
#   --gbeta-stopgrad-head-features False \
#   --frozen-beta-none-mode model \
#   --frozen-beta-refresh 10 \
#   --start-step "$START_STEP" --max-steps "$MAX_STEPS" \
#   --device "$DEVICE" --seed 2 \
#   --output-dir "$OUT_DIR/joint_fullgrad" \
#   2>&1 | tail -5

echo "============================================"
echo "Joint training launched (arm A: stop-gradient)."
echo "After completion, run:"
echo "  python3 analyses/head_drift_diagnostic.py \\"
echo "    --arms random=<path> feedback=$OUT_DIR/joint_stopgrad/ \\"
echo "    --start-step $START_STEP --steps 30000 40000 50000 60000 \\"
echo "    --gbeta-ckpt $G_BETA_CKPT"
echo "============================================"
