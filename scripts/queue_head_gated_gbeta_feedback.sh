#!/usr/bin/env bash
# Frozen-feedback launcher for offline-distilled head-gated g_beta variants.
#
# Usage:
#   bash scripts/queue_head_gated_gbeta_feedback.sh smoke <base_ckpt.pt> <out_dir> <gated_ckpt.pt> [single_ckpt.pt] [mean_ckpt.pt]
#   bash scripts/queue_head_gated_gbeta_feedback.sh full  <base_ckpt.pt> <out_dir> <gated_ckpt.pt> <single_ckpt.pt> <mean_ckpt.pt>
#
# The single/mean/gated checkpoints here are produced by
# scripts/train_head_gated_gbeta.py. They all run through
# --gbeta-input-mode layer_heads so the provider can load the variant-specific
# checkpoint config.

set -euo pipefail

MODE="${1:?usage: $0 <smoke|full> <base_ckpt.pt> <out_dir> <gated_ckpt.pt> [single_ckpt.pt] [mean_ckpt.pt]}"
BASE_CKPT="${2:?usage: $0 <smoke|full> <base_ckpt.pt> <out_dir> <gated_ckpt.pt> [single_ckpt.pt] [mean_ckpt.pt]}"
OUT_DIR="${3:?usage: $0 <smoke|full> <base_ckpt.pt> <out_dir> <gated_ckpt.pt> [single_ckpt.pt] [mean_ckpt.pt]}"
GATED_CKPT="${4:?usage: $0 <smoke|full> <base_ckpt.pt> <out_dir> <gated_ckpt.pt> [single_ckpt.pt] [mean_ckpt.pt]}"
SINGLE_CKPT="${5:-}"
MEAN_CKPT="${6:-}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi
if [[ "$MODE" == "full" && ( -z "$SINGLE_CKPT" || -z "$MEAN_CKPT" ) ]]; then
  echo "full mode requires <single_ckpt.pt> and <mean_ckpt.pt>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/block_lo_arm_order_network/train_clean_aogpt.py"
START_STEP="$(echo "$BASE_CKPT" | grep -oP 'step\K\d+' || echo "20000")"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2}"
PERMUTE_SEED="${PERMUTE_SEED:-2}"

if [[ "$MODE" == "smoke" ]]; then
  MAX_STEPS=$((START_STEP + ${SMOKE_STEPS:-500}))
  EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
  LOG_INTERVAL="${LOG_INTERVAL:-25}"
  STREAM_EVAL_WINDOWS="${STREAM_EVAL_WINDOWS:-200}"
  SAVE_STEPS="$MAX_STEPS"
else
  MAX_STEPS="${MAX_STEPS:-60000}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
  LOG_INTERVAL="${LOG_INTERVAL:-50}"
  STREAM_EVAL_WINDOWS="${STREAM_EVAL_WINDOWS:-2000}"
  SAVE_STEPS="${SAVE_STEPS:-30000,40000,50000,60000}"
fi

COMMON_ARGS=(
  --resume-ckpt "$BASE_CKPT"
  --data-source continuous
  --device "$DEVICE"
  --seed "$SEED"
  --permute-seed "$PERMUTE_SEED"
  --batch-size "${BATCH_SIZE:-64}"
  --grad-accum "${GRAD_ACCUM:-2}"
  --lr "${LR:-0.001}"
  --min-lr "${MIN_LR:-0.0001}"
  --lr-decay-steps "${LR_DECAY_STEPS:-50000}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --beta1 "${BETA1:-0.9}"
  --beta2 "${BETA2:-0.99}"
  --grad-clip "${GRAD_CLIP:-1.0}"
  --n-layer "${N_LAYER:-4}"
  --n-head "${N_HEAD:-8}"
  --n-embd "${N_EMBD:-384}"
  --dropout "${DROPOUT:-0.0}"
  --vocab-size "${VOCAB_SIZE:-50304}"
  --stream-eval-windows "$STREAM_EVAL_WINDOWS"
  --eval-interval "$EVAL_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
)

mkdir -p "$OUT_DIR"

echo "============================================"
echo "Head-gated g_beta frozen feedback ($MODE)"
echo "base:   $BASE_CKPT"
echo "steps:  $START_STEP -> $MAX_STEPS"
echo "device: $DEVICE"
echo "out:    $OUT_DIR"
echo "============================================"

run_random() {
  local out="$OUT_DIR/random_continuation"
  mkdir -p "$out"
  echo ""
  echo "[random] $out"
  python3 "$TRAIN" \
    --run-kind random_continuation \
    --output-dir "$out" \
    "${COMMON_ARGS[@]}"
}

run_gbeta() {
  local tag="$1"
  local ckpt="$2"
  local out="$OUT_DIR/$tag"
  mkdir -p "$out"
  echo ""
  echo "[$tag] $out"
  python3 "$TRAIN" \
    --run-kind frozen_beta \
    --output-dir "$out" \
    --frozen-beta-ckpt "$ckpt" \
    --gbeta-input-mode layer_heads \
    --gbeta-layer "${GBETA_LAYER:-0}" \
    --gbeta-topk "${GBETA_TOPK:-2}" \
    --gbeta-frozen True \
    --frozen-beta-none-mode "${GBETA_NONE_MODE:-model}" \
    --frozen-beta-refresh "${GBETA_REFRESH:-10}" \
    --alpha-start "${ALPHA_START:-0.0}" \
    --alpha-target "${ALPHA_TARGET:-1.0}" \
    --alpha-warmup-steps "${ALPHA_WARMUP_STEPS:-5000}" \
    --alpha-ramp-from-resume \
    "${COMMON_ARGS[@]}"
}

run_random
run_gbeta "head_gated" "$GATED_CKPT"

if [[ -n "$SINGLE_CKPT" ]]; then
  run_gbeta "single_head" "$SINGLE_CKPT"
fi
if [[ -n "$MEAN_CKPT" ]]; then
  run_gbeta "mean_head" "$MEAN_CKPT"
fi

echo ""
echo "Feedback runs finished. Outputs are under $OUT_DIR"
