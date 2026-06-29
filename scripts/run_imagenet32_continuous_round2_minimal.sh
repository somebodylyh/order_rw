#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-cuda:0}
SEED=${2:-42}
MAX_STEPS=${3:-3000}

ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
TRAIN=$ROOT/scripts/train_imagenet32_continuous_round2.py
OUT_ROOT=$ROOT/probe_results_image/imagenet32_continuous_round2_minimal_seed${SEED}
BASE=$ROOT/probe_results_image/baseline_imagenet32/baseline10k/ckpt_step10000.pt
A_PATH=$ROOT/probe_results_image/baseline_imagenet32/baseline10k/attention/A_global.npy

mkdir -p "$OUT_ROOT"
{
  echo "ImageNet32 continuous Phase-B positive minimal"
  date -Is
  echo "device=$DEVICE seed=$SEED max_steps=$MAX_STEPS"
  echo "baseline=$BASE"
  echo "A=$A_PATH"
  echo "arms=random Bcov_balanced distance_only_coverage shuffled_Bcov_balanced"
} > "$OUT_ROOT/master.log"

run_arm() {
  local policy=$1
  local out=$OUT_ROOT/cont_$policy
  mkdir -p "$out"
  echo "=== START $policy $(date -Is) ===" | tee -a "$OUT_ROOT/master.log"
  "$PY" "$TRAIN" \
    --policy "$policy" \
    --baseline-ckpt "$BASE" \
    --a-path "$A_PATH" \
    --output-dir "$out" \
    --max-steps "$MAX_STEPS" \
    --batch-size 64 \
    --lr 3e-5 \
    --min-lr 3e-6 \
    --warmup-iters 100 \
    --eval-interval 500 \
    --max-eval-batches 16 \
    --val-images 1000 \
    --alpha 0.9 \
    --alpha-warmup 1000 \
    --seed "$SEED" \
    --device "$DEVICE" \
    2>&1 | tee -a "$OUT_ROOT/master.log"
  echo "=== DONE $policy $(date -Is) ===" | tee -a "$OUT_ROOT/master.log"
}

run_arm random
run_arm Bcov_balanced
run_arm distance_only_coverage
run_arm shuffled_Bcov_balanced

echo "DONE $(date -Is)" | tee -a "$OUT_ROOT/master.log"
