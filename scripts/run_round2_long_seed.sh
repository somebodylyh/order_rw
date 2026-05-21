#!/usr/bin/env bash
# Phase Long-1: 10k continuation, 4 minimal arms, single seed.
# Same trainer + baseline + A_block + data + optimizer + alpha + eval_interval as the
# Round-2 minimal protocol (scripts/run_round2_seed_minimal_param.sh). The ONLY
# differences are: max_steps 5000 -> 10000, save_steps include 5000+10000 (so the 5k
# step matches the existing Round-2 seed run for sanity, and 10k is the new endpoint),
# output dir is *_long10k_*. Identical seed => 5k row will reproduce committed Round-2.
set -euo pipefail

SEED=${1:?seed required}
DEVICE=${2:?device required, e.g. cuda:0}
MAX_STEPS=${3:-10000}
ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
TRAIN=$ROOT/block_lo_arm_order_network/train_imagelarge_round2.py
OUT_ROOT=$ROOT/probe_results_image_large/grw_e3ctrlsmall_round2_long${MAX_STEPS}_seed${SEED}
BASE_CKPT=$ROOT/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
A_BLOCK=$ROOT/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy
DATA_DIR=$ROOT/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
MASTER_LOG=$OUT_ROOT/seed${SEED}_long${MAX_STEPS}_master.log

mkdir -p "$OUT_ROOT"
{
  echo "Phase Long-1: seed=$SEED max_steps=$MAX_STEPS"
  date -Is
  echo "arms=random Bcov_balanced distance_only_coverage shuffled_Bcov_balanced"
  echo "device=$DEVICE  trainer=$TRAIN"
  echo "out_root=$OUT_ROOT"
  echo "save_steps=5000,${MAX_STEPS}  (5k point should bit-match committed Round-2 seed=$SEED)"
  git -C "$ROOT" status --short --branch > "$OUT_ROOT/git_status_at_launch.txt" || true
} > "$MASTER_LOG"

run_arm() {
  local policy=$1; local name=$2; local out=$OUT_ROOT/$name
  mkdir -p "$out"
  echo "=== START $name policy=$policy seed=$SEED device=$DEVICE $(date -Is) ===" | tee -a "$MASTER_LOG"
  "$PY" "$TRAIN" \
    --policy "$policy" \
    --baseline-ckpt "$BASE_CKPT" \
    --a-block-path "$A_BLOCK" \
    --data-train "$DATA_DIR/train.bin" \
    --data-val "$DATA_DIR/val.bin" \
    --meta "$DATA_DIR/meta.pkl" \
    --output-dir "$out" \
    --max-steps "$MAX_STEPS" \
    --batch-size 16 \
    --grad-accum 16 \
    --lr 1e-4 \
    --alpha 0.9 \
    --alpha-warmup 1000 \
    --eval-interval 500 \
    --max-eval-batches 16 \
    --save-steps "5000,${MAX_STEPS}" \
    --device "$DEVICE" \
    --seed "$SEED" \
    2>&1 | tee -a "$MASTER_LOG"
  echo "=== DONE $name seed=$SEED $(date -Is) ===" | tee -a "$MASTER_LOG"
}

run_arm random cont_random
run_arm Bcov_balanced cont_Bcov_balanced
run_arm distance_only_coverage cont_distance_only_coverage
run_arm shuffled_Bcov_balanced cont_shuffled_Bcov_balanced

echo "Phase Long-1 seed=$SEED max_steps=$MAX_STEPS finished $(date -Is)" | tee -a "$MASTER_LOG"
