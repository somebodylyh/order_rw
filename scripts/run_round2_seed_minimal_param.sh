#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?seed required}
DEVICE=${2:?device required, e.g. cuda:0}
ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
TRAIN=$ROOT/block_lo_arm_order_network/train_imagelarge_round2.py
OUT_ROOT=$ROOT/probe_results_image_large/grw_e3ctrlsmall_round2_seed${SEED}_minimal
BASE_CKPT=$ROOT/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
A_BLOCK=$ROOT/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy
DATA_DIR=$ROOT/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
MASTER_LOG=$OUT_ROOT/seed${SEED}_master.log

mkdir -p "$OUT_ROOT"
{
  echo "Round-2 seed=$SEED minimal replication"
  date -Is
  echo "arms=random Bcov_balanced distance_only_coverage shuffled_Bcov_balanced"
  echo "device=$DEVICE"
  echo "trainer=$TRAIN"
  echo "out_root=$OUT_ROOT"
  echo "same baseline/A/data/eval settings as seed=42 pilot"
  git -C "$ROOT" status --short --branch > "$OUT_ROOT/git_status_at_launch.txt" || true
  sed -n '1,80p' "$OUT_ROOT/git_status_at_launch.txt" || true
} > "$MASTER_LOG"

run_arm() {
  local policy=$1
  local name=$2
  local out=$OUT_ROOT/$name
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
    --max-steps 5000 \
    --batch-size 16 \
    --grad-accum 16 \
    --lr 1e-4 \
    --alpha 0.9 \
    --alpha-warmup 1000 \
    --eval-interval 500 \
    --max-eval-batches 16 \
    --device "$DEVICE" \
    --seed "$SEED" \
    2>&1 | tee -a "$MASTER_LOG"
  echo "=== DONE $name seed=$SEED $(date -Is) ===" | tee -a "$MASTER_LOG"
}

run_arm random cont_random
run_arm Bcov_balanced cont_Bcov_balanced
run_arm distance_only_coverage cont_distance_only_coverage
run_arm shuffled_Bcov_balanced cont_shuffled_Bcov_balanced

echo "seed=$SEED minimal finished $(date -Is)" | tee -a "$MASTER_LOG"
