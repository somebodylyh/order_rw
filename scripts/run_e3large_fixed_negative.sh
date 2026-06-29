#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-cuda:1}
SEED=${2:-42}
ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
TRAIN=$ROOT/block_lo_arm_order_network/train_imagelarge_round2.py
OUT_ROOT=$ROOT/probe_results_image_large/e3large_fixed_round2_negative_seed${SEED}
BASE_CKPT=$ROOT/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline/ckpt.pt
A_BLOCK=$ROOT/probe_results_image_large/imagenet64_vqf4_full_l8h8e512_patch2x2_fixed/A_block_8x8.npy
DATA_DIR=$ROOT/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_patch2x2
MASTER_LOG=$OUT_ROOT/master.log
mkdir -p "$OUT_ROOT"
{
  echo "E3-large-fixed no-structure/fallback validation"
  date -Is
  echo "device=$DEVICE seed=$SEED"
  echo "diagnostic_regime=random_or_no_structure_fallback"
  echo "arms=random hilbert Bcov_balanced distance_only_coverage shuffled_Bcov_balanced"
  echo "baseline=$BASE_CKPT"
  echo "a_block=$A_BLOCK"
  echo "data=$DATA_DIR"
} > "$MASTER_LOG"
run_arm() {
  local policy=$1
  local name=$2
  local out=$OUT_ROOT/$name
  mkdir -p "$out"
  echo "=== START $name policy=$policy $(date -Is) ===" | tee -a "$MASTER_LOG"
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
  echo "=== DONE $name $(date -Is) ===" | tee -a "$MASTER_LOG"
}
run_arm random cont_random
run_arm hilbert cont_hilbert
run_arm Bcov_balanced cont_Bcov_balanced
run_arm distance_only_coverage cont_distance_only_coverage
run_arm shuffled_Bcov_balanced cont_shuffled_Bcov_balanced
echo "E3-large-fixed validation finished $(date -Is)" | tee -a "$MASTER_LOG"
