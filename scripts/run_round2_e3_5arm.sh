#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
TRAIN=$ROOT/block_lo_arm_order_network/train_imagelarge_round2.py
OUT_ROOT=$ROOT/probe_results_image_large/grw_e3ctrlsmall_round2_5arm
BASE_CKPT=$ROOT/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
A_BLOCK=$ROOT/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy
DATA_DIR=$ROOT/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
MASTER_LOG=$OUT_ROOT/round2_master.log

mkdir -p "$OUT_ROOT"
{
  echo "Round-2 E3 5-arm launch"
  date -Is
  echo "root=$ROOT"
  echo "trainer=$TRAIN"
  echo "out_root=$OUT_ROOT"
  echo "baseline_ckpt=$BASE_CKPT"
  echo "a_block=$A_BLOCK"
  echo "device=cuda:0"
  echo "Bcov_balanced: score = minmax(B[last,v]) - minmax(manh(last,v)); gamma_B=1.0 gamma_d=1.0 deterministic argmax; cyclic starts"
  echo "physical orders remapped inside trainer by inverse_block_perm when baseline ckpt has permute_data=True"
  git -C "$ROOT" status --short --branch > "$OUT_ROOT/git_status_at_launch.txt" || true
  sed -n "1,80p" "$OUT_ROOT/git_status_at_launch.txt" || true
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
    --device cuda:0 \
    --seed 42 \
    2>&1 | tee -a "$MASTER_LOG"
  echo "=== DONE $name $(date -Is) ===" | tee -a "$MASTER_LOG"
}

run_arm random cont_random
run_arm graph_rw cont_v1_graph_rw
run_arm hilbert cont_hilbert
run_arm Bcov_balanced cont_Bcov_balanced
run_arm raster cont_raster

echo "Round-2 5-arm finished $(date -Is)" | tee -a "$MASTER_LOG"
