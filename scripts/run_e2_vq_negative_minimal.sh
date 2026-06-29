#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
TRAIN=$ROOT/scripts/train_vq64_round2.py
OUT_ROOT=$ROOT/probe_results_image/e2_vq_round2_negative_seed42
BASE_CKPT=$ROOT/nanogpt-learned-order/out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256/ckpt.pt
A_BLOCK=$ROOT/probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy
DATA_DIR=$ROOT/block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64
MASTER_LOG=$OUT_ROOT/master.log
SEED=42

mkdir -p "$OUT_ROOT"
{
  echo "E2 VQ no-structure/fallback negative control"
  date -Is
  echo "arms=random Bcov_balanced distance_only_coverage shuffled_Bcov_balanced"
  echo "trainer=$TRAIN"
  echo "baseline_ckpt=$BASE_CKPT"
  echo "A_block=$A_BLOCK"
  echo "data_dir=$DATA_DIR"
  echo "max_steps=3000 batch_size=16 grad_accum=16 lr=1e-4 alpha=0.9 alpha_warmup=1000"
  git -C "$ROOT" status --short --branch > "$OUT_ROOT/git_status_at_launch.txt" || true
  sed -n '1,80p' "$OUT_ROOT/git_status_at_launch.txt" || true
} > "$MASTER_LOG"

run_arm() {
  local policy=$1
  local name=$2
  local device=$3
  local out=$OUT_ROOT/$name
  mkdir -p "$out"
  {
    echo "=== START $name policy=$policy seed=$SEED device=$device $(date -Is) ==="
    "$PY" "$TRAIN" \
      --policy "$policy" \
      --baseline-ckpt "$BASE_CKPT" \
      --a-block-path "$A_BLOCK" \
      --data-train "$DATA_DIR/train.bin" \
      --data-val "$DATA_DIR/val.bin" \
      --meta "$DATA_DIR/meta.pkl" \
      --output-dir "$out" \
      --max-steps 3000 \
      --batch-size 16 \
      --grad-accum 16 \
      --lr 1e-4 \
      --alpha 0.9 \
      --alpha-warmup 1000 \
      --eval-interval 500 \
      --max-eval-batches 16 \
      --save-steps 1000,3000 \
      --device "$device" \
      --seed "$SEED"
    echo "=== DONE $name seed=$SEED $(date -Is) ==="
  } 2>&1 | tee "$out/run.log" >> "$MASTER_LOG"
}

run_arm random cont_random cuda:0 &
run_arm Bcov_balanced cont_Bcov_balanced cuda:1 &
wait

run_arm distance_only_coverage cont_distance_only_coverage cuda:0 &
run_arm shuffled_Bcov_balanced cont_shuffled_Bcov_balanced cuda:1 &
wait

"$PY" "$ROOT/scripts/summarize_e2_vq_negative_minimal.py"
echo "E2 negative minimal finished $(date -Is)" | tee -a "$MASTER_LOG"
