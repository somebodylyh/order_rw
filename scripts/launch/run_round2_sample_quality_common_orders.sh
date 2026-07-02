#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=${1:-1024}
REAL_SAMPLES=${2:-2048}

ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
SCRIPT=$ROOT/scripts/sample_quality_fid_imagenet64_vq.py
ROUND2=$ROOT/probe_results_image_large/grw_e3ctrlsmall_round2_5arm
OUT_ROOT=$ROOT/probe_results_image_large/grw_e3ctrlsmall_round2_sample_quality_common_orders_n${NUM_SAMPLES}
ARMS=cont_random,cont_hilbert,cont_Bcov_balanced,cont_distance_only_coverage,cont_shuffled_Bcov_balanced,cont_raster

mkdir -p "$OUT_ROOT"
{
  echo "Round-2 common-order sample quality / proxy FID"
  date -Is
  echo "num_samples=$NUM_SAMPLES"
  echo "fid_real_samples=$REAL_SAMPLES"
  echo "arms=$ARMS"
  echo "policies=random hilbert Bcov_balanced"
  echo "fid_reference=VQ-decoded validation tokens"
  git -C "$ROOT" status --short --branch > "$OUT_ROOT/git_status_at_launch.txt" || true
  sed -n '1,80p' "$OUT_ROOT/git_status_at_launch.txt" || true
} > "$OUT_ROOT/run.log"

run_policy() {
  local policy=$1
  local device=$2
  local out=$OUT_ROOT/common_${policy}
  mkdir -p "$out"
  echo "=== START common policy=$policy device=$device $(date -Is) ===" | tee -a "$OUT_ROOT/run.log"
  "$PY" "$SCRIPT" \
    --round2-root "$ROUND2" \
    --arms "$ARMS" \
    --out-dir "$out" \
    --data-dir "$ROOT/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8" \
    --a-block-path "$ROOT/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy" \
    --device "$device" \
    --num-samples "$NUM_SAMPLES" \
    --fid-real-samples "$REAL_SAMPLES" \
    --batch-size 16 \
    --decode-batch-size 64 \
    --seed 42 \
    --temperature 1.0 \
    --top-k 0 \
    --force-policy "$policy" \
    2>&1 | tee "$out/run.log" >> "$OUT_ROOT/run.log"
  echo "=== DONE common policy=$policy $(date -Is) ===" | tee -a "$OUT_ROOT/run.log"
}

run_policy random cuda:0 &
run_policy hilbert cuda:1 &
wait

run_policy Bcov_balanced cuda:0

"$PY" "$ROOT/scripts/summarize_sample_quality_common_orders.py" "$OUT_ROOT"
echo "Common-order sample quality finished $(date -Is)" | tee -a "$OUT_ROOT/run.log"
