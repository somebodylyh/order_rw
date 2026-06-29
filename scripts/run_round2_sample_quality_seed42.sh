#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-cuda:0}
NUM_SAMPLES=${2:-256}
REAL_SAMPLES=${3:-1024}

ROOT=/home/admin/lyuyuhuan/order_lyu
PY=/home/admin/anaconda3/envs/X1/bin/python
OUT=$ROOT/probe_results_image_large/grw_e3ctrlsmall_round2_sample_quality_seed42
SCRIPT=$ROOT/scripts/sample_quality_fid_imagenet64_vq.py

mkdir -p "$OUT"
{
  echo "Round-2 seed42 sample quality / FID"
  date -Is
  echo "device=$DEVICE"
  echo "num_samples=$NUM_SAMPLES"
  echo "fid_real_samples=$REAL_SAMPLES"
  echo "fid_reference=VQ-decoded validation tokens"
  echo "arms=random,v1_graph_rw,hilbert,Bcov_balanced,distance_only_coverage,shuffled_Bcov_balanced,raster"
} > "$OUT/run.log"

"$PY" "$SCRIPT" \
  --round2-root "$ROOT/probe_results_image_large/grw_e3ctrlsmall_round2_5arm" \
  --arms cont_random,cont_v1_graph_rw,cont_hilbert,cont_Bcov_balanced,cont_distance_only_coverage,cont_shuffled_Bcov_balanced,cont_raster \
  --out-dir "$OUT" \
  --data-dir "$ROOT/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8" \
  --a-block-path "$ROOT/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy" \
  --device "$DEVICE" \
  --num-samples "$NUM_SAMPLES" \
  --fid-real-samples "$REAL_SAMPLES" \
  --batch-size 16 \
  --decode-batch-size 64 \
  --seed 42 \
  --temperature 1.0 \
  --top-k 0 \
  2>&1 | tee -a "$OUT/run.log"
