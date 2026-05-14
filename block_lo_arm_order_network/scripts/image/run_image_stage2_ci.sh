#!/usr/bin/env bash
# Image Stage-2 3-seed CI: 3 seeds × 4 configs × 5000 steps from baseline10k.
# Sequential within a single GPU (parallel runs OOM each other on shared 4090).

set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

GPU="${CUDA_VISIBLE_DEVICES:-0}"
BASE_CKPT="probe_results_image/baseline/baseline10k/ckpt_step10000.pt"
B_PATH="probe_results_image/attention/baseline10k/B_global.npy"
COMBINED_LOG="probe_results_image/graph_rw_ci/stage2_ci.log"
mkdir -p "$(dirname "$COMBINED_LOG")"
: > "$COMBINED_LOG"

for SEED in 0 1 42; do
  for spec in "cont_random:0.0:4:0.0" "cont_top4:0.9:4:0.0" "cont_eps015:0.9:0:0.15" "cont_top8:0.9:8:0.0"; do
    IFS=: read -r name alpha topk eps <<< "$spec"
    OUT="probe_results_image/graph_rw_ci/${name}/seed${SEED}"
    mkdir -p "$OUT"
    {
      echo "=== [seed=${SEED}] START ${name} ($(date +%H:%M:%S)) ==="
      CUDA_VISIBLE_DEVICES="$GPU" python -u image_order/train_image_graph_rw.py \
          --baseline-ckpt "$BASE_CKPT" --b-path "$B_PATH" \
          --output-dir "$OUT" \
          --max-steps 5000 --batch-size 64 \
          --alpha "$alpha" --rw-top-k "$topk" --rw-epsilon "$eps" \
          --eval-interval 5000 --log-interval 1000 \
          --seed "$SEED"

      echo "  -- [seed=${SEED}] frozen 5-order eval ${name} --"
      CUDA_VISIBLE_DEVICES="$GPU" python -u image_order/eval_image_orders.py \
          --ckpt "$OUT/ckpt_step5000.pt" \
          --b-path "$B_PATH" \
          --output-dir "probe_results_image/eval_ci/${name}/seed${SEED}" \
          --val-images 1000 --batch-size 64 --n-seed-rounds 3 \
          --seed-base "$SEED"
      echo "=== [seed=${SEED}] END ${name} ($(date +%H:%M:%S)) ==="
    } 2>&1 | tee -a "$COMBINED_LOG"
  done
done

echo "=== STAGE2 CI ALL DONE $(date +%H:%M:%S) ===" | tee -a "$COMBINED_LOG"
