#!/usr/bin/env bash
# Image 25k long-run continuation: cont_random_long vs cont_top4_long.
# Both fork from baseline10k. Used for the persistent-gain analysis.

set -euo pipefail
cd "$(dirname "$0")/../../.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
BASE_CKPT="probe_results_image/baseline/baseline10k/ckpt_step10000.pt"
B_PATH="probe_results_image/attention/baseline10k/B_global.npy"

LONG_LOG="probe_results_image/graph_rw_ci/long_run.log"
mkdir -p "$(dirname "$LONG_LOG")"

for spec in "cont_random_long:0.0:4:0.0" "cont_top4_long:0.9:4:0.0"; do
  IFS=: read -r name alpha topk eps <<< "$spec"
  OUT="probe_results_image/graph_rw_ci/${name}"
  mkdir -p "$OUT"
  {
    echo "=== START ${name} ($(date +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES="$GPU" python -u image_order/train_image_graph_rw.py \
        --baseline-ckpt "$BASE_CKPT" --b-path "$B_PATH" \
        --output-dir "$OUT" \
        --max-steps 25000 --batch-size 64 \
        --alpha "$alpha" --rw-top-k "$topk" --rw-epsilon "$eps" \
        --eval-interval 1000 --log-interval 500 \
        --seed 42
    echo "=== END ${name} ($(date +%H:%M:%S)) ==="
  } 2>&1 | tee -a "$LONG_LOG"
done

echo "=== LONG RUN ALL DONE $(date +%H:%M:%S) ===" | tee -a "$LONG_LOG"
