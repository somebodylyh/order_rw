#!/bin/bash
# Check large model training progress and run per-head scan when done.
# Usage: bash scripts/check_large_model_progress.sh
set -euo pipefail

TRAIN_LOG=/tmp/large_model_final.log
OUT_DIR=/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d

echo "=== Training Progress ==="
if ps aux | grep -q "[t]rain_clean_aogpt.*large"; then
    echo "Training is RUNNING"
    tail -c 300 "$TRAIN_LOG" | tr '\r' '\n' | grep "step" | tail -1
else
    echo "Training has FINISHED"
fi

echo ""
echo "=== Checkpoints ==="
ls -lh "$OUT_DIR"/ckpt_step*.pt 2>/dev/null || echo "No checkpoints yet"

echo ""
echo "=== Loss curve (last 5 evals) ==="
grep "^[0-9]" "$OUT_DIR"/eval_curve.tsv 2>/dev/null | tail -5 || echo "No eval curve yet"

echo ""
echo "=== To scan after training ==="
echo "  # For the large model (256 heads), use smaller M to fit in GPU memory:"
echo "  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=block_lo_arm_order_network \\"
echo "    python block_lo_arm_order_network/per_head_order_scan.py \\"
echo "      --ckpt $OUT_DIR/ckpt_step5000.pt \\"
echo "      --M 20 --batch-size 8 --seed 42 --device cuda:0 \\"
echo "      --alpha-dep 0.5 --none-mode b0 \\"
echo "      --out $OUT_DIR/head_scan_step5000.json"
