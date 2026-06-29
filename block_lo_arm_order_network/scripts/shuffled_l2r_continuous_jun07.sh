#!/bin/bash
# shuffled-L2R control baseline (continuous), 6.8-gate arm #4 — re-run.
#
# Waits for from20k frozen-beta on GPU0 to finish, then launches shuffled_AR
# from scratch on GPU0.
set -u
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network

GPU=0
TARGET_PID=$(pgrep -f "frozen_beta.*from20k" | head -1)
OUT=probe_results/shuffled_l2r_continuous_jun07

COMMON="--data-source continuous --max-steps 50000 --lr-decay-steps 50000 \
 --batch-size 64 --grad-accum 2 --lr 1e-3 --min-lr 1e-4 \
 --eval-interval 1000 --stream-eval-windows 2000 \
 --save-steps 0,1000,5000,10000,20000,30000,40000,50000 --log-interval 50"

if [ -n "${TARGET_PID:-}" ] && ps -p "$TARGET_PID" >/dev/null 2>&1; then
  echo "[$(date)] waiting for frozen-beta from20k PID $TARGET_PID to finish..."
  while ps -p "$TARGET_PID" >/dev/null 2>&1; do sleep 120; done
  echo "[$(date)] from20k done."
else
  echo "[$(date)] frozen-beta from20k not running (or already finished), proceeding."
fi

echo "[$(date)] launching shuffled_l2r on GPU$GPU -> $OUT"
rm -rf "$OUT"
python3 train_clean_aogpt.py --run-kind shuffled_l2r --device cuda:$GPU --output-dir "$OUT" $COMMON
echo "[$(date)] shuffled_l2r exited rc=$?"
echo "[$(date)] === shuffled-L2R DONE; arm #4 -> $OUT/eval_curve.tsv ==="
