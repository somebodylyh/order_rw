#!/bin/bash
# Serial on ONE card (GPU0): wait for the running l2r verdict run to finish,
# then random-order baseline 0->50k, then g_beta hook from its 5k ckpt (alpha 0->1 over 5k).
set -u
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
GPU=cuda:0
RAND=probe_results/random_baseline_continuous_jun05
GBETA=probe_results/gbeta_from5000_continuous_jun05
GBETA_CKPT=batch_readout/logs/phase33_gbeta/full/g_beta_best.pt
COMMON="--data-source continuous --device $GPU --max-steps 50000 --lr-decay-steps 50000 \
 --batch-size 64 --grad-accum 2 --lr 1e-3 --min-lr 1e-4 \
 --eval-interval 1000 --stream-eval-windows 2000 \
 --save-steps 0,1000,5000,10000,20000,30000,40000,50000 --log-interval 50"

echo "[$(date)] waiting for l2r run (PID 290627) to free GPU0..."
while kill -0 290627 2>/dev/null; do sleep 120; done
echo "[$(date)] GPU0 free."

echo "[$(date)] === ARM A: random baseline (continuous, GPU0) ==="
rm -rf "$RAND"
python3 train_clean_aogpt.py --run-kind baseline --output-dir "$RAND" $COMMON
RC=$?
echo "[$(date)] random baseline exited rc=$RC"
if [ $RC -ne 0 ] || [ ! -f "$RAND/ckpt_step5000.pt" ]; then
  echo "[$(date)] ABORT: random baseline failed or no ckpt_step5000.pt; not starting g_beta."
  exit 1
fi

echo "[$(date)] === ARM B: g_beta hook from 5k ckpt (alpha 0->1 over 5k, ramp-from-resume, GPU0) ==="
rm -rf "$GBETA"
python3 train_clean_aogpt.py --run-kind frozen_beta --output-dir "$GBETA" $COMMON \
  --resume-ckpt "$RAND/ckpt_step5000.pt" \
  --frozen-beta-ckpt "$GBETA_CKPT" --frozen-beta-head 0 0 \
  --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 --alpha-ramp-from-resume
echo "[$(date)] g_beta exited rc=$?"
echo "[$(date)] === SERIAL CHAIN DONE ==="
