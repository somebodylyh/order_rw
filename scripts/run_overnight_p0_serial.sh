#!/usr/bin/env bash
# Serial overnight: GPU 0, P0 priority order.
#   (1) Resume shuffled-L2R control jun05 from 10k → 50k (P0#2, critical confound fix)
#   (2) Random baseline seed2 → 50k (P0#1)
#
# shuffled-L2R ~40k remaining → ~10h. Random ~50k → ~12h. Total ~22h on one GPU.
set -euo pipefail

GPU="${1:-cuda:0}"
PKG="/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network"
cd "$PKG"
TS="$(date +%Y%m%d_%H%M)"

echo "================================================================"
echo "Serial overnight — started $(date)"
echo "GPU=$GPU  PKG=$PKG"
echo "================================================================"

# ═══════════════════════════════════════════════════════════════════
# Step 1: Shuffled-L2R control (P0#2)
# Resume jun05 run from ckpt_step10000.pt (ran to 16.8k on GPU1 before chenhe took over).
# This MUST complete — it's the key confound fix for position leakage.
# ═══════════════════════════════════════════════════════════════════
SHUFDIR="probe_results/shuffled_l2r_continuous_jun05"

if [ -f "$SHUFDIR/ckpt_step10000.pt" ]; then
    echo ""
    echo "=== [1/2] shuffled-L2R control: resuming jun05 from step 10000 ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_clean_aogpt.py \
      --run-kind shuffled_l2r \
      --resume-ckpt "$SHUFDIR/ckpt_step10000.pt" \
      --data-source continuous \
      --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
      --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
      --stream-eval-windows 2000 \
      --seed 42 --permute-seed 42 \
      --max-steps 50000 --lr-decay-steps 50000 \
      --batch-size 64 --grad-accum 2 \
      --lr 1e-3 --min-lr 1e-4 \
      --output-dir "$SHUFDIR" \
      --eval-interval 1000 --log-interval 50 \
      --save-steps "20000,30000,40000,50000" \
      --device "$GPU" \
      2>&1 | tee -a "$SHUFDIR/train_log_${TS}.txt"
    echo "=== [1/2] shuffled-L2R DONE at $(date) ==="
else
    echo "=== [1/2] shuffled-L2R: ckpt_step10000.pt not found, starting from scratch ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_clean_aogpt.py \
      --run-kind shuffled_l2r \
      --data-source continuous \
      --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
      --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
      --stream-eval-windows 2000 \
      --seed 42 --permute-seed 42 \
      --max-steps 50000 --lr-decay-steps 50000 \
      --batch-size 64 --grad-accum 2 \
      --lr 1e-3 --min-lr 1e-4 \
      --output-dir "$SHUFDIR" \
      --eval-interval 1000 --log-interval 50 \
      --save-steps "0,1000,5000,10000,20000,30000,40000,50000" \
      --device "$GPU" \
      2>&1 | tee -a "$SHUFDIR/train_log_${TS}.txt"
    echo "=== [1/2] shuffled-L2R DONE at $(date) ==="
fi

# ═══════════════════════════════════════════════════════════════════
# Step 2: Random baseline seed2 (P0#1)
# From scratch with seed=123, save ckpts at 5k/10k for future frozen_beta use.
# ═══════════════════════════════════════════════════════════════════
SEED2DIR="probe_results/random_baseline_continuous_jun08_seed2"

echo ""
echo "=== [2/2] random baseline seed2: from scratch ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_clean_aogpt.py \
  --run-kind baseline \
  --data-source continuous \
  --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
  --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
  --stream-eval-windows 2000 \
  --seed 123 --permute-seed 123 \
  --max-steps 50000 --lr-decay-steps 50000 \
  --batch-size 64 --grad-accum 2 \
  --lr 1e-3 --min-lr 1e-4 \
  --output-dir "$SEED2DIR" \
  --eval-interval 1000 --log-interval 50 \
  --save-steps "0,1000,5000,10000,20000,30000,40000,50000" \
  --device "$GPU" \
  2>&1 | tee "$SEED2DIR/train_log.txt"
echo "=== [2/2] random baseline seed2 DONE at $(date) ==="

echo ""
echo "================================================================"
echo "All overnight runs completed — $(date)"
echo "================================================================"
