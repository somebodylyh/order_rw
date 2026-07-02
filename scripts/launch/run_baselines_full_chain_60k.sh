#!/bin/bash
# Full chain: seed2+seed42 ori-L2R + random baselines, 0→60k, GPU 1
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"
TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
COMMON="--data-source continuous --n-layer 4 --n-head 8 --n-embd 384 \
  --batch-size 64 --grad-accum 2 --max-steps 60000 \
  --lr 1e-3 --min-lr 1e-4 --lr-decay-steps 60000 \
  --save-steps 0,5000,10000,20000,30000,40000,50000,60000 \
  --eval-interval 1000 --log-interval 100 --device cuda:0"

run_one() {
    local KIND="$1" SEED="$2" OUT="$3"
    echo "============================================"
    echo "[$(date)] $KIND seed=$SEED → $OUT (0→60k)"
    echo "============================================"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=block_lo_arm_order_network \
      python "$TRAIN" \
        --run-kind "$KIND" --seed "$SEED" --permute-seed "$SEED" \
        --output-dir "$OUT" $COMMON \
        2>&1 | tee "${OUT}/train.log"
    echo "[$(date)] $KIND seed=$SEED done."
    echo ""
}

# 1. ori-L2R seed2
run_one l2r 2 block_lo_arm_order_network/probe_results/l2r_continuous_seed2

# 2. ori-L2R seed42
run_one l2r 42 block_lo_arm_order_network/probe_results/l2r_continuous_seed42

# 3. Random baseline seed2
run_one baseline 2 block_lo_arm_order_network/probe_results/random_baseline_continuous_seed2

# 4. Random baseline seed42
run_one baseline 42 block_lo_arm_order_network/probe_results/random_baseline_continuous_seed42

echo ""
echo "[$(date)] All 4 baselines complete."
echo "  l2r seed2:    probe_results/l2r_continuous_seed2/"
echo "  l2r seed42:   probe_results/l2r_continuous_seed42/"
echo "  random seed2: probe_results/random_baseline_continuous_seed2/"
echo "  random seed42: probe_results/random_baseline_continuous_seed42/"
