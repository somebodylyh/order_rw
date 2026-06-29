#!/bin/bash
# Tonight experiments: A_cont 40k->60k, B alpha=0.95 30k->60k, no-refresh 20k->40k
# Sequential on GPU 0
set -euo pipefail

ROOT="/home/admin/lyuyuhuan/order_lyu"
SCRIPT="block_lo_arm_order_network/train_method_from_base.py"
COMMON="--run-kind graph_rw --device cuda:0 --seed 42 --permute-seed 42 \
  --eval-interval 1000 --log-interval 10 --batch-size 64 --grad-accum 2 \
  --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
  --tau-start 0.1 --tau-step 0.1 --rw-top-k 4 --rw-order-bag-k 1 --epsilon-uniform 0.0 \
  --refresh-n-chunks 200 --refresh-ema-beta 0.9 --eval-order-seeds 42 123 456"

cd "$ROOT"

# ── P0: A_cont 40k→60k ──────────────────────────────────────────────────
echo "=== [P0] Starting A_cont 40k→60k at $(date) ==="
python -u $SCRIPT \
  --resume-ckpt block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a09_30k40k/ckpt_step40000.pt \
  --output-dir block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a09_30k40k \
  --max-steps 60000 --save-steps 50000,60000 \
  --alpha-start 0.9 --alpha-target 0.9 --alpha-warmup-steps 1 \
  --refresh-interval 2000 \
  $COMMON \
  > block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a09_30k40k/stdout_60k.log 2>&1
echo "=== [P0] Done at $(date) ==="

# ── P1: B α=1.0→0.95 30k→60k ───────────────────────────────────────────
echo "=== [P1] Starting B alpha=0.95 30k→60k at $(date) ==="
OUTDIR1="block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k"
mkdir -p "$OUTDIR1"
python -u $SCRIPT \
  --resume-ckpt block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_from20k/ckpt_step30000.pt \
  --output-dir "$OUTDIR1" \
  --max-steps 60000 --save-steps 35000,40000,50000,60000 \
  --alpha-start 0.95 --alpha-target 0.95 --alpha-warmup-steps 1 \
  --refresh-interval 2000 \
  $COMMON \
  > "$OUTDIR1/stdout.log" 2>&1
echo "=== [P1] Done at $(date) ==="

# ── P2: no-refresh 20k→40k ──────────────────────────────────────────────
echo "=== [P2] Starting no-refresh 20k→40k at $(date) ==="
OUTDIR2="block_lo_arm_order_network/probe_results/clean_method_graph_rw_a09_v2_no_refresh_20k40k"
mkdir -p "$OUTDIR2"
python -u $SCRIPT \
  --resume-ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step20000.pt \
  --output-dir "$OUTDIR2" \
  --max-steps 40000 --save-steps 30000,40000 \
  --alpha-start 0.0 --alpha-target 0.9 --alpha-warmup-steps 10000 \
  --refresh-interval 0 \
  $COMMON \
  > "$OUTDIR2/stdout.log" 2>&1
echo "=== [P2] Done at $(date) ==="

echo "=== ALL DONE at $(date) ==="
