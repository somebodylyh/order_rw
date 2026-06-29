#!/bin/bash
# Serial execution: after P0 (v3 40k->60k) finishes, run P1 then P2
# P0 is PID 3454862 on GPU 0

set -e

REPO_DIR="/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network"
cd "$REPO_DIR"
DEVICE="cuda:0"

echo "=== Waiting for P0 (v3 40k->60k) to finish ==="
while kill -0 3454862 2>/dev/null; do
    sleep 30
done
echo "=== P0 finished at $(date) ==="

echo ""
echo "=== P1: no-readiness ablation (rho=0) from baseline 20k -> 40k ==="
python -u /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/train_clean_aogpt.py \
  --run-kind graph_rw \
  --resume-ckpt /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step20000.pt \
  --output-dir /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results/clean_method_graph_rw_no_readiness_from20k \
  --max-steps 40000 --save-steps 30000,40000 \
  --rw-policy progressive_rw_v3 --rw-lam 0.75 --rw-rho 0.0 \
  --device "$DEVICE"
echo "=== P1 finished at $(date) ==="

echo ""
echo "=== P2: ori_l2r from scratch -> 60k ==="
python -u /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/train_clean_aogpt.py \
  --run-kind l2r \
  --output-dir /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results/clean_ori_l2r_from_scratch \
  --max-steps 60000 --save-steps 20000,40000,50000,60000 \
  --device "$DEVICE"
echo "=== P2 finished at $(date) ==="

echo ""
echo "=== All serial tasks done ==="
