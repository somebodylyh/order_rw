#!/bin/bash
# Overnight: once the improved gβ (bm16 x n_groups=1500) is trained AND GPU 0
# frees (current old-gβ method run finishes), launch the full frozen-gβ method
# with the improved gβ (10k -> 50k) to get its origin_l2r.
set -e
cd /home/admin/lyuyuhuan/order_lyu/chenhe_rerun
X1=/data/users/chenhe/conda_envs/X1/bin/python
GB=out/rerun/gbeta_bm16_g1500/g_beta_best.pt

echo "[wait] improved gβ to finish training..."
while [ ! -f "$GB" ]; do sleep 30; done
sleep 10   # let provenance.json flush

echo "[wait] GPU 0 to free (current method run to finish)..."
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)" -lt 18000 ]; do sleep 60; done
sleep 30

echo "[run] frozen-gβ method (improved gβ) 10k->50k on GPU 0"
CUDA_VISIBLE_DEVICES=0 $X1 train.py \
  config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py \
  --init_from_ckpt=out/rerun/parent_random_10k/ckpt.pt \
  --gbeta_ckpt="$GB" \
  --out_dir=out/rerun/method_gbeta_bm16g1500_50k \
  --gbeta_refresh_every=25 \
  --wandb_log=True --wandb_project=order-rerun-block64 \
  --wandb_run_name=seq256-gbeta-L1H6-bm16g1500-from10k-50k \
  --compile=False
echo "[done]"
