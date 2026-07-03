#!/bin/bash
# Full frozen-gβ production pipeline (chained). Run from chenhe_rerun/.
#   1. random parent -> 10k (keep lr_decay_iters=50000 for lr continuity)
#   2. CDL-pretrain gβ from that parent (M=1000)
#   3. frozen-gβ method 10k->50k (refresh_every=25 for tractable speed)
set -e
cd /home/admin/lyuyuhuan/order_lyu/chenhe_rerun
X1=/data/users/chenhe/conda_envs/X1/bin/python
export CUDA_VISIBLE_DEVICES=0

echo "=== [1/3] parent random -> 10k ==="
$X1 train.py config/WikiText103/seq256/permute/block64/random.py \
  --max_iters=10000 --out_dir=out/rerun/parent_random_10k \
  --wandb_log=False --compile=False

echo "=== [2/3] CDL-pretrain gβ (Stage-A head select + single-head NodewiseReadout + batch-mean) ==="
$X1 -c "from gbeta_cdl_pretrain import pretrain_gbeta_cdl; \
print('GBETA:', pretrain_gbeta_cdl('out/rerun/parent_random_10k/ckpt.pt', \
  'out/rerun/gbeta_from_parent10k', n_select=800, n_groups=300, batch_mean_size=16, \
  n_reveal=8, epochs=40, device='cuda'))"

echo "=== [3/3] frozen-gβ method 10k->50k (refresh_every=25) ==="
$X1 train.py config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py \
  --gbeta_refresh_every=25 \
  --wandb_log=True --wandb_project=order-rerun-block64 --compile=False

echo "=== DONE ==="
