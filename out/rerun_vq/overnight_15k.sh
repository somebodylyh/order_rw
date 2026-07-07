#!/bin/bash
# Overnight: gβ(15k warmup) + frozen + pgonly deploys to 50k. No baseline rerun.
set -e
REPO=/home/admin/lyuyuhuan/order_lyu
export CUDA_VISIBLE_DEVICES=1
LOG=$REPO/out/rerun_vq

echo "==== WAIT for FID (PID 588353) $(date) ===="
while kill -0 588353 2>/dev/null; do sleep 60; done
echo "==== FID done, GPU1 free $(date) ===="

echo "==== [1/3] gβ pretrain on 15k warmup $(date) ===="
cd $REPO
python -u chenhe_rerun/run_gbeta_vq15k_pretrain.py > $LOG/gbeta_vq15k_pretrain.log 2>&1
echo "gβ done: $(grep GBETA $LOG/gbeta_vq15k_pretrain.log | tail -1)"

echo "==== [2/3] frozen deploy 15k->50k $(date) ===="
cd $REPO/chenhe_rerun
python -u train.py config/imagenet64vq/deploy15k_frozen.py > $LOG/deploy15k_frozen.log 2>&1
echo "frozen done: $(grep -E 'step [0-9]+: train loss' $LOG/deploy15k_frozen.log | tail -1)"

echo "==== [3/3] pgonly deploy 15k->50k (PG@20k) $(date) ===="
python -u train.py config/imagenet64vq/deploy15k_pgonly.py > $LOG/deploy15k_pgonly.log 2>&1
echo "pgonly done: $(grep -E 'step [0-9]+: train loss' $LOG/deploy15k_pgonly.log | tail -1)"

echo "==== ALL OVERNIGHT DONE $(date) ===="
