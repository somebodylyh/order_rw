#!/bin/bash
# Three-arm VQ-image AMOR deployment: random / frozen gβ / PG-only gβ.
# Each continues the 30k random VQ backbone for 20k steps; only order policy differs.
# own-order val CE (each arm under its own deployment order) is the headline.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
LOG=../out/rerun_vq

for arm in random frozen pgonly; do
  echo "==================== ARM: $arm  $(date) ===================="
  python -u train.py config/imagenet64vq/deploy_${arm}_vq.py \
    > ${LOG}/deploy_${arm}.log 2>&1
  echo "---- $arm done: $(grep -E 'step [0-9]+: train loss' ${LOG}/deploy_${arm}.log | tail -1)"
done
echo "==================== ALL THREE ARMS DONE  $(date) ===================="
