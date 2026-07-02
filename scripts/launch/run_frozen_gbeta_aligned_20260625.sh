#!/usr/bin/env bash
# frozen_gbeta aligned run — launched 2026-06-25
# Protocol: docs/superpowers/specs/2026-06-25-aligned-training-protocol.md
#
# Waits for GPU 0 memory to drop below 8 GiB (baseline job finished),
# then launches frozen_gbeta from the same step-20000 checkpoint.
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu

OUT_DIR="block_lo_arm_order_network/probe_results/frozen_gbeta_aligned_20260625"
GBETA_CKPT="block_lo_arm_order_network/batch_readout/checkpoints/l0_dynamic_gbeta_step20k_M2000/g_beta_best.pt"
RESUME_CKPT="block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt"

echo "[$(date)] Waiting for GPU 0 to free (< 8 GiB used)..."
while true; do
    mem_used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | awk -F',' '{print $2}' | tr -d ' ')
    if [ "${mem_used:-99999}" -lt 8000 ]; then
        echo "[$(date)] GPU 0 free (${mem_used} MiB used). Launching."
        break
    fi
    sleep 60
done

exec python block_lo_arm_order_network/train_clean_aogpt.py \
    --run-kind frozen_beta \
    --resume-ckpt "$RESUME_CKPT" \
    --output-dir "$OUT_DIR" \
    --frozen-beta-ckpt "$GBETA_CKPT" \
    --frozen-beta-mode argsort \
    --frozen-beta-tau 1.0 \
    --frozen-beta-refresh 10 \
    --frozen-beta-none-mode model \
    --batch-mean-probes 4 \
    --seed 123 \
    --device cuda:0 \
    --max-steps 60000 \
    --lr 1e-3 \
    --min-lr 1e-4 \
    --lr-decay-steps 60000 \
    --warmup-iters 0 \
    --weight-decay 0.1 \
    --beta1 0.9 \
    --beta2 0.99 \
    --grad-clip 1.0 \
    --eval-interval 1000 \
    --log-interval 50 \
    --batch-size 64 \
    --grad-accum 2 \
    --eval-batch-size 16 \
    --stream-eval-windows 2000 \
    --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --wandb-log \
    --wandb-project order-lyu \
    --wandb-run-name frozen_gbeta_aligned_20260625 \
    --wandb-tags aligned,frozen_gbeta,v0_label_free \
    --wandb-mode online \
    2>&1 | tee "$OUT_DIR/train_log.txt"
