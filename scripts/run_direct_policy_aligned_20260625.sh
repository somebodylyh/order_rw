#!/usr/bin/env bash
# Aligned direct-policy continuations from the same step-20000 checkpoint.
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu

RESUME_CKPT="block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt"
ROOT_OUT="block_lo_arm_order_network/probe_results/direct_policy_aligned_20260625"
WANDB_MODE="${WANDB_MODE:-online}"

run_policy() {
    local policy="$1"
    local lambda_dep="$2"
    local out_dir="${ROOT_OUT}/${policy}"

    mkdir -p "$out_dir"
    python block_lo_arm_order_network/train_clean_aogpt.py \
        --run-kind direct_policy \
        --direct-policy "$policy" \
        --direct-policy-lambda-dep "$lambda_dep" \
        --direct-policy-refresh 10 \
        --batch-mean-probes 4 \
        --resume-ckpt "$RESUME_CKPT" \
        --output-dir "$out_dir" \
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
        --wandb-run-name "direct_${policy}_aligned_20260625" \
        --wandb-tags aligned,direct_policy,"${policy}" \
        --wandb-mode "$WANDB_MODE" \
        2>&1 | tee "$out_dir/train_log.txt"
}

run_policy initial_cdl_one_shot 1.0
run_policy source_mass 1.0
run_policy readiness 1.0

