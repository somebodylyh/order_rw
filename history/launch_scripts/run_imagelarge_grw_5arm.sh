#!/bin/bash
# Sequential 5-arm Graph-RW continuation on E3-control-small.
# Phase 2 of docs/superpowers/plans/2026-05-19-image-grw-continuation-e3-control-small.md
#
# Trainer applies the baseline data_permutation and remaps physical block orders
# to model coordinates via inverse_block_perm (see trainer module docstring).
# B is fixed at run-start (no EMA refresh); float32 eval; eval_interval=500.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
CKPT=$REPO/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
ABLOCK=$REPO/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy
DATA_DIR=$REPO/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
ROOT=$REPO/probe_results_image_large/grw_e3ctrlsmall

MAX_STEPS="${MAX_STEPS:-5000}"
GPU="${GPU:-0}"

cd "$REPO"

run_arm() {
    local POL="$1"; shift
    local OUT="$ROOT/cont_${POL}"
    mkdir -p "$OUT"
    echo "================================================="
    echo "[$(date -Iseconds)] arm=$POL  max_steps=$MAX_STEPS  ->  $OUT"
    echo "================================================="
    CUDA_VISIBLE_DEVICES="$GPU" python -u \
        block_lo_arm_order_network/train_imagelarge_graph_rw.py \
        --policy "$POL" \
        --baseline-ckpt "$CKPT" \
        --a-block-path "$ABLOCK" \
        --data-train "$DATA_DIR/train.bin" \
        --data-val "$DATA_DIR/val.bin" \
        --meta "$DATA_DIR/meta.pkl" \
        --output-dir "$OUT" \
        --max-steps "$MAX_STEPS" \
        --eval-interval 500 \
        --save-steps "1000,3000,$MAX_STEPS" \
        "$@" 2>&1 | tee "$OUT/cmd.log"
}

run_arm random
run_arm graph_rw
run_arm raster
run_arm shuffled_B
run_arm eps015 --rw-top-k 0 --rw-epsilon 0.15

echo "[$(date -Iseconds)] ALL 5 ARMS DONE"
