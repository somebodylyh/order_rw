#!/bin/bash
# Task 6: post-train attention re-extraction + dual-level diagnostic.
# DESCRIPTIVE attention-drift diagnostic only — NOT a decision gate.
# Extracts baseline fresh (procedure-consistent) + each of the 5 arms, so drift
# is measured in one consistent extraction frame.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
BASE_CKPT=$REPO/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
DATA_DIR=$REPO/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
ROOT=$REPO/probe_results_image_large/grw_e3ctrlsmall
GPU="${GPU:-0}"
NIMG="${NIMG:-500}"
MPASS="${MPASS:-3}"

cd "$REPO"

extract_and_diag() {
    local CKPT="$1"; local ATT="$2"; local LABEL="$3"
    mkdir -p "$ATT"
    echo "=================================================="
    echo "[$(date -Iseconds)] $LABEL"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES="$GPU" python block_lo_arm_order_network/extract_image_attention_e2.py \
        --ckpt "$CKPT" --data "$DATA_DIR/val.bin" --meta "$DATA_DIR/meta.pkl" \
        --out "$ATT" --n-images "$NIMG" --M-passes "$MPASS" --device cuda:0
    python scripts/diagnose_e3_control_dual_level.py \
        --a-global "$ATT/A_global.npy" --outdir "$ATT" --label "$LABEL"
}

# Fresh baseline extraction (reference for drift, same procedure as arms)
extract_and_diag "$BASE_CKPT" "$ROOT/baseline_attention" "baseline (pre-continuation, fresh)"

for POL in random graph_rw raster shuffled_B eps015; do
    CKPT="$ROOT/cont_$POL/ckpt_final.pt"
    [[ -f "$CKPT" ]] || { echo "[skip] $POL: $CKPT missing"; continue; }
    extract_and_diag "$CKPT" "$ROOT/cont_$POL/attention" "cont_$POL (post 5000 steps)"
done

echo "[$(date -Iseconds)] ALL POST-DIAGNOSE DONE"
