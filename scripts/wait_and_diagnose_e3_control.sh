#!/bin/bash
# E3-control-small post-train pipeline.
# Waits for the training PID to exit, then runs attention extraction +
# dual-granularity diagnostic + summary.
set -euo pipefail

TRAIN_PID="${TRAIN_PID:-14658}"
REPO=/home/admin/lyuyuhuan/order_lyu
CKPT=$REPO/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
DATA=$REPO/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/val.bin
META=$REPO/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/meta.pkl
OUTDIR=$REPO/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control

cd "$REPO"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "==================================================================="
echo "[`date -Iseconds`] E3-control-small post-train pipeline starting"
echo "TRAIN_PID=$TRAIN_PID  CKPT=$CKPT"
echo "==================================================================="

# 1. Wait for training PID to exit
echo "[wait] Polling PID $TRAIN_PID every 60s..."
while ps -p "$TRAIN_PID" > /dev/null 2>&1; do
    sleep 60
done
echo "[wait] Train PID $TRAIN_PID has exited at `date -Iseconds`."
sleep 5

# Verify ckpt exists
if [[ ! -f "$CKPT" ]]; then
    echo "[FATAL] ckpt not found: $CKPT"
    exit 1
fi
echo "[ok] ckpt found: $(stat -c '%s bytes, mtime=%y' "$CKPT")"

# 2. Extract A_global (N=256, token-level)
echo "==================================================================="
echo "[`date -Iseconds`] Step 1/3: extracting attention (N=256)"
echo "==================================================================="
CUDA_VISIBLE_DEVICES=0 python block_lo_arm_order_network/extract_image_attention_e2.py \
    --ckpt "$CKPT" --data "$DATA" --meta "$META" --out "$OUTDIR" \
    --n-images 500 --M-passes 3 --device cuda:0

# 3. Dual-level diagnostic + summary
echo "==================================================================="
echo "[`date -Iseconds`] Step 2/3: dual-level diagnostic + summary"
echo "==================================================================="
python scripts/diagnose_e3_control_dual_level.py \
    --a-global "$OUTDIR/A_global.npy" \
    --outdir "$OUTDIR" \
    --label "E3-control-small l4h8e256 patch2x2 20k"

echo "==================================================================="
echo "[`date -Iseconds`] Step 3/3: DONE"
echo "Output dir: $OUTDIR"
echo "  - A_global.npy        (256x256 token-level)"
echo "  - A_block_8x8.npy     (64x64 block-aggregated)"
echo "  - diagnostics.json    (dual-level metrics + controls)"
echo "  - SUMMARY.md          (cross-experiment comparison)"
echo "==================================================================="
