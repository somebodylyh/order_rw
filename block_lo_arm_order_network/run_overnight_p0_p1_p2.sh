#!/bin/bash
# Overnight experiments: P0 (baseline 20k→50k) + P1 (layer/head scan) + P2 (batch-level diagnostic)
# Launch from repo root: bash block_lo_arm_order_network/run_overnight_p0_p1_p2.sh
# GPU 0: P0 training (background)
# GPU 1: P1 → P2 → post-training diagnostics (sequential)

set -e

REPO="/home/admin/lyuyuhuan/order_lyu"
NANO="${REPO}/nanogpt-learned-order"
BLON="${REPO}/block_lo_arm_order_network"
DATA="${NANO}/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin"
CKPT_DIR="${NANO}/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline"
CKPT_20K="${CKPT_DIR}/checkpoints/ckpt_iter0020000.pt"
CONFIG_50K="${BLON}/configs/image_large/imagenet64_l8h8e512_patch2x2_full_50k.py"
LOGS="${REPO}/logs"

mkdir -p "${LOGS}"

echo "=== Overnight Experiments: $(date) ==="
echo "  P0: Baseline 20k→50k on GPU 0"
echo "  P1: Layer/head scan on GPU 1"
echo "  P2: Batch-level diagnostic on GPU 1"
echo ""

# ─── P0: Resume training from 20k to 50k (GPU 0) ───
echo "[P0] Starting baseline training 20k→50k on GPU 0..."
cd "${NANO}"
CUDA_VISIBLE_DEVICES=0 python train.py "${CONFIG_50K}" \
    > "${LOGS}/p0_training_50k.log" 2>&1 &
P0_PID=$!
cd "${REPO}"
echo "  P0 PID: ${P0_PID}"
echo "  Log: ${LOGS}/p0_training_50k.log"

# ─── P1: Layer/head locality scan (GPU 1, step 20k) ───
echo ""
echo "[P1] Starting layer/head locality scan (step 20k) on GPU 1..."
python "${BLON}/layer_head_locality_scan.py" \
    --ckpt "${CKPT_20K}" \
    --data "${DATA}" \
    --step 20000 \
    --n-images 200 \
    --device cuda:1 \
    > "${LOGS}/p1_layer_head_20k.log" 2>&1
echo "[P1] Done. $(date)"
echo "  Log: ${LOGS}/p1_layer_head_20k.log"

# ─── P2: Batch-level block graph diagnostic (GPU 1, step 20k) ───
echo ""
echo "[P2] Starting batch-level diagnostic (step 20k) on GPU 1..."
python "${BLON}/batch_level_graph_diagnostic.py" \
    --ckpt "${CKPT_20K}" \
    --data "${DATA}" \
    --step 20000 \
    --n-batches 100 \
    --batch-size 8 \
    --device cuda:1 \
    > "${LOGS}/p2_batch_level_20k.log" 2>&1
echo "[P2] Done. $(date)"
echo "  Log: ${LOGS}/p2_batch_level_20k.log"

# ─── Wait for P0 training to complete ───
echo ""
echo "[WAIT] Waiting for P0 training to finish (PID ${P0_PID})..."
wait ${P0_PID}
P0_EXIT=$?
echo "[P0] Training done (exit=${P0_EXIT}). $(date)"

# ─── Post-training: Layer/head scan on 50k ───
CKPT_50K="${CKPT_DIR}/checkpoints/ckpt_iter0050000.pt"
if [ -f "${CKPT_50K}" ]; then
    echo ""
    echo "[P1-50k] Running layer/head scan on 50k checkpoint..."
    python "${BLON}/layer_head_locality_scan.py" \
        --ckpt "${CKPT_50K}" \
        --data "${DATA}" \
        --step 50000 \
        --n-images 200 \
        --device cuda:1 \
        > "${LOGS}/p1_layer_head_50k.log" 2>&1
    echo "[P1-50k] Done. $(date)"
else
    echo "[P1-50k] SKIP: checkpoint ${CKPT_50K} not found"
fi

# ─── Post-training: Frozen diagnostic on 30k, 40k, 50k ───
for STEP in 30000 40000 50000; do
    CKPT="${CKPT_DIR}/checkpoints/ckpt_iter$(printf '%07d' ${STEP}).pt"
    if [ -f "${CKPT}" ]; then
        OUT="${BLON}/probe_results_image_large/diagnostics_patch8x8/step_${STEP}"
        echo ""
        echo "[Diag-${STEP}] Running frozen diagnostic..."
        python "${BLON}/run_frozen_diagnostic.py" \
            --ckpt "${CKPT}" \
            --data "${DATA}" \
            --step ${STEP} \
            --out-dir "${OUT}" \
            --device cuda:1 \
            > "${LOGS}/diag_step${STEP}.log" 2>&1
        echo "[Diag-${STEP}] Done. $(date)"
    else
        echo "[Diag-${STEP}] SKIP: checkpoint ${CKPT} not found"
    fi
done

echo ""
echo "=== ALL OVERNIGHT EXPERIMENTS COMPLETE: $(date) ==="
echo ""
echo "Results:"
echo "  P0 log: ${LOGS}/p0_training_50k.log"
echo "  P1 output: ${BLON}/probe_results_image_large/layer_head_scan/"
echo "  P2 output: ${BLON}/probe_results_image_large/batch_level_diagnostic/"
echo "  Frozen diagnostics: ${BLON}/probe_results_image_large/diagnostics_patch8x8/"
