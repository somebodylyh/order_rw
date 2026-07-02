#!/usr/bin/env bash
# B1 g_β pretrain at 20k and 40k (seed2 random baseline).
# Builds B1 selected-head datasets, trains g_β on each, then launches frozen_beta.
set -euo pipefail

CD="$(dirname "$0")/../block_lo_arm_order_network"
CKPT_DIR="$CD/probe_results/random_baseline_continuous_jun08_seed2"
HEAD="0 2"   # L0H2 — strongest B1 head at both 20k and 40k (τ=1.0000)
M=2000
BATCH_SIZE=16
EPOCHS=40

run_gbeta() {
    local STEP=$1
    local GPU=$2
    local OUT="$CD/batch_readout/logs/gbeta_b1_seed2/step${STEP}"
    mkdir -p "$OUT"

    echo "=== B1 g_β @ step${STEP} (GPU $GPU) ==="
    python "$(dirname "$0")/run_phase33_gbeta.py" \
        --step $STEP --layer 0 --head 2 \
        --M $M --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --none-mode b1 --seed 42 --device "cuda:$GPU" --fwd-batch 8 \
        --ckpt-dir "$CKPT_DIR" \
        --cross-steps 10000 30000 50000 \
        --out "$CD/batch_readout/logs/gbeta_b1_seed2"
    echo "Done: step${STEP}  g_β -> $OUT/g_beta_best.pt"
}

GPU0="${1:-0}"
GPU1="${2:-1}"

echo "g_β B1 @ 20k → GPU $GPU0"
echo "g_β B1 @ 40k → GPU $GPU1"

# Launch both in background
run_gbeta 20000 $GPU0 > /tmp/gbeta_b1_20k.log 2>&1 &
PID20=$!
run_gbeta 40000 $GPU1 > /tmp/gbeta_b1_40k.log 2>&1 &
PID40=$!

echo "20k PID=$PID20  40k PID=$PID40"
echo "tail -f /tmp/gbeta_b1_20k.log"
echo "tail -f /tmp/gbeta_b1_40k.log"
wait
echo "=== Both g_β done ==="

# ── Launch frozen_beta from 20k → 60k ──
echo ""
echo "=== Launching frozen_beta from 20k → 60k (GPU $GPU0) ==="
python "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta \
    --resume-ckpt "$CKPT_DIR/ckpt_step20000.pt" \
    --frozen-beta-ckpt "$CD/batch_readout/logs/gbeta_b1_seed2/step20000/g_beta_best.pt" \
    --frozen-beta-head $HEAD \
    --frozen-beta-none-mode b1 \
    --frozen-beta-refresh 10 \
    --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
    --alpha-ramp-from-resume \
    --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --output-dir "$CKPT_DIR/../frozen_beta_b1_seed2_from20k" \
    --max-steps 60000 \
    --save-steps 10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000 \
    --device "cuda:$GPU0" \
    --seed 42 --permute-seed 42 \
    --batch-size 64 --grad-accum 2 \
    --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
    --weight-decay 0.1 --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --n-layer 4 --n-head 8 --n-embd 384 --dropout 0.0 \
    --vocab-size 50304 \
    --stream-eval-windows 2000 \
    --eval-interval 500 --log-interval 50 \
    > /tmp/frozen_beta_b1_from20k.log 2>&1 &
PIDFB20=$!

echo "frozen_beta from20k PID=$PIDFB20"
echo "tail -f /tmp/frozen_beta_b1_from20k.log"

# ── Launch frozen_beta from 40k → 60k ──
echo ""
echo "=== Launching frozen_beta from 40k → 60k (GPU $GPU1) ==="
python "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta \
    --resume-ckpt "$CKPT_DIR/ckpt_step40000.pt" \
    --frozen-beta-ckpt "$CD/batch_readout/logs/gbeta_b1_seed2/step40000/g_beta_best.pt" \
    --frozen-beta-head $HEAD \
    --frozen-beta-none-mode b1 \
    --frozen-beta-refresh 10 \
    --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
    --alpha-ramp-from-resume \
    --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --output-dir "$CKPT_DIR/../frozen_beta_b1_seed2_from40k" \
    --max-steps 60000 \
    --save-steps 10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000 \
    --device "cuda:$GPU1" \
    --seed 42 --permute-seed 42 \
    --batch-size 64 --grad-accum 2 \
    --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
    --weight-decay 0.1 --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --n-layer 4 --n-head 8 --n-embd 384 --dropout 0.0 \
    --vocab-size 50304 \
    --stream-eval-windows 2000 \
    --eval-interval 500 --log-interval 50 \
    > /tmp/frozen_beta_b1_from40k.log 2>&1 &
PIDFB40=$!

echo "frozen_beta from40k PID=$PIDFB40"
echo "tail -f /tmp/frozen_beta_b1_from40k.log"

wait $PIDFB20 $PIDFB40
echo "=== All done ==="
