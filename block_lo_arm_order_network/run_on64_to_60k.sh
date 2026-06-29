#!/bin/bash
# Auto-pipeline: wait for ON64 training → train AO-GPT 50k→60k with ON sampling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ON_OUT_DIR="$SCRIPT_DIR/probe_results/on64_from_50k_nn10k"
ON_CKPT="$ON_OUT_DIR/grpo_on64_from_50k_nn10k.bc.pt"
A64_DATA="$SCRIPT_DIR/probe_results/A64_from_N64_50k_10k.npy"
AO_GPT_CKPT=~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt
TOKENS_PATH="$SCRIPT_DIR/probe_results/A32_from_N64_10k.tokens.npy"

echo "=== Step 0: Waiting for ON64 training to finish ==="
echo "ON64 pid: ${1:-<detect>}"

if [ -n "${1:-}" ]; then
    while kill -0 "$1" 2>/dev/null; do
        sleep 5
    done
    echo "ON64 training process $1 exited."
else
    while [ ! -f "$ON_CKPT" ]; do
        sleep 5
    done
    echo "ON64 checkpoint found: $ON_CKPT"
fi

sleep 2

echo ""
echo "=== Step 1: Training AO-GPT 50k → 60k with ON sampling ==="
echo "  alpha warmup: 0→0.9 over 5000 steps"
echo "  temperature: 1.0 (multinomial)"
echo "  max-iters: 7000 (5k ramp + 2k stable)"
cd "$SCRIPT_DIR"
python3 -u train_on_mixed.py \
    --ao-gpt-ckpt "$AO_GPT_CKPT" \
    --tokens-path "$TOKENS_PATH" \
    --a32-path "$A64_DATA" \
    --num-blocks 64 \
    --order-source on \
    --on-ckpt "$ON_CKPT" \
    --on-temperature 1.0 \
    --alpha 0.9 \
    --alpha-warmup 5000 \
    --max-iters 7000 \
    --batch-size 8 \
    --lr 3e-5 \
    --eval-interval 250 \
    --eval-iters 50 \
    --out-dir "$ON_OUT_DIR" \
    --device cuda:0

echo ""
echo "=== Done ==="
echo "Output: $ON_OUT_DIR"
