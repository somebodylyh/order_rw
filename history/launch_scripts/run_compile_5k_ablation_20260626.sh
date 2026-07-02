#!/usr/bin/env bash
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="block_lo_arm_order_network/probe_results/compile_ablation_5k_${STAMP}"
TRAIN="block_lo_arm_order_network/train_clean_aogpt.py"
DEVICE="cuda:0"
SEED=123

COMMON_ARGS=(
  --run-kind baseline
  --data-source continuous
  --n-layer 4
  --n-head 8
  --n-embd 384
  --batch-size 64
  --grad-accum 2
  --lr 1e-3
  --min-lr 1e-4
  --warmup-iters 0
  --weight-decay 0.1
  --beta1 0.9
  --beta2 0.99
  --grad-clip 1.0
  --dropout 0.0
  --seed "$SEED"
  --permute-seed "$SEED"
  --vocab-size 50304
  --stream-eval-windows 2000
  --eval-batch-size 16
  --eval-interval 1000
  --log-interval 100
  --max-steps 5000
  --lr-decay-steps 50000
  --save-steps 0,5000
  --device "$DEVICE"
)

run_one() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo
  echo "============================================================"
  echo "[$(date)] START $name"
  echo "out=$out"
  echo "extra=$*"
  echo "============================================================"
  PYTHONPATH=block_lo_arm_order_network python "$TRAIN" \
    --output-dir "$out" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$out/train.log"
  echo "[$(date)] DONE $name"
}

mkdir -p "$ROOT"
{
  echo "root=$ROOT"
  echo "device=$DEVICE"
  echo "seed=$SEED"
  echo "common_args=${COMMON_ARGS[*]}"
} | tee "$ROOT/manifest.txt"

run_one compile_a
run_one compile_b
run_one nocompile_a --no-compile-model
run_one nocompile_b --no-compile-model

echo
echo "All 5k compile ablation runs finished: $ROOT"
