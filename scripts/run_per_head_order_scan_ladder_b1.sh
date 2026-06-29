#!/usr/bin/env bash
# B1 predictor-aligned per-head order-scan ladder.
#
# Same ladder as run_per_head_order_scan_ladder_b0.sh, but with
# --none-mode predictor. This keeps AO-GPT's original predictor frame
# (attn[:-1, :-1]) and writes to a separate dir for B0-vs-B1 comparison.
#
# Usage: scripts/run_per_head_order_scan_ladder_b1.sh [DEVICE] [M] [BATCH_SIZE] [SEEDS...]
# Defaults: DEVICE=cuda:0 M=100 BATCH_SIZE=32 SEEDS="0 1 2 3 4"
set -euo pipefail

REPO="/home/admin/lyuyuhuan/order_lyu"
PKG="$REPO/block_lo_arm_order_network"
CKPT_DIR="$PKG/probe_results/clean_base_random_perm"
OUT_DIR="$PKG/batch_readout/logs/per_head_scan_b1"

DEVICE="${1:-cuda:0}"
M="${2:-100}"
BATCH_SIZE="${3:-32}"
shift $(( $# > 3 ? 3 : $# )) || true
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3 4)

STEPS=(0 1000 5000 10000 20000 30000 40000 50000 60000)

mkdir -p "$OUT_DIR"
cd "$REPO"

total=$(( ${#STEPS[@]} * ${#SEEDS[@]} ))
done_count=0
echo "[b1-ladder] DEVICE=$DEVICE M=$M BATCH_SIZE=$BATCH_SIZE SEEDS=${SEEDS[*]}"
echo "[b1-ladder] $total jobs -> $OUT_DIR"

for step in "${STEPS[@]}"; do
  ckpt="$CKPT_DIR/ckpt_step${step}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[b1-ladder] MISSING ckpt: $ckpt - skipping step $step"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    done_count=$(( done_count + 1 ))
    out="$OUT_DIR/ckpt${step}_seed${seed}.json"
    if [ -s "$out" ]; then
      echo "[b1-ladder] ($done_count/$total) skip existing $out"
      continue
    fi
    echo "[b1-ladder] ($done_count/$total) scan step=$step seed=$seed -> $out"
    python "$PKG/per_head_order_scan.py" \
      --ckpt "$ckpt" --M "$M" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --device "$DEVICE" --none-mode predictor --out "$out"
  done
done

echo "[b1-ladder] DONE: $(ls "$OUT_DIR"/ckpt*_seed*.json 2>/dev/null | wc -l) JSON files in $OUT_DIR"
