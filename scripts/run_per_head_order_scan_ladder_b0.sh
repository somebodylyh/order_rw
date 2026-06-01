#!/usr/bin/env bash
# B0 (none->block0) per-head order-scan ladder — §3.0 gate for the
# auto-order-head spec. Same ladder as run_per_head_order_scan_ladder.sh but with
# --none-mode b0, writing to a SEPARATE dir so OLD dumps stay intact for OLD-vs-B0
# comparison. Idempotent: non-empty (step,seed) JSON is skipped.
#
# Usage: scripts/run_per_head_order_scan_ladder_b0.sh [DEVICE] [M] [BATCH_SIZE] [SEEDS...]
# Defaults: DEVICE=cuda:0 M=100 BATCH_SIZE=32 SEEDS="0 1 2 3 4"
set -euo pipefail

REPO="/home/admin/lyuyuhuan/order_lyu"
PKG="$REPO/block_lo_arm_order_network"
CKPT_DIR="$PKG/probe_results/clean_base_random_perm"
OUT_DIR="$PKG/batch_readout/logs/per_head_scan_b0"

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
done=0
echo "[b0-ladder] DEVICE=$DEVICE M=$M BATCH_SIZE=$BATCH_SIZE SEEDS=${SEEDS[*]}"
echo "[b0-ladder] $total jobs -> $OUT_DIR"

for step in "${STEPS[@]}"; do
  ckpt="$CKPT_DIR/ckpt_step${step}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[b0-ladder] MISSING ckpt: $ckpt — skipping step $step"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    done=$(( done + 1 ))
    out="$OUT_DIR/ckpt${step}_seed${seed}.json"
    if [ -s "$out" ]; then
      echo "[b0-ladder] ($done/$total) skip existing $out"
      continue
    fi
    echo "[b0-ladder] ($done/$total) scan step=$step seed=$seed -> $out"
    python "$PKG/per_head_order_scan.py" \
      --ckpt "$ckpt" --M "$M" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --device "$DEVICE" --none-mode b0 --out "$out"
  done
done

echo "[b0-ladder] DONE: $(ls "$OUT_DIR"/ckpt*_seed*.json 2>/dev/null | wc -l) JSON files in $OUT_DIR"
