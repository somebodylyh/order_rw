#!/usr/bin/env bash
# Expensive per-head CDL order-scan ladder (stability-spec deliverable).
#
# Runs per_head_order_scan.scan_checkpoint over the clean_base_random_perm ladder
# (9 ckpt x 5 seed) -> batch_readout/logs/per_head_scan/ckpt{STEP}_seed{S}.json.
# These JSONs are the expensive ground-truth consumed by the quick-head-selector
# validation driver (validate_quick_selector.py, plan Task 6/7) and by the
# BR-1 §8 B1 head-selection contract.
#
# Idempotent: an existing non-empty JSON for (step, seed) is skipped, so the
# script can be killed and re-run to resume.
#
# Usage:
#   scripts/run_per_head_order_scan_ladder.sh [DEVICE] [M] [BATCH_SIZE] [SEEDS...]
# Defaults: DEVICE=cuda:0  M=100  BATCH_SIZE=32  SEEDS="0 1 2 3 4"
set -euo pipefail

REPO="/home/admin/lyuyuhuan/order_lyu"
PKG="$REPO/block_lo_arm_order_network"
CKPT_DIR="$PKG/probe_results/clean_base_random_perm"
OUT_DIR="$PKG/batch_readout/logs/per_head_scan"

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
echo "[ladder] DEVICE=$DEVICE M=$M BATCH_SIZE=$BATCH_SIZE SEEDS=${SEEDS[*]}"
echo "[ladder] $total jobs -> $OUT_DIR"

for step in "${STEPS[@]}"; do
  ckpt="$CKPT_DIR/ckpt_step${step}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[ladder] MISSING ckpt: $ckpt — skipping step $step"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    done=$(( done + 1 ))
    out="$OUT_DIR/ckpt${step}_seed${seed}.json"
    if [ -s "$out" ]; then
      echo "[ladder] ($done/$total) skip existing $out"
      continue
    fi
    echo "[ladder] ($done/$total) scan step=$step seed=$seed -> $out"
    python "$PKG/per_head_order_scan.py" \
      --ckpt "$ckpt" --M "$M" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --device "$DEVICE" --out "$out"
  done
done

echo "[ladder] DONE: $(ls "$OUT_DIR"/ckpt*_seed*.json 2>/dev/null | wc -l) JSON files in $OUT_DIR"
