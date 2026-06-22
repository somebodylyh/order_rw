#!/usr/bin/env bash
# Smoke: train all three variants for 10 epochs on M=20 headset.
# Verifies: no NaN, tau improves from random, scripts/ pipeline works.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:-/tmp/headset_smoke_train.npz}"
OUT_BASE="${2:-/tmp/head_gated_smoke}"

echo "=== Smoke: single_head ==="
python3 "$ROOT/scripts/train_head_gated_gbeta.py" \
  --dataset "$DATASET" \
  --variant single_head --head-idx 2 \
  --epochs 10 --batch-size 8 \
  --out-dir "$OUT_BASE/single_head" \
  --device cpu

echo "=== Smoke: mean_head ==="
python3 "$ROOT/scripts/train_head_gated_gbeta.py" \
  --dataset "$DATASET" \
  --variant mean_head \
  --epochs 10 --batch-size 8 \
  --out-dir "$OUT_BASE/mean_head" \
  --device cpu

echo "=== Smoke: head_gated soft_all ==="
python3 "$ROOT/scripts/train_head_gated_gbeta.py" \
  --dataset "$DATASET" \
  --variant head_gated --gate-mode soft_all \
  --epochs 10 --batch-size 8 \
  --out-dir "$OUT_BASE/head_gated_soft_all" \
  --device cpu

echo "=== Smoke: head_gated topk=2 ==="
python3 "$ROOT/scripts/train_head_gated_gbeta.py" \
  --dataset "$DATASET" \
  --variant head_gated --gate-mode topk --topk 2 \
  --epochs 10 --batch-size 8 \
  --out-dir "$OUT_BASE/head_gated_topk2" \
  --device cpu

echo ""
echo "All smoke runs complete."
echo "Outputs in $OUT_BASE"
