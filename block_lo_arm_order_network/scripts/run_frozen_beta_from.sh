#!/usr/bin/env bash
# Launch ONE frozen-g_β hook continuation run: resume clean_base from START step,
# train to 60k under run-kind=frozen_beta (g_β-driven order, frozen, no CDL).
# LR anchored at lr-decay-steps=50000 (matches the original clean_base extension to
# 60k: cosine to 50k, then min_lr) so resuming from any start step shares the same
# LR trajectory. Separate output dir per start step; never writes clean_base.
#
# Usage: run_frozen_beta_from.sh START_STEP GPU   (e.g. 5000 cuda:1)
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
START="${1:?need START step}"
GPU="${2:?need GPU e.g. cuda:0}"
REFRESH="${3:-25}"

CKPT="$PKG/probe_results/clean_base_random_perm/ckpt_step${START}.pt"
GBETA="$PKG/batch_readout/logs/phase33_gbeta/full/g_beta_best.pt"
OUT="$PKG/probe_results/frozen_beta_from${START}"
mkdir -p "$OUT"

[ -f "$CKPT" ] || { echo "MISSING ckpt: $CKPT"; exit 1; }
[ -f "$GBETA" ] || { echo "MISSING g_beta: $GBETA"; exit 1; }

echo "[frozen_beta] start=$START gpu=$GPU refresh=$REFRESH -> $OUT"
cd "$PKG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exec python -u train_clean_aogpt.py \
  --run-kind frozen_beta \
  --resume-ckpt "$CKPT" \
  --max-steps 60000 --lr-decay-steps 50000 \
  --frozen-beta-ckpt "$GBETA" \
  --frozen-beta-head 0 0 --frozen-beta-mode argsort --frozen-beta-refresh "$REFRESH" \
  --batch-size 32 --grad-accum 4 \
  --output-dir "$OUT" \
  --eval-interval 1000 --log-interval 50 --save-steps "60000" \
  --device "$GPU"
