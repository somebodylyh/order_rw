#!/usr/bin/env bash
set -euo pipefail

# Step 0: Train ON_round3 on A_after and compare with ON_round2.
# Quick diagnostic (~15-20 min) before committing to Round 1 co-train.
#
# Usage:
#   bash run_step0_on_round3.sh
#   DEVICE=cuda:1 EPOCHS=20 bash run_step0_on_round3.sh

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-30}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p probe_results/round0

echo "================================================================"
echo "Step 0a: Train ON_round3 (${EPOCHS} epochs)"
echo "  A = A_after_2k.npy (from updated AO-GPT)"
echo "  warm start = grpo_on_round2.pt"
echo "  AO-GPT = aogpt_round0.pt (updated)"
echo "  Device = ${DEVICE}"
echo "================================================================"
date

python -u train_grpo_on.py \
  --a-matrices probe_results/round0/A_after_2k.npy \
  --on-ckpt probe_results/grpo_on_round2.pt \
  --ao-gpt-ckpt probe_results/round0/aogpt_round0.pt \
  --output probe_results/round0/grpo_on_round3.pt \
  --data-source train \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --skip-tests \
  2>&1 | tee probe_results/round0/step0_train.log

echo ""
echo "================================================================"
echo "Step 0b: Compare ON_round2 vs ON_round3"
echo "================================================================"
date

python -u compare_on_versions.py \
  --on2 probe_results/grpo_on_round2.pt \
  --on3 probe_results/round0/grpo_on_round3.pt \
  --a-before probe_results/round0/A_before_2k.npy \
  --a-after probe_results/round0/A_after_2k.npy \
  --device "${DEVICE}" \
  2>&1 | tee probe_results/round0/step0_compare.log

echo ""
echo "================================================================"
echo "Step 0 done"
echo "================================================================"
date
echo ""
echo "Review:"
echo "  cat probe_results/round0/step0_compare.log  # decision at the bottom"
echo "  grep 'best_val_reward' probe_results/round0/step0_train.log"
