#!/bin/bash
# Overnight 2026-06-18: CDL from20k/40k + frozen_beta from10k/20k/40k
# L0H2, b1, continuous stream, seed2, source=random_baseline_continuous_jun08_seed2
set -euo pipefail

CD="$(dirname "$0")/.."
SRC="$CD/probe_results/random_baseline_continuous_jun08_seed2"
GB10="$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step10k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"
GB20="$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"
GB40="$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step40k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"

COMMON="--data-source continuous --seed 2 --cdl-teacher-head 0 2 --cdl-teacher-none-mode b1 \
  --cdl-teacher-tau 1.0 --cdl-teacher-refresh 10 --max-steps 60000 --device"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ========== GPU0: CDL from20k → CDL from40k → frozen_beta from40k ==========
(
  log "GPU0: CDL from20k (20k→60k)"
  python -u train_clean_aogpt.py \
    --run-kind cdl_teacher \
    --resume-ckpt "$SRC/ckpt_step20000.pt" \
    --output-dir "$CD/probe_results/cdl_teacher_seed2_from20k_l0h2_b1" \
    $COMMON cuda:0 \
    2>&1 | tee "$CD/probe_results/cdl_teacher_seed2_from20k_l0h2_b1.log"

  log "GPU0: CDL from40k (40k→60k)"
  python -u train_clean_aogpt.py \
    --run-kind cdl_teacher \
    --resume-ckpt "$SRC/ckpt_step40000.pt" \
    --output-dir "$CD/probe_results/cdl_teacher_seed2_from40k_l0h2_b1" \
    $COMMON cuda:0 \
    2>&1 | tee "$CD/probe_results/cdl_teacher_seed2_from40k_l0h2_b1.log"

  log "GPU0: frozen_beta from40k (40k→60k)"
  python -u train_clean_aogpt.py \
    --run-kind frozen_beta \
    --resume-ckpt "$SRC/ckpt_step40000.pt" \
    --output-dir "$CD/probe_results/frozen_beta_seed2_from40k_l0h2_b1" \
    --frozen-beta-ckpt "$GB40" --frozen-beta-head 0 2 --frozen-beta-none-mode b1 \
    --frozen-beta-refresh 10 --frozen-beta-mode argsort \
    $COMMON cuda:0 \
    2>&1 | tee "$CD/probe_results/frozen_beta_seed2_from40k_l0h2_b1.log"

  log "GPU0: ALL DONE"
) &

# ========== GPU1: frozen_beta from10k → frozen_beta from20k ==========
(
  log "GPU1: frozen_beta from10k (10k→60k)"
  python -u train_clean_aogpt.py \
    --run-kind frozen_beta \
    --resume-ckpt "$SRC/ckpt_step10000.pt" \
    --output-dir "$CD/probe_results/frozen_beta_seed2_from10k_l0h2_b1" \
    --frozen-beta-ckpt "$GB10" --frozen-beta-head 0 2 --frozen-beta-none-mode b1 \
    --frozen-beta-refresh 10 --frozen-beta-mode argsort \
    $COMMON cuda:1 \
    2>&1 | tee "$CD/probe_results/frozen_beta_seed2_from10k_l0h2_b1.log"

  log "GPU1: frozen_beta from20k (20k→60k)"
  python -u train_clean_aogpt.py \
    --run-kind frozen_beta \
    --resume-ckpt "$SRC/ckpt_step20000.pt" \
    --output-dir "$CD/probe_results/frozen_beta_seed2_from20k_l0h2_b1" \
    --frozen-beta-ckpt "$GB20" --frozen-beta-head 0 2 --frozen-beta-none-mode b1 \
    --frozen-beta-refresh 10 --frozen-beta-mode argsort \
    $COMMON cuda:1 \
    2>&1 | tee "$CD/probe_results/frozen_beta_seed2_from20k_l0h2_b1.log"

  log "GPU1: ALL DONE"
) &

wait
log "BOTH GPUS DONE"
