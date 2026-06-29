#!/bin/bash
# Serial overnight 2026-06-19 (v2):
#   gbeta20k → CDL20k → CDL40k → frozen_beta10k → frozen_beta20k → frozen_beta40k
# Single GPU (cuda:0), all use continuous baseline checkpoints
set -euo pipefail

CD=/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
SRC=$CD/probe_results/random_baseline_continuous_jun08_seed2
GPU=cuda:0
TS=20260619_2116

GB10=$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step10k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt
GB20=$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt
GB40=$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step40k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt

log() { echo "[$(date '+%H:%M:%S')] $*"; }
trap 'log "INTERRUPTED at $(date)"' INT TERM

log "=========================================="
log "Serial overnight: gbeta20k → CDL20k → CDL40k → frozen10k → frozen20k → frozen40k"
log "GPU=$GPU  Started $(date)"
log "=========================================="

# ═══════════════════════════════════════════════════════════════
# 1. g_β step20k — pretrain g_β on L0H2 @ step20000
# ═══════════════════════════════════════════════════════════════
log ""
log "[1/6] g_β step20k start"
rm -rf $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/scripts/run_phase33_gbeta.py \
  --layer 0 --head 2 --none-mode b1 --M 2000 --batch-size 16 --epochs 40 \
  --device $GPU \
  --ckpt-dir $SRC \
  --step 20000 \
  --cross-steps 10000 40000 \
  --out $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k \
  2>&1 | tee $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k_${TS}.log
log "[1/6] g_β step20k DONE at $(date)"

# ═══════════════════════════════════════════════════════════════
# 2. CDL teacher seed124 from20k L0H0 B1
# ═══════════════════════════════════════════════════════════════
log ""
log "[2/6] CDL seed124 from20k start"
CDL_FLAGS="--run-kind cdl_teacher --data-source continuous \
  --seed 124 --permute-seed 124 \
  --cdl-teacher-head 0 0 --cdl-teacher-none-mode b1 --cdl-teacher-tau 1.0 --cdl-teacher-refresh 10 \
  --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 --alpha-warmup-start 0 --alpha-ramp-from-resume \
  --max-steps 60000 \
  --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
  --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
  --device $GPU"

mkdir -p $CD/probe_results/cdl_teacher_seed124_from20k_l0h0_b1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/train_clean_aogpt.py \
  $CDL_FLAGS \
  --resume-ckpt $SRC/ckpt_step20000.pt \
  --save-steps "30000,40000,50000,60000" \
  --output-dir $CD/probe_results/cdl_teacher_seed124_from20k_l0h0_b1 \
  2>&1 | tee $CD/probe_results/cdl_teacher_seed124_from20k_l0h0_b1/train_log_${TS}.txt
log "[2/6] CDL from20k DONE at $(date)"

# ═══════════════════════════════════════════════════════════════
# 3. CDL teacher seed124 from40k L0H0 B1
# ═══════════════════════════════════════════════════════════════
log ""
log "[3/6] CDL seed124 from40k start"
mkdir -p $CD/probe_results/cdl_teacher_seed124_from40k_l0h0_b1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/train_clean_aogpt.py \
  $CDL_FLAGS \
  --resume-ckpt $SRC/ckpt_step40000.pt \
  --save-steps "50000,60000" \
  --output-dir $CD/probe_results/cdl_teacher_seed124_from40k_l0h0_b1 \
  2>&1 | tee $CD/probe_results/cdl_teacher_seed124_from40k_l0h0_b1/train_log_${TS}.txt
log "[3/6] CDL from40k DONE at $(date)"

# ═══════════════════════════════════════════════════════════════
# Common flags for frozen_beta (L0H2, seed=2)
# ═══════════════════════════════════════════════════════════════
FB_FLAGS="--run-kind frozen_beta --data-source continuous \
  --seed 2 --permute-seed 2 \
  --frozen-beta-head 0 2 --frozen-beta-none-mode b1 \
  --frozen-beta-refresh 10 --frozen-beta-mode argsort \
  --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 --alpha-warmup-start 0 --alpha-ramp-from-resume \
  --max-steps 60000 \
  --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
  --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
  --device $GPU"

# ═══════════════════════════════════════════════════════════════
# 4. frozen_beta from10k (g_β @ step10000 → AOGPT)
# ═══════════════════════════════════════════════════════════════
log ""
log "[4/6] frozen_beta from10k start"
mkdir -p $CD/probe_results/frozen_beta_b1_seed2_from10000_l0h2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/train_clean_aogpt.py \
  $FB_FLAGS \
  --resume-ckpt $SRC/ckpt_step10000.pt \
  --frozen-beta-ckpt $GB10 \
  --save-steps "20000,30000,40000,50000,60000" \
  --output-dir $CD/probe_results/frozen_beta_b1_seed2_from10000_l0h2 \
  2>&1 | tee $CD/probe_results/frozen_beta_b1_seed2_from10000_l0h2/train_log_${TS}.txt
log "[4/6] frozen_beta from10k DONE at $(date)"

# ═══════════════════════════════════════════════════════════════
# 5. frozen_beta from20k (g_β @ step20000 → AOGPT)
# ═══════════════════════════════════════════════════════════════
log ""
log "[5/6] frozen_beta from20k start"
mkdir -p $CD/probe_results/frozen_beta_b1_seed2_from20000_l0h2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/train_clean_aogpt.py \
  $FB_FLAGS \
  --resume-ckpt $SRC/ckpt_step20000.pt \
  --frozen-beta-ckpt $GB20 \
  --save-steps "30000,40000,50000,60000" \
  --output-dir $CD/probe_results/frozen_beta_b1_seed2_from20000_l0h2 \
  2>&1 | tee $CD/probe_results/frozen_beta_b1_seed2_from20000_l0h2/train_log_${TS}.txt
log "[5/6] frozen_beta from20k DONE at $(date)"

# ═══════════════════════════════════════════════════════════════
# 6. frozen_beta from40k (g_β @ step40000 → AOGPT)
# ═══════════════════════════════════════════════════════════════
log ""
log "[6/6] frozen_beta from40k start"
mkdir -p $CD/probe_results/frozen_beta_b1_seed2_from40000_l0h2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/train_clean_aogpt.py \
  $FB_FLAGS \
  --resume-ckpt $SRC/ckpt_step40000.pt \
  --frozen-beta-ckpt $GB40 \
  --save-steps "50000,60000" \
  --output-dir $CD/probe_results/frozen_beta_b1_seed2_from40000_l0h2 \
  2>&1 | tee $CD/probe_results/frozen_beta_b1_seed2_from40000_l0h2/train_log_${TS}.txt
log "[6/6] frozen_beta from40k DONE at $(date)"

log ""
log "=========================================="
log "ALL 6 JOBS DONE — $(date)"
log "=========================================="
