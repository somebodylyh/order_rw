#!/bin/bash
# Serial overnight 2026-06-19: gbeta20k → CDL20k → CDL40k → gbeta10k → gbeta20k → gbeta40k
# Single GPU (cuda:0), all use continuous baseline checkpoints
set -euo pipefail

CD=/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
SRC=probe_results/random_baseline_continuous_jun08_seed2
GPU=cuda:0
TS=20260619_2115

log() { echo "[$(date '+%H:%M:%S')] $*"; }
trap 'log "INTERRUPTED at $(date)"' INT TERM

log "=========================================="
log "Serial overnight: gbeta20k → CDL20k → CDL40k → gbeta10k → gbeta20k → gbeta40k"
log "GPU=$GPU  Started $(date)"
log "=========================================="

# ═══ 1. g_β step20k ═══
log ""
log "[1/6] g_β step20k start"
rm -rf $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/scripts/run_phase33_gbeta.py \
  --layer 0 --head 2 --none-mode b1 --M 2000 --batch-size 16 --epochs 40 \
  --device $GPU \
  --ckpt-dir $CD/$SRC \
  --step 20000 \
  --cross-steps 10000 40000 \
  --out $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k \
  2>&1 | tee $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k_${TS}_rerun.log
log "[1/6] g_β step20k DONE at $(date)"

# ═══ 2. CDL teacher from20k ═══
log ""
log "[2/6] CDL seed124 from20k start"
COMMON_CDL="--run-kind cdl_teacher --data-source continuous \
  --seed 124 --permute-seed 124 \
  --cdl-teacher-head 0 0 --cdl-teacher-none-mode b1 --cdl-teacher-tau 1.0 --cdl-teacher-refresh 10 \
  --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 --alpha-warmup-start 0 --alpha-ramp-from-resume \
  --max-steps 60000 \
  --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
  --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
  --device $GPU"

mkdir -p $CD/probe_results/cdl_teacher_seed124_from20k_l0h0_b1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/train_clean_aogpt.py \
  $COMMON_CDL \
  --resume-ckpt $CD/$SRC/ckpt_step20000.pt \
  --save-steps "30000,40000,50000,60000" \
  --output-dir $CD/probe_results/cdl_teacher_seed124_from20k_l0h0_b1 \
  2>&1 | tee $CD/probe_results/cdl_teacher_seed124_from20k_l0h0_b1/train_log_${TS}.txt
log "[2/6] CDL from20k DONE at $(date)"

# ═══ 3. CDL teacher from40k ═══
log ""
log "[3/6] CDL seed124 from40k start"
mkdir -p $CD/probe_results/cdl_teacher_seed124_from40k_l0h0_b1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/train_clean_aogpt.py \
  $COMMON_CDL \
  --resume-ckpt $CD/$SRC/ckpt_step40000.pt \
  --save-steps "50000,60000" \
  --output-dir $CD/probe_results/cdl_teacher_seed124_from40k_l0h0_b1 \
  2>&1 | tee $CD/probe_results/cdl_teacher_seed124_from40k_l0h0_b1/train_log_${TS}.txt
log "[3/6] CDL from40k DONE at $(date)"

# ═══ 4. g_β step10k ═══
log ""
log "[4/6] g_β step10k start"
rm -rf $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step10k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/scripts/run_phase33_gbeta.py \
  --layer 0 --head 2 --none-mode b1 --M 2000 --batch-size 16 --epochs 40 \
  --device $GPU \
  --ckpt-dir $CD/$SRC \
  --step 10000 \
  --cross-steps 20000 40000 \
  --out $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step10k \
  2>&1 | tee $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step10k_${TS}_rerun.log
log "[4/6] g_β step10k DONE at $(date)"

# ═══ 5. g_β step20k ═══
log ""
log "[5/6] g_β step20k (rerun in set) start"
rm -rf $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/scripts/run_phase33_gbeta.py \
  --layer 0 --head 2 --none-mode b1 --M 2000 --batch-size 16 --epochs 40 \
  --device $GPU \
  --ckpt-dir $CD/$SRC \
  --step 20000 \
  --cross-steps 10000 40000 \
  --out $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k \
  2>&1 | tee $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step20k_${TS}_rerun.log
log "[5/6] g_β step20k DONE at $(date)"

# ═══ 6. g_β step40k ═══
log ""
log "[6/6] g_β step40k start"
rm -rf $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step40k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u $CD/scripts/run_phase33_gbeta.py \
  --layer 0 --head 2 --none-mode b1 --M 2000 --batch-size 16 --epochs 40 \
  --device $GPU \
  --ckpt-dir $CD/$SRC \
  --step 40000 \
  --cross-steps 10000 20000 \
  --out $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step40k \
  2>&1 | tee $CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step40k_${TS}_rerun.log
log "[6/6] g_β step40k DONE at $(date)"

log ""
log "=========================================="
log "ALL 6 JOBS DONE — $(date)"
log "=========================================="
