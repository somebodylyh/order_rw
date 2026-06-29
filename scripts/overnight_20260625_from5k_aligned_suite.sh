#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Overnight 2026-06-25: from5k aligned comparison suite (GPU 0 only)
#
# Runs:
#   1. random_baseline         (from0, seed=123, 0→80k)
#   2. g_beta ListMLE training (from random step5k teacher labels)
#   3. ori_l2r_from5k          (from random step5k ckpt)
#   4. cdl_from5k              (from random step5k ckpt)
#   5. gbeta_listmle_from5k    (from random step5k ckpt)
#   6. ori_l2r_baseline        (from0, seed=123, 0→80k)
#
# All runs use --attn-trajectory for L0 attention-map evolution tracking.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN="block_lo_arm_order_network/train_clean_aogpt.py"
BUILD_DS="analyses/build_l0_dynamic_gbeta_dataset.py"
TRAIN_GBETA="block_lo_arm_order_network/batch_readout/train_l0_dynamic_gbeta.py"
PROBE_DIR="block_lo_arm_order_network/probe_results"
GBETA_CKPT_DIR="block_lo_arm_order_network/batch_readout/checkpoints"

# ── Shared config ──────────────────────────────────────────────────────────
DEVICE="cuda:0"
SEED=123
COMMON_BASE="--data-source continuous --n-layer 4 --n-head 8 --n-embd 384 \
  --batch-size 64 --grad-accum 2 \
  --lr 1e-3 --min-lr 1e-4 \
  --warmup-iters 0 --weight-decay 0.1 --beta1 0.9 --beta2 0.99 \
  --grad-clip 1.0 --dropout 0.0 \
  --seed $SEED --permute-seed $SEED \
  --vocab-size 50304 \
  --stream-eval-windows 2000 --eval-batch-size 16 \
  --eval-interval 1000 --log-interval 100 \
  --device $DEVICE \
  --wandb-log --wandb-project order-lyu"

# Attention trajectory: L0 all-head B maps at eval time
ATTN_TRAJ="--attn-trajectory --attn-trajectory-samples 8 --attn-trajectory-heatmap-interval 5000"

# Save steps for 60k runs
SAVE_STEPS_60K="0,5000,10000,20000,30000,40000,50000,60000"
# For from5k: same effective steps
SAVE_STEPS_FROM5K="5000,10000,20000,30000,40000,50000,60000"

# ── Output dirs ────────────────────────────────────────────────────────────
RANDOM_BL_DIR="$PROBE_DIR/overnight_20260625_random_baseline"
ORI_L2R_BL_DIR="$PROBE_DIR/overnight_20260625_ori_l2r_baseline"
ORI_L2R_F5K_DIR="$PROBE_DIR/overnight_20260625_ori_l2r_from5k"
CDL_F5K_DIR="$PROBE_DIR/overnight_20260625_cdl_from5k"
GBETA_F5K_DIR="$PROBE_DIR/overnight_20260625_gbeta_listmle_from5k"

RANDOM_5K_CKPT="$RANDOM_BL_DIR/ckpt_step5000.pt"
GBETA_DS_PATH="$GBETA_CKPT_DIR/l0_dynamic_gbeta_ds_step5k_M2000_s123.npz"
GBETA_OUT_DIR="$GBETA_CKPT_DIR/l0_dynamic_gbeta_step5k_M2000_listmle_s123"
GBETA_BEST="$GBETA_OUT_DIR/g_beta_best.pt"

# ── Logging ────────────────────────────────────────────────────────────────
LOGDIR="$REPO/scripts/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/overnight_20260625_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "════════════════════════════════════════════════════════════════════"
echo "  Overnight 2026-06-25: from5k aligned comparison suite"
echo "  Started: $(date)"
echo "  Device:  $DEVICE"
echo "  Seed:    $SEED"
echo "  Log:     $LOGFILE"
echo "════════════════════════════════════════════════════════════════════"

# ── Pre-flight: GPU check ─────────────────────────────────────────────────
echo ""
echo "[PRE-FLIGHT] GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

GPU0_BUSY=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | grep "00000000:01:00.0" || true)
if [ -n "$GPU0_BUSY" ]; then
    echo "[PRE-FLIGHT] GPU 0 is BUSY. Exiting (will not preempt)."
    echo "  Active processes on GPU 0:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -i "00000000:01:00.0" || true
    exit 1
fi
echo "[PRE-FLIGHT] GPU 0 is free. Proceeding."

# ── Verify Python env ──────────────────────────────────────────────────────
echo ""
echo "[PRE-FLIGHT] Python environment:"
python -c "
import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}')
import numpy as np; print(f'  numpy={np.__version__}')
try:
    import matplotlib; print(f'  matplotlib={matplotlib.__version__}')
except ImportError:
    print('  matplotlib=MISSING (heatmaps disabled)')
try:
    import wandb; print(f'  wandb={wandb.__version__}')
except ImportError:
    print('  wandb=MISSING (W&B logging disabled)')
print(f'  PYTHONPATH=block_lo_arm_order_network')
"

# ═══════════════════════════════════════════════════════════════════════════
# Helper: run training and wait
# ═══════════════════════════════════════════════════════════════════════════
run_training() {
    local NAME="$1"
    local EXTRA_ARGS="$2"
    local OUT="$3"
    local WANDB_NAME="$4"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "[$(date)] START: $NAME"
    echo "  Output:    $OUT"
    echo "  W&B name:  $WANDB_NAME"
    echo "════════════════════════════════════════════════════════════════"

    mkdir -p "$OUT"

    PYTHONPATH=block_lo_arm_order_network python "$TRAIN" \
        --output-dir "$OUT" \
        --wandb-run-name "$WANDB_NAME" \
        $COMMON_BASE \
        $ATTN_TRAJ \
        $EXTRA_ARGS \
        2>&1 | tee "${OUT}/train.log"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] ❌ FAILED: $NAME (exit=$EXIT_CODE)"
        echo "  Log: ${OUT}/train.log"
        return $EXIT_CODE
    fi
    echo "[$(date)] ✅ DONE: $NAME"
}

# ═══════════════════════════════════════════════════════════════════════════
# Pre-launch audit table
# ═══════════════════════════════════════════════════════════════════════════
print_audit_table() {
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
    echo "│                              PRE-LAUNCH AUDIT TABLE                                              │"
    echo "├────────────────────┬──────────────┬────────────┬──────────────────────┬────────────┬─────────────┤"
    echo "│ Run                │ Start ckpt   │ Start step │ Policy               │ Max steps  │ LR decay    │"
    echo "├────────────────────┼──────────────┼────────────┼──────────────────────┼────────────┼─────────────┤"
    echo "│ random_baseline    │ none         │ 0          │ random               │ 60000      │ 60000       │"
    echo "│ ori_l2r_baseline   │ none         │ 0          │ ori_l2r              │ 60000      │ 60000       │"
    echo "│ ori_l2r_from5k     │ random 5k    │ 5000       │ ori_l2r              │ 60000      │ 55000       │"
    echo "│ cdl_from5k         │ random 5k    │ 5000       │ full seq CDL         │ 60000      │ 55000       │"
    echo "│ gbeta_listmle_f5k  │ random 5k    │ 5000       │ frozen g_beta+ListMLE│ 60000      │ 55000       │"
    echo "└────────────────────┴──────────────┴────────────┴──────────────────────┴────────────┴─────────────┘"
    echo ""
    echo "  Device:       $DEVICE (ONLY)"
    echo "  Seed:         123"
    echo "  Model:        4L/8H/d=384 (47M)"
    echo "  Dataset:      WikiText-103 continuous stream"
    echo "  Seq/Block:    256 / 4 → 64 blocks"
    echo "  Eval indices: fixed, from permute_seed=123"
    echo "  Attn traj:    L0 all-head strict65, 8 fixed samples, heatmap every 5k"
    echo "  W&B project:  order-lyu"
    echo ""
    echo "  CDL teacher:  full_sequential_cdl, C-D+L, L0 all-head, model-frame strict65, batch-mean probes"
    echo "  g_beta loss:  ListMLE (Plackett-Luce), L0 all-head dynamic g_beta"
    echo ""
    echo "  Step 5k ckpt: $RANDOM_5K_CKPT"
    echo "  g_beta model: $GBETA_BEST"
    echo ""
}

print_audit_table

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: random_baseline (from0, 0→80k) — produces step5k checkpoint
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1: random_baseline (from0, seed=$SEED, 0→80k)"
echo "  Purpose: produce random step5k checkpoint + random baseline trajectory"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_training \
    "random_baseline" \
    "--run-kind baseline --max-steps 60000 --lr-decay-steps 60000 --save-steps $SAVE_STEPS_60K" \
    "$RANDOM_BL_DIR" \
    "overnight_20260625_random_baseline"

if [ ! -f "$RANDOM_5K_CKPT" ]; then
    echo ""
    echo "❌ FATAL: random step5k checkpoint not found at $RANDOM_5K_CKPT"
    echo "   random_baseline finished but did not save step5000 checkpoint."
    echo "   Check save_steps=$SAVE_STEPS_80K"
    exit 1
fi
echo "✅ Random step5k checkpoint: $RANDOM_5K_CKPT"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: g_beta ListMLE training from random step5k
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2: g_beta ListMLE training from random step5k"
echo "  Dataset:  $GBETA_DS_PATH"
echo "  Output:   $GBETA_OUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 2a: Build L0 all-head dataset
if [ -f "$GBETA_DS_PATH" ]; then
    echo "[BUILD] Dataset already exists: $GBETA_DS_PATH"
else
    echo "[BUILD] Building L0 dynamic g_beta dataset from random step5k..."
    echo "  ckpt=$RANDOM_5K_CKPT  M=2000  batch_mean_size=16  seed=$SEED"

    PYTHONPATH=block_lo_arm_order_network python "$BUILD_DS" \
        --ckpt "$RANDOM_5K_CKPT" \
        --M 2000 --batch-mean-size 16 \
        --seed $SEED --device "$DEVICE" \
        --split train --forward-batch 8 \
        --destroy-replicas 1 \
        --teacher-temperature 1.0 --teacher-smoothing 0.05 \
        --out "$GBETA_DS_PATH"

    if [ ! -f "$GBETA_DS_PATH" ]; then
        echo "❌ FATAL: Dataset build failed — no output at $GBETA_DS_PATH"
        exit 1
    fi
    echo "[BUILD] Dataset saved: $GBETA_DS_PATH"
fi

# Step 2b: Train g_beta with ListMLE loss
echo ""
echo "[TRAIN] Training L0 dynamic g_beta with ListMLE loss..."
echo "  loss_type=listmle  epochs=40  batch_size=32  lr=3e-4"

mkdir -p "$GBETA_OUT_DIR"
PYTHONPATH=block_lo_arm_order_network python "$TRAIN_GBETA" \
    --dataset "$GBETA_DS_PATH" \
    --out-dir "$GBETA_OUT_DIR" \
    --loss-type listmle \
    --epochs 40 --batch-size 32 \
    --lr 3e-4 --weight-decay 1e-2 \
    --lambda-aux 0.05 --lambda-ent 0.001 \
    --min-gate-entropy 1.5 \
    --seed $SEED --device "$DEVICE" --heads 8 \
    2>&1 | tee "${GBETA_OUT_DIR}/train.log"

GBETA_EXIT=${PIPESTATUS[0]}
if [ $GBETA_EXIT -ne 0 ]; then
    echo "❌ FATAL: g_beta training failed (exit=$GBETA_EXIT)"
    exit 1
fi

# Step 2c: Validate g_beta before launching continuation
echo ""
echo "[VALIDATE] Checking g_beta ListMLE on held-out splits..."
if [ -f "$GBETA_BEST" ]; then
    echo "  Best model: $GBETA_BEST"
    # Extract validation metrics from the training output
    if [ -f "${GBETA_OUT_DIR}/evaluation/summary.json" ]; then
        echo "  Evaluation summary:"
        cat "${GBETA_OUT_DIR}/evaluation/summary.json" | python -m json.tool 2>/dev/null || true
    fi
else
    echo "❌ FATAL: g_beta best model not found at $GBETA_BEST"
    echo "  Available files in $GBETA_OUT_DIR:"
    ls -la "$GBETA_OUT_DIR/" || true
    exit 1
fi
echo "✅ g_beta ListMLE training complete."
echo "  g_beta checkpoint: $GBETA_BEST"

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: from5k continuation runs + from0 baselines
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3: from5k interventions + from0 baselines"
echo "  Random 5k ckpt: $RANDOM_5K_CKPT"
echo "  g_beta ckpt:    $GBETA_BEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Common continuation flags for from5k runs (5k→60k, lr decay over 55k steps)
FROM5K_COMMON="--resume-ckpt $RANDOM_5K_CKPT --max-steps 60000 --lr-decay-steps 55000 --save-steps $SAVE_STEPS_FROM5K"

# 3a. ori_l2r_from5k
run_training \
    "ori_l2r_from5k" \
    "--run-kind l2r $FROM5K_COMMON" \
    "$ORI_L2R_F5K_DIR" \
    "overnight_20260625_ori_l2r_from5k"

# 3b. cdl_from5k (full sequential CDL)
# CDL teacher config: L0H7, C-D+L, b1 none-mode, refresh every 10 steps
run_training \
    "cdl_from5k" \
    "--run-kind cdl_teacher $FROM5K_COMMON \
     --cdl-teacher-head 0 7 --cdl-teacher-tau 1.0 --cdl-teacher-refresh 10 \
     --cdl-teacher-none-mode b1 \
     --alpha-start 0.0 --alpha-target 0.9 --alpha-warmup-steps 10000" \
    "$CDL_F5K_DIR" \
    "overnight_20260625_cdl_from5k"

# 3c. gbeta_listmle_from5k (frozen g_beta with ListMLE-trained model)
run_training \
    "gbeta_listmle_from5k" \
    "--run-kind frozen_beta $FROM5K_COMMON \
     --frozen-beta-ckpt $GBETA_BEST \
     --batch-mean-probes 4 \
     --frozen-beta-refresh 10 \
     --frozen-beta-mode argsort \
     --frozen-beta-none-mode model \
     --alpha-start 0.0 --alpha-target 0.9 --alpha-warmup-steps 10000" \
    "$GBETA_F5K_DIR" \
    "overnight_20260625_gbeta_listmle_from5k"

# 3d. ori_l2r_baseline (from0)
run_training \
    "ori_l2r_baseline" \
    "--run-kind l2r --max-steps 60000 --lr-decay-steps 60000 --save-steps $SAVE_STEPS_60K" \
    "$ORI_L2R_BL_DIR" \
    "overnight_20260625_ori_l2r_baseline"

# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL RUNS COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Output directories:"
echo "    random_baseline:        $RANDOM_BL_DIR"
echo "    ori_l2r_baseline:       $ORI_L2R_BL_DIR"
echo "    ori_l2r_from5k:         $ORI_L2R_F5K_DIR"
echo "    cdl_from5k:             $CDL_F5K_DIR"
echo "    gbeta_listmle_from5k:   $GBETA_F5K_DIR"
echo "    g_beta model:           $GBETA_BEST"
echo ""
echo "  Attention trajectories:"
echo "    $RANDOM_BL_DIR/attention_trajectory/"
echo "    $ORI_L2R_BL_DIR/attention_trajectory/"
echo "    $ORI_L2R_F5K_DIR/attention_trajectory/"
echo "    $CDL_F5K_DIR/attention_trajectory/"
echo "    $GBETA_F5K_DIR/attention_trajectory/"
echo ""
echo "  W&B: https://wandb.ai/order-lyu/order-lyu"
echo "  Log: $LOGFILE"
echo "════════════════════════════════════════════════════════════════════"
