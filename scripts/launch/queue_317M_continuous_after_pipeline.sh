#!/usr/bin/env bash
# Queue: wait for from20k_fixseed pipeline to finish, then launch
# 317M from-0→10k continuous training + strict 65-node scan on GPU 1.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

GPU=1
OUT_DIR=block_lo_arm_order_network/probe_results/large_317M_continuous_from0_10k
LOG="$OUT_DIR/queue.log"
SCAN_DIR=reports/large_317M_strict_65_scan_10k_continuous

mkdir -p "$OUT_DIR" "$SCAN_DIR"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# ── Wait for from20k_fixseed pipeline to finish ──
PIPELINE_PID=231468
log "waiting for from20k_fixseed pipeline PID=$PIPELINE_PID to finish..."
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    sleep 120
done
log "pipeline PID=$PIPELINE_PID exited"

# Extra safety: wait for GPU 1 memory to free up
log "waiting for GPU $GPU memory >= 20000 MiB free"
while true; do
    free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', *' -v gpu="$GPU" '$1==gpu{print $2}')
    free_mb=${free_mb:-0}
    if [ "$free_mb" -ge 20000 ]; then break; fi
    log "GPU$GPU free=${free_mb} MiB; sleeping 60s"
    sleep 60
done

# ── Clean old chunks-trained checkpoints ──
OLD_DIR=block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d
log "removing old chunks-trained ckpts from $OLD_DIR"
rm -f "$OLD_DIR/ckpt_step0.pt"
rm -f "$OLD_DIR/ckpt_step1000.pt"
rm -f "$OLD_DIR/ckpt_step5000.pt"
rm -f "$OLD_DIR/ckpt_step10000.pt"
log "old ckpts removed"

# ── Train 317M from 0 to 10k (continuous streaming) ──
log "launching 317M from-0→10k continuous training on cuda:$GPU"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u block_lo_arm_order_network/train_clean_aogpt.py \
  --run-kind baseline \
  --data-source continuous \
  --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
  --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
  --seed 42 --permute-seed 42 \
  --n-layer 16 --n-head 16 --n-embd 1024 \
  --block-size 256 --block-order-block-len 4 \
  --batch-size 8 --grad-accum 16 \
  --max-steps 10000 \
  --save-steps 5000,10000 \
  --lr 0.0003 --min-lr 3e-05 --lr-decay-steps 10000 \
  --warmup-iters 200 \
  --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
  --device "cuda:$GPU" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "317M training exited rc=$rc"

if [ "$rc" -ne 0 ]; then
    log "TRAINING FAILED, aborting"
    exit "$rc"
fi

# ── Strict 65-node scan on 10k checkpoint ──
CKPT_10K="$OUT_DIR/ckpt_step10000.pt"
if [ ! -f "$CKPT_10K" ]; then
    log "ERROR: ckpt_step10000.pt not found at $CKPT_10K"
    exit 1
fi

log "launching strict 65-node scan on $CKPT_10K"

python -u scripts/search_strict_label_free_65.py \
  --ckpt "$CKPT_10K" \
  --out-dir "$SCAN_DIR" \
  --device "cuda:$GPU" \
  --M 40 --batch-size 4 --fwd-batch 4 \
  --seed 42 \
  --methods L C-D+L \
  --control-seeds 0 1 2 3 4 \
  2>&1 | tee -a "$LOG"

rc_scan=${PIPESTATUS[0]}
log "strict 65-node scan exited rc=$rc_scan"
log "DONE at $(date)"
