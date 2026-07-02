#!/usr/bin/env bash
# Minimal diagnostic for the seed-123 L2R 50k->51k jump.
#
# Runs two 1000-step arms from the same 50k checkpoint:
#   1. lr=0 no-update control: tests whether resume/eval itself changes the metric.
#   2. lr=1e-5 low-LR control: tests whether a safer continuation still jumps.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

PYTHON=${PYTHON:-python}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
POLL=${POLL:-60}
RUN_TS=$(date '+%Y%m%d_%H%M%S')

TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
SRC=block_lo_arm_order_network/probe_results/l2r_continuous_seed123
CKPT="${SRC}/ckpt_step50000.pt"
LOG_DIR=block_lo_arm_order_network/probe_results/logs
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/l2r_seed123_51000_smoke_${RUN_TS}.log"

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

require_file() {
    if [ ! -f "$1" ]; then
        log "ERROR: missing required file: $1"
        exit 1
    fi
}

require_file "$CKPT"
require_file "$TB"
require_file "$VB"

if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    log "ERROR: ${PYTHON} cannot access CUDA. Activate the GPU-capable environment or run outside sandbox."
    exit 1
fi

log "Waiting for a GPU with >= ${MIN_FREE_MB} MiB free"
GPU=""
while true; do
    for gpu in 0 1; do
        free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F', *' -v gpu="$gpu" '$1==gpu{print $2}')
        free_mb=${free_mb:-0}
        if [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
            GPU="$gpu"
            log "Selected GPU ${GPU} with ${free_mb} MiB free"
            break 2
        fi
    done
    log "No GPU has >= ${MIN_FREE_MB} MiB free; sleeping ${POLL}s"
    sleep "$POLL"
done

run_arm() {
    local label="$1"
    local lr="$2"
    local out="/tmp/order_lyu_l2r_seed123_${label}_51000_${RUN_TS}"
    mkdir -p "$out"

    log "Starting arm=${label} lr=${lr} out=${out}"
    "$PYTHON" -u "$TRAIN" \
        --run-kind l2r \
        --data-source continuous \
        --train-bin "$TB" \
        --val-bin "$VB" \
        --seed 123 \
        --permute-seed 123 \
        --resume-ckpt "$CKPT" \
        --output-dir "$out" \
        --max-steps 51000 \
        --save-steps "51000" \
        --lr "$lr" \
        --min-lr "$lr" \
        --lr-decay-steps 50000 \
        --batch-size 64 \
        --grad-accum 2 \
        --eval-interval 1000 \
        --log-interval 50 \
        --stream-eval-windows 2000 \
        --eval-batch-size 16 \
        --device "cuda:${GPU}" \
        2>&1 | tee -a "$LOG"

    local rc=${PIPESTATUS[0]}
    log "arm=${label} exited rc=${rc}"
    if [ "$rc" -eq 0 ]; then
        log "arm=${label} final eval:"
        tail -n 4 "${out}/eval_curve.tsv" | tee -a "$LOG"
    fi
    return "$rc"
}

run_arm "zero_lr" "0.0"
run_arm "lr1e5" "0.00001"

log "Done. Log: $LOG"
