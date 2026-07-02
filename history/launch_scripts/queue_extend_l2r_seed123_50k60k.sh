#!/usr/bin/env bash
# Queue and extend seed-123 ori-L2R from 50k to 60k with a stable LR schedule.
#
# This intentionally appends to the original output directory so eval_curve.tsv
# remains one continuous curve. It backs up the 50k curve before training.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
SOURCE_OUT=block_lo_arm_order_network/probe_results/l2r_continuous_seed123
RUN_TS=$(date '+%Y%m%d_%H%M%S')
SMOKE_TEST=${SMOKE_TEST:-0}
PYTHON=${PYTHON:-python}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
POLL=${POLL:-60}

if [ "$SMOKE_TEST" = "1" ]; then
    OUT="${SMOKE_OUT:-/tmp/order_lyu_smoke_l2r_seed123_50k60k_${RUN_TS}}"
    MAX_STEPS=50001
    SAVE_STEPS=50001
    BATCH_SIZE=2
    GRAD_ACCUM=1
    EVAL_INTERVAL=999999
    LOG_INTERVAL=1
    STREAM_EVAL_WINDOWS=2
    EVAL_BATCH_SIZE=1
    EXTRA_EVAL_ARGS=(--eval-order-seeds 42)
else
    OUT="$SOURCE_OUT"
    MAX_STEPS=60000
    SAVE_STEPS="55000,60000"
    BATCH_SIZE=64
    GRAD_ACCUM=2
    EVAL_INTERVAL=1000
    LOG_INTERVAL=50
    STREAM_EVAL_WINDOWS=2000
    EVAL_BATCH_SIZE=16
    EXTRA_EVAL_ARGS=()
fi

CKPT="${SOURCE_OUT}/ckpt_step50000.pt"
LOG="${OUT}/extend_50k60k_${RUN_TS}.log"

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin

mkdir -p "$OUT"

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
require_file "${SOURCE_OUT}/eval_curve.tsv"
require_file "$TB"
require_file "$VB"

if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    log "ERROR: ${PYTHON} cannot access CUDA. Activate the GPU-capable environment or run outside sandbox."
    exit 1
fi

if [ "$SMOKE_TEST" = "1" ]; then
    log "SMOKE_TEST=1: output will go to $OUT and the formal curve will not be touched."
else
    BACKUP="${OUT}/eval_curve.pre_extend_50k_${RUN_TS}.tsv"
    if [ -f "${OUT}/ckpt_step60000.pt" ]; then
        log "SKIP: ${OUT}/ckpt_step60000.pt already exists."
        exit 0
    fi
    cp "${OUT}/eval_curve.tsv" "$BACKUP"
    log "Backed up eval curve: $BACKUP"
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

log "Starting seed-123 ori-L2R extension 50k -> 60k"
log "Resume ckpt: $CKPT"
log "Output dir:  $OUT"
log "LR policy:   lr=1e-3 min_lr=1e-4 lr_decay_steps=50000 (keeps LR at min after 50k)"

"$PYTHON" -u "$TRAIN" \
    --run-kind l2r \
    --data-source continuous \
    --train-bin "$TB" \
    --val-bin "$VB" \
    --seed 123 \
    --permute-seed 123 \
    --resume-ckpt "$CKPT" \
    --output-dir "$OUT" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --lr 0.001 \
    --min-lr 0.0001 \
    --lr-decay-steps 50000 \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --eval-interval "$EVAL_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --stream-eval-windows "$STREAM_EVAL_WINDOWS" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --device "cuda:${GPU}" \
    "${EXTRA_EVAL_ARGS[@]}" \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "Training exited rc=${rc}"

if [ "$rc" -eq 0 ]; then
    log "Final eval rows:"
    tail -n 12 "${OUT}/eval_curve.tsv" | tee -a "$LOG"
fi

exit "$rc"
