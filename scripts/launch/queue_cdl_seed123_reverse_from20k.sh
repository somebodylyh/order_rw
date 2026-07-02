#!/usr/bin/env bash
# Queue a strict seed-123 CDL reverse-order negative control from the 20k baseline.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
BASE=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2
OD=block_lo_arm_order_network/probe_results
OUT="${OD}/cdl_teacher_seed123_from20k_l0h2_reverse"
RUN_TS=$(date '+%Y%m%d_%H%M%S')
SMOKE_TEST=${SMOKE_TEST:-0}
PYTHON=${PYTHON:-python}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
POLL=${POLL:-60}

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin
CKPT="${BASE}/ckpt_step20000.pt"

LOG_DIR="${OD}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/cdl_seed123_reverse_from20k_${RUN_TS}.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

require_file() {
    if [ ! -f "$1" ]; then
        log "ERROR: missing required file: $1"
        exit 1
    fi
}

require_file "$TB"
require_file "$VB"
require_file "$CKPT"

if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    log "ERROR: ${PYTHON} cannot access CUDA. Activate the GPU-capable environment or run outside sandbox."
    exit 1
fi

if [ "$SMOKE_TEST" = "1" ]; then
    OUT="/tmp/order_lyu_smoke_cdl_seed123_reverse_from20k_${RUN_TS}"
    MAX_STEPS=20001
    SAVE_STEPS=20001
    BATCH_SIZE=2
    GRAD_ACCUM=1
    EVAL_INTERVAL=999999
    LOG_INTERVAL=1
    STREAM_EVAL_WINDOWS=2
    EVAL_BATCH_SIZE=1
    EXTRA_EVAL_ARGS=(--eval-order-seeds 42)
    log "SMOKE_TEST=1: output=$OUT"
else
    MAX_STEPS=60000
    SAVE_STEPS="25000,30000,35000,40000,45000,50000,55000,60000"
    BATCH_SIZE=64
    GRAD_ACCUM=2
    EVAL_INTERVAL=500
    LOG_INTERVAL=50
    STREAM_EVAL_WINDOWS=2000
    EVAL_BATCH_SIZE=16
    EXTRA_EVAL_ARGS=()
    if [ -f "${OUT}/ckpt_step60000.pt" ]; then
        log "SKIP: ${OUT}/ckpt_step60000.pt already exists."
        exit 0
    fi
fi

mkdir -p "$OUT"

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

log "Starting seed-123 CDL reverse control from20k"
log "Resume ckpt: $CKPT"
log "Output dir:  $OUT"

"$PYTHON" -u "$TRAIN" \
    --run-kind cdl_teacher \
    --data-source continuous \
    --train-bin "$TB" \
    --val-bin "$VB" \
    --seed 123 \
    --permute-seed 123 \
    --resume-ckpt "$CKPT" \
    --cdl-teacher-head 0 2 \
    --cdl-teacher-refresh 10 \
    --cdl-teacher-tau 1.0 \
    --cdl-teacher-rev \
    --alpha-start 0.0 \
    --alpha-target 1.0 \
    --alpha-warmup-steps 5000 \
    --alpha-warmup-start 0 \
    --alpha-ramp-from-resume \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --output-dir "$OUT" \
    --device "cuda:${GPU}" \
    --lr 0.001 \
    --min-lr 0.0001 \
    --lr-decay-steps 50000 \
    --eval-interval "$EVAL_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --stream-eval-windows "$STREAM_EVAL_WINDOWS" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    "${EXTRA_EVAL_ARGS[@]}" \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "Training exited rc=${rc}"

if [ "$rc" -eq 0 ]; then
    log "Final eval rows:"
    tail -n 12 "${OUT}/eval_curve.tsv" | tee -a "$LOG"
fi

exit "$rc"
