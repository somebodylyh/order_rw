#!/usr/bin/env bash
# Wait for the current closed-loop pipeline, then rerun seed-123 L2R 50k->60k
# in a separate output directory with constant low LR.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

WAIT_PID=${WAIT_PID:-450190}
PYTHON=${PYTHON:-python}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
POLL=${POLL:-60}
RUN_TS=$(date '+%Y%m%d_%H%M%S')

TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
SRC=block_lo_arm_order_network/probe_results/l2r_continuous_seed123
OUT=block_lo_arm_order_network/probe_results/l2r_continuous_seed123_from50k_lr1e5_60k
CKPT="${SRC}/ckpt_step50000.pt"
LOG_DIR=block_lo_arm_order_network/probe_results/logs
mkdir -p "$LOG_DIR" "$OUT"
LOG="${LOG_DIR}/l2r_seed123_low_lr_after_closed_loop_${RUN_TS}.log"

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

if [ "$WAIT_PID" != "0" ]; then
    log "Waiting for closed-loop pipeline PID ${WAIT_PID} to finish"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep "$POLL"
    done
    log "PID ${WAIT_PID} is no longer running"
fi

if [ -f "${OUT}/ckpt_step60000.pt" ]; then
    log "SKIP: ${OUT}/ckpt_step60000.pt already exists."
    exit 0
fi

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

log "Starting seed-123 L2R low-LR rerun 50k -> 60k"
log "Resume ckpt: $CKPT"
log "Output dir:  $OUT"
log "LR policy:   lr=min_lr=1e-5 constant after 50k"

"$PYTHON" -u "$TRAIN" \
    --run-kind l2r \
    --data-source continuous \
    --train-bin "$TB" \
    --val-bin "$VB" \
    --seed 123 \
    --permute-seed 123 \
    --resume-ckpt "$CKPT" \
    --output-dir "$OUT" \
    --max-steps 60000 \
    --save-steps "55000,60000" \
    --lr 0.00001 \
    --min-lr 0.00001 \
    --lr-decay-steps 50000 \
    --batch-size 64 \
    --grad-accum 2 \
    --eval-interval 500 \
    --log-interval 50 \
    --stream-eval-windows 2000 \
    --eval-batch-size 16 \
    --device "cuda:${GPU}" \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "Training exited rc=${rc}"

if [ "$rc" -eq 0 ]; then
    log "Final eval rows:"
    tail -n 16 "${OUT}/eval_curve.tsv" | tee -a "$LOG"
fi

exit "$rc"
