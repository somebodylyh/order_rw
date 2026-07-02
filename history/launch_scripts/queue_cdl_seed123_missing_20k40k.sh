#!/usr/bin/env bash
# Queue the missing strict seed-123 CDL teacher runs: from20k and from40k.
#
# The from10k strict seed-123 CDL run already exists, so this script only fills
# the missing starts needed for a clean seed-123 matched comparison.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
BASE=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2
OD=block_lo_arm_order_network/probe_results
RUN_TS=$(date '+%Y%m%d_%H%M%S')
SMOKE_TEST=${SMOKE_TEST:-0}
PYTHON=${PYTHON:-python}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
POLL=${POLL:-60}

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin

LOG_DIR="${OD}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/cdl_seed123_missing_20k40k_${RUN_TS}.log"

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
require_file "${BASE}/ckpt_step20000.pt"
require_file "${BASE}/ckpt_step40000.pt"

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

run_cdl() {
    local label="$1"
    local ckpt="$2"
    local out="$3"
    local save_steps="$4"
    local smoke_out=""

    require_file "$ckpt"

    if [ "$SMOKE_TEST" = "1" ]; then
        smoke_out="/tmp/order_lyu_smoke_${label}_${RUN_TS}"
        out="$smoke_out"
        max_steps=$(( ${label#from} + 1 ))
        save_steps="$max_steps"
        batch_size=2
        grad_accum=1
        eval_interval=999999
        log_interval=1
        stream_eval_windows=2
        eval_batch_size=1
        extra_eval_args=(--eval-order-seeds 42)
        log "SMOKE_TEST=1 for ${label}: output=$out max_steps=$max_steps"
    else
        max_steps=60000
        batch_size=64
        grad_accum=2
        eval_interval=500
        log_interval=50
        stream_eval_windows=2000
        eval_batch_size=16
        extra_eval_args=()
        if [ -f "${out}/ckpt_step60000.pt" ]; then
            log "SKIP ${label}: ${out}/ckpt_step60000.pt already exists"
            return 0
        fi
        mkdir -p "$out"
    fi

    log "Starting ${label}: ckpt=$ckpt out=$out gpu=${GPU}"
    "$PYTHON" -u "$TRAIN" \
        --run-kind cdl_teacher \
        --data-source continuous \
        --train-bin "$TB" \
        --val-bin "$VB" \
        --seed 123 \
        --permute-seed 123 \
        --resume-ckpt "$ckpt" \
        --cdl-teacher-head 0 2 \
        --cdl-teacher-refresh 10 \
        --cdl-teacher-tau 1.0 \
        --alpha-start 0.0 \
        --alpha-target 1.0 \
        --alpha-warmup-steps 5000 \
        --alpha-warmup-start 0 \
        --alpha-ramp-from-resume \
        --max-steps "$max_steps" \
        --save-steps "$save_steps" \
        --output-dir "$out" \
        --device "cuda:${GPU}" \
        --lr 0.001 \
        --min-lr 0.0001 \
        --lr-decay-steps 50000 \
        --eval-interval "$eval_interval" \
        --log-interval "$log_interval" \
        --stream-eval-windows "$stream_eval_windows" \
        --eval-batch-size "$eval_batch_size" \
        --batch-size "$batch_size" \
        --grad-accum "$grad_accum" \
        "${extra_eval_args[@]}" \
        2>&1 | tee -a "$LOG"

    local rc=${PIPESTATUS[0]}
    log "${label} exited rc=${rc}"
    return "$rc"
}

run_cdl \
    "from20000" \
    "${BASE}/ckpt_step20000.pt" \
    "${OD}/cdl_teacher_seed123_from20k_l0h2" \
    "25000,30000,35000,40000,45000,50000,55000,60000"

run_cdl \
    "from40000" \
    "${BASE}/ckpt_step40000.pt" \
    "${OD}/cdl_teacher_seed123_from40k_l0h2" \
    "45000,50000,55000,60000"

log "Done."
