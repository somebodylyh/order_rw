#!/usr/bin/env bash
# Queue shuffled-AR model-frame order diagnostic when a GPU is free.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

PYTHON=${PYTHON:-python3}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
POLL=${POLL:-60}
RUN_TS=$(date '+%Y%m%d_%H%M%S')
LOG_DIR=block_lo_arm_order_network/probe_results/logs
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/shuffled_ar_model_frame_diag_${RUN_TS}.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

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

log "Starting shuffled-AR model-frame diagnostic"
"$PYTHON" analyses/diag_shuffled_ar_model_frame_order.py \
    --ckpt block_lo_arm_order_network/probe_results/shuffled_l2r_continuous_jun05/ckpt_step50000.pt \
    --out-dir reports/shuffled_ar_model_frame_diag \
    --device "cuda:${GPU}" \
    --M 12 \
    --batch-size 4 \
    --fwd-batch 4 \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "Diagnostic exited rc=${rc}"
exit "$rc"
