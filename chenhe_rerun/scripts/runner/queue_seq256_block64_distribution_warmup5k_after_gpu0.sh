#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/chenhe/nanogpt-learned-order"
CONFIG="config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_fast_b256_update20_top32_loss1024_warmup5k_anneal15k.py"
RUN_DIR="${1:-${REPO_DIR}/Report/logs/block64_distribution_warmup5k_after_gpu0_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-2}"

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

MONITOR_LOG="${RUN_DIR}/launcher.log"
TRAIN_LOG="${RUN_DIR}/train.log"
PID_FILE="${RUN_DIR}/launcher.pid"
COMMAND_FILE="${RUN_DIR}/command.txt"

printf '%s\n' "$$" > "${PID_FILE}"
printf 'CUDA_VISIBLE_DEVICES=%s PYTHONUNBUFFERED=1 python train.py %s\n' "${GPU_ID}" "${CONFIG}" > "${COMMAND_FILE}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${MONITOR_LOG}"
}

gpu_compute_pids() {
  nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' || true
}

log "queue started; waiting for GPU ${GPU_ID}"
log "run_dir=${RUN_DIR}"
log "config=${CONFIG}"
log "requires ${REQUIRED_IDLE_CHECKS} consecutive idle checks, interval=${CHECK_INTERVAL_SECONDS}s"

idle_checks=0
while true; do
  snapshot="$(
    nvidia-smi -i "${GPU_ID}" --query-gpu=index,memory.free,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      || true
  )"
  pids="$(gpu_compute_pids)"
  log "gpu snapshot: ${snapshot:-unavailable}; compute_pids=${pids:-none}"

  if [ -z "${pids}" ]; then
    idle_checks=$((idle_checks + 1))
    log "GPU ${GPU_ID} idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS}"
    if [ "${idle_checks}" -ge "${REQUIRED_IDLE_CHECKS}" ]; then
      log "GPU ${GPU_ID} confirmed idle; launching training"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 python train.py "${CONFIG}" >> "${TRAIN_LOG}" 2>&1
      code=$?
      log "training exited with code ${code}"
      exit "${code}"
    fi
  else
    idle_checks=0
    log "GPU ${GPU_ID} busy; waiting"
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
