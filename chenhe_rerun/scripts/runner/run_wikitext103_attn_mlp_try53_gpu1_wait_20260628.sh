#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/chenhe/nanogpt-learned-order"
GPU_ID="1"
CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try53_seed2036_shadow_teacher_score_mse_train8k12k_attn1024_train15val3_fixed12k_anneal12k35k.py"
RUN_DIR="${1:-Report/logs/wikitext103_attn_mlp_try53_seed2036_train8k12k_attn1024_train15val3_gpu1_$(date +%Y%m%d_%H%M%S)}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
STABLE_IDLE_SECONDS="${STABLE_IDLE_SECONDS:-60}"

cd "${ROOT_DIR}"
mkdir -p "${RUN_DIR}"

STATUS_FILE="${RUN_DIR}/status.txt"
DECISION_FILE="${RUN_DIR}/decision.txt"
LOG_FILE="${RUN_DIR}/train_try53_gpu1.log"
SUPERVISOR_LOG="${RUN_DIR}/supervisor.log"
PID_FILE="${RUN_DIR}/runner.pid"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${SUPERVISOR_LOG}"
}

gpu_compute_pids() {
  nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {print $1}'
}

wait_for_gpu_idle() {
  local idle_start=""
  while true; do
    local pids
    pids="$(gpu_compute_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    if [[ -z "${pids}" ]]; then
      if [[ -z "${idle_start}" ]]; then
        idle_start="$(date +%s)"
        log "GPU ${GPU_ID} has no compute process; starting stable-idle timer"
      fi
      local now
      now="$(date +%s)"
      if (( now - idle_start >= STABLE_IDLE_SECONDS )); then
        log "GPU ${GPU_ID} stable idle for ${STABLE_IDLE_SECONDS}s"
        return 0
      fi
      printf 'waiting_gpu%s_stable_idle\n' "${GPU_ID}" > "${STATUS_FILE}"
    else
      idle_start=""
      printf 'waiting_gpu%s_busy\n' "${GPU_ID}" > "${STATUS_FILE}"
      log "GPU ${GPU_ID} busy with compute pids: ${pids}"
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

echo "$$" > "${PID_FILE}"
cat > "${DECISION_FILE}" <<EOF
task=try53_seed2036_shadow_teacher_score_mse_train8k12k_attn1024_train15val3
gpu=${GPU_ID}
config=${CONFIG}
root=${ROOT_DIR}
run_dir=${RUN_DIR}
log_file=${LOG_FILE}
supervisor_log=${SUPERVISOR_LOG}
launch_time=$(date --iso-8601=seconds)
check_interval_seconds=${CHECK_INTERVAL_SECONDS}
stable_idle_seconds=${STABLE_IDLE_SECONDS}
cancel_hint=kill -TERM -- -$(ps -o pgid= $$ | tr -d ' ')
EOF

printf 'queued\n' > "${STATUS_FILE}"
trap 'printf "terminated\n" > "${STATUS_FILE}"; log "terminated"; exit 143' TERM INT

log "runner started"
wait_for_gpu_idle

printf 'running\n' > "${STATUS_FILE}"
log "launching try53 on GPU ${GPU_ID}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
python train.py "${CONFIG}" > "${LOG_FILE}" 2>&1
status=$?

if [[ "${status}" -eq 0 ]]; then
  printf 'completed\n' > "${STATUS_FILE}"
  log "completed"
else
  printf 'failed_status_%s\n' "${status}" > "${STATUS_FILE}"
  log "failed with status ${status}"
fi
exit "${status}"
