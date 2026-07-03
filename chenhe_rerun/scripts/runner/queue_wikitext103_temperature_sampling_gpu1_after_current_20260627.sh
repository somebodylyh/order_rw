#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/home/chenhe/nanogpt-learned-order}"
GPU_ID="${GPU_ID:-1}"
WAIT_FOR_PID="${WAIT_FOR_PID:-${1:-}}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-1}"
RUN_DIR="${2:-${REPO_DIR}/Report/logs/wikitext103_temperature_sampling_l0_gpu1_after_current_$(date +%Y%m%d_%H%M%S)}"

SEED2027_CONFIG="config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_temperature_sampling_t2to01_update20_warmup15k_sample35k_freeze35k_seed2027_sameperm.py"
SEED2028_CONFIG="config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_temperature_sampling_t2to01_update20_warmup15k_sample35k_freeze35k_seed2028_sameperm.py"

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

SUPERVISOR_LOG="${RUN_DIR}/supervisor.log"
SUPERVISOR_PID_FILE="${RUN_DIR}/supervisor.pid"
COMMAND_FILE="${RUN_DIR}/command.txt"
DECISION_FILE="${RUN_DIR}/decision.txt"
TRAIN_LOG="${RUN_DIR}/train_serial_gpu${GPU_ID}.log"
STATUS_FILE="${RUN_DIR}/status.txt"

printf '%s\n' "$$" > "${SUPERVISOR_PID_FILE}"

exec >> "${SUPERVISOR_LOG}" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*"
}

gpu_compute_rows() {
  nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
}

one_line_rows() {
  if [ -z "$1" ]; then
    printf 'none'
  else
    printf '%s' "$1" | tr '\n' ';'
  fi
}

write_command_file() {
  cat > "${COMMAND_FILE}" <<EOF
Policy:
- Use physical GPU ${GPU_ID} only.
- Never fallback to GPU0 or multi-GPU.
- Wait for existing GPU1 queue PID ${WAIT_FOR_PID:-none} to exit before launch.
- Poll GPU ${GPU_ID} every ${CHECK_INTERVAL_SECONDS}s.
- Launch after ${REQUIRED_IDLE_CHECKS} consecutive idle compute checks.
- Run the two temperature-sampling configs serially.
- Stop if any task exits non-zero.

Task 1:
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py ${SEED2027_CONFIG}

Task 2:
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py ${SEED2028_CONFIG}
EOF
}

preflight() {
  if [ "${GPU_ID}" != "1" ]; then
    log "refusing to run: GPU_ID=${GPU_ID}; this task is constrained to physical GPU1"
    printf 'failed_preflight_wrong_gpu\n' > "${STATUS_FILE}"
    exit 2
  fi
  local missing=0
  for path in "${SEED2027_CONFIG}" "${SEED2028_CONFIG}"; do
    if [ ! -e "${path}" ]; then
      log "missing required config: ${path}"
      missing=1
    fi
  done
  if [ "${missing}" -ne 0 ]; then
    printf 'failed_preflight_missing_config\n' > "${STATUS_FILE}"
    exit 2
  fi
}

wait_for_existing_queue() {
  if [ -z "${WAIT_FOR_PID}" ]; then
    log "no WAIT_FOR_PID provided; skipping parent queue wait"
    return 0
  fi
  if ! [[ "${WAIT_FOR_PID}" =~ ^[0-9]+$ ]]; then
    log "invalid WAIT_FOR_PID=${WAIT_FOR_PID}"
    printf 'failed_preflight_bad_wait_pid\n' > "${STATUS_FILE}"
    exit 2
  fi
  while ps -p "${WAIT_FOR_PID}" >/dev/null 2>&1; do
    log "waiting for existing GPU1 queue pid=${WAIT_FOR_PID}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
  log "existing GPU1 queue pid=${WAIT_FOR_PID} has exited"
}

wait_for_gpu_idle() {
  local idle_checks=0
  local rows

  while true; do
    rows="$(gpu_compute_rows)"
    log "GPU ${GPU_ID} snapshot: compute=$(one_line_rows "${rows}") idle_checks=${idle_checks}/${REQUIRED_IDLE_CHECKS}"

    if [ -z "${rows}" ]; then
      idle_checks=$((idle_checks + 1))
      log "GPU ${GPU_ID} idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS}"
      if [ "${idle_checks}" -ge "${REQUIRED_IDLE_CHECKS}" ]; then
        return 0
      fi
    else
      if [ "${idle_checks}" -ne 0 ]; then
        log "GPU ${GPU_ID} became busy again; resetting idle checks"
      fi
      idle_checks=0
    fi

    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

run_task() {
  local name="$1"
  shift

  log "starting ${name}: $*"
  printf '[%s] START %s: %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "${name}" "$*" >> "${TRAIN_LOG}"

  set +e
  "$@" >> "${TRAIN_LOG}" 2>&1
  local cmd_status="$?"
  set -e

  printf '[%s] END %s status=%s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "${name}" "${cmd_status}" >> "${TRAIN_LOG}"
  printf '%s=%s\n' "${name}" "${cmd_status}" >> "${STATUS_FILE}"

  if [ "${cmd_status}" -ne 0 ]; then
    log "${name} failed with status ${cmd_status}; serial queue stopped"
    exit "${cmd_status}"
  fi
}

main() {
  write_command_file
  preflight

  log "queue started"
  log "run_dir=${RUN_DIR}"
  log "repo=${REPO_DIR}"
  log "gpu=${GPU_ID}; wait_for_pid=${WAIT_FOR_PID:-none}; interval=${CHECK_INTERVAL_SECONDS}s"
  log "supervisor_pid=$$"
  log "train_log=${TRAIN_LOG}"
  printf 'waiting_existing_queue\n' > "${STATUS_FILE}"

  wait_for_existing_queue
  printf 'waiting_gpu_idle\n' >> "${STATUS_FILE}"
  wait_for_gpu_idle

  cat > "${DECISION_FILE}" <<EOF
gpu=${GPU_ID}
reason=after_existing_gpu1_queue_then_gpu${GPU_ID}_stable_idle
wait_for_pid=${WAIT_FOR_PID:-none}
launched_at=$(date '+%Y-%m-%d %H:%M:%S%z')
train_log=${TRAIN_LOG}
seed2027_config=${SEED2027_CONFIG}
seed2028_config=${SEED2028_CONFIG}
sequence=seed2027,seed2028
EOF

  log "launching serial tasks on GPU ${GPU_ID}"
  printf 'started\n' >> "${STATUS_FILE}"

  export WANDB_MODE="${WANDB_MODE:-online}"
  export PYTHONUNBUFFERED=1

  run_task "task1_temperature_sampling_seed2027" env CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py "${SEED2027_CONFIG}"
  run_task "task2_temperature_sampling_seed2028" env CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py "${SEED2028_CONFIG}"

  printf 'complete\n' >> "${STATUS_FILE}"
  log "serial queue completed"
}

main "$@"
