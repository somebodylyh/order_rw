#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/home/chenhe/nanogpt-learned-order}"
GPU_ID="${GPU_ID:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-10}"
RUN_DIR="${1:-${REPO_DIR}/Report/logs/wikitext103_mlp_try31_try32_gpu1_serial_$(date +%Y%m%d_%H%M%S)}"

TRY31_SCRIPT="Report/language/wikitext103/mlp/distillation/try_31/run_main_try31_seed2027_attnema095_update1_warmup20k_cuda0.sh"
TRY32_SCRIPT="Report/language/wikitext103/mlp/distillation/try_32/run_main_try32_seed2028_attnema095_update1_warmup20k_cuda0.sh"
TRY31_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try31_seed2027_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_attnema095_update1_warmup20k_anneal35k_freeze35k.py"
TRY32_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try32_seed2028_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_attnema095_update1_warmup20k_anneal35k_freeze35k.py"

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
- GPU-only queue for physical GPU ${GPU_ID}.
- Poll GPU ${GPU_ID} every ${CHECK_INTERVAL_SECONDS}s.
- Launch after ${REQUIRED_IDLE_CHECKS} consecutive idle checks.
- Run tasks serially; stop if any task exits non-zero.

Task 1:
CUDA_VISIBLE_DEVICES=${GPU_ID} bash ${TRY31_SCRIPT}

Task 2:
CUDA_VISIBLE_DEVICES=${GPU_ID} bash ${TRY32_SCRIPT}
EOF
}

preflight() {
  local missing=0
  for path in "${TRY31_SCRIPT}" "${TRY32_SCRIPT}" "${TRY31_CONFIG}" "${TRY32_CONFIG}"; do
    if [ ! -e "${path}" ]; then
      log "missing required path: ${path}"
      missing=1
    fi
  done
  if [ "${missing}" -ne 0 ]; then
    printf 'failed_preflight\n' > "${STATUS_FILE}"
    exit 2
  fi
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

main() {
  write_command_file
  preflight

  log "queue started"
  log "run_dir=${RUN_DIR}"
  log "repo=${REPO_DIR}"
  log "gpu=${GPU_ID}; interval=${CHECK_INTERVAL_SECONDS}s; required_idle_checks=${REQUIRED_IDLE_CHECKS}"
  log "supervisor_pid=$$"
  log "train_log=${TRAIN_LOG}"

  wait_for_gpu_idle

  cat > "${DECISION_FILE}" <<EOF
gpu=${GPU_ID}
reason=gpu${GPU_ID}_stable_idle
launched_at=$(date '+%Y-%m-%d %H:%M:%S%z')
train_log=${TRAIN_LOG}
EOF

  log "launching serial tasks on GPU ${GPU_ID}"
  printf 'started\n' > "${STATUS_FILE}"

  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTHONUNBUFFERED=1

  run_task "task1_try31" env CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${TRY31_SCRIPT}"
  run_task "task2_try32" env CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${TRY32_SCRIPT}"

  printf 'complete\n' >> "${STATUS_FILE}"
  log "serial queue completed"
}

main "$@"
