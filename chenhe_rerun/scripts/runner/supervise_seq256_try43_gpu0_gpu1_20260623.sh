#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/chenhe/nanogpt-learned-order}"
RUN_DIR="${1:-${REPO_DIR}/Report/logs/supervise_seq256_try43_gpu0_gpu1_$(date +%Y%m%d_%H%M%S)}"

GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
GPU0_REQUIRED_IDLE_CHECKS="${GPU0_REQUIRED_IDLE_CHECKS:-10}"
GPU1_WATCH_SECONDS="${GPU1_WATCH_SECONDS:-12600}"

TASK1_CONFIG="config/WikiText103/seq256/permute/block64/online_spectral_direct_asym_eig_l0headmean_withnone_map_update20_warmup10k_anneal35k_freeze35k.py"
TASK2_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try43_fromscratch_l0headavg_withoutnone_directed_ribbon_seed2029_sameperm_noheadselect_warmup10k_anneal35k_fixed50k_wandb.py"

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

SUPERVISOR_LOG="${RUN_DIR}/supervisor.log"
SUPERVISOR_PID_FILE="${RUN_DIR}/supervisor.pid"
COMMAND_FILE="${RUN_DIR}/command.txt"
DECISION_FILE="${RUN_DIR}/decision.txt"

printf '%s\n' "$$" > "${SUPERVISOR_PID_FILE}"

cat > "${COMMAND_FILE}" <<EOF
Policy:
- Poll GPU0 and GPU1 every ${CHECK_INTERVAL_SECONDS}s.
- GPU0 is usable after ${GPU0_REQUIRED_IDLE_CHECKS} consecutive idle checks.
- GPU1 is usable only if no compute process appears for ${GPU1_WATCH_SECONDS}s (3h30m by default).
- If GPU1 gets any compute process during that window, abandon GPU1 and wait for GPU0.
- Launch the two requested jobs serially on the first eligible GPU.

Task 1:
CUDA_VISIBLE_DEVICES=<gpu> python train.py ${TASK1_CONFIG}

Task 2:
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 python train.py ${TASK2_CONFIG}
EOF

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*" >> "${SUPERVISOR_LOG}"
}

gpu_compute_rows() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
}

gpu_visible_train_rows() {
  local gpu="$1"
  local pid visible cmd
  while IFS= read -r pid; do
    if [ -z "${pid}" ] || [ ! -r "/proc/${pid}/environ" ] || [ ! -r "/proc/${pid}/cmdline" ]; then
      continue
    fi
    visible="$(tr '\0' '\n' < "/proc/${pid}/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1 || true)"
    if [ -z "${visible}" ]; then
      continue
    fi
    case ",${visible}," in
      *,"${gpu}",*)
        cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" || true)"
        printf 'visible-cuda:%s,%s\n' "${pid}" "${cmd}"
        ;;
    esac
  done < <(pgrep -f 'python train.py' || true)
}

gpu_busy_rows() {
  local gpu="$1"
  {
    gpu_compute_rows "${gpu}"
    gpu_visible_train_rows "${gpu}"
  } | sed '/^[[:space:]]*$/d' || true
}

one_line_rows() {
  if [ -z "$1" ]; then
    printf 'none'
  else
    printf '%s' "$1" | tr '\n' ';'
  fi
}

launch_serial_on_gpu() {
  local gpu="$1"
  local reason="$2"
  local train_log="${RUN_DIR}/train_serial_gpu${gpu}.log"
  local train_pid_file="${RUN_DIR}/train_serial_gpu${gpu}.pid"
  local train_status_file="${RUN_DIR}/train_serial_gpu${gpu}.status"

  printf 'gpu=%s\nreason=%s\nlaunched_at=%s\ntrain_log=%s\n' \
    "${gpu}" "${reason}" "$(date '+%Y-%m-%d %H:%M:%S%z')" "${train_log}" > "${DECISION_FILE}"
  log "decision: launch serial jobs on GPU ${gpu}; reason=${reason}; train_log=${train_log}"

  CHILD_PID_FILE="${train_pid_file}" \
  STATUS_FILE="${train_status_file}" \
  GPU_ID="${gpu}" \
  REPO_DIR="${REPO_DIR}" \
  TASK1_CONFIG="${TASK1_CONFIG}" \
  TASK2_CONFIG="${TASK2_CONFIG}" \
  setsid -f bash -c '
    echo "$$" > "${CHILD_PID_FILE}"
    cd "${REPO_DIR}" || exit 1
    printf "[%s] serial train shell started on GPU %s\n" "$(date "+%Y-%m-%d %H:%M:%S%z")" "${GPU_ID}"
    printf "[%s] task1: CUDA_VISIBLE_DEVICES=%s python train.py %s\n" "$(date "+%Y-%m-%d %H:%M:%S%z")" "${GPU_ID}" "${TASK1_CONFIG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py "${TASK1_CONFIG}"
    task1_code=$?
    printf "%s\n" "${task1_code}" > "${STATUS_FILE}.task1"
    printf "[%s] task1 exited with code %s\n" "$(date "+%Y-%m-%d %H:%M:%S%z")" "${task1_code}"
    if [ "${task1_code}" -ne 0 ]; then
      printf "%s\n" "${task1_code}" > "${STATUS_FILE}"
      exit "${task1_code}"
    fi

    printf "[%s] task2: CUDA_VISIBLE_DEVICES=%s PYTHONUNBUFFERED=1 python train.py %s\n" "$(date "+%Y-%m-%d %H:%M:%S%z")" "${GPU_ID}" "${TASK2_CONFIG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 python train.py "${TASK2_CONFIG}"
    task2_code=$?
    printf "%s\n" "${task2_code}" > "${STATUS_FILE}.task2"
    printf "%s\n" "${task2_code}" > "${STATUS_FILE}"
    printf "[%s] task2 exited with code %s\n" "$(date "+%Y-%m-%d %H:%M:%S%z")" "${task2_code}"
    exit "${task2_code}"
  ' >> "${train_log}" 2>&1

  for _ in $(seq 1 10); do
    if [ -s "${train_pid_file}" ]; then
      break
    fi
    sleep 1
  done

  local train_pid
  train_pid="$(cat "${train_pid_file}" 2>/dev/null || true)"
  log "detached serial launcher started; child_pid=${train_pid:-pending}; supervisor exiting"
  exit 0
}

start_epoch="$(date +%s)"
deadline_epoch=$((start_epoch + GPU1_WATCH_SECONDS))
gpu0_idle_checks=0
gpu1_disqualified=0

log "supervisor started"
log "run_dir=${RUN_DIR}"
log "repo=${REPO_DIR}"
log "gpu0=${GPU0_ID}; gpu1=${GPU1_ID}; interval=${CHECK_INTERVAL_SECONDS}s; gpu0_idle_checks=${GPU0_REQUIRED_IDLE_CHECKS}; gpu1_watch_seconds=${GPU1_WATCH_SECONDS}"
log "task1=${TASK1_CONFIG}"
log "task2=${TASK2_CONFIG}"

while true; do
  now_epoch="$(date +%s)"
  gpu0_rows="$(gpu_busy_rows "${GPU0_ID}")"
  gpu1_rows="$(gpu_busy_rows "${GPU1_ID}")"
  elapsed=$((now_epoch - start_epoch))
  remaining=$((deadline_epoch - now_epoch))
  if [ "${remaining}" -lt 0 ]; then
    remaining=0
  fi

  log "snapshot: elapsed=${elapsed}s remaining_gpu1_watch=${remaining}s gpu0_compute=$(one_line_rows "${gpu0_rows}") gpu1_compute=$(one_line_rows "${gpu1_rows}")"

  if [ -z "${gpu0_rows}" ]; then
    gpu0_idle_checks=$((gpu0_idle_checks + 1))
    log "GPU ${GPU0_ID} idle check ${gpu0_idle_checks}/${GPU0_REQUIRED_IDLE_CHECKS}"
  else
    if [ "${gpu0_idle_checks}" -ne 0 ]; then
      log "GPU ${GPU0_ID} busy again; resetting idle checks"
    fi
    gpu0_idle_checks=0
  fi

  if [ "${gpu1_disqualified}" -eq 0 ] && [ -n "${gpu1_rows}" ]; then
    gpu1_disqualified=1
    log "GPU ${GPU1_ID} saw a compute process during the 3h30m watch; disqualifying GPU ${GPU1_ID}"
  fi

  if [ "${gpu0_idle_checks}" -ge "${GPU0_REQUIRED_IDLE_CHECKS}" ]; then
    launch_serial_on_gpu "${GPU0_ID}" "gpu0_stable_idle"
  fi

  if [ "${gpu1_disqualified}" -eq 0 ] && [ "${now_epoch}" -ge "${deadline_epoch}" ]; then
    gpu1_rows_at_deadline="$(gpu_busy_rows "${GPU1_ID}")"
    if [ -z "${gpu1_rows_at_deadline}" ]; then
      launch_serial_on_gpu "${GPU1_ID}" "gpu1_idle_for_full_watch_window"
    fi
    gpu1_disqualified=1
    log "GPU ${GPU1_ID} was busy at the watch deadline; falling back to GPU ${GPU0_ID}"
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
