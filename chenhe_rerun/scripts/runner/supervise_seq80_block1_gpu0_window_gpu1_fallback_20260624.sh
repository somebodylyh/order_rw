#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/chenhe/nanogpt-learned-order}"
RUN_DIR="${1:-${REPO_DIR}/Report/logs/supervise_seq80_block1_gpu0_window_gpu1_fallback_$(date +%Y%m%d_%H%M%S)}"

GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
TOTAL_WAIT_SECONDS="${TOTAL_WAIT_SECONDS:-14400}"
GPU0_FINAL_WINDOW_SECONDS="${GPU0_FINAL_WINDOW_SECONDS:-1800}"
GPU1_REQUIRED_IDLE_CHECKS="${GPU1_REQUIRED_IDLE_CHECKS:-10}"

TASK1_CONFIG="config/WikiText103/seq80/permute/block1/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py"
TASK2_CONFIG="config/WikiText103/seq80/permute/block1/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py"

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
- Start time is the supervisor's logical start time; this can be preserved across a supervisor restart.
- If GPU1 is stably idle for ${GPU1_REQUIRED_IDLE_CHECKS} consecutive checks before the 4h deadline, launch on GPU1 immediately.
- During the final ${GPU0_FINAL_WINDOW_SECONDS}s of the 4h window, record whether GPU0 ever has a compute/train process.
- At the 4h deadline, launch on GPU0 if GPU0 had no detected task during that final window and is still idle.
- If GPU0 had a task during the final window, or is busy at the deadline, wait for GPU1 to become stably idle and launch there.

Task 1:
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 python train.py ${TASK1_CONFIG}

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
    visible="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1 || true)"
    if [ -z "${visible}" ]; then
      continue
    fi
    case ",${visible}," in
      *,"${gpu}",*)
        cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        printf 'visible-cuda:%s,%s\n' "${pid}" "${cmd}"
        ;;
    esac
  done < <(pgrep -f 'python.*train.py' || true)
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

launch_on_gpu() {
  local gpu="$1"
  local reason="$2"
  local train_log="${RUN_DIR}/train_serial_gpu${gpu}.log"
  local train_pid_file="${RUN_DIR}/train_serial_gpu${gpu}.pid"
  local train_status_file="${RUN_DIR}/train_serial_gpu${gpu}.status"

  printf 'gpu=%s\nreason=%s\nlaunched_at=%s\ntrain_log=%s\ntask1_config=%s\ntask2_config=%s\n' \
    "${gpu}" "${reason}" "$(date '+%Y-%m-%d %H:%M:%S%z')" "${train_log}" "${TASK1_CONFIG}" "${TASK2_CONFIG}" > "${DECISION_FILE}"
  log "decision: launch serial tasks on GPU ${gpu}; reason=${reason}; train_log=${train_log}"

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
    printf "[%s] task1: CUDA_VISIBLE_DEVICES=%s PYTHONUNBUFFERED=1 python train.py %s\n" "$(date "+%Y-%m-%d %H:%M:%S%z")" "${GPU_ID}" "${TASK1_CONFIG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 python train.py "${TASK1_CONFIG}"
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
  log "detached train launcher started; child_pid=${train_pid:-pending}; supervisor exiting"
  exit 0
}

if [ -n "${SCHEDULER_START_EPOCH:-}" ]; then
  start_epoch="${SCHEDULER_START_EPOCH}"
else
  start_epoch="$(date +%s)"
fi
deadline_epoch=$((start_epoch + TOTAL_WAIT_SECONDS))
final_window_start_epoch=$((deadline_epoch - GPU0_FINAL_WINDOW_SECONDS))

gpu1_idle_checks=0
gpu0_final_window_started=0
gpu0_final_window_busy="${GPU0_FINAL_WINDOW_BUSY_INITIAL:-0}"
fallback_to_gpu1=0

log "supervisor started"
log "run_dir=${RUN_DIR}"
log "repo=${REPO_DIR}"
log "gpu0=${GPU0_ID}; gpu1=${GPU1_ID}; interval=${CHECK_INTERVAL_SECONDS}s; total_wait=${TOTAL_WAIT_SECONDS}s; gpu0_final_window=${GPU0_FINAL_WINDOW_SECONDS}s; gpu1_idle_checks=${GPU1_REQUIRED_IDLE_CHECKS}"
log "start_epoch=${start_epoch}; final_window_start_epoch=${final_window_start_epoch}; deadline_epoch=${deadline_epoch}"
log "task1_config=${TASK1_CONFIG}"
log "task2_config=${TASK2_CONFIG}"

while true; do
  now_epoch="$(date +%s)"
  gpu0_rows="$(gpu_busy_rows "${GPU0_ID}")"
  gpu1_rows="$(gpu_busy_rows "${GPU1_ID}")"
  elapsed=$((now_epoch - start_epoch))
  until_final=$((final_window_start_epoch - now_epoch))
  until_deadline=$((deadline_epoch - now_epoch))
  if [ "${until_final}" -lt 0 ]; then
    until_final=0
  fi
  if [ "${until_deadline}" -lt 0 ]; then
    until_deadline=0
  fi

  log "snapshot: elapsed=${elapsed}s until_final_window=${until_final}s until_deadline=${until_deadline}s gpu0_compute=$(one_line_rows "${gpu0_rows}") gpu1_compute=$(one_line_rows "${gpu1_rows}") gpu0_final_busy=${gpu0_final_window_busy} fallback_gpu1=${fallback_to_gpu1}"

  if [ -z "${gpu1_rows}" ]; then
    gpu1_idle_checks=$((gpu1_idle_checks + 1))
    log "GPU ${GPU1_ID} idle check ${gpu1_idle_checks}/${GPU1_REQUIRED_IDLE_CHECKS}"
  else
    if [ "${gpu1_idle_checks}" -ne 0 ]; then
      log "GPU ${GPU1_ID} busy again; resetting idle checks"
    fi
    gpu1_idle_checks=0
  fi

  if [ "${now_epoch}" -ge "${final_window_start_epoch}" ] && [ "${now_epoch}" -lt "${deadline_epoch}" ]; then
    if [ "${gpu0_final_window_started}" -eq 0 ]; then
      gpu0_final_window_started=1
      log "GPU ${GPU0_ID} final 30m observation window started"
    fi
    if [ -n "${gpu0_rows}" ] && [ "${gpu0_final_window_busy}" -eq 0 ]; then
      gpu0_final_window_busy=1
      log "GPU ${GPU0_ID} had work during the final 30m window; GPU0 deadline launch disqualified"
    fi
  fi

  if [ "${now_epoch}" -lt "${deadline_epoch}" ] && [ "${gpu1_idle_checks}" -ge "${GPU1_REQUIRED_IDLE_CHECKS}" ]; then
    launch_on_gpu "${GPU1_ID}" "gpu1_stable_idle_before_4h_deadline"
  fi

  if [ "${now_epoch}" -ge "${deadline_epoch}" ]; then
    if [ -n "${gpu0_rows}" ] && [ "${gpu0_final_window_busy}" -eq 0 ]; then
      gpu0_final_window_busy=1
      log "GPU ${GPU0_ID} was busy at the 4h deadline; GPU0 deadline launch disqualified"
    fi

    if [ "${gpu0_final_window_busy}" -eq 0 ] && [ -z "${gpu0_rows}" ]; then
      launch_on_gpu "${GPU0_ID}" "gpu0_idle_through_final_30m_at_4h_deadline"
    fi

    if [ "${fallback_to_gpu1}" -eq 0 ]; then
      fallback_to_gpu1=1
      log "falling back to GPU ${GPU1_ID}; waiting for stable idle behind the GPU1 queue"
    fi

    if [ "${gpu1_idle_checks}" -ge "${GPU1_REQUIRED_IDLE_CHECKS}" ]; then
      launch_on_gpu "${GPU1_ID}" "gpu0_final_30m_busy_waited_for_gpu1"
    fi
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
