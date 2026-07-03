#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/home/chenhe/nanogpt-learned-order}"
GPU_ID="${GPU_ID:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-2}"
RUN_DIR="${1:-${REPO_DIR}/Report/logs/wikitext103_score_mlp_try50_gpu1_after_current_serial_$(date +%Y%m%d_%H%M%S)}"

UPSTREAM_QUEUE_PATTERN="queue_wikitext103_score_mlp_try46_try48_try47_try49_gpu1_serial_20260627.sh"
TRY50_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try50_seed2031_resume10k_frozen_try33_score_mlp_noema_teacherdiag_update1_anneal10k35k_freeze35k.py"
BASE_CKPT="out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-clean-10k-for-attn-mlp-resume/ckpt.pt"
TRY50_OUT_DIR="out/base/permute/seq256/block64/out-wikitext103-seq256-try50-seed2031-resume10k-frozen-try33-score-mlp-noema-teacherdiag-update1-anneal10k35k-freeze35k-b64-permute-block"
TRY50_CKPT="${TRY50_OUT_DIR}/ckpt.pt"
MLP_CKPT="checkpoints/attn_mlp_distillation/try33_joint_try20_try24_l0_layermean_fiedler_score_mlp_h2048_1024/best_by_val_tau.pt"

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

upstream_queue_pids() {
  pgrep -f "${UPSTREAM_QUEUE_PATTERN}" 2>/dev/null | awk -v self="$$" '$1 != self {print $1}' || true
}

write_command_file() {
  cat > "${COMMAND_FILE}" <<EOF
Policy:
- Use physical GPU ${GPU_ID} only.
- Never fallback to GPU0 or multi-GPU.
- Wait for existing upstream queue pattern: ${UPSTREAM_QUEUE_PATTERN}
- Poll GPU ${GPU_ID} every ${CHECK_INTERVAL_SECONDS}s after upstream queue exits.
- Launch after ${REQUIRED_IDLE_CHECKS} consecutive idle checks.
- Run try50 as a single serial task.

Resume source:
${BASE_CKPT}

Runtime checkpoint:
${TRY50_CKPT}

MLP checkpoint:
${MLP_CKPT}

Task:
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py ${TRY50_CONFIG}
EOF
}

preflight() {
  if [ "${GPU_ID}" != "1" ]; then
    log "refusing to run: GPU_ID=${GPU_ID}; this task is constrained to physical GPU1"
    printf 'failed_preflight_wrong_gpu\n' > "${STATUS_FILE}"
    exit 2
  fi
  local missing=0
  for path in "${TRY50_CONFIG}" "${BASE_CKPT}" "${MLP_CKPT}"; do
    if [ ! -e "${path}" ]; then
      log "missing required path: ${path}"
      missing=1
    fi
  done
  if [ "${missing}" -ne 0 ]; then
    printf 'failed_preflight_missing_path\n' > "${STATUS_FILE}"
    exit 2
  fi
  mkdir -p "${TRY50_OUT_DIR}"
  if [ ! -e "${TRY50_CKPT}" ]; then
    log "copying clean Random 10k checkpoint into try50 out_dir"
    cp "${BASE_CKPT}" "${TRY50_CKPT}"
  else
    log "try50 checkpoint already exists; preserving it for resume: ${TRY50_CKPT}"
  fi
}

wait_for_upstream_queue() {
  local pids
  while true; do
    pids="$(upstream_queue_pids)"
    if [ -z "${pids}" ]; then
      log "no upstream queue process remains"
      return 0
    fi
    log "waiting for upstream queue pids: $(printf '%s' "${pids}" | tr '\n' ' ')"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
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

  log "try50 queue started"
  log "run_dir=${RUN_DIR}"
  log "repo=${REPO_DIR}"
  log "gpu=${GPU_ID}; interval=${CHECK_INTERVAL_SECONDS}s; required_idle_checks=${REQUIRED_IDLE_CHECKS}"
  log "supervisor_pid=$$"
  log "train_log=${TRAIN_LOG}"

  wait_for_upstream_queue
  wait_for_gpu_idle

  cat > "${DECISION_FILE}" <<EOF
gpu=${GPU_ID}
reason=upstream_queue_complete_and_gpu${GPU_ID}_stable_idle
launched_at=$(date '+%Y-%m-%d %H:%M:%S%z')
train_log=${TRAIN_LOG}
config=${TRY50_CONFIG}
base_checkpoint=${BASE_CKPT}
runtime_checkpoint=${TRY50_CKPT}
mlp_checkpoint=${MLP_CKPT}
sequence=try50
EOF

  log "launching try50 on GPU ${GPU_ID}"
  printf 'started\n' > "${STATUS_FILE}"

  export WANDB_MODE="${WANDB_MODE:-online}"
  export PYTHONUNBUFFERED=1

  run_task "task1_try50_seed2031_resume10k_noema_teacherdiag" env CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py "${TRY50_CONFIG}"

  printf 'complete\n' >> "${STATUS_FILE}"
  log "try50 queue completed"
}

main "$@"
