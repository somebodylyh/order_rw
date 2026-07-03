#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/chenhe/nanogpt-learned-order}"
GPU_ID="${GPU_ID:-1}"
WAIT_FOR_PID="${WAIT_FOR_PID:-2358913}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/Report/logs/wikitext103_seq80_teacher_try2_gpu1_after_try1_${STAMP}}"

TRY2_CONFIG="config/WikiText103/seq80/permute/block1/seq_teacher/online_spectral_seq_teacher_try2_l0_allheads_targettoobserved_noema_versionB.py"

cd "$ROOT"
mkdir -p "$RUN_DIR"

SUPERVISOR_LOG="$RUN_DIR/supervisor.log"
STATUS_FILE="$RUN_DIR/status.txt"
DECISION_FILE="$RUN_DIR/decision.txt"
COMMAND_FILE="$RUN_DIR/command.txt"
QUEUE_PID_FILE="$RUN_DIR/queue.pid"
QUEUE_PGID_FILE="$RUN_DIR/queue.pgid"
CURRENT_CHILD=""

QUEUE_PGID="$(ps -o pgid= "$$" | tr -d ' ')"
printf '%s\n' "$$" > "$QUEUE_PID_FILE"
printf '%s\n' "$QUEUE_PGID" > "$QUEUE_PGID_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*" >> "$SUPERVISOR_LOG"
}

terminate() {
  log "received termination; stopping seq80 teacher try2 GPU1 queue"
  printf 'terminated\n' > "$STATUS_FILE"
  if [[ -n "$CURRENT_CHILD" ]] && kill -0 "$CURRENT_CHILD" 2>/dev/null; then
    kill -TERM "$CURRENT_CHILD" 2>/dev/null || true
    wait "$CURRENT_CHILD" 2>/dev/null || true
  fi
  exit 143
}

trap terminate TERM INT

gpu_compute_count() {
  nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF { n += 1 } END { print n + 0 }'
}

wait_for_pid_exit() {
  if [[ -z "$WAIT_FOR_PID" ]] || [[ "$WAIT_FOR_PID" == "0" ]]; then
    log "no WAIT_FOR_PID requested; skipping process wait"
    return
  fi

  if ! kill -0 "$WAIT_FOR_PID" 2>/dev/null; then
    log "WAIT_FOR_PID=${WAIT_FOR_PID} is not running; continuing to GPU idle checks"
    return
  fi

  printf 'waiting_for_pid_%s\n' "$WAIT_FOR_PID" > "$STATUS_FILE"
  log "waiting for current GPU1 run PID=${WAIT_FOR_PID} to exit"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep "$CHECK_INTERVAL_SECONDS"
  done
  log "WAIT_FOR_PID=${WAIT_FOR_PID} exited"
}

wait_for_gpu_idle() {
  local idle_checks=0
  local count

  while [[ "$idle_checks" -lt "$REQUIRED_IDLE_CHECKS" ]]; do
    count="$(gpu_compute_count)"
    if [[ "$count" == "0" ]]; then
      idle_checks=$((idle_checks + 1))
      log "GPU${GPU_ID} idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS}"
    else
      idle_checks=0
      log "GPU${GPU_ID} busy with ${count} compute app(s); waiting"
    fi

    if [[ "$idle_checks" -lt "$REQUIRED_IDLE_CHECKS" ]]; then
      sleep "$CHECK_INTERVAL_SECONDS"
    fi
  done
}

preflight() {
  if [[ "$GPU_ID" != "1" ]]; then
    log "refusing to run: GPU_ID=${GPU_ID}; this queue is for physical GPU1 only"
    printf 'failed_preflight_wrong_gpu\n' > "$STATUS_FILE"
    exit 2
  fi

  if [[ ! -f "$TRY2_CONFIG" ]]; then
    log "missing config: $TRY2_CONFIG"
    printf 'failed_preflight_missing_config\n' > "$STATUS_FILE"
    exit 2
  fi
}

cat > "$COMMAND_FILE" <<EOF
CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONUNBUFFERED=1 ${PYTHON_BIN} train.py ${TRY2_CONFIG}
EOF

cat > "$DECISION_FILE" <<EOF
queue=seq80_teacher_try2_after_try1_gpu1
gpu=CUDA_VISIBLE_DEVICES=1
root=${ROOT}
run_dir=${RUN_DIR}
queue_pid=$$
queue_pgid=${QUEUE_PGID}
cancel_hint=kill -TERM -- -${QUEUE_PGID}
wait_for_pid=${WAIT_FOR_PID}
try2_config=${TRY2_CONFIG}
readout=target_to_observed
head_signal_probe_heads=0:mean
teacher_method=pairwise_max_fiedler_linear_profile_candidate
attention_samples=1024
loss_samples=512
required_idle_checks=${REQUIRED_IDLE_CHECKS}
check_interval_seconds=${CHECK_INTERVAL_SECONDS}
created_at=$(date --iso-8601=seconds)
EOF

main() {
  preflight
  printf 'queued\n' > "$STATUS_FILE"
  log "seq80 teacher try2 GPU1 serial queue started"
  log "run_dir=${RUN_DIR}"
  log "cancel with: kill -TERM -- -${QUEUE_PGID}"

  wait_for_pid_exit
  printf 'waiting_for_gpu1_idle\n' > "$STATUS_FILE"
  wait_for_gpu_idle

  printf 'running_try2\n' > "$STATUS_FILE"
  log "starting try2: ${TRY2_CONFIG}"

  set +e
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py "$TRY2_CONFIG" \
    > "$RUN_DIR/train_try2_gpu${GPU_ID}.log" 2>&1 &
  CURRENT_CHILD="$!"
  printf '%s\n' "$CURRENT_CHILD" > "$RUN_DIR/try2.pid"
  wait "$CURRENT_CHILD"
  code="$?"
  CURRENT_CHILD=""
  set -e

  if [[ "$code" -eq 0 ]]; then
    printf 'completed\n' > "$STATUS_FILE"
    log "try2 completed"
  else
    printf 'failed_try2_status_%s\n' "$code" > "$STATUS_FILE"
    log "try2 failed with status ${code}"
    exit "$code"
  fi
}

main "$@"
