#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/chenhe/nanogpt-learned-order}"
GPU_ID="${GPU_ID:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-2}"
PYTHON_BIN="${PYTHON_BIN:-/data/users/chenhe/conda_envs/X1/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/Report/logs/wikitext103_attn_mlp_try62_try63_try64_gpu1_after_try61_${STAMP}}"

TRY62_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try62_seed2050_trainable_mlp_mse_noema_shadow512_policy1024_train10k32k_stop32k_prob08_fixed35k.py"
TRY63_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try63_seed2051_trainable_mlp_mse_noema_shadow512_policy512_train10k32k_stop32k_prob08_fixed35k.py"
TRY64_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try64_seed2052_trainable_mlp_mse_noema_shadow512_policy512_train10k32k_stop32k_prob08_fixed35k.py"

cd "$ROOT"
mkdir -p "$RUN_DIR"

SUPERVISOR_LOG="$RUN_DIR/supervisor.log"
STATUS_FILE="$RUN_DIR/status.txt"
DECISION_FILE="$RUN_DIR/decision.txt"
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
  log "received termination; stopping try62/try63/try64 GPU1 queue"
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

  local missing=0
  for config in "$TRY62_CONFIG" "$TRY63_CONFIG" "$TRY64_CONFIG"; do
    if [[ ! -f "$config" ]]; then
      log "missing config: $config"
      missing=1
    fi
  done

  if [[ "$missing" -ne 0 ]]; then
    printf 'failed_preflight_missing_config\n' > "$STATUS_FILE"
    exit 2
  fi
}

run_one() {
  local name="$1"
  local config="$2"
  local seed="$3"
  local train_log="$RUN_DIR/train_${name}_seed${seed}_gpu${GPU_ID}.log"

  wait_for_gpu_idle
  printf 'running_%s\n' "$name" > "$STATUS_FILE"
  printf 'running\n' > "$RUN_DIR/${name}.status"
  log "starting ${name}: ${config}"

  set +e
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py "$config" \
    > "$train_log" 2>&1 &
  CURRENT_CHILD="$!"
  printf '%s\n' "$CURRENT_CHILD" > "$RUN_DIR/${name}.pid"
  wait "$CURRENT_CHILD"
  local code="$?"
  CURRENT_CHILD=""
  set -e

  if [[ "$code" -eq 0 ]]; then
    printf 'completed\n' > "$RUN_DIR/${name}.status"
    log "${name} completed"
  else
    printf 'failed_status_%s\n' "$code" > "$RUN_DIR/${name}.status"
    printf 'failed_%s_status_%s\n' "$name" "$code" > "$STATUS_FILE"
    log "${name} failed with status ${code}; stopping queue"
    exit "$code"
  fi
}

cat > "$DECISION_FILE" <<EOF
queue=try62_then_try63_then_try64_after_try61_gpu1
gpu=CUDA_VISIBLE_DEVICES=1
root=${ROOT}
run_dir=${RUN_DIR}
queue_pid=$$
queue_pgid=${QUEUE_PGID}
cancel_hint=kill -TERM -- -${QUEUE_PGID}
try62_config=${TRY62_CONFIG}
try63_config=${TRY63_CONFIG}
try64_config=${TRY64_CONFIG}
sequence=try62,try63,try64
counterparts=try62<-try57,try63<-try58,try64<-try59
required_idle_checks=${REQUIRED_IDLE_CHECKS}
check_interval_seconds=${CHECK_INTERVAL_SECONDS}
created_at=$(date --iso-8601=seconds)
EOF

main() {
  preflight
  printf 'waiting_for_gpu1_idle\n' > "$STATUS_FILE"
  log "try62/try63/try64 GPU1 serial queue started"
  log "run_dir=${RUN_DIR}"
  log "cancel with: kill -TERM -- -${QUEUE_PGID}"

  run_one try62 "$TRY62_CONFIG" 2050
  run_one try63 "$TRY63_CONFIG" 2051
  run_one try64 "$TRY64_CONFIG" 2052

  printf 'completed\n' > "$STATUS_FILE"
  log "try62/try63/try64 GPU1 serial queue completed"
}

main "$@"
