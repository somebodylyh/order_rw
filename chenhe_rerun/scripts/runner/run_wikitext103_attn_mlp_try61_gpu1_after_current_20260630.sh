#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/chenhe/nanogpt-learned-order}"
GPU_ID="${GPU_ID:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-2}"
PYTHON_BIN="${PYTHON_BIN:-/data/users/chenhe/conda_envs/X1/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/Report/logs/wikitext103_attn_mlp_try61_gpu1_after_current_${STAMP}}"

TRY61_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try61_seed2053_trainable_mlp_mse_noema_shadow512_policy1024_train10k32k_stop32k_prob08_fixed35k.py"

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
  log "received termination; stopping try61 GPU1 queue"
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

  if [[ ! -f "$TRY61_CONFIG" ]]; then
    log "missing config: $TRY61_CONFIG"
    printf 'failed_preflight_missing_config\n' > "$STATUS_FILE"
    exit 2
  fi
}

cat > "$DECISION_FILE" <<EOF
queue=try61_after_current_gpu1
gpu=CUDA_VISIBLE_DEVICES=1
root=${ROOT}
run_dir=${RUN_DIR}
queue_pid=$$
queue_pgid=${QUEUE_PGID}
cancel_hint=kill -TERM -- -${QUEUE_PGID}
try61_config=${TRY61_CONFIG}
seed=2053
policy_attention_samples=1024
shadow_mse_window=10000-32000
required_idle_checks=${REQUIRED_IDLE_CHECKS}
check_interval_seconds=${CHECK_INTERVAL_SECONDS}
created_at=$(date --iso-8601=seconds)
EOF

main() {
  preflight
  printf 'waiting_for_gpu1_idle\n' > "$STATUS_FILE"
  log "try61 GPU1 queue started"
  log "run_dir=${RUN_DIR}"
  log "cancel with: kill -TERM -- -${QUEUE_PGID}"

  wait_for_gpu_idle
  printf 'running_try61\n' > "$STATUS_FILE"
  log "starting try61: ${TRY61_CONFIG}"

  set +e
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py "$TRY61_CONFIG" \
    > "$RUN_DIR/train_try61_seed2053_gpu${GPU_ID}.log" 2>&1 &
  CURRENT_CHILD="$!"
  printf '%s\n' "$CURRENT_CHILD" > "$RUN_DIR/try61.pid"
  wait "$CURRENT_CHILD"
  code="$?"
  CURRENT_CHILD=""
  set -e

  if [[ "$code" -eq 0 ]]; then
    printf 'completed\n' > "$STATUS_FILE"
    log "try61 completed"
  else
    printf 'failed_try61_status_%s\n' "$code" > "$STATUS_FILE"
    log "try61 failed with status ${code}"
    exit "$code"
  fi
}

main "$@"
