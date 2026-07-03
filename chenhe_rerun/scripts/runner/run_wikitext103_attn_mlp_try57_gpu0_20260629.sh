#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/chenhe/nanogpt-learned-order}"
GPU_ID="${GPU_ID:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/Report/logs/wikitext103_attn_mlp_try57_gpu0_${STAMP}}"

TRY57_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try57_seed2050_shadow_teacher_score_mse_train10k20k_step10val2_fixed20k_anneal20k35k.py"

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
  log "received termination; stopping try57 GPU0 run"
  printf 'terminated\n' > "$STATUS_FILE"
  if [[ -n "$CURRENT_CHILD" ]] && kill -0 "$CURRENT_CHILD" 2>/dev/null; then
    kill -TERM "$CURRENT_CHILD" 2>/dev/null || true
    wait "$CURRENT_CHILD" 2>/dev/null || true
  fi
  exit 143
}

trap terminate TERM INT

preflight() {
  if [[ "$GPU_ID" != "0" ]]; then
    log "refusing to run: GPU_ID=${GPU_ID}; this queue is for physical GPU0 only"
    printf 'failed_preflight_wrong_gpu\n' > "$STATUS_FILE"
    exit 2
  fi
  if [[ ! -f "$TRY57_CONFIG" ]]; then
    log "missing config: $TRY57_CONFIG"
    printf 'failed_preflight_missing_config\n' > "$STATUS_FILE"
    exit 2
  fi
}

cat > "$DECISION_FILE" <<EOF
queue=try57_gpu0
gpu=CUDA_VISIBLE_DEVICES=0
root=${ROOT}
run_dir=${RUN_DIR}
queue_pid=$$
queue_pgid=${QUEUE_PGID}
cancel_hint=kill -TERM -- -${QUEUE_PGID}
try57_config=${TRY57_CONFIG}
sequence=try57
created_at=$(date --iso-8601=seconds)
EOF

main() {
  preflight
  printf 'running_try57\n' > "$STATUS_FILE"
  log "try57 GPU0 run started"
  log "run_dir=${RUN_DIR}"
  log "cancel with: kill -TERM -- -${QUEUE_PGID}"

  set +e
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 python train.py "$TRY57_CONFIG" > "$RUN_DIR/train_try57_gpu${GPU_ID}.log" 2>&1 &
  CURRENT_CHILD="$!"
  printf '%s\n' "$CURRENT_CHILD" > "$RUN_DIR/try57.pid"
  wait "$CURRENT_CHILD"
  code="$?"
  CURRENT_CHILD=""
  set -e

  if [[ "$code" -eq 0 ]]; then
    printf 'completed\n' > "$STATUS_FILE"
    log "try57 completed"
  else
    printf 'failed_try57_status_%s\n' "$code" > "$STATUS_FILE"
    log "try57 failed with status ${code}"
    exit "$code"
  fi
}

main "$@"
