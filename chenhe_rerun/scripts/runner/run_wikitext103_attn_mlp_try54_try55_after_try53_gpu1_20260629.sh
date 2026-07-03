#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/chenhe/nanogpt-learned-order}"
cd "$ROOT"

UPSTREAM_PID="${UPSTREAM_PID:-684557}"
UPSTREAM_STATUS="${UPSTREAM_STATUS:-Report/logs/wikitext103_attn_mlp_try53_seed2036_train8k12k_attn1024_train15val3_gpu1_20260628_223235/status.txt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-Report/logs/wikitext103_attn_mlp_try54_try55_after_try53_gpu1_${STAMP}}"

TRY54_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try54_seed2027_shadow_teacher_score_mse_train8k10k_step15val3_fixed10k_anneal10k35k.py"
TRY55_CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try55_seed2027_shadow_teacher_score_mse_train8k12k_attn1024_train15val3_fixed12k_anneal12k35k.py"

mkdir -p "$RUN_DIR"

QUEUE_PGID="$(ps -o pgid= "$$" | tr -d ' ')"
CURRENT_CHILD=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUN_DIR/supervisor.log"
}

terminate() {
  log "received termination; stopping queue"
  printf cancelled > "$RUN_DIR/status.txt"
  if [[ -n "$CURRENT_CHILD" ]] && kill -0 "$CURRENT_CHILD" 2>/dev/null; then
    kill -TERM "$CURRENT_CHILD" 2>/dev/null || true
    wait "$CURRENT_CHILD" 2>/dev/null || true
  fi
  exit 143
}

trap terminate TERM INT

cat > "$RUN_DIR/decision.txt" <<EOF
Queue: try54 -> try55 after current try53
GPU: CUDA_VISIBLE_DEVICES=1 only
Upstream try53 PID/PGID: ${UPSTREAM_PID}
Upstream status file: ${UPSTREAM_STATUS}
Queue PID: $$
Queue PGID: ${QUEUE_PGID}
Cancel this queued chain: kill -TERM -- -${QUEUE_PGID}
Run dir: ${RUN_DIR}
TRY54_CONFIG=${TRY54_CONFIG}
TRY55_CONFIG=${TRY55_CONFIG}
EOF

printf waiting_try53 > "$RUN_DIR/status.txt"
printf '%s\n' "$$" > "$RUN_DIR/queue.pid"
printf '%s\n' "$QUEUE_PGID" > "$RUN_DIR/queue.pgid"

gpu1_app_count() {
  nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF { n += 1 } END { print n + 0 }'
}

wait_for_gpu1_idle() {
  local consecutive=0
  while [[ "$consecutive" -lt 3 ]]; do
    local count
    count="$(gpu1_app_count)"
    if [[ "$count" == "0" ]]; then
      consecutive=$((consecutive + 1))
      log "GPU1 idle check ${consecutive}/3"
    else
      consecutive=0
      log "GPU1 still busy with ${count} compute app(s); waiting"
    fi
    if [[ "$consecutive" -lt 3 ]]; then
      sleep 30
    fi
  done
}

wait_for_upstream_try53() {
  log "waiting for upstream try53 pid ${UPSTREAM_PID}"
  if [[ -n "$UPSTREAM_PID" ]] && [[ "$UPSTREAM_PID" != "0" ]]; then
    while kill -0 "$UPSTREAM_PID" 2>/dev/null; do
      sleep 60
    done
  fi

  local upstream_status="missing"
  if [[ -f "$UPSTREAM_STATUS" ]]; then
    upstream_status="$(cat "$UPSTREAM_STATUS")"
  fi
  log "upstream try53 status: ${upstream_status}"
  if [[ "$upstream_status" != "completed" ]]; then
    printf blocked_try53_${upstream_status} > "$RUN_DIR/status.txt"
    log "try53 did not complete cleanly; not launching try54/try55"
    exit 1
  fi
}

run_one() {
  local name="$1"
  local config="$2"
  local log_file="$RUN_DIR/train_${name}_gpu1.log"

  if [[ ! -f "$config" ]]; then
    printf failed_missing_${name}_config > "$RUN_DIR/status.txt"
    log "missing config for ${name}: ${config}"
    exit 1
  fi

  wait_for_gpu1_idle
  printf running_${name} > "$RUN_DIR/status.txt"
  printf running > "$RUN_DIR/${name}.status"
  log "starting ${name}: ${config}"

  export CUDA_VISIBLE_DEVICES=1
  export PYTHONUNBUFFERED=1

  set +e
  python train.py "$config" > "$log_file" 2>&1 &
  CURRENT_CHILD="$!"
  printf '%s\n' "$CURRENT_CHILD" > "$RUN_DIR/${name}.pid"
  wait "$CURRENT_CHILD"
  local code="$?"
  CURRENT_CHILD=""
  set -e

  if [[ "$code" -eq 0 ]]; then
    printf completed > "$RUN_DIR/${name}.status"
    log "${name} completed"
  else
    printf "failed_status_%s" "$code" > "$RUN_DIR/${name}.status"
    printf "failed_${name}_status_%s" "$code" > "$RUN_DIR/status.txt"
    log "${name} failed with status ${code}; stopping queue"
    exit "$code"
  fi
}

log "queue started"
wait_for_upstream_try53
run_one try54 "$TRY54_CONFIG"
run_one try55 "$TRY55_CONFIG"
printf completed > "$RUN_DIR/status.txt"
log "queue completed"
