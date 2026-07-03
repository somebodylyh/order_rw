#!/usr/bin/env bash
set -u

REPO_DIR="/home/chenhe/nanogpt-learned-order"
cd "$REPO_DIR" || exit 1

RUN_DIR="${1:-Report/logs/gpu0_idle_launcher/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

MONITOR_LOG="$RUN_DIR/monitor.log"
TRAIN_LOG="$RUN_DIR/gpu0_train.log"

PYTHON_BIN="/data/users/chenhe/conda_envs/X1/bin/python"
AR_CONFIG="config/Imagenet64VQ_f4/seq256/non_permute/block64_rect1x4/ar.py"
CURRICULUM_RUNNER="scripts/runner/hierarchical_segment_curriculum_runner.py"
CURRICULUM_CONFIG="config/Imagenet64VQ_f4/seq256/permute/block64_rect1x4/segment_curriculum_highmem_strict.py"

check_gpu0_busy() {
  nvidia-smi pmon -c 1 | awk '$1 == 0 && $3 ~ /C/ {found=1} END {exit found ? 0 : 1}'
}

echo "[$(date -Is)] GPU 0 idle launcher started" >> "$MONITOR_LOG"
echo "run_dir=$RUN_DIR" >> "$MONITOR_LOG"

while true; do
  ts="$(date -Is)"
  echo "[$ts] gpu0-only snapshot" >> "$MONITOR_LOG"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$MONITOR_LOG" 2>&1
  nvidia-smi pmon -c 1 >> "$MONITOR_LOG" 2>&1

  if check_gpu0_busy; then
    echo "[$ts] GPU 0 has a compute process; waiting" >> "$MONITOR_LOG"
    sleep 30
    continue
  fi

  echo "[$ts] GPU 0 has no compute process; confirming after 10s" >> "$MONITOR_LOG"
  sleep 10

  if check_gpu0_busy; then
    echo "[$(date -Is)] GPU 0 became busy during confirmation; waiting" >> "$MONITOR_LOG"
    sleep 30
    continue
  fi

  echo "[$(date -Is)] launching requested command on GPU 0" >> "$MONITOR_LOG"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py "$AR_CONFIG" >> "$TRAIN_LOG" 2>&1 && \
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$CURRICULUM_RUNNER" "$CURRICULUM_CONFIG" >> "$TRAIN_LOG" 2>&1
  code=$?
  echo "[$(date -Is)] requested command exited with code $code" >> "$MONITOR_LOG"
  exit "$code"
done
