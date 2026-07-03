#!/usr/bin/env bash
set -euo pipefail

cd /home/devbox/project/AOGPT-test-order/nanogpt_learned_order

CMD=(
  python
  scripts/runner/hierarchical_segment_curriculum_runner.py
  config/WikiText103/seq256/permute/block64/segment_curriculum_early_stop_open_level.py
)

CHECK_INTERVAL_SECONDS=60
REQUIRED_IDLE_CHECKS=3
idle_0=0
idle_1=0

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] queue started; waiting for GPU 0 or 1"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launch requires ${REQUIRED_IDLE_CHECKS} consecutive idle checks, interval=${CHECK_INTERVAL_SECONDS}s"

while true; do
  for gpu in 0 1; do
    pids="$(
      nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:space:]' || true
    )"
    if [ -z "${pids}" ]; then
      if [ "${gpu}" = "0" ]; then
        idle_0=$((idle_0 + 1))
        idle_count="${idle_0}"
      else
        idle_1=$((idle_1 + 1))
        idle_count="${idle_1}"
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU ${gpu} idle check ${idle_count}/${REQUIRED_IDLE_CHECKS}"
      if [ "${idle_count}" -ge "${REQUIRED_IDLE_CHECKS}" ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU ${gpu} stayed idle; launching with CUDA_VISIBLE_DEVICES=${gpu}"
        . /home/devbox/project/bin/activate
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec "${CMD[@]}"
      fi
    else
      if [ "${gpu}" = "0" ]; then
        idle_0=0
      else
        idle_1=0
      fi
    fi
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] no confirmed idle GPU yet; rechecking in ${CHECK_INTERVAL_SECONDS}s"
  sleep "${CHECK_INTERVAL_SECONDS}"
done
