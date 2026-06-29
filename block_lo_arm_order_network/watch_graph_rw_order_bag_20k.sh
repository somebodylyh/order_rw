#!/bin/bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
GPU_INDEX="${DEVICE#cuda:}"
if [ "$GPU_INDEX" = "$DEVICE" ]; then
    GPU_INDEX=0
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="block_lo_arm_order_network/probe_results/graph_rw_order_bag"
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/launch_20k_abd.log"
LOCK_FILE="$OUT_DIR/launch_20k_abd.lockfile"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date)] watcher already active: $LOCK_FILE" >> "$LOG"
    exit 0
fi

echo "[$(date)] watcher started for RUNS=control,A,B,D MAX_STEPS=20000 WARMUP_STEPS=5000 EVAL_INTERVAL=1000 on $DEVICE" >> "$LOG"

while true; do
    query=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1 || true)
    line=$(printf "%s\n" "$query" | awk -F', ' -v gpu="$GPU_INDEX" '$1 == gpu {print $2, $3}')
    mem=$(echo "$line" | awk '{print $1}')
    util=$(echo "$line" | awk '{print $2}')

    if [ -z "${line:-}" ]; then
        echo "[$(date)] gpu${GPU_INDEX} query unavailable: $query" >> "$LOG"
        sleep 300
        continue
    fi

    echo "[$(date)] gpu${GPU_INDEX} mem=${mem}MiB util=${util}%" >> "$LOG"

    if [ -n "${mem:-}" ] && [ -n "${util:-}" ] && [ "$mem" -lt 2000 ] && [ "$util" -lt 20 ]; then
        echo "[$(date)] gpu${GPU_INDEX} free; launching long run" >> "$LOG"
        RUNS=control,A,B,D \
        MAX_STEPS=20000 \
        WARMUP_STEPS=5000 \
        EVAL_INTERVAL=1000 \
        ./block_lo_arm_order_network/run_graph_rw_order_bag_ablation.sh "$DEVICE" >> "$LOG" 2>&1
        status=$?
        echo "[$(date)] long run exited with status $status" >> "$LOG"
        exit "$status"
    fi

    sleep 300
done
