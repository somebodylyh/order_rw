#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/probe_results/graph_rw_order_bag"
mkdir -p "$OUT_DIR"

DEVICE="${1:-cuda:0}"
WATCH_LOG="$OUT_DIR/watcher_nohup.log"

setsid "$SCRIPT_DIR/watch_graph_rw_order_bag_20k.sh" "$DEVICE" >> "$WATCH_LOG" 2>&1 < /dev/null &
pid=$!

echo "$pid" > "$OUT_DIR/watcher_20k_abd.pid"
echo "$pid"
