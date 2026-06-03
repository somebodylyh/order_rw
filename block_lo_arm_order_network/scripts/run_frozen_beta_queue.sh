#!/usr/bin/env bash
# Orchestrate the frozen-β ladder on a SINGLE GPU with bounded concurrency.
# 5k and 40k are launched separately; this queues the rest, keeping at most
# MAXJOBS run-kind=frozen_beta training processes alive at once.
#
# Usage: run_frozen_beta_queue.sh GPU "START1 START2 ..."   (e.g. cuda:1 "20000 10000")
set -uo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:?need GPU}"
QUEUE="${2:?need queue of start steps}"
MAXJOBS="${3:-2}"

active() { pgrep -fc "train_clean_aogpt.py --run-kind frozen_beta" 2>/dev/null || echo 0; }

cd "$PKG"
for START in $QUEUE; do
  while [ "$(active)" -ge "$MAXJOBS" ]; do sleep 60; done
  echo "[queue] launching from${START} on ${GPU} (active=$(active))"
  nohup bash scripts/run_frozen_beta_from.sh "$START" "$GPU" \
    > "probe_results/frozen_beta_from${START}.log" 2>&1 &
  sleep 45  # let it register as an active process before re-checking
done
echo "[queue] all queued runs launched"
