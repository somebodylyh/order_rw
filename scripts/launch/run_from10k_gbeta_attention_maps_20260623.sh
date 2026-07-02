#!/usr/bin/env bash
# Repeat the previous frozen_beta from10k L0H2/b1 run, adding only raw all-head
# attention-map snapshots for later offline analysis.
set -euo pipefail

ROOT=/home/admin/lyuyuhuan/order_lyu
CD="$ROOT/block_lo_arm_order_network"
SRC="$CD/probe_results/random_baseline_continuous_jun08_seed2"
GB10="$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step10k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"
GPU="${GPU:-cuda:1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$CD/probe_results/frozen_beta_b1_seed2_from10000_l0h2_headmaps_${TS}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-order-lyu}"
TRACK_HEAD_MAP_INTERVAL="${TRACK_HEAD_MAP_INTERVAL:-200}"
TRACK_HEAD_MAP_SAMPLES="${TRACK_HEAD_MAP_SAMPLES:-4}"
TRACK_HEAD_MAP_DTYPE="${TRACK_HEAD_MAP_DTYPE:-float32}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"

TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="$OUT_DIR/wandb"
export WANDB_CACHE_DIR="$OUT_DIR/wandb_cache"
export WANDB_CONFIG_DIR="$OUT_DIR/wandb_config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
}

require_file "$SRC/ckpt_step10000.pt"
require_file "$GB10"
require_file "$TB"
require_file "$VB"

gpu_index() {
  if [[ "$GPU" =~ cuda:([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

wait_for_gpu() {
  local idx
  idx="$(gpu_index)"
  if [[ "$WAIT_FOR_GPU" != "1" || -z "$idx" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; skipping GPU wait"
    return
  fi
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | tr -d ' ')"
    if [[ -z "$used" ]]; then
      echo "could not read GPU $idx memory; continuing"
      return
    fi
    if (( used <= GPU_MAX_USED_MB )); then
      echo "GPU $idx ready: used=${used}MiB <= ${GPU_MAX_USED_MB}MiB"
      return
    fi
    echo "GPU $idx busy: used=${used}MiB > ${GPU_MAX_USED_MB}MiB; waiting ${GPU_WAIT_SECONDS}s"
    sleep "$GPU_WAIT_SECONDS"
  done
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] frozen_beta from10k L0H2/b1 with raw head maps"
echo "  GPU=$GPU"
echo "  OUT_DIR=$OUT_DIR"
echo "  GBETA=$GB10"
echo "  maps: interval=$TRACK_HEAD_MAP_INTERVAL samples=$TRACK_HEAD_MAP_SAMPLES dtype=$TRACK_HEAD_MAP_DTYPE"

wait_for_gpu

python -u "$CD/train_clean_aogpt.py" \
  --run-kind frozen_beta \
  --data-source continuous \
  --train-bin "$TB" \
  --val-bin "$VB" \
  --seed 2 \
  --permute-seed 2 \
  --resume-ckpt "$SRC/ckpt_step10000.pt" \
  --frozen-beta-ckpt "$GB10" \
  --frozen-beta-head 0 2 \
  --frozen-beta-none-mode b1 \
  --frozen-beta-refresh 10 \
  --frozen-beta-mode argsort \
  --alpha-start 0.0 \
  --alpha-target 1.0 \
  --alpha-warmup-steps 5000 \
  --alpha-warmup-start 0 \
  --alpha-ramp-from-resume \
  --max-steps 60000 \
  --save-steps 20000,30000,40000,50000,60000 \
  --output-dir "$OUT_DIR" \
  --device "$GPU" \
  --lr 0.001 \
  --min-lr 0.0001 \
  --lr-decay-steps 50000 \
  --eval-interval 500 \
  --log-interval 50 \
  --stream-eval-windows 2000 \
  --max-eval-seqs 200 \
  --eval-batch-size 16 \
  --batch-size 64 \
  --grad-accum 2 \
  --track-head-maps \
  --track-head-map-interval "$TRACK_HEAD_MAP_INTERVAL" \
  --track-head-map-samples "$TRACK_HEAD_MAP_SAMPLES" \
  --track-head-map-dtype "$TRACK_HEAD_MAP_DTYPE" \
  --track-head-none-mode b1 \
  --wandb-log \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$(basename "$OUT_DIR")" \
  --wandb-tags from10k frozen-gbeta l0h2 b1 raw-head-maps \
  2>&1 | tee "$OUT_DIR/train_log_${TS}.txt"
