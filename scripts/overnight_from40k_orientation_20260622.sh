#!/usr/bin/env bash
# Overnight orientation controls for the from40k seed124 g_beta failure.
#
# Default RUN_SET=main launches only the priority arm:
#   1. frozen_beta from40k seed124 L0H0 rev
#
# RUN_SET=remaining runs only the lower-priority follow-up controls:
#   1. L0H2 fwd  - near-tied small-train candidate
#   2. L0H0 fwd  - old-orientation reproduction control
#
# RUN_SET=all runs:
#   1. L0H0 rev  - audition-selected orientation, priority arm
#   2. L0H2 fwd  - near-tied small-train candidate
#   3. L0H0 fwd  - old-orientation reproduction control
#
# Example:
#   GPU=cuda:1 WANDB_MODE=online RUN_SET=main bash scripts/overnight_from40k_orientation_20260622.sh
set -euo pipefail

ROOT=/home/admin/lyuyuhuan/order_lyu
CD="$ROOT/block_lo_arm_order_network"
SRC="$CD/probe_results/random_baseline_continuous_jun08_seed2"
GPU="${GPU:-cuda:1}"
RUN_SET="${RUN_SET:-main}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-order-lyu}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

GB_L0H0="$CD/batch_readout/logs/gbeta_b1_L0H0_from40k_seed124/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"
GB_L0H2="$CD/batch_readout/logs/gbeta_b1_L0H2_seed2_step40k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"

BASE_OUT="$CD/probe_results/from40k_orientation_controls_${TS}"
mkdir -p "$BASE_OUT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="$BASE_OUT/wandb"
export WANDB_CACHE_DIR="$BASE_OUT/wandb_cache"
export WANDB_CONFIG_DIR="$BASE_OUT/wandb_config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
}

require_file "$SRC/ckpt_step40000.pt"
require_file "$GB_L0H0"
require_file "$GB_L0H2"

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
    log "nvidia-smi not found; skipping GPU wait"
    return
  fi
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | tr -d ' ')"
    if [[ -z "$used" ]]; then
      log "could not read GPU $idx memory; continuing"
      return
    fi
    if (( used <= GPU_MAX_USED_MB )); then
      log "GPU $idx ready: used=${used}MiB <= ${GPU_MAX_USED_MB}MiB"
      return
    fi
    log "GPU $idx busy: used=${used}MiB > ${GPU_MAX_USED_MB}MiB; waiting ${GPU_WAIT_SECONDS}s"
    sleep "$GPU_WAIT_SECONDS"
  done
}

COMMON_FLAGS=(
  --run-kind frozen_beta
  --resume-ckpt "$SRC/ckpt_step40000.pt"
  --device "$GPU"
  --seed 2
  --permute-seed 2
  --data-source continuous
  --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
  --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin
  --stream-eval-windows 2000
  --max-eval-seqs 200
  --eval-batch-size 16
  --eval-interval 500
  --log-interval 50
  --max-steps 60000
  --save-steps 50000,60000
  --batch-size 64
  --grad-accum 2
  --lr 0.001
  --min-lr 0.0001
  --lr-decay-steps 50000
  --weight-decay 0.1
  --beta1 0.9
  --beta2 0.99
  --grad-clip 1.0
  --alpha-start 0.0
  --alpha-target 1.0
  --alpha-warmup-steps 5000
  --alpha-warmup-start 0
  --alpha-ramp-from-resume
  --frozen-beta-none-mode b1
  --frozen-beta-refresh 10
  --frozen-beta-mode argsort
  --wandb-log
  --wandb-project "$WANDB_PROJECT"
  --wandb-mode "$WANDB_MODE"
  --wandb-tags from40k orientation-control seed124 frozen-gbeta
)

run_arm() {
  local name="$1"
  local ckpt="$2"
  local layer="$3"
  local head="$4"
  local orientation="$5"
  local out_dir="$BASE_OUT/$name"
  local log_file="$out_dir/train_log_${TS}.txt"

  mkdir -p "$out_dir"
  log "START $name on $GPU"
  log "  out_dir=$out_dir"
  log "  gbeta=$ckpt"
  log "  head=L${layer}H${head} orientation=$orientation"

  local flags=(
    "${COMMON_FLAGS[@]}"
    --output-dir "$out_dir"
    --frozen-beta-ckpt "$ckpt"
    --frozen-beta-head "$layer" "$head"
    --wandb-run-name "$name"
  )
  if [[ "$orientation" == "rev" ]]; then
    flags+=(--frozen-beta-rev)
  fi

  python -u "$CD/train_clean_aogpt.py" "${flags[@]}" 2>&1 | tee "$log_file"
  log "DONE $name"
}

log "=========================================="
log "from40k orientation controls"
log "RUN_SET=$RUN_SET GPU=$GPU WANDB_MODE=$WANDB_MODE BASE_OUT=$BASE_OUT"
log "=========================================="

case "$RUN_SET" in
  main)
    wait_for_gpu
    run_arm "from40k_seed124_l0h0_rev" "$GB_L0H0" 0 0 rev
    ;;
  remaining)
    wait_for_gpu
    run_arm "from40k_seed124_l0h2_fwd" "$GB_L0H2" 0 2 fwd
    run_arm "from40k_seed124_l0h0_fwd" "$GB_L0H0" 0 0 fwd
    ;;
  all)
    wait_for_gpu
    run_arm "from40k_seed124_l0h0_rev" "$GB_L0H0" 0 0 rev
    run_arm "from40k_seed124_l0h2_fwd" "$GB_L0H2" 0 2 fwd
    run_arm "from40k_seed124_l0h0_fwd" "$GB_L0H0" 0 0 fwd
    ;;
  *)
    echo "Unknown RUN_SET=$RUN_SET; expected main, remaining, or all" >&2
    exit 2
    ;;
esac

log "ALL REQUESTED ARMS DONE"
