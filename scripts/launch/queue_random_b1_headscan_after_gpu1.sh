#!/usr/bin/env bash
# Queue a fresh random baseline with lightweight B1 all-head signal tracking
# after the current GPU1 baseline process exits.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

WAIT_PID=${WAIT_PID:-827734}
GPU_ID=${GPU_ID:-1}
SEED=${SEED:-124}
OUT_DIR=${OUT_DIR:-block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed${SEED}}
WAIT_DONE_FILE=${WAIT_DONE_FILE:-block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2_ext60k/ckpt_step60000.pt}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
POLL=${POLL:-60}

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/queued_launch.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

log "queued random baseline B1 head scan"
log "waiting for ${WAIT_DONE_FILE} before using GPU${GPU_ID}"

while [ ! -s "$WAIT_DONE_FILE" ]; do
  sleep "$POLL"
done

log "${WAIT_DONE_FILE} exists; waiting for GPU${GPU_ID} free memory >= ${MIN_FREE_MB} MiB"
while true; do
  free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v gpu="$GPU_ID" '$1==gpu{print $2}')
  free_mb=${free_mb:-0}
  if [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
    break
  fi
  log "GPU${GPU_ID} free=${free_mb} MiB; sleeping ${POLL}s"
  sleep "$POLL"
done

log "launching on cuda:${GPU_ID}; output=${OUT_DIR}"

PYTHONPATH=block_lo_arm_order_network \
python -u block_lo_arm_order_network/train_clean_aogpt.py \
  --run-kind baseline --data-source continuous \
  --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
  --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
  --seed "$SEED" --permute-seed "$SEED" \
  --max-steps 60000 \
  --save-steps 0,1000,5000,10000,20000,30000,40000,50000,60000 \
  --output-dir "$OUT_DIR" \
  --device "cuda:${GPU_ID}" \
  --lr 1e-3 --min-lr 1e-4 --lr-decay-steps 50000 \
  --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
  --batch-size 64 --grad-accum 2 \
  --track-all-heads --track-head-interval 10 --track-head-m 4 \
  --track-head-none-mode predictor \
  2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "training exited rc=${rc}"
exit "$rc"
