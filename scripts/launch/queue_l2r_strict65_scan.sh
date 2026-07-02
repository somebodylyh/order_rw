#!/usr/bin/env bash
# Queue: wait for GPU 1 or 0 to free, then run strict 65-node scan on L2R baseline.
set -euo pipefail
REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

LOG="reports/l2r_baseline_strict_65_scan/queue.log"
mkdir -p reports/l2r_baseline_strict_65_scan

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# Wait for any GPU with >= 20000 MB free
while true; do
    for gpu in 0 1; do
        free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F', *' -v gpu="$gpu" '$1==gpu{print $2}')
        free_mb=${free_mb:-0}
        if [ "$free_mb" -ge 20000 ]; then
            log "GPU $gpu free=${free_mb} MiB, launching"
            GPU=$gpu
            break 2
        fi
    done
    log "no GPU with >= 20000 MB free; sleeping 60s"
    sleep 60
done

# ── Scan L2R-trained baseline ──
CKPT="block_lo_arm_order_network/probe_results/l2r_continuous_seed2/ckpt_step60000.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="block_lo_arm_order_network/probe_results/l2r_continuous_seed123/ckpt_step50000.pt"
    log "using seed-123 L2R ckpt"
fi
log "ckpt: $CKPT"

python -u scripts/search_strict_label_free_65.py \
  --ckpt "$CKPT" \
  --out-dir reports/l2r_baseline_strict_65_scan \
  --device "cuda:$GPU" \
  --M 40 --batch-size 4 --fwd-batch 4 \
  --seed 42 \
  --methods L C-D+L \
  --control-seeds 0 1 2 3 4 \
  2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "L2R baseline strict 65-node scan exited rc=$rc"
log "DONE at $(date)"
