#!/usr/bin/env bash
# Queue model-frame diagnostics for feedback/controller checkpoint ladders.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

MIN_FREE_MB=${MIN_FREE_MB:-18000}
POLL=${POLL:-60}
RUN_TS=$(date '+%Y%m%d_%H%M%S')
LOG_DIR=block_lo_arm_order_network/probe_results/logs
OUT_DIR=reports/model_frame_feedback_diag_${RUN_TS}
mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG="${LOG_DIR}/model_frame_feedback_diag_${RUN_TS}.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

GPU=""
log "Waiting for a GPU with >= ${MIN_FREE_MB} MiB free"
while true; do
    for gpu in 0 1; do
        free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F', *' -v g="$gpu" '$1==g {print $2}')
        free_mb=${free_mb:-0}
        if [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
            GPU="$gpu"
            log "Selected GPU ${GPU} with ${free_mb} MiB free"
            break 2
        fi
    done
    log "No GPU has >= ${MIN_FREE_MB} MiB free; sleeping ${POLL}s"
    sleep "$POLL"
done

log "Output: ${OUT_DIR}"
log "Primary coordinate system: MODEL frame"
log "Primary metrics: tau_model_vs_semantic_path, tau_model_vs_identity"

CUDA_VISIBLE_DEVICES="$GPU" python3 -u analyses/diag_model_frame_feedback.py \
    --device cuda:0 \
    --M 12 \
    --batch-size 4 \
    --fwd-batch 4 \
    --tracked-head 0 2 \
    --out-dir "$OUT_DIR" \
    --ckpt random20=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt \
    --ckpt random30=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step30000.pt \
    --ckpt random40=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step40000.pt \
    --ckpt random50=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step50000.pt \
    --ckpt random60=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2_ext60k/ckpt_step60000.pt \
    --ckpt gbeta30=block_lo_arm_order_network/probe_results/frozen_beta_from20k_fixseed/ckpt_step30000.pt \
    --ckpt gbeta40=block_lo_arm_order_network/probe_results/frozen_beta_from20k_fixseed/ckpt_step40000.pt \
    --ckpt gbeta50=block_lo_arm_order_network/probe_results/frozen_beta_from20k_fixseed/ckpt_step50000.pt \
    --ckpt gbeta60=block_lo_arm_order_network/probe_results/frozen_beta_from20k_fixseed/ckpt_step60000.pt \
    --ckpt cdl30=block_lo_arm_order_network/probe_results/cdl_teacher_from20k_fixseed/ckpt_step30000.pt \
    --ckpt cdl40=block_lo_arm_order_network/probe_results/cdl_teacher_from20k_fixseed/ckpt_step40000.pt \
    --ckpt cdl50=block_lo_arm_order_network/probe_results/cdl_teacher_from20k_fixseed/ckpt_step50000.pt \
    --ckpt cdl60=block_lo_arm_order_network/probe_results/cdl_teacher_from20k_fixseed/ckpt_step60000.pt \
    --ckpt cdl_seed123_60=block_lo_arm_order_network/probe_results/cdl_teacher_seed123_from20k_l0h2/ckpt_step60000.pt \
    --ckpt cdl_seed123_reverse60=block_lo_arm_order_network/probe_results/cdl_teacher_seed123_from20k_l0h2_reverse/ckpt_step60000.pt \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
log "Finished rc=${rc}"
log "TSV: ${OUT_DIR}/model_frame_feedback.tsv"
log "Summary: ${OUT_DIR}/summary.json"
exit "$rc"
