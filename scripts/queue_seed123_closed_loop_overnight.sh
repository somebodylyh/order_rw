#!/usr/bin/env bash
# Full seed-123 single-seed closed-loop overnight pipeline.
#
# Produces the matched panel:
#   random 60k, L2R 60k, g_beta from10/20/40 60k,
#   CDL from10/20/40 60k, and CDL reverse-from20k negative control.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

RUN_TS=$(date '+%Y%m%d_%H%M%S')
LOG_DIR=block_lo_arm_order_network/probe_results/logs
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/seed123_closed_loop_overnight_${RUN_TS}.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

run_stage() {
    local label="$1"
    shift
    log "===== START ${label} ====="
    set +e
    "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    log "===== END ${label} rc=${rc} ====="
    return "$rc"
}

log "Seed123 closed-loop overnight pipeline"
log "Repo: $REPO"
log "Log:  $LOG"

run_stage "extend L2R seed123 50k->60k" \
    bash scripts/queue_extend_l2r_seed123_50k60k.sh

run_stage "CDL seed123 missing from20k/from40k" \
    bash scripts/queue_cdl_seed123_missing_20k40k.sh

run_stage "CDL seed123 reverse from20k negative control" \
    bash scripts/queue_cdl_seed123_reverse_from20k.sh

run_stage "redraw matched CDL/g_beta plots" \
    python3 analyses/plot_cdl_gbeta_matched.py

log "Closed-loop pipeline finished."
log "Plots:"
log "  /home/admin/lyuyuhuan/order_lyu/cdl_matched_by_seed.png"
log "  /home/admin/lyuyuhuan/order_lyu/gbeta_matched_by_seed.png"
log "  /home/admin/lyuyuhuan/order_lyu/cdl_gbeta_matched_by_seed.png"
log "Summary:"
log "  /home/admin/lyuyuhuan/order_lyu/cdl_gbeta_matched_by_seed.json"
