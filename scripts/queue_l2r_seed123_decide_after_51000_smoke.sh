#!/usr/bin/env bash
# Wait for the 50k->51k diagnostic smoke, then choose the next L2R action:
#   - if lr=1e-5 at 51k stays within +0.02 of the 50k baseline, continue low-LR to 60k;
#   - otherwise start a fresh seed123 L2R 0->60k rerun.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

WAIT_PID=${WAIT_PID:-486997}
BASELINE=${BASELINE:-3.305407}
THRESHOLD=${THRESHOLD:-0.02}
POLL=${POLL:-60}
RUN_TS=$(date '+%Y%m%d_%H%M%S')

LOG_DIR=block_lo_arm_order_network/probe_results/logs
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/l2r_seed123_decide_after_51000_smoke_${RUN_TS}.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

latest_dir() {
    local pattern="$1"
    find /tmp -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' \
        | sort -nr \
        | awk 'NR==1{print $2}'
}

read_final_value() {
    local curve="$1"
    python3 - "$curve" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path) as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if not rows:
    raise SystemExit(f"empty curve: {path}")
row = rows[-1]
print(row["step"], row["val_ori_l2r_block"])
PY
}

log "Waiting for 51000 smoke PID ${WAIT_PID}"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL"
done
log "PID ${WAIT_PID} finished"

zero_dir=$(latest_dir 'order_lyu_l2r_seed123_zero_lr_51000_*')
low_dir=$(latest_dir 'order_lyu_l2r_seed123_lr1e5_51000_*')

if [ -z "${low_dir:-}" ] || [ ! -f "${low_dir}/eval_curve.tsv" ]; then
    log "ERROR: could not find lr1e5 smoke eval_curve.tsv"
    exit 1
fi

if [ -n "${zero_dir:-}" ] && [ -f "${zero_dir}/eval_curve.tsv" ]; then
    read -r zero_step zero_val < <(read_final_value "${zero_dir}/eval_curve.tsv")
    log "zero_lr final: step=${zero_step} val_ori_l2r_block=${zero_val} dir=${zero_dir}"
else
    log "WARN: zero_lr smoke result not found; continuing decision from lr1e5 only"
fi

read -r low_step low_val < <(read_final_value "${low_dir}/eval_curve.tsv")
log "lr1e5 final: step=${low_step} val_ori_l2r_block=${low_val} dir=${low_dir}"
log "decision baseline=${BASELINE} threshold=${THRESHOLD}"

decision=$(python3 - "$BASELINE" "$THRESHOLD" "$low_val" <<'PY'
import sys
baseline = float(sys.argv[1])
threshold = float(sys.argv[2])
value = float(sys.argv[3])
print("continue_low_lr" if value <= baseline + threshold else "fresh60k")
PY
)

if [ "$decision" = "continue_low_lr" ]; then
    log "Decision: no material 51k jump; continue low-LR 50k->60k."
    WAIT_PID=0 bash scripts/queue_l2r_seed123_low_lr_after_closed_loop.sh 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
else
    log "Decision: 51k still jumps; start fresh seed123 L2R 0->60k."
    bash scripts/queue_l2r_seed123_fresh60k.sh 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
fi

log "Decision branch exited rc=${rc}"
exit "$rc"
