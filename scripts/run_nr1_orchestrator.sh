#!/bin/bash
# NR-1 fully-detached orchestrator: waits for the currently-running Task 12
# smoke (PID passed as $1), checks the smoke gate, then chains Tasks 13-16
# sequentially.
#
# Run with `setsid nohup bash ... &` so it survives parent-shell exit.
# Status is written to STATUS_FILE; chain script output goes to LOG_FILE.

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <task12_pid> <task12_output_file>" >&2
    exit 2
fi

TASK12_PID="$1"
TASK12_OUT="$2"

REPO=/home/admin/lyuyuhuan/order_lyu
STATUS_FILE="$REPO/analyses/neural_readout_nr1_2026-05-28/orchestrator_status.txt"
LOG_FILE="$REPO/analyses/neural_readout_nr1_2026-05-28/orchestrator_chain.log"

cd "$REPO"
mkdir -p "$(dirname "$STATUS_FILE")"

# Stage 1: wait for Task 12 to finish.
echo "[$(date -Is)] waiting for Task 12 pid=$TASK12_PID to exit" > "$STATUS_FILE"
while kill -0 "$TASK12_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date -Is)] Task 12 pid=$TASK12_PID exited" >> "$STATUS_FILE"

# Stage 2: check smoke gate
if grep -q "smoke gate PASS" "$TASK12_OUT"; then
    echo "[$(date -Is)] smoke gate PASS -- chaining Tasks 13-16" >> "$STATUS_FILE"
else
    echo "[$(date -Is)] smoke gate did NOT PASS -- chain aborted" >> "$STATUS_FILE"
    grep -E "(smoke gate|Traceback|Error|FAIL)" "$TASK12_OUT" | tail -20 >> "$STATUS_FILE"
    exit 1
fi

# Stage 3: chain Tasks 13-16
echo "[$(date -Is)] starting chain_13_to_16" >> "$STATUS_FILE"
if bash scripts/run_nr1_chain_13_to_16.sh > "$LOG_FILE" 2>&1; then
    echo "[$(date -Is)] chain_13_to_16 PASS" >> "$STATUS_FILE"
else
    RC=$?
    echo "[$(date -Is)] chain_13_to_16 FAILED (rc=$RC)" >> "$STATUS_FILE"
    tail -50 "$LOG_FILE" >> "$STATUS_FILE"
    exit $RC
fi

echo "[$(date -Is)] orchestrator DONE" >> "$STATUS_FILE"
