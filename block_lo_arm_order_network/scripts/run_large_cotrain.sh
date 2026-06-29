#!/bin/bash
# Large-scale ON-refresh co-training + fixed-teacher baseline
# Usage: bash scripts/run_large_cotrain.sh

set -e
ROOT=$(dirname "$0")/..
cd "$ROOT/.."  # back to order_lyu root
SCRIPT="block_lo_arm_order_network/cotrain_aogpt_on.py"
OUT_BASE="block_lo_arm_order_network/probe_results/cotrain_large"
LOG_DIR="$OUT_BASE/logs"

mkdir -p "$OUT_BASE" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_one() {
    local method=$1    # "refresh" or "fixed"
    local seed=$2
    local gpu=$3
    local steps=${4:-9000}
    local refresh_every=${5:-1500}

    local tag="${method}_s${seed}"
    local out_dir="${OUT_BASE}/${tag}"
    local log_file="${LOG_DIR}/${tag}_${TIMESTAMP}.log"

    # Build args
    local extra_args=""
    if [ "$method" = "fixed" ]; then
        extra_args="--no-refresh"
    fi

    echo "=== Launching $tag on GPU $gpu, $steps steps ==="
    echo "  output: $out_dir"
    echo "  log: $log_file"

    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
        --output-dir "$out_dir" \
        --max-total-steps "$steps" \
        --refresh-every "$refresh_every" \
        --phase1-epochs 10 \
        --phase3-bc-epochs 3 \
        --phase3-extract-subset 2000 \
        --batch-size 4 --grad-accum 4 \
        --lr 3e-5 --alpha-target 0.7 --alpha-warmup 1500 \
        --temperature 1.0 --top-k 4 \
        --eval-interval 100 --log-interval 20 \
        --seed "$seed" --device cuda:0 \
        $extra_args \
        2>&1 | tee "$log_file"
}

# ── Main ──

echo "=============================================="
echo "Large-Scale Co-Training Suite"
echo "Methods: refresh (3 seeds) + fixed baseline (1 seed)"
echo "Steps: 9000 each"
echo "Timestamp: $TIMESTAMP"
echo "=============================================="

# Phase A: ON-refresh-sample, 2 seeds on 2 GPUs in parallel
echo ""
echo "=== Phase A: ON-refresh-sample (seed 42, 123) ==="

run_one "refresh" 42 0 9000 1500 &
PID1=$!

run_one "refresh" 123 1 9000 1500 &
PID2=$!

echo "Waiting for Phase A to complete..."
wait $PID1 $PID2
echo "Phase A done."

# Phase B: ON-refresh-sample seed 456 + fixed baseline seed 42
echo ""
echo "=== Phase B: ON-refresh-sample (seed 456) + fixed baseline (seed 42) ==="

run_one "refresh" 456 0 9000 1500 &
PID3=$!

run_one "fixed" 42 1 9000 1500 &
PID4=$!

echo "Waiting for Phase B to complete..."
wait $PID3 $PID4
echo "Phase B done."

echo ""
echo "=== All runs complete ==="
echo "Results in: $OUT_BASE/"
ls -d "$OUT_BASE"/*/
