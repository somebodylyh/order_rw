#!/usr/bin/env bash
# scripts/run_handoff_circuit_trajectory.sh
# Online order-signal handoff-circuit trajectory: 3 seeds, 0->10k, record every 200.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=block_lo_arm_order_network

SEEDS=(2 42 123)
COMMON=(--run-kind baseline --data-source continuous
        --attn-trajectory --attn-trajectory-samples 16
        --attn-composition-pairs all
        --eval-interval 200 --max-steps 10000)
# NOTE: --save-steps uses comma-separated list (confirmed from --help + DEFAULT_SAVE_STEPS).
SAVE_FLAG="--save-steps 0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000"
DEVICE=${DEVICE:-cuda:0}

# ── Calibration gate: seed 2 to 400 steps, measure snapshot overhead ──
echo "[calib] seed 2 -> 400 steps"
python block_lo_arm_order_network/train_clean_aogpt.py \
  --seed 2 "${COMMON[@]}" $SAVE_FLAG --max-steps 400 \
  --device "$DEVICE" --output-dir runs/handoff_calib_seed2 2>&1 | tee runs/handoff_calib_seed2.log

echo "[calib] inspect extraction_time_s in runs/handoff_calib_seed2/attention_trajectory/summaries/"
echo "[calib] If snapshot overhead > 10% of a 200-step interval, apply the spec degradation ladder before continuing."
read -r -p "Calibration acceptable? continue to full 3-seed run? [y/N] " ok
[[ "$ok" == "y" ]] || { echo "stopping at calibration gate"; exit 0; }

# ── Full runs ──
for s in "${SEEDS[@]}"; do
  echo "[run] seed $s -> 10000"
  python block_lo_arm_order_network/train_clean_aogpt.py \
    --seed "$s" "${COMMON[@]}" $SAVE_FLAG \
    --device "$DEVICE" --output-dir "runs/handoff_seed${s}" 2>&1 | tee "runs/handoff_seed${s}.log"
done
echo "done. trajectories in runs/handoff_seed{2,42,123}/attention_trajectory/"
