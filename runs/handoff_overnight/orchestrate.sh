#!/usr/bin/env bash
# Non-interactive overnight orchestrator for the order-signal handoff-circuit
# trajectory runs. Waits for GPU0 to free (the user's frozen_beta pid), runs an
# overhead calibration, then the 3-seed x 10k trajectory runs sequentially.
set -uo pipefail
cd /home/admin/lyuyuhuan/order_lyu
export PYTHONPATH=block_lo_arm_order_network
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0           # GPU0 only
GPU=0
WAIT_PID=67178                          # user's frozen_beta on GPU0
ROOT=runs/handoff_overnight
mkdir -p "$ROOT"

echo "=== orchestrator start $(date) ==="

# ── 1. wait for GPU0 to free ────────────────────────────────────────────────
while kill -0 "$WAIT_PID" 2>/dev/null; do
  echo "[wait $(date +%H:%M)] frozen_beta pid $WAIT_PID still running; sleeping 5m"
  sleep 300
done
echo "[wait $(date +%H:%M)] pid $WAIT_PID gone; waiting for GPU$GPU memory to drop"
for i in $(seq 1 120); do
  MEM=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  echo "[wait $(date +%H:%M)] GPU$GPU mem=${MEM}MiB"
  [ -n "$MEM" ] && [ "$MEM" -lt 3000 ] && { echo "[wait] GPU$GPU free"; break; }
  sleep 60
done

COMMON=(--run-kind baseline --data-source continuous
        --attn-trajectory --attn-composition-pairs all
        --eval-interval 200 --device cuda:0)
SAVE="--save-steps 0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000"

# ── 2. overhead calibration: seed2 -> 400 steps, samples=16 ─────────────────
echo "=== calibration seed2 -> 400 ($(date)) ==="
T0=$(date +%s)
python block_lo_arm_order_network/train_clean_aogpt.py \
  --seed 2 "${COMMON[@]}" --attn-trajectory-samples 16 $SAVE \
  --max-steps 400 --output-dir "$ROOT/calib_seed2" || { echo "CALIB FAILED"; exit 1; }
T1=$(date +%s)
CALIB_WALL=$((T1 - T0))

# decide snapshot sample count from measured overhead (spec: target<=5%, cap<=10%)
SAMPLES=$(python - "$ROOT/calib_seed2" "$CALIB_WALL" <<'PY'
import sys, json, glob, os
root, wall = sys.argv[1], float(sys.argv[2])
times = []
for f in sorted(glob.glob(os.path.join(root, "attention_trajectory", "summaries", "summary_step*.json"))):
    try: times.append(float(json.load(open(f)).get("extraction_time_s", 0.0)))
    except Exception: pass
n = max(len(times), 1)
snap_mean = sum(times) / n
train_wall = max(wall - sum(times), 1.0)
per200 = train_wall / 2.0            # 400 steps = 2 intervals (snapshots at 0/200/400)
overhead = snap_mean / per200 if per200 > 0 else 1.0
samples = 16 if overhead <= 0.10 else 8
sys.stderr.write(f"[calib] snapshots={n} snap_mean={snap_mean:.2f}s per200_train={per200:.1f}s "
                 f"overhead={overhead*100:.1f}% -> samples={samples}\n")
print(samples)
PY
)
echo "[calib] chosen attn-trajectory-samples=$SAMPLES (see [calib] line above for overhead)"

# ── 3. full 3-seed x 10k runs ───────────────────────────────────────────────
for s in 2 42 123; do
  echo "=== run seed $s -> 10000 (samples=$SAMPLES) ($(date)) ==="
  python block_lo_arm_order_network/train_clean_aogpt.py \
    --seed "$s" "${COMMON[@]}" --attn-trajectory-samples "$SAMPLES" $SAVE \
    --max-steps 10000 --output-dir "$ROOT/seed${s}" \
    || { echo "RUN seed $s FAILED ($(date))"; continue; }
  echo "=== seed $s done ($(date)) ==="
done

echo "=== orchestrator complete $(date) ==="
echo "trajectories in $ROOT/seed{2,42,123}/attention_trajectory/"
