#!/usr/bin/env bash
set -euo pipefail

# Round 0 orchestrator: A_before → AOGPT cotrain (500 steps) → A_after → diff.
#
# Inputs (existing):
#   probe_results/A_train_10k.npy        — initial A from clean ych ckpt
#   probe_results/grpo_on_round2.pt      — current best ON
#   ~/ych/.../ckpt.pt                    — clean Random_CL AOGPT
#
# Outputs (round 0):
#   probe_results/round0/A_before_2k.npy
#   probe_results/round0/aogpt_round0.pt
#   probe_results/round0/A_after_2k.npy
#   probe_results/round0/diff_report.txt

cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda:0}"
ROUND_DIR="probe_results/round0"
NUM_SEQS="${NUM_SEQS:-2000}"
MAX_ITERS="${MAX_ITERS:-500}"
BATCH="${BATCH:-8}"

mkdir -p "$ROUND_DIR"

# ── 1. Slice A_before from existing 10k cache ───────────────────────────────
echo "[1/4] Slicing A_before (first $NUM_SEQS of A_train_10k.npy)..."
python -c "
import numpy as np
A = np.load('probe_results/A_train_10k.npy')
np.save('$ROUND_DIR/A_before_2k.npy', A[:$NUM_SEQS])
print(f'  saved A_before shape={A[:$NUM_SEQS].shape}')
"

# ── 2. Train AOGPT for 500 steps with ON-sampled per-batch orders ───────────
echo "[2/4] Training AOGPT ($MAX_ITERS steps, per-batch ON sampling)..."
python -u train_aogpt_cotrain.py \
    --device "$DEVICE" \
    --max-iters "$MAX_ITERS" \
    --batch-size "$BATCH" \
    --num-seqs "$NUM_SEQS" \
    --a-matrices "$ROUND_DIR/A_before_2k.npy" \
    --on-ckpt probe_results/grpo_on_round2.pt \
    --output-dir "$ROUND_DIR" \
    --output-name aogpt_round0.pt \
    --tau-init 1.0 --tau-final 0.3 \
    --log-interval 25 --eval-interval 250 \
    2>&1 | tee "$ROUND_DIR/train.log"

# ── 3. Re-extract A from updated AOGPT on the same 2k seqs ──────────────────
echo "[3/4] Re-extracting A_after from AOGPT_round0 (attention-based, 1 fwd/seq)..."
python -u extract_train_A.py \
    --n-chunks "$NUM_SEQS" \
    --ckpt "$(realpath $ROUND_DIR/aogpt_round0.pt)" \
    --output "$ROUND_DIR/A_after_2k.npy" \
    --device "$DEVICE" \
    2>&1 | tee "$ROUND_DIR/extract.log"

# ── 4. Diff report ──────────────────────────────────────────────────────────
echo "[4/4] Computing A diff..."
set +e
python -u diag_a_diff.py \
    --before "$ROUND_DIR/A_before_2k.npy" \
    --after  "$ROUND_DIR/A_after_2k.npy" \
    2>&1 | tee "$ROUND_DIR/diff_report.txt"
DIFF_RC=$?
set -e

echo
echo "======================================================="
echo "Round 0 complete. Reports:"
echo "  $ROUND_DIR/train.log"
echo "  $ROUND_DIR/extract.log"
echo "  $ROUND_DIR/diff_report.txt"
echo
if [ "$DIFF_RC" = "0" ]; then
    echo "GO Round 1 — A moved meaningfully, proceed to next round"
else
    echo "STOP — A barely moved; debug bottleneck before continuing"
fi
exit $DIFF_RC
