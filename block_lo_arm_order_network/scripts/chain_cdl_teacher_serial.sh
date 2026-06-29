#!/bin/bash
# Serial chain on GPU 1: CDL teacher L0H0 (65-node B1) multi-start 10k/20k/40k → 60k
# Uses seed 124 (new, distinct from existing seeds 2, 42, 123).
set -euo pipefail

CD="$(dirname "$0")/.."
GPU="${1:-cuda:1}"
TS="$(date +%Y%m%d_%H%M)"

echo "================================================================"
echo "Serial CDL teacher chain (B1/65-node) — started $(date)"
echo "GPU=$GPU  CD=$CD"
echo "================================================================"

# Common flags for all runs
COMMON_FLAGS=(
    --run-kind cdl_teacher
    --data-source continuous
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin
    --seed 124 --permute-seed 124
    --cdl-teacher-head 0 0
    --cdl-teacher-none-mode b1
    --cdl-teacher-tau 1.0
    --cdl-teacher-refresh 10
    --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000
    --alpha-warmup-start 0 --alpha-ramp-from-resume
    --max-steps 60000
    --device "$GPU"
    --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000
    --eval-interval 500 --log-interval 50
    --stream-eval-windows 2000
)

BASE_CKPT="$CD/probe_results/clean_base_random_perm"

# ═══════════════════════════════════════════════════════════════════
# Run 1: CDL teacher L0H0 from 10k → 60k
# ═══════════════════════════════════════════════════════════════════
OUT1="$CD/probe_results/cdl_teacher_seed124_from10k_l0h0_b1"
echo ""
echo "=== [1/3] CDL teacher L0H0 B1: 10k → 60k ==="
mkdir -p "$OUT1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
    "${COMMON_FLAGS[@]}" \
    --resume-ckpt "$BASE_CKPT/ckpt_step10000.pt" \
    --save-steps "20000,30000,40000,50000,60000" \
    --output-dir "$OUT1" \
    2>&1 | tee "$OUT1/train_log_${TS}.txt"
echo "=== [1/3] from10k DONE at $(date) ==="

# ═══════════════════════════════════════════════════════════════════
# Run 2: CDL teacher L0H0 from 20k → 60k
# ═══════════════════════════════════════════════════════════════════
OUT2="$CD/probe_results/cdl_teacher_seed124_from20k_l0h0_b1"
echo ""
echo "=== [2/3] CDL teacher L0H0 B1: 20k → 60k ==="
mkdir -p "$OUT2"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
    "${COMMON_FLAGS[@]}" \
    --resume-ckpt "$BASE_CKPT/ckpt_step20000.pt" \
    --save-steps "30000,40000,50000,60000" \
    --output-dir "$OUT2" \
    2>&1 | tee "$OUT2/train_log_${TS}.txt"
echo "=== [2/3] from20k DONE at $(date) ==="

# ═══════════════════════════════════════════════════════════════════
# Run 3: CDL teacher L0H0 from 40k → 60k
# ═══════════════════════════════════════════════════════════════════
OUT3="$CD/probe_results/cdl_teacher_seed124_from40k_l0h0_b1"
echo ""
echo "=== [3/3] CDL teacher L0H0 B1: 40k → 60k ==="
mkdir -p "$OUT3"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
    "${COMMON_FLAGS[@]}" \
    --resume-ckpt "$BASE_CKPT/ckpt_step40000.pt" \
    --save-steps "50000,60000" \
    --output-dir "$OUT3" \
    2>&1 | tee "$OUT3/train_log_${TS}.txt"
echo "=== [3/3] from40k DONE at $(date) ==="

echo ""
echo "================================================================"
echo "All CDL teacher multi-start runs complete — $(date)"
echo "Outputs:"
echo "  $OUT1"
echo "  $OUT2"
echo "  $OUT3"
echo "================================================================"
