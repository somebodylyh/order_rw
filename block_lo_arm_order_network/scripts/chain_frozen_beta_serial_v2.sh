#!/bin/bash
# Serial chain on GPU 0 after g_β seed42 L0H4 completes.
set -euo pipefail

CD="$(dirname "$0")/.."
GPU="${1:-cuda:0}"
TS="$(date +%Y%m%d_%H%M)"

echo "================================================================"
echo "Serial frozen_beta chain — started $(date)"
echo "GPU=$GPU"
echo "================================================================"

# ═══════════════════════════════════════════════════════════════════
# Step 1: frozen_beta seed2 L0H2 from 10k → 60k
# ═══════════════════════════════════════════════════════════════════
GBETA_S2="$CD/batch_readout/logs/phase33_gbeta_seed2_from10k_l0h2/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt"
RESUME_S2="$CD/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step10000.pt"
OUT_S2="$CD/probe_results/frozen_beta_seed2_from10k_l0h2"

echo ""
echo "=== [1/4] frozen_beta seed2 L0H2: 10k → 60k ==="
mkdir -p "$OUT_S2"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --seed 123 --permute-seed 123 \
    --resume-ckpt "$RESUME_S2" \
    --frozen-beta-ckpt "$GBETA_S2" --frozen-beta-head 0 2 \
    --frozen-beta-mode argsort --frozen-beta-tau 1.0 --frozen-beta-refresh 10 \
    --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
    --alpha-warmup-start 0 --alpha-ramp-from-resume \
    --max-steps 60000 --save-steps "20000,30000,40000,50000,60000" \
    --output-dir "$OUT_S2" --device "$GPU" \
    --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
    --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
    2>&1 | tee "$OUT_S2/train_log_${TS}.txt"
echo "=== [1/4] seed2 from10k DONE at $(date) ==="

# ═══════════════════════════════════════════════════════════════════
# Step 2: frozen_beta seed42 L0H4 from 10k → 60k
# ═══════════════════════════════════════════════════════════════════
GBETA_42="$CD/batch_readout/logs/phase33_gbeta_seed42_from10k_l0h4/random_baseline_continuous_jun05/full/g_beta_best.pt"
RESUME_42_10K="$CD/probe_results/random_baseline_continuous_jun05/ckpt_step10000.pt"
OUT_42_10="$CD/probe_results/frozen_beta_seed42_from10k_l0h4"

echo ""
echo "=== [2/4] frozen_beta seed42 L0H4: 10k → 60k ==="
mkdir -p "$OUT_42_10"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --seed 42 --permute-seed 42 \
    --resume-ckpt "$RESUME_42_10K" \
    --frozen-beta-ckpt "$GBETA_42" --frozen-beta-head 0 4 \
    --frozen-beta-mode argsort --frozen-beta-tau 1.0 --frozen-beta-refresh 10 \
    --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
    --alpha-warmup-start 0 --alpha-ramp-from-resume \
    --max-steps 60000 --save-steps "20000,30000,40000,50000,60000" \
    --output-dir "$OUT_42_10" --device "$GPU" \
    --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
    --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
    2>&1 | tee "$OUT_42_10/train_log_${TS}.txt"
echo "=== [2/4] seed42 from10k DONE at $(date) ==="

# ═══════════════════════════════════════════════════════════════════
# Step 3: frozen_beta seed42 L0H4 from 20k → 60k
# ═══════════════════════════════════════════════════════════════════
RESUME_42_20K="$CD/probe_results/random_baseline_continuous_jun05/ckpt_step20000.pt"
OUT_42_20="$CD/probe_results/frozen_beta_seed42_from20k_l0h4"

echo ""
echo "=== [3/4] frozen_beta seed42 L0H4: 20k → 60k ==="
mkdir -p "$OUT_42_20"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --seed 42 --permute-seed 42 \
    --resume-ckpt "$RESUME_42_20K" \
    --frozen-beta-ckpt "$GBETA_42" --frozen-beta-head 0 4 \
    --frozen-beta-mode argsort --frozen-beta-tau 1.0 --frozen-beta-refresh 10 \
    --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
    --alpha-warmup-start 0 --alpha-ramp-from-resume \
    --max-steps 60000 --save-steps "30000,40000,50000,60000" \
    --output-dir "$OUT_42_20" --device "$GPU" \
    --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
    --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
    2>&1 | tee "$OUT_42_20/train_log_${TS}.txt"
echo "=== [3/4] seed42 from20k DONE at $(date) ==="

# ═══════════════════════════════════════════════════════════════════
# Step 4: frozen_beta seed42 L0H4 from 40k → 60k
# ═══════════════════════════════════════════════════════════════════
RESUME_42_40K="$CD/probe_results/random_baseline_continuous_jun05/ckpt_step40000.pt"
OUT_42_40="$CD/probe_results/frozen_beta_seed42_from40k_l0h4"

echo ""
echo "=== [4/4] frozen_beta seed42 L0H4: 40k → 60k ==="
mkdir -p "$OUT_42_40"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
    --run-kind frozen_beta --data-source continuous \
    --train-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin \
    --val-bin /home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin \
    --seed 42 --permute-seed 42 \
    --resume-ckpt "$RESUME_42_40K" \
    --frozen-beta-ckpt "$GBETA_42" --frozen-beta-head 0 4 \
    --frozen-beta-mode argsort --frozen-beta-tau 1.0 --frozen-beta-refresh 10 \
    --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
    --alpha-warmup-start 0 --alpha-ramp-from-resume \
    --max-steps 60000 --save-steps "50000,60000" \
    --output-dir "$OUT_42_40" --device "$GPU" \
    --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
    --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
    2>&1 | tee "$OUT_42_40/train_log_${TS}.txt"
echo "=== [4/4] seed42 from40k DONE at $(date) ==="

echo ""
echo "================================================================"
echo "All frozen_beta chain completed — $(date)"
echo "================================================================"
