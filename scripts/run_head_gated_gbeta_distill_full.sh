#!/usr/bin/env bash
# Full offline distillation grid: train all variant×teacher combinations.
#
# Variants:       single_head, mean_head, head_gated(soft_all), head_gated(topk=2)
# Teachers:       manual_l0h2_cd, full_cdl, minus_d_only
# Total:          4 × 3 = 12 runs per (dataset, layer)
#
# Usage:
#   bash scripts/run_head_gated_gbeta_distill_full.sh \
#     /path/to/headset_l0_full_cdl_ckpt_step20000.npz \
#     /path/to/output_dir
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:?usage: $0 <dataset.npz> <out-dir>}"
OUT_BASE="${2:?usage: $0 <dataset.npz> <out-dir>}"

EPOCHS=40
BATCH_SIZE=64
DEVICE="${DEVICE:-cuda:0}"

echo "============================================"
echo "Full head-gated g_beta distillation grid"
echo "Dataset: $DATASET"
echo "Output:  $OUT_BASE"
echo "Epochs:  $EPOCHS"
echo "Device:  $DEVICE"
echo "============================================"

run_one() {
    local variant="$1"
    local extra_args="$2"
    local tag="$3"
    local out="$OUT_BASE/${tag}"
    echo ""
    echo "--- $tag ---"
    python3 "$ROOT/scripts/train_head_gated_gbeta.py" \
      --dataset "$DATASET" \
      --variant "$variant" \
      $extra_args \
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
      --out-dir "$out" \
      --device "$DEVICE" \
      2>&1 | tail -5
    echo "  saved to $out"
}

# --- single_head L0H2 ---
run_one single_head "--head-idx 2" "single_H2"

# --- mean_head ---
run_one mean_head "" "mean_all"

# --- head_gated soft_all ---
run_one head_gated "--gate-mode soft_all" "gated_soft"

# --- head_gated topk=2 ---
run_one head_gated "--gate-mode topk --topk 2" "gated_topk2"

echo ""
echo "============================================"
echo "Full grid complete. Results in $OUT_BASE"
echo "Next: python3 analyses/eval_head_gated_gbeta.py --runs $OUT_BASE/*"
echo "============================================"
