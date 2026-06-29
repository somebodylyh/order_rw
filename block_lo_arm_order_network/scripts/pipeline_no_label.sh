#!/bin/bash
# Pipeline: audition (no-label small train) → CDL teacher → g_β pretrain → frozen_beta
# Usage: pipeline_no_label.sh <ckpt_path> <step> <output_tag> [gpu] [seed]
#
# Dynamically selects the best head via audition (small-train, no oracle τ),
# then runs the full chain: CDL → g_β → frozen_beta on a single GPU.
# ALL phases use the SAME seed for data-stream consistency.
#
# Example:
#   pipeline_no_label.sh probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt 20000 from20k cuda:0 2
set -euo pipefail

CKPT="$1"
STEP="$2"
TAG="$3"
GPU="${4:-cuda:0}"
SEED="${5:-2}"

CD="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$(dirname "$CKPT")"
SRC_NAME="$(basename "$SRC_DIR")"
AUDITION_DIR="$CD/probe_results/audition"
GBETA_LOG_DIR="$CD/batch_readout/logs"
TS="$(date +%Y%m%d_%H%M)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$AUDITION_DIR"

# Cross-steps: evaluate g_β generalization on other checkpoints
case "$STEP" in
  10000) CROSS_STEPS=(20000 40000) ;;
  20000) CROSS_STEPS=(10000 40000) ;;
  40000) CROSS_STEPS=(10000 20000) ;;
  *)     CROSS_STEPS=(10000 20000) ;;
esac

# ═══════════════════════════════════════════════════════════════
# Phase 0: Audition — no-label small train, find best head
# ═══════════════════════════════════════════════════════════════
log "=========================================="
log "[$TAG] Phase 0: Audition @ step=$STEP (small train, no-label)"
log "=========================================="

AUDITION_CSV="$AUDITION_DIR/audition_${TAG}_step${STEP}.csv"

# Candidate heads: all Layer0 + top performers from headscan
CANDIDATES=("0,0" "0,1" "0,2" "0,3" "0,4" "0,5" "0,6" "0,7" "1,0" "1,5" "2,3" "3,0")

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/scripts/audition_heads.py" \
  --ckpt "$CKPT" \
  --heads "${CANDIDATES[@]}" \
  --n-steps 100 \
  --device "$GPU" \
  --seed "$SEED" \
  --out "$AUDITION_CSV" \
  2>&1 | grep -E "Best:|L[0-9]H[0-9]|random baseline|Saved"

# Parse best head from CSV (sort by total_advantage descending)
BEST_LINE=$(tail -n +2 "$AUDITION_CSV" | sort -t',' -k6 -nr | head -1)
BEST_HEAD=$(echo "$BEST_LINE" | cut -d',' -f1)      # e.g. "L0H4"
BEST_ORIENT=$(echo "$BEST_LINE" | cut -d',' -f2)    # "fwd" or "rev"
BEST_LAYER=$(echo "$BEST_HEAD" | sed 's/L\([0-9]*\)H.*/\1/')
BEST_H=$(echo "$BEST_HEAD" | sed 's/L[0-9]*H\(.*\)/\1/')
BEST_TOTAL_ADV=$(echo "$BEST_LINE" | cut -d',' -f6)

log "[$TAG] Audition winner: $BEST_HEAD $BEST_ORIENT  total_adv=$BEST_TOTAL_ADV"

# ═══════════════════════════════════════════════════════════════
# Phase 1: CDL teacher with audition-selected head
# ═══════════════════════════════════════════════════════════════
log ""
log "[$TAG] Phase 1: CDL teacher (head=$BEST_HEAD, orient=$BEST_ORIENT)"

CDL_DIR="$CD/probe_results/cdl_teacher_${TAG}"
rm -rf "$CDL_DIR"
mkdir -p "$CDL_DIR"

CDL_REV_FLAG=""
if [ "$BEST_ORIENT" = "rev" ]; then
    CDL_REV_FLAG="--cdl-teacher-rev"
    log "  (reversed CDL order)"
fi

# Compute save steps relative to start
SAVE_STEPS=""
for offset in 10000 20000 30000 40000; do
    s=$((STEP + offset))
    if [ $s -le 60000 ]; then
        if [ -z "$SAVE_STEPS" ]; then
            SAVE_STEPS="$s"
        else
            SAVE_STEPS="$SAVE_STEPS,$s"
        fi
    fi
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
  --run-kind cdl_teacher \
  --data-source continuous \
  --seed "$SEED" --permute-seed "$SEED" \
  --cdl-teacher-head "$BEST_LAYER" "$BEST_H" \
  --cdl-teacher-none-mode b1 \
  --cdl-teacher-tau 1.0 \
  --cdl-teacher-refresh 10 \
  $CDL_REV_FLAG \
  --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
  --alpha-warmup-start 0 --alpha-ramp-from-resume \
  --max-steps 60000 \
  --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
  --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
  --device "$GPU" \
  --resume-ckpt "$CKPT" \
  --save-steps "$SAVE_STEPS" \
  --output-dir "$CDL_DIR" \
  2>&1 | tee "$CDL_DIR/train_log_${TS}.txt"

log "[$TAG] CDL teacher DONE"

# ═══════════════════════════════════════════════════════════════
# Phase 2: g_β pretrain on the same audition-selected head (B1/65-node)
# ═══════════════════════════════════════════════════════════════
log ""
log "[$TAG] Phase 2: g_β pretrain (head=$BEST_HEAD, B1/65-node)"

GBETA_OUT="$GBETA_LOG_DIR/gbeta_b1_${BEST_HEAD}_${TAG}"
rm -rf "$GBETA_OUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/scripts/run_phase33_gbeta.py" \
  --layer "$BEST_LAYER" --head "$BEST_H" \
  --none-mode b1 \
  --M 2000 --batch-size 16 --epochs 40 \
  --device "$GPU" \
  --seed "$SEED" \
  --ckpt-dir "$SRC_DIR" \
  --step "$STEP" \
  --cross-steps "${CROSS_STEPS[@]}" \
  --out "$GBETA_OUT" \
  2>&1 | tee "$GBETA_OUT.log"

# g_beta_best.pt is under <src_name>/full/
GBETA_CKPT="$GBETA_OUT/$SRC_NAME/full/g_beta_best.pt"
if [ ! -f "$GBETA_CKPT" ]; then
    log "ERROR: g_beta_best.pt not found at $GBETA_CKPT"
    log "Trying to find it..."
    GBETA_CKPT=$(find "$GBETA_OUT" -name "g_beta_best.pt" -type f | head -1)
    log "  found: $GBETA_CKPT"
fi
log "[$TAG] g_β pretrain DONE → $GBETA_CKPT"

# ═══════════════════════════════════════════════════════════════
# Phase 3: frozen_beta (g_β → AOGPT training)
# ═══════════════════════════════════════════════════════════════
log ""
log "[$TAG] Phase 3: frozen_beta (g_β hook → AOGPT)"

FB_DIR="$CD/probe_results/frozen_beta_${TAG}"
rm -rf "$FB_DIR"
mkdir -p "$FB_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u "$CD/train_clean_aogpt.py" \
  --run-kind frozen_beta \
  --data-source continuous \
  --seed "$SEED" --permute-seed "$SEED" \
  --frozen-beta-ckpt "$GBETA_CKPT" \
  --frozen-beta-head "$BEST_LAYER" "$BEST_H" \
  --frozen-beta-none-mode b1 \
  --frozen-beta-refresh 10 \
  --frozen-beta-mode argsort \
  --alpha-start 0.0 --alpha-target 1.0 --alpha-warmup-steps 5000 \
  --alpha-warmup-start 0 --alpha-ramp-from-resume \
  --max-steps 60000 \
  --lr 0.001 --min-lr 0.0001 --lr-decay-steps 50000 \
  --eval-interval 500 --log-interval 50 --stream-eval-windows 2000 \
  --device "$GPU" \
  --resume-ckpt "$CKPT" \
  --save-steps "$SAVE_STEPS" \
  --output-dir "$FB_DIR" \
  2>&1 | tee "$FB_DIR/train_log_${TS}.txt"

log "[$TAG] frozen_beta DONE"

log ""
log "=========================================="
log "[$TAG] FULL PIPELINE DONE — $(date)"
log "  Audition:    $AUDITION_CSV → winner $BEST_HEAD $BEST_ORIENT"
log "  CDL:         $CDL_DIR"
log "  g_β:         $GBETA_CKPT"
log "  frozen_beta: $FB_DIR"
log "=========================================="
