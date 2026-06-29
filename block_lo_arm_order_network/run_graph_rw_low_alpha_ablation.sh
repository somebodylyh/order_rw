#!/bin/bash
# Graph-RW low-alpha ablation: 6 configs × 1500 steps, no refresh.
# Tests whether low-intensity RW signal avoids the U-shape degradation.
#
# Configs:
#   A: alpha_max=0.1, tau=0.05, lr=1e-5
#   B: alpha_max=0.2, tau=0.05, lr=1e-5
#   C: alpha_max=0.3, tau=0.05, lr=1e-5
#   D: alpha_max=0.2, tau=0.10, lr=1e-5
#   E: alpha_max=0.2, tau=0.20, lr=1e-5
#   F: alpha_max=0.2, tau=0.10, lr=5e-6

set -euo pipefail
set +e  # don't exit on summary extraction errors

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_OUT="$SCRIPT_DIR/probe_results/graph_rw_low_alpha"
mkdir -p "$BASE_OUT"

DEVICE="${1:-cuda:0}"
MAX_STEPS=1500
EVAL_INTERVAL=250
SEED=42

declare -A LABELS=( [A]="a0.1_t0.05_l1e-5"  [B]="a0.2_t0.05_l1e-5"
                    [C]="a0.3_t0.05_l1e-5"  [D]="a0.2_t0.1_l1e-5"
                    [E]="a0.2_t0.2_l1e-5"   [F]="a0.2_t0.1_l5e-6" )

declare -A ALPHAS=( [A]=0.1  [B]=0.2  [C]=0.3  [D]=0.2  [E]=0.2  [F]=0.2 )
declare -A TAUS=(   [A]=0.05 [B]=0.05 [C]=0.05 [D]=0.1  [E]=0.2  [F]=0.1  )
declare -A LRS=(    [A]=1e-5 [B]=1e-5 [C]=1e-5 [D]=1e-5 [E]=1e-5 [F]=5e-6 )

SUMMARY_TSV="$BASE_OUT/summary.tsv"
echo -e "label\talpha_max\ttau_step\tlr\tbest_step\tbest_rw_order\tfinal_rw_order\tbest_ori_l2r\tfinal_ori_l2r\tbest_model_order\tfinal_model_order\tori_l2r_delta" > "$SUMMARY_TSV"

for RUN in A B C D E F; do
    OUT_DIR="$BASE_OUT/${LABELS[$RUN]}"
    ALPHA="${ALPHAS[$RUN]}"
    TAU="${TAUS[$RUN]}"
    LR="${LRS[$RUN]}"
    MIN_LR=$(python3 -c "print($LR / 10)")

    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  RUN $RUN: alpha_max=$ALPHA, tau_step=$TAU, lr=$LR"
    echo "  Output: $OUT_DIR"
    echo "══════════════════════════════════════════════════════════════════"

    python -u "$SCRIPT_DIR/train_aogpt_graph_rw.py" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --max-steps "$MAX_STEPS" \
        --eval-interval "$EVAL_INTERVAL" \
        --alpha-target "$ALPHA" \
        --alpha-warmup "$MAX_STEPS" \
        --tau-start "$TAU" \
        --tau-step "$TAU" \
        --lr "$LR" \
        --min-lr "$MIN_LR" \
        --no-refresh \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/train_log.txt"

    # Extract summary from eval_curve.tsv
    EVAL="$OUT_DIR/eval_curve.tsv"
    if [ -f "$EVAL" ]; then
        # Best val_rw_order step (min of col 4)
        BEST_LINE=$(tail -n +2 "$EVAL" | sort -t$'\t' -k4 -n | head -1)
        FINAL_LINE=$(tail -n 1 "$EVAL")

        best_step=$(echo "$BEST_LINE" | cut -f1)
        best_rw=$(echo "$BEST_LINE" | cut -f4)
        best_ori=$(echo "$BEST_LINE" | cut -f7)
        best_model=$(echo "$BEST_LINE" | cut -f5)

        final_rw=$(echo "$FINAL_LINE" | cut -f4)
        final_ori=$(echo "$FINAL_LINE" | cut -f7)
        final_model=$(echo "$FINAL_LINE" | cut -f5)

        # ori_l2r degradation: final - best (negative = improved)
        ori_delta=$(awk "BEGIN {printf \"%.4f\", $final_ori - $best_ori}" 2>/dev/null || echo "N/A")

        echo -e "${LABELS[$RUN]}\t$ALPHA\t$TAU\t$LR\t$best_step\t$best_rw\t$final_rw\t$best_ori\t$final_ori\t$best_model\t$final_model\t$ori_delta" >> "$SUMMARY_TSV"
    else
        echo -e "${LABELS[$RUN]}\t$ALPHA\t$TAU\t$LR\tFAILED" >> "$SUMMARY_TSV"
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  ABLATION SUMMARY"
echo "══════════════════════════════════════════════════════════════════"
column -t -s $'\t' "$SUMMARY_TSV"
echo ""
echo "Summary saved to: $SUMMARY_TSV"
