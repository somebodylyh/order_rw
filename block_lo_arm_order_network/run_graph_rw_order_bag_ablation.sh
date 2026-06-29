#!/bin/bash
# Graph-RW order-bag ablation.
#
# Goal:
#   Compare low-variance RW loss mixing against alpha=0 random continuation.
#   Refresh is disabled so this isolates the training distribution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_OUT="$SCRIPT_DIR/probe_results/graph_rw_order_bag"
mkdir -p "$BASE_OUT"

DEVICE="${1:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
ALPHA_WARMUP="${WARMUP_STEPS:-${ALPHA_WARMUP:-1500}}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
RUNS="${RUNS:-control,A,B,C,D,E}"

SUMMARY_TSV="$BASE_OUT/summary.tsv"
DELTA_TSV="$BASE_OUT/delta_vs_alpha0.tsv"

echo -e "label\talpha_max\tbag_k\ttau_step\ttop_k\tlr\tbest_step\tbest_rw_order\tfinal_rw_order\tbest_ori_l2r\tfinal_ori_l2r\tbest_model_order\tfinal_model_order" > "$SUMMARY_TSV"

run_one() {
    local label="$1"
    local alpha="$2"
    local bag_k="$3"
    local tau="$4"
    local top_k="$5"

    local out_dir="$BASE_OUT/$label"
    mkdir -p "$out_dir"

    echo ""
    echo "=================================================================="
    echo "  RUN $label: alpha=$alpha, K=$bag_k, tau=$tau, top_k=$top_k, lr=$LR"
    echo "  Output: $out_dir"
    echo "=================================================================="

    python -u "$SCRIPT_DIR/train_aogpt_graph_rw.py" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --max-steps "$MAX_STEPS" \
        --eval-interval "$EVAL_INTERVAL" \
        --alpha-target "$alpha" \
        --alpha-warmup "$ALPHA_WARMUP" \
        --tau-start "$tau" \
        --tau-step "$tau" \
        --rw-top-k "$top_k" \
        --mixing-mode loss_bag \
        --rw-order-bag-k "$bag_k" \
        --lr "$LR" \
        --min-lr "$MIN_LR" \
        --no-refresh \
        --output-dir "$out_dir" \
        2>&1 | tee "$out_dir/train_log.txt"

    local eval_file="$out_dir/eval_curve.tsv"
    local best_line
    local final_line
    best_line=$(tail -n +2 "$eval_file" | sort -t$'\t' -k4 -n | head -1)
    final_line=$(tail -n 1 "$eval_file")

    local best_step best_rw best_ori best_model final_rw final_ori final_model
    best_step=$(echo "$best_line" | cut -f1)
    best_rw=$(echo "$best_line" | cut -f4)
    best_model=$(echo "$best_line" | cut -f5)
    best_ori=$(echo "$best_line" | cut -f7)
    final_rw=$(echo "$final_line" | cut -f4)
    final_model=$(echo "$final_line" | cut -f5)
    final_ori=$(echo "$final_line" | cut -f7)

    echo -e "$label\t$alpha\t$bag_k\t$tau\t$top_k\t$LR\t$best_step\t$best_rw\t$final_rw\t$best_ori\t$final_ori\t$best_model\t$final_model" >> "$SUMMARY_TSV"
}

run_selected() {
    local run_id="$1"
    case "$run_id" in
        control) run_one "control_a0" 0.0 1 0.1 4 ;;
        A) run_one "bag_A_a0.2_k2_t0.1_top4" 0.2 2 0.1 4 ;;
        B) run_one "bag_B_a0.3_k2_t0.1_top4" 0.3 2 0.1 4 ;;
        C) run_one "bag_C_a0.2_k4_t0.1_top4" 0.2 4 0.1 4 ;;
        D) run_one "bag_D_a0.2_k2_t0.2_top4" 0.2 2 0.2 4 ;;
        E) run_one "bag_E_a0.2_k2_t0.1_top8" 0.2 2 0.1 8 ;;
        *) echo "Unknown RUNS entry: $run_id" >&2; exit 2 ;;
    esac
}

IFS=',' read -ra SELECTED_RUNS <<< "$RUNS"
for run_id in "${SELECTED_RUNS[@]}"; do
    run_selected "$run_id"
done

python - "$BASE_OUT" "$DELTA_TSV" <<'PY'
import csv
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
control_path = base / "control_a0" / "eval_curve.tsv"

def read_curve(path):
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {int(r["step"]): r for r in rows}

control = read_curve(control_path)
metric_names = [
    "val_rw_order",
    "val_model_order",
    "val_unstructured_order",
    "val_ori_l2r",
]

with out_path.open("w", newline="") as f:
    fieldnames = [
        "label",
        "step",
        "alpha",
        *metric_names,
        *(f"delta_{m}_vs_alpha0" for m in metric_names),
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for run_dir in sorted(base.iterdir()):
        eval_path = run_dir / "eval_curve.tsv"
        if not eval_path.exists() or run_dir.name == "control_a0":
            continue
        for step, row in sorted(read_curve(eval_path).items()):
            if step not in control:
                continue
            out = {
                "label": run_dir.name,
                "step": step,
                "alpha": row["alpha"],
            }
            for metric in metric_names:
                value = float(row[metric])
                ctrl = float(control[step][metric])
                out[metric] = f"{value:.6f}"
                out[f"delta_{metric}_vs_alpha0"] = f"{value - ctrl:.6f}"
            writer.writerow(out)
PY

echo ""
echo "Summary: $SUMMARY_TSV"
echo "Delta vs alpha=0: $DELTA_TSV"
