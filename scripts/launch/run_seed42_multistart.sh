#!/bin/bash
# seed42 L0H2 multi-start frozen_beta chain
# ==========================================
# g_β status: ONLY g_β@10k exists (L0H2, Phase1.5=REVIEW, τ=0.569 < 0.6 gate)
#   - from10k: matched deployment (g_β@10k → hook@10k)
#   - from20k: cross-step deployment (g_β@10k → hook@20k)
#   - from40k: cross-step deployment (g_β@10k → hook@40k)
#
# Usage: bash scripts/run_seed42_multistart.sh [--dry-run]
# Set GPU=1 (or 0) to choose device.

set -euo pipefail
cd /home/admin/lyuyuhuan/order_lyu

# ── config ──────────────────────────────────────────────────────────────────
GPU="${GPU:-1}"
TB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin
VB=/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin

# g_β: seed42 L0H2 @10k (Phase1.5=REVIEW, τ=0.569)
GBETA=block_lo_arm_order_network/batch_readout/logs/phase33_gbeta_seed42_from10k_l0h2/random_baseline_continuous_jun05/full/g_beta_best.pt

# Baseline checkpoints (seed42, continuous, jun05)
BL_DIR=block_lo_arm_order_network/probe_results/random_baseline_continuous_jun05
OUT_BASE=block_lo_arm_order_network/probe_results

DRY_RUN="${1:-}"

# ── helper ──────────────────────────────────────────────────────────────────
die() { echo "FATAL: $*" >&2; exit 1; }

sanity_check() {
    local LABEL="$1"; shift
    echo "  [$LABEL]"

    # Check resume ckpt exists
    local RESUME="$1"
    test -f "$RESUME" || die "resume ckpt not found: $RESUME"
    echo "    resume_ckpt: $RESUME ($(du -h "$RESUME" | cut -f1))"

    # Check g_β exists
    test -f "$GBETA" || die "g_β not found: $GBETA"
    echo "    gbeta_ckpt:  $GBETA"

    # Check protocol
    python3 -c "
import torch, hashlib, json
ckpt = torch.load('$RESUME', map_location='cpu', weights_only=False)
p = ckpt['clean_protocol']
bp = p['block_perm_phys_to_model']
seed = ckpt.get('args',{}).get('permute_seed','?')
ds = ckpt.get('args',{}).get('data_source','?')
h = hashlib.sha256(str(bp).encode()).hexdigest()[:12]
print(f'    permute_seed={seed}  data_source={ds}  perm_hash={h}  n_blocks={len(bp)}')
" || die "failed to read protocol from $RESUME"

    # Print gbeta info
    python3 -c "
import torch, json
st = torch.load('$GBETA', map_location='cpu', weights_only=False)
cfg = st['config']
print(f'    gbeta_model={cfg[\"model_name\"]}  N={cfg[\"N\"]}  loss={cfg.get(\"loss_name\",\"?\")}')
" || die "failed to read gbeta config"

    # Print gbeta quality
    local REPORT_DIR
    REPORT_DIR=$(dirname "$GBETA")
    if [ -f "$REPORT_DIR/phase15_report.json" ]; then
        python3 -c "
import json
with open('$REPORT_DIR/phase15_report.json') as f:
    d = json.load(f)
print(f'    gbeta_tau(val)={d[\"splits\"][\"5k_val\"][\"gbeta\"][\"kendall_tau\"]:.3f}  verdict={d[\"verdict\"][\"result\"]}  M={d[\"config\"][\"M\"]}')
" || true
    fi

    echo "    seed=42  head=L0H2  refresh=10  alpha=0→1/warmup5k  max_steps=60000"
    echo "    output: $OUTDIR"
}

mark_done() {
    local OUTDIR="$1"
    mkdir -p "$OUTDIR"
    echo "DONE $(date)" > "$OUTDIR/DONE.txt"
}

is_done() {
    local OUTDIR="$1"
    test -f "$OUTDIR/DONE.txt"
}

run_one() {
    local START_STEP="$1"
    local RESUME_CKPT="$2"
    local OUTDIR="$3"
    local MATCHED="$4"  # "matched" or "cross-step"

    echo ""
    echo "================================================================"
    echo "RUN: seed42 from${START_STEP} [${MATCHED}]"
    echo "================================================================"

    sanity_check "seed42_from${START_STEP}" "$RESUME_CKPT"

    if is_done "$OUTDIR"; then
        echo "  ⏭ SKIP: already done ($(cat "$OUTDIR/DONE.txt"))"
        return 0
    fi

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "  [DRY-RUN] would run:"
        echo "    python train_clean_aogpt.py --run-kind frozen_beta ..."
        return 0
    fi

    echo "  ▶ START @ $(date)"
    python -u block_lo_arm_order_network/train_clean_aogpt.py \
        --run-kind frozen_beta \
        --data-source continuous \
        --train-bin "$TB" \
        --val-bin "$VB" \
        --seed 42 \
        --permute-seed 42 \
        --resume-ckpt "$RESUME_CKPT" \
        --frozen-beta-ckpt "$GBETA" \
        --frozen-beta-head 0 2 \
        --frozen-beta-mode argsort \
        --frozen-beta-tau 1.0 \
        --frozen-beta-refresh 10 \
        --alpha-start 0.0 \
        --alpha-target 1.0 \
        --alpha-warmup-steps 5000 \
        --alpha-warmup-start 0 \
        --alpha-ramp-from-resume \
        --max-steps 60000 \
        --save-steps "${SAVE_STEPS}" \
        --output-dir "$OUTDIR" \
        --device "cuda:${GPU}" \
        --lr 0.001 \
        --min-lr 0.0001 \
        --lr-decay-steps 50000 \
        --eval-interval 500 \
        --log-interval 50 \
        --stream-eval-windows 2000 \
        --batch-size 64 \
        --grad-accum 2 \
        2>&1 | tee "probe_results/frozen_beta_seed42_from${START_STEP}_l0h2.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        mark_done "$OUTDIR"
        echo "  ✅ DONE @ $(date)"
    else
        echo "  ❌ FAILED @ $(date)"
        return 1
    fi
}

# ── main ────────────────────────────────────────────────────────────────────
echo "============================================"
echo "seed42 L0H2 multi-start chain"
echo "GPU: cuda:${GPU}"
echo "g_β: 10k only (Phase1.5=REVIEW, τ=0.569)"
echo "Start: $(date)"
echo "============================================"

# Run 1: from10k (matched)
SAVE_STEPS="20000,30000,40000,50000,60000"
run_one 10000 \
    "${BL_DIR}/ckpt_step10000.pt" \
    "${OUT_BASE}/frozen_beta_seed42_from10k_l0h2" \
    "matched"

# Run 2: from20k (cross-step)
SAVE_STEPS="25000,30000,35000,40000,45000,50000,55000,60000"
run_one 20000 \
    "${BL_DIR}/ckpt_step20000.pt" \
    "${OUT_BASE}/frozen_beta_seed42_from20k_l0h2_xstep" \
    "cross-step"

# Run 3: from40k (cross-step)
SAVE_STEPS="45000,50000,55000,60000"
run_one 40000 \
    "${BL_DIR}/ckpt_step40000.pt" \
    "${OUT_BASE}/frozen_beta_seed42_from40k_l0h2_xstep" \
    "cross-step"

echo ""
echo "============================================"
echo "seed42 chain DONE: $(date)"
echo "Outputs:"
echo "  ${OUT_BASE}/frozen_beta_seed42_from10k_l0h2/eval_curve.tsv"
echo "  ${OUT_BASE}/frozen_beta_seed42_from20k_l0h2_xstep/eval_curve.tsv"
echo "  ${OUT_BASE}/frozen_beta_seed42_from40k_l0h2_xstep/eval_curve.tsv"
echo "============================================"
