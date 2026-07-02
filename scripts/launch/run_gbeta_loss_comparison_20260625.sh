#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/block_lo_arm_order_network:${ROOT}/analyses:${PYTHONPATH:-}"

SOURCE_DATASET="${SOURCE_DATASET:-${ROOT}/block_lo_arm_order_network/batch_readout/checkpoints/l0_dynamic_gbeta_ds_step20k_M2000.npz}"
DATASET="${DATASET:-${ROOT}/block_lo_arm_order_network/batch_readout/checkpoints/l0_dynamic_gbeta_ds_step20k_M2000_order.npz}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/block_lo_arm_order_network/batch_readout/checkpoints/gbeta_loss_comparison_20260625}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-0}"

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${DATASET}" ]]; then
    python3 "${ROOT}/analyses/add_consensus_order_to_gbeta_dataset.py" \
        --src "${SOURCE_DATASET}" \
        --dst "${DATASET}"
fi

run_one() {
    local name="$1"
    local loss_type="$2"
    local temperature="$3"
    local out="${OUT_ROOT}/${name}"
    mkdir -p "${out}"

    python3 -m batch_readout.train_l0_dynamic_gbeta \
        --dataset "${DATASET}" \
        --out-dir "${out}" \
        --epochs "${EPOCHS}" \
        --batch-size 32 \
        --lr 3e-4 \
        --weight-decay 1e-2 \
        --lambda-aux 0.05 \
        --lambda-ent 0.001 \
        --min-gate-entropy 1.5 \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --loss-type "${loss_type}" \
        --rank-kl-temperature "${temperature}" \
        2>&1 | tee "${out}.log"

    python3 "${ROOT}/analyses/eval_l0_dynamic_gbeta.py" \
        --dataset "${DATASET}" \
        --ckpt "${out}/g_beta_best.pt" \
        --device "${DEVICE}" \
        --out-dir "${out}/evaluation"
}

run_one listmle listmle 4
run_one pairwise_bce pairwise_bce 4
for tau in 2 4 8 16; do
    run_one "rank_kl_tau${tau}" rank_kl "${tau}"
done

python3 "${ROOT}/analyses/summarize_gbeta_loss_comparison.py" \
    --root "${OUT_ROOT}"
