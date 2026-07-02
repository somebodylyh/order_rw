#!/bin/bash
# Chain-run shuffle granularity experiments: 32 → 64 → 128 block
# Usage: bash scripts/run_shuffle_granularity_chain.sh [device_id]
set -euo pipefail

DEVICE_ID="${1:-1}"
REPO=/home/admin/lyuyuhuan/order_lyu
cd "$REPO"

TRAIN=block_lo_arm_order_network/train_clean_aogpt.py
COMMON="--run-kind baseline --data-source chunks \
  --n-layer 4 --n-head 8 --n-embd 384 \
  --batch-size 64 --grad-accum 2 --max-steps 10000 \
  --lr 1e-3 --min-lr 1e-4 --lr-decay-steps 10000 \
  --save-steps 0,1000,5000,10000 \
  --eval-interval 1000 --log-interval 100 --seed 42 \
  --device cuda:0"

for GRAN in 32 64 128; do
  OUT_DIR="block_lo_arm_order_network/probe_results/shuffle_gran_${GRAN}"
  mkdir -p "$OUT_DIR"

  # Skip if already completed
  if [ -f "${OUT_DIR}/ckpt_step10000.pt" ]; then
    echo "[$(date)] SKIP gran=${GRAN}: ckpt_step10000.pt already exists"
    continue
  fi

  echo "============================================"
  echo "[$(date)] Starting gran=${GRAN} (groups of $((256 / GRAN)) tokens)"
  echo "Output: ${OUT_DIR}"
  echo "GPU: cuda:${DEVICE_ID}"
  echo "============================================"

  CUDA_VISIBLE_DEVICES="${DEVICE_ID}" PYTHONPATH=block_lo_arm_order_network \
    python "$TRAIN" \
      --shuffle-granularity "$GRAN" \
      --output-dir "$OUT_DIR" \
      $COMMON \
      2>&1 | tee "${OUT_DIR}/train.log"

  echo ""
  echo "[$(date)] gran=${GRAN} training complete."

  # Run head scan
  CKPT="${OUT_DIR}/ckpt_step10000.pt"
  if [ -f "$CKPT" ]; then
    echo "[$(date)] Running per_head_order_scan for gran=${GRAN}..."
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" PYTHONPATH=block_lo_arm_order_network \
      python block_lo_arm_order_network/per_head_order_scan.py \
        --ckpt "$CKPT" \
        --M 100 --batch-size 32 --seed 42 \
        --device cuda:0 \
        --alpha-dep 0.5 --none-mode b0 \
        --out "${OUT_DIR}/head_scan_10k.json" \
        2>&1 | tee -a "${OUT_DIR}/scan.log"
    echo "[$(date)] Scan complete for gran=${GRAN}"
  fi
done

echo ""
echo "[$(date)] All shuffle granularity experiments complete!"
echo "Results:"
for GRAN in 32 64 128; do
  OUT_DIR="block_lo_arm_order_network/probe_results/shuffle_gran_${GRAN}"
  echo "  ${GRAN}-block: ${OUT_DIR}/"
  if [ -f "${OUT_DIR}/head_scan_10k.json" ]; then
    python3 -c "
import json
d = json.load(open('${OUT_DIR}/head_scan_10k.json'))
hb = d['heavy_baseline']
print(f'    heavy tau_vs_l2r: {hb[\"tau_vs_l2r\"]:.4f}')
top = d['per_head_layer_sorted_by_abs_tau_vs_l2r'][:3]
for h in top:
    print(f'    L{h[\"layer\"]}H{h[\"head\"]}: tau={h[\"tau_vs_l2r\"]:.4f}  pairwise={h[\"mean_pairwise_tau\"]:.4f}')
"
  fi
done
