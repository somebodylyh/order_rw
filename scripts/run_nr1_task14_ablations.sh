#!/bin/bash
# NR-1 Task 14: four MVP ablations (reverse / sym / row_shuffle / b_global).
# Each is a separate training run on the SAME full 10k dataset; only the
# input transform differs. Teacher labels are NEVER re-derived from the
# transformed B (the ablation studies whether the student can still recover
# the original teacher order when its input is degraded).
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu

DATA="block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz"

if [[ ! -f "${DATA}" ]]; then
    echo "ERROR: ${DATA} not found; run Task 13 first" >&2
    exit 1
fi

for AB in reverse sym row_shuffle b_global; do
    echo "=== Task 14.${AB}: train with --ablation ${AB} ==="
    python -m neural_readout.train_nr1 \
        --dataset "${DATA}" \
        --out-dir "block_lo_arm_order_network/neural_readout/checkpoints/ablation_${AB}" \
        --train-n 8000 --val-n 1000 --epochs 40 --batch 64 --lr 3e-4 \
        --device cuda:0 \
        --ablation "${AB}"
done

echo "=== Task 14.summary: aggregate ablation metrics ==="
python -c "
import json, torch, pathlib

base_dir = pathlib.Path('block_lo_arm_order_network/neural_readout/checkpoints')
summary = {}

# Identity (from Task 13)
identity_ckpt = base_dir / 'full_10k' / 'g_beta_best.pt'
if identity_ckpt.exists():
    state = torch.load(identity_ckpt, map_location='cpu', weights_only=False)
    summary['identity'] = state['metrics']
else:
    summary['identity'] = None

# Ablations
for ab in ['reverse', 'sym', 'row_shuffle', 'b_global']:
    p = base_dir / f'ablation_{ab}' / 'g_beta_best.pt'
    if p.exists():
        state = torch.load(p, map_location='cpu', weights_only=False)
        summary[ab] = state['metrics']
    else:
        summary[ab] = None

with open(base_dir / 'ablation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
"

echo "=== Task 14 DONE ==="
