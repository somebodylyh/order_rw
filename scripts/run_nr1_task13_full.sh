#!/bin/bash
# NR-1 Task 13: full 10k dataset + full training run + §5.1 hard gates check
# + frozen-theta NLL diagnostic.
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt"
DATA="block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz"
DIV="block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.diversity.json"
OUT="block_lo_arm_order_network/neural_readout/checkpoints/full_10k"
NLL_OUT="${OUT}/frozen_nll_diag.json"
GATE_OUT="${OUT}/hard_gate_check.json"

echo "=== Task 13.1: build full 10k dataset (M=10000, seed=42, split=train) ==="
python -c "
import sys; sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import build_dataset_from_ckpt
build_dataset_from_ckpt(
    ckpt_path='${CKPT}',
    M=10000, seed=42, alpha_dep=0.5,
    out_path='${DATA}',
    device='cuda:0',
    split='train',
)
print('saved', '${DATA}')
"

echo "=== Task 13.2: compute teacher diversity on full set ==="
python -c "
import sys, json
sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import load_dataset
from neural_readout.diversity_stats import teacher_diversity
_, sigma, _, _, _ = load_dataset('${DATA}')
stats = teacher_diversity(sigma)
with open('${DIV}', 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
"

echo "=== Task 13.3: train full (8000/1000/1000, 40 epochs, batch=64) ==="
python -m neural_readout.train_nr1 \
    --dataset "${DATA}" \
    --out-dir "${OUT}" \
    --train-n 8000 --val-n 1000 --epochs 40 --batch 64 --lr 3e-4 \
    --device cuda:0

echo "=== Task 13.4: apply §5.1 hard gates ==="
python -c "
import json, torch
state = torch.load('${OUT}/g_beta_best.pt', map_location='cpu', weights_only=False)
m = state['metrics']
print('best epoch:', state['epoch'])
print('best metrics:', json.dumps(m, indent=2))

gates = {
    'kendall_tau':                {'value': m['kendall_tau'],               'threshold': 0.80, 'pass': m['kendall_tau'] >= 0.80},
    'pairwise_precedence_acc':    {'value': m['pairwise_precedence_acc'],   'threshold': 0.90, 'pass': m['pairwise_precedence_acc'] >= 0.90},
    'spearman_rho':               {'value': m['spearman_rho'],              'threshold': 0.85, 'pass': m['spearman_rho'] >= 0.85},
}
all_pass = all(g['pass'] for g in gates.values())
gates['NR1_HARD_GATE_PASS'] = all_pass
with open('${GATE_OUT}', 'w') as f:
    json.dump(gates, f, indent=2)
print(json.dumps(gates, indent=2))
if not all_pass:
    print('NR-1 §5.1 hard gates FAILED (see above). Subsequent steps will still run for diagnostic.')
"

echo "=== Task 13.5: post-selection frozen-theta NLL diagnostic (NOT a gate) ==="
python -m neural_readout.eval_frozen_nll \
    --ckpt "${CKPT}" \
    --g-beta "${OUT}/g_beta_best.pt" \
    --dataset "${DATA}" \
    --out "${NLL_OUT}" \
    --device cuda:0

echo "=== Task 13 DONE ==="
