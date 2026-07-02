#!/bin/bash
# NR-1 Task 12: smoke 1k dataset + smoke training run.
# Verifies the full pipeline runs end-to-end on the 5k clean_base_random_perm
# ckpt with M=1000. Smoke threshold (NOT the §5.1 hard gate): val tau > 0.5.
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt"
DATA="block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.npz"
DIV="block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.diversity.json"
OUT="block_lo_arm_order_network/neural_readout/checkpoints/smoke_1k"

echo "=== Task 12.1: build smoke 1k dataset (M=1000, seed=42, split=train) ==="
python -c "
import sys; sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import build_dataset_from_ckpt
build_dataset_from_ckpt(
    ckpt_path='${CKPT}',
    M=1000, seed=42, alpha_dep=0.5,
    out_path='${DATA}',
    device='cuda:0',
    split='train',
)
print('saved', '${DATA}')
"

echo "=== Task 12.2: compute teacher diversity ==="
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

echo "=== Task 12.3: train smoke (800/100/100 split, 30 epochs) ==="
python -m neural_readout.train_nr1 \
    --dataset "${DATA}" \
    --out-dir "${OUT}" \
    --train-n 800 --val-n 100 --epochs 30 --batch 32 --lr 3e-4 \
    --device cuda:0

echo "=== Task 12.4: print final smoke metrics from g_beta_best.pt ==="
python -c "
import json, torch
state = torch.load('${OUT}/g_beta_best.pt', map_location='cpu', weights_only=False)
print('best epoch:', state['epoch'])
print('best metrics:', json.dumps(state['metrics'], indent=2))
tau = state['metrics']['kendall_tau']
if tau < 0.5:
    raise SystemExit(f'smoke gate FAILED: val tau {tau:.4f} < 0.5')
print(f'smoke gate PASS (tau {tau:.4f} >= 0.5)')
"
echo "=== Task 12 DONE ==="
