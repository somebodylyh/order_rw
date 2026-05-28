#!/bin/bash
# NR-1 Task 15: cross-ckpt diagnostic (10k/20k/30k/40k/50k/60k).
# NOT a hard gate — diagnostic for NR-4 generalization. Builds 1k-sample
# (B, sigma_T) datasets on each cross-ckpt, evaluates frozen g_beta from
# Task 13, reports per-ckpt tau + diversity.
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu

CKPT_DIR="block_lo_arm_order_network/probe_results/clean_base_random_perm"
GBETA="block_lo_arm_order_network/neural_readout/checkpoints/full_10k/g_beta_best.pt"
OUT_DIR="block_lo_arm_order_network/neural_readout/checkpoints/full_10k"

if [[ ! -f "${GBETA}" ]]; then
    echo "ERROR: ${GBETA} not found; run Task 13 first" >&2
    exit 1
fi

for STEP in 10000 20000 30000 40000 50000 60000; do
    DATA="block_lo_arm_order_network/neural_readout/data/text_cross_${STEP}_1k.npz"
    echo "=== Task 15.${STEP}: build 1k dataset from ckpt_step${STEP}.pt ==="
    python -c "
import sys
sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import build_dataset_from_ckpt
build_dataset_from_ckpt(
    ckpt_path='${CKPT_DIR}/ckpt_step${STEP}.pt',
    M=1000, seed=42, alpha_dep=0.5,
    out_path='${DATA}',
    device='cuda:0',
    split='train',
)
"
done

echo "=== Task 15.eval: evaluate frozen g_beta on each cross-ckpt + diversity ==="
python -c "
import sys, json, numpy as np, torch
sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import load_dataset
from neural_readout.graph_transformer_readout import GraphTransformerReadout
from neural_readout.eval_metrics import compute_matching_metrics
from neural_readout.diversity_stats import teacher_diversity

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
g = GraphTransformerReadout(N=64).to(dev)
state = torch.load('${GBETA}', map_location=dev, weights_only=False)
g.load_state_dict(state['model'])
g.eval()

results = {}
for step in [10000, 20000, 30000, 40000, 50000, 60000]:
    path = f'block_lo_arm_order_network/neural_readout/data/text_cross_{step}_1k.npz'
    B, sigma, rank, _ci, _split = load_dataset(path)
    with torch.no_grad():
        scores = g(torch.from_numpy(B).to(dev)).cpu().numpy()
    metrics = compute_matching_metrics(scores=scores, rank=rank)
    diversity = teacher_diversity(sigma)
    results[str(step)] = {'metrics': metrics, 'teacher_diversity': diversity}

with open('${OUT_DIR}/cross_ckpt_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
"

echo "=== Task 15 DONE ==="
