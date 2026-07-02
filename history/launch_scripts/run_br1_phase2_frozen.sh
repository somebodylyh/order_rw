#!/usr/bin/env bash
# BR-1 Task 13: Phase-2 frozen-theta NLL diagnostic over 4 arms.
# Output: JSON dump of {arm: {mean_nll, delta_vs_random}} on the val split.
# Gate (spec §5.2): mlp_argsort must beat random by >= 0.005, else STOP.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
GBETA="block_lo_arm_order_network/batch_readout/checkpoints/g_beta_phase1_best.pt"
DS="block_lo_arm_order_network/batch_readout/data/text_5k_M1000_B32_seed0.npz"
LOG_DIR="block_lo_arm_order_network/batch_readout/logs"
mkdir -p "$LOG_DIR"

GPU=$(MIN_FREE_MB=6000 bash scripts/_br1_wait_for_gpu.sh)
echo "[BR-1 Phase-2] GPU=$GPU"

CUDA_VISIBLE_DEVICES=$GPU python - <<PY
import json
from batch_readout.eval_frozen_phase2 import compute_arms
out = compute_arms(
    ckpt_path="$CKPT",
    g_beta_path="$GBETA",
    dataset_path="$DS",
    arms=("random", "teacher", "mlp_argsort", "mlp_sample"),
    tau=0.3,
    seed=0,
    device="cuda:0",
)
print(json.dumps(out, indent=2))
PY
