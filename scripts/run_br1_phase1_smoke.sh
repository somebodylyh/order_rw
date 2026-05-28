#!/usr/bin/env bash
# BR-1 Phase-1 smoke: train FlattenReadout on the M=100 dataset built by
# Phase-0. Goal is pipeline health (pairwise_acc > 0.55 above chance);
# §5.1 hard gates are checked by Phase-1 full (Task 12) on M=1000.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

DS="block_lo_arm_order_network/batch_readout/data/text_5k_M100_B32_seed0.npz"
OUT="block_lo_arm_order_network/batch_readout/checkpoints/smoke_flatten_pairwise"
LOG="block_lo_arm_order_network/batch_readout/logs/phase1_smoke.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

if [ ! -f "$DS" ]; then
  echo "ERROR: missing dataset $DS — run scripts/run_br1_phase0_smoke.sh first" >&2
  exit 1
fi

GPU=$(MIN_FREE_MB=4000 bash scripts/_br1_wait_for_gpu.sh)
echo "[BR-1 phase 1 smoke] GPU=$GPU dataset=$DS out=$OUT"

CUDA_VISIBLE_DEVICES=$GPU python - <<PY 2>&1 | tee "$LOG"
from batch_readout.train_offline import train
out = train(
    dataset_path="$DS",
    model_name="flatten", loss_name="pairwise",
    N=64, batch_size=16, lr=3e-4, epochs=60, seed=0,
    out_dir="$OUT", device="cuda:0",
)
m = out["best_metrics"]
print(f"[BR-1] best epoch={out['best_epoch']} pairwise_acc={m['pairwise_acc']:.4f} "
      f"kendall_tau={m['kendall_tau']:.4f} spearman_rho={m['spearman_rho']:.4f} "
      f"top1={m['top1']:.4f} first3={m['first3']:.4f} "
      f"final_train_loss={out['final_train_loss']:.4f}")
PY
