#!/usr/bin/env bash
# BR-1 Phase-1 full: build the M=1000 B=32 dataset and train all four
# (model, loss) configurations sequentially on the same GPU. Picks the
# winner by val pairwise_acc and symlinks it to g_beta_phase1_best.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="block_lo_arm_order_network:${PYTHONPATH:-}"

CKPT="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt"
DS="block_lo_arm_order_network/batch_readout/data/text_5k_M1000_B32_seed0.npz"
LOG_DIR="block_lo_arm_order_network/batch_readout/logs"
CKPT_DIR="block_lo_arm_order_network/batch_readout/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$(dirname "$DS")"

GPU=$(MIN_FREE_MB=8000 bash scripts/_br1_wait_for_gpu.sh)
echo "[BR-1 phase 1 full] GPU=$GPU"

# 1) build dataset if absent
if [ ! -f "$DS" ]; then
  echo "[BR-1] building M=1000 dataset (~50 min) -> $DS"
  CUDA_VISIBLE_DEVICES=$GPU python - <<PY 2>&1 | tee "$LOG_DIR/phase1_build_M1000.log"
import time
from batch_readout.dataset_batch import build_dataset
from batch_readout.dataset_batch import load_dataset
from batch_readout.diversity_batch import teacher_diversity_stats
import json, numpy as np

t0 = time.time()
info = build_dataset(
    ckpt_path="$CKPT", M=1000, batch_size=32, seed=0, alpha_dep=0.5,
    out_path="$DS", train_frac=0.8, val_frac=0.1, device="cuda:0",
)
print(f"[BR-1] build_dataset done in {time.time()-t0:.1f}s; split sizes: {info}")
ds = load_dataset("$DS")
all_sig = np.concatenate([ds["train_sigma_T"], ds["val_sigma_T"], ds["test_sigma_T"]], axis=0)
print("teacher diversity (all M=1000):", json.dumps(teacher_diversity_stats(all_sig), indent=2))
PY
else
  echo "[BR-1] dataset already exists -> $DS"
fi

# 2) train 4 configs sequentially
BEST_ACC=-1
BEST_TAG=""
for MODEL in flatten nodewise; do
  for LOSS in pairwise pl; do
    TAG="full_M1000_B32_${MODEL}_${LOSS}"
    OUT="$CKPT_DIR/$TAG"
    mkdir -p "$OUT"
    LOG="$LOG_DIR/phase1_${TAG}.log"
    echo "[BR-1] train $TAG -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU python - <<PY 2>&1 | tee "$LOG"
from batch_readout.train_offline import train
out = train(
    dataset_path="$DS",
    model_name="$MODEL", loss_name="$LOSS",
    N=64, batch_size=64, lr=3e-4, epochs=40, seed=0,
    out_dir="$OUT", device="cuda:0",
)
m = out["best_metrics"]
print(f"[BR-1] $TAG best epoch={out['best_epoch']} "
      f"pairwise={m['pairwise_acc']:.4f} tau={m['kendall_tau']:.4f} "
      f"rho={m['spearman_rho']:.4f} top1={m['top1']:.4f} first3={m['first3']:.4f}")
PY
    ACC=$(grep "best epoch" "$LOG" | tail -1 | sed -n 's/.*pairwise=\([0-9.]*\).*/\1/p')
    if awk -v a="$ACC" -v b="$BEST_ACC" 'BEGIN{exit !(a>b)}'; then
      BEST_ACC=$ACC
      BEST_TAG=$TAG
    fi
  done
done

# 3) symlink winner
if [ -n "$BEST_TAG" ]; then
  cd "$CKPT_DIR"
  ln -sfn "$BEST_TAG/g_beta_best.pt" g_beta_phase1_best.pt
  echo "[BR-1] winner: $BEST_TAG (val pairwise=$BEST_ACC) -> g_beta_phase1_best.pt"
fi
