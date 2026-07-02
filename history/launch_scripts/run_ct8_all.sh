#!/usr/bin/env bash
# CT8 — Causal-Hidden Order Probe real GPU runs (frozen diagnostic, no training).
# Three runs, sequential on one GPU to avoid contention/OOM (GPU0 shared w/ chenhe idle proc).
#   1) text primary   : clean_base_random_perm @30k   (n_chunks16 / n_roll16 / n_patterns4)
#   2) text contrast   : alt_from0_mlp_finetune @30k
#   3) image primary   : vq64_alt_from0_mlp_patch2x2_l8h8e512 @30k (n_eval256 / n_roll16)
# Verdict gate = Path X only (candidate-conditioned). Path Y is control. ev_mode=content_token
# (NOT a deployable controller — flagged in each report). After this: synthesize SUMMARY.md +
# Case A/B/C decision matrix. NOTE: extract_A_matrices randperm is unseeded -> B_A run-variance;
# decision margin ~0.01-0.05 nats, so verify verdict is not sampling noise (n_chunks16 helps).
set -u
cd "$(dirname "$0")/.."
PY=/home/admin/anaconda3/envs/X1/bin/python
export CUDA_VISIBLE_DEVICES=0          # physical GPU0 (mostly free); maps to cuda:0
OUT=probe_results/causal_hidden_probe
LOG=$OUT/ct8_logs
mkdir -p "$LOG"

echo "===== CT8 START $(date '+%F %T') on physical GPU0 ====="

echo "----- [1/3] text primary (clean_base_random_perm) -----"
$PY -u scripts/run_causal_hidden_probe.py \
  --modality text \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt \
  --out-dir "$OUT/text_clean_random_primary" \
  --n-chunks 16 --n-roll 16 --n-patterns 4 \
  > "$LOG/text_primary.log" 2>&1
echo "  [1/3] exit=$? $(date '+%T')"

echo "----- [2/3] text contrast (alt_from0_mlp_finetune) -----"
$PY -u scripts/run_causal_hidden_probe.py \
  --modality text \
  --ckpt probe_results/attention_order_mlp/alt_from0_mlp_finetune/ckpt_step30000.pt \
  --out-dir "$OUT/text_alt_mlp_contrast" \
  --n-chunks 16 --n-roll 16 --n-patterns 4 \
  > "$LOG/text_contrast.log" 2>&1
echo "  [2/3] exit=$? $(date '+%T')"

echo "----- [3/3] image primary (vq64_alt_from0_mlp_patch2x2_l8h8e512) -----"
$PY -u scripts/run_causal_hidden_probe.py \
  --modality image \
  --ckpt probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/ckpt_step30000.pt \
  --a-global-path probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/A_global_step30000.npy \
  --data-val nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/val.bin \
  --meta nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/meta.pkl \
  --out-dir "$OUT/image_vq64_alt_mlp_primary" \
  --n-eval 256 --n-roll 16 --n-patterns 4 \
  > "$LOG/image_primary.log" 2>&1
echo "  [3/3] exit=$? $(date '+%T')"

echo "===== CT8 DONE $(date '+%F %T') ====="
