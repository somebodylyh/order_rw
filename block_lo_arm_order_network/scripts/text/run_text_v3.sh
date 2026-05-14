#!/usr/bin/env bash
# Text Graph-RW v3 readiness-guided continuation from clean_base@20k.
# Reproduces clean_method_graph_rw_v3_from20k (best ori_l2r ≈ 3.458 @50k).

set -euo pipefail
cd "$(dirname "$0")/../.."

START_CKPT="${START_CKPT:-probe_results/clean_base_random_perm/ckpt_step20000.pt}"

python -u train_clean_aogpt.py \
    --run-kind graph_rw \
    --resume-ckpt "$START_CKPT" \
    --output-dir probe_results/clean_method_graph_rw_v3_from20k_repro \
    --max-iters 60000 \
    --lr 1e-3 --min-lr 1e-4 \
    --batch-size 64 --gradient-accumulation-steps 2 \
    --rw-policy progressive_rw_v3 \
    --rw-top-k 4 \
    --tau-start 0.10 --tau-step 0.10 \
    --rw-lam 0.75 --rw-rho 0.2 \
    --alpha-start 0.0 --alpha-target 0.9 --alpha-warmup-steps 10000 \
    --eval-interval 1000 --log-interval 100 \
    "$@"
