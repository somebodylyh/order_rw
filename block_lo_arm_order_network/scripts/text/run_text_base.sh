#!/usr/bin/env bash
# Text random-perm baseline. Mirrors the collaborator's WikiText-103 seq256 block64
# 50 000-iter random baseline. The rw_* CLI flags are intentionally NOT passed —
# alpha_for_step short-circuits to 0 for run_kind=baseline anyway.

set -euo pipefail
cd "$(dirname "$0")/../.."

python -u train_clean_aogpt.py \
    --run-kind baseline \
    --output-dir probe_results/clean_base_random_perm_v2 \
    --max-iters 50000 \
    --lr 1e-3 --min-lr 1e-4 \
    --batch-size 64 --gradient-accumulation-steps 2 \
    --eval-interval 250 --log-interval 10 \
    "$@"
