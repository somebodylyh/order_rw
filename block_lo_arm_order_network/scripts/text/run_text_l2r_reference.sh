#!/usr/bin/env bash
# Text L2R reference (run_kind=l2r). Used ONLY as upper-bound reference for
# Graph-RW. Two flavors selectable via CKPT env var:
#   (a) From scratch  — leave START_CKPT unset; max_iters=60000
#   (b) Continuation oracle — pass START_CKPT to a previous L2R checkpoint

set -euo pipefail
cd "$(dirname "$0")/../.."

START_CKPT="${START_CKPT:-}"
RESUME_FLAG=""
[ -n "$START_CKPT" ] && RESUME_FLAG="--resume-ckpt $START_CKPT"

python -u train_clean_aogpt.py \
    --run-kind l2r \
    $RESUME_FLAG \
    --output-dir "${OUTPUT_DIR:-probe_results/clean_ori_l2r_from_scratch_repro}" \
    --max-iters 60000 \
    --lr 1e-3 --min-lr 1e-4 \
    --batch-size 64 --gradient-accumulation-steps 2 \
    --eval-interval 1000 --log-interval 100 \
    "$@"
