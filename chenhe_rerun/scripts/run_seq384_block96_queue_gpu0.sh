#!/usr/bin/env bash
set -euo pipefail

cd /home/chenhe/nanogpt-learned-order

GPU_INDEX="${GPU_INDEX:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="Report/logs"
mkdir -p "${LOG_DIR}"

GPU_UUID="$(
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F, -v idx="${GPU_INDEX}" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $1);
          gsub(/^[ \t]+|[ \t]+$/, "", $2);
          if ($1 == idx) print $2;
        }'
)"

if [[ -z "${GPU_UUID}" ]]; then
  echo "Could not resolve GPU UUID for GPU index ${GPU_INDEX}" >&2
  exit 1
fi

echo "seq384/block96 queue started at $(date)"
echo "target GPU index: ${GPU_INDEX}"
echo "target GPU uuid: ${GPU_UUID}"
echo "poll seconds: ${POLL_SECONDS}"

gpu_pids() {
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits \
    | awk -F, -v uuid="${GPU_UUID}" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $1);
          gsub(/^[ \t]+|[ \t]+$/, "", $2);
          if ($2 == uuid) print $1;
        }' || true
}

while true; do
  current_pids="$(gpu_pids | xargs || true)"
  if [[ -z "${current_pids}" ]]; then
    echo "GPU${GPU_INDEX} is free at $(date); starting seq384 queue."
    break
  fi
  echo "GPU${GPU_INDEX} busy at $(date); waiting. pids=${current_pids}"
  sleep "${POLL_SECONDS}"
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1

run_one() {
  local name="$1"
  local cfg="$2"
  local run_log="${LOG_DIR}/seq384_block96_gpu${GPU_INDEX}_${STAMP}_${name}.log"
  echo "===== START ${name} at $(date) ====="
  echo "config: ${cfg}"
  echo "log: ${run_log}"
  python train.py "${cfg}" 2>&1 | tee "${run_log}"
  local rc="${PIPESTATUS[0]}"
  echo "===== END ${name} at $(date), exit=${rc} ====="
  return "${rc}"
}

run_one "nonpermute_ar_70k" \
  "config/WikiText103/seq384/non_permute/block96/ar.py"

run_one "permute_random_70k" \
  "config/WikiText103/seq384/permute/block96/random.py"

run_one "permute_ar_70k" \
  "config/WikiText103/seq384/permute/block96/ar.py"

run_one "mlp_try1_warmup15k_anneal40k_fixed70k" \
  "config/WikiText103/seq384/permute/block96/attn_mlp_try1_fromscratch_autohead_eigloss_withoutnone_directed_ribbon_lazyinit_warmup15k_anneal40k_fixed70k_wandb.py"

echo "seq384/block96 queue finished at $(date)"
