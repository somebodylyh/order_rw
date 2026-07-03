#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/chenhe/nanogpt-learned-order"
GPU_ID="0"
CONFIG="config/WikiText103/seq256/permute/block64/attn_mlp_try53_seed2036_shadow_teacher_score_mse_train8k12k_attn1024_train15val3_fixed12k_anneal12k35k.py"
RUN_DIR="${1:-Report/logs/wikitext103_attn_mlp_try53_seed2036_train8k12k_attn1024_train15val3_gpu0_$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT_DIR}"
mkdir -p "${RUN_DIR}"

STATUS_FILE="${RUN_DIR}/status.txt"
DECISION_FILE="${RUN_DIR}/decision.txt"
LOG_FILE="${RUN_DIR}/train_try53_gpu0.log"
PID_FILE="${RUN_DIR}/runner.pid"

echo "$$" > "${PID_FILE}"
cat > "${DECISION_FILE}" <<EOF
task=try53_seed2036_shadow_teacher_score_mse_train8k12k_attn1024_train15val3
gpu=${GPU_ID}
config=${CONFIG}
root=${ROOT_DIR}
run_dir=${RUN_DIR}
log_file=${LOG_FILE}
launch_time=$(date --iso-8601=seconds)
cancel_hint=kill -TERM -- -$(ps -o pgid= $$ | tr -d ' ')
EOF

printf 'running\n' > "${STATUS_FILE}"

trap 'printf "terminated\n" > "${STATUS_FILE}"; exit 143' TERM INT

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
python train.py "${CONFIG}" > "${LOG_FILE}" 2>&1
status=$?

if [[ "${status}" -eq 0 ]]; then
  printf 'completed\n' > "${STATUS_FILE}"
else
  printf 'failed_status_%s\n' "${status}" > "${STATUS_FILE}"
fi
exit "${status}"
