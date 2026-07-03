#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/runner/run_when_vram_free.sh [options] -- <command> [args...]

Wait until the requested GPU(s) have enough free VRAM, then run the command.

Options:
  --gpus IDS                 Comma-separated GPU ids to watch. Default: 0,1
  --min-free-gb GB           Required free VRAM per selected GPU, in GiB. Default: 20
  --min-free-mib MIB         Required free VRAM per selected GPU, in MiB.
  --mode all|any             all: require every listed GPU; any: pick first eligible GPU. Default: all
  --interval SECONDS         Poll interval. Default: 60
  --checks N                 Required consecutive passing checks. Default: 1
  --once                     Check once and exit 1 if VRAM is not enough.
  --max-wait-seconds SEC     Stop waiting after this many seconds. Default: wait forever
  --no-cuda-visible-devices  Do not set CUDA_VISIBLE_DEVICES before running command.
  -h, --help                 Show this help.

Examples:
  # Run a two-GPU job when GPU 0 and 1 both have at least 40 GiB free:
  scripts/runner/run_when_vram_free.sh --gpus 0,1 --min-free-gb 40 -- \
    torchrun --nproc_per_node=2 train.py config/foo.py

  # Run a one-GPU job on whichever of GPU 0 or 1 first has at least 20 GiB free:
  scripts/runner/run_when_vram_free.sh --gpus 0,1 --mode any --min-free-gb 20 -- \
    python train.py config/foo.py
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

trim_spaces() {
  local value="$1"
  value="${value//[[:space:]]/}"
  printf '%s' "${value}"
}

join_by() {
  local sep="$1"
  shift || true

  local first=1
  local item
  for item in "$@"; do
    if [ "${first}" -eq 1 ]; then
      printf '%s' "${item}"
      first=0
    else
      printf '%s%s' "${sep}" "${item}"
    fi
  done
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

gpu_list="0,1"
min_free_mib=$((20 * 1024))
mode="all"
interval_seconds=60
required_checks=1
check_once=0
max_wait_seconds=0
set_cuda_visible_devices=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus)
      [ "$#" -ge 2 ] || die "--gpus requires a value"
      gpu_list="$2"
      shift 2
      ;;
    --gpus=*)
      gpu_list="${1#*=}"
      shift
      ;;
    --min-free-gb)
      [ "$#" -ge 2 ] || die "--min-free-gb requires a value"
      is_positive_int "$2" || die "--min-free-gb must be a positive integer"
      min_free_mib=$(("$2" * 1024))
      shift 2
      ;;
    --min-free-gb=*)
      gb="${1#*=}"
      is_positive_int "${gb}" || die "--min-free-gb must be a positive integer"
      min_free_mib=$((gb * 1024))
      shift
      ;;
    --min-free-mib)
      [ "$#" -ge 2 ] || die "--min-free-mib requires a value"
      is_positive_int "$2" || die "--min-free-mib must be a positive integer"
      min_free_mib="$2"
      shift 2
      ;;
    --min-free-mib=*)
      min_free_mib="${1#*=}"
      is_positive_int "${min_free_mib}" || die "--min-free-mib must be a positive integer"
      shift
      ;;
    --mode)
      [ "$#" -ge 2 ] || die "--mode requires a value"
      mode="$2"
      shift 2
      ;;
    --mode=*)
      mode="${1#*=}"
      shift
      ;;
    --interval)
      [ "$#" -ge 2 ] || die "--interval requires a value"
      is_positive_int "$2" || die "--interval must be a positive integer"
      interval_seconds="$2"
      shift 2
      ;;
    --interval=*)
      interval_seconds="${1#*=}"
      is_positive_int "${interval_seconds}" || die "--interval must be a positive integer"
      shift
      ;;
    --checks)
      [ "$#" -ge 2 ] || die "--checks requires a value"
      is_positive_int "$2" || die "--checks must be a positive integer"
      required_checks="$2"
      shift 2
      ;;
    --checks=*)
      required_checks="${1#*=}"
      is_positive_int "${required_checks}" || die "--checks must be a positive integer"
      shift
      ;;
    --once)
      check_once=1
      shift
      ;;
    --max-wait-seconds)
      [ "$#" -ge 2 ] || die "--max-wait-seconds requires a value"
      is_positive_int "$2" || die "--max-wait-seconds must be a positive integer"
      max_wait_seconds="$2"
      shift 2
      ;;
    --max-wait-seconds=*)
      max_wait_seconds="${1#*=}"
      is_positive_int "${max_wait_seconds}" || die "--max-wait-seconds must be a positive integer"
      shift
      ;;
    --no-cuda-visible-devices)
      set_cuda_visible_devices=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[ "$#" -gt 0 ] || {
  usage >&2
  exit 2
}

case "${mode}" in
  all|any) ;;
  *) die "--mode must be all or any" ;;
esac

[ -n "${gpu_list}" ] || die "--gpus cannot be empty"
IFS=',' read -r -a requested_gpus <<< "${gpu_list}"
[ "${#requested_gpus[@]}" -gt 0 ] || die "--gpus cannot be empty"

declare -A requested=()
normalized_requested_gpus=()
for gpu in "${requested_gpus[@]}"; do
  gpu="$(trim_spaces "${gpu}")"
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU id must be a non-negative integer: ${gpu}"
  requested["${gpu}"]=1
  normalized_requested_gpus+=("${gpu}")
done
requested_gpus=("${normalized_requested_gpus[@]}")
gpu_list="$(join_by ',' "${requested_gpus[@]}")"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

passing_checks=0
start_time=$(date +%s)
selected_gpus=""

describe_status() {
  local -n free_map_ref=$1
  local -n used_map_ref=$2
  local -n total_map_ref=$3
  local parts=()
  local gpu free used total

  for gpu in "${requested_gpus[@]}"; do
    gpu="$(trim_spaces "${gpu}")"
    free="${free_map_ref[${gpu}]:-unknown}"
    used="${used_map_ref[${gpu}]:-unknown}"
    total="${total_map_ref[${gpu}]:-unknown}"
    parts+=("GPU ${gpu}: free=${free} MiB used=${used} MiB total=${total} MiB")
  done

  join_by '; ' "${parts[@]}"
}

check_vram() {
  local -n selected_ref=$1
  local line index free used total
  local eligible=()
  declare -A free_mib=()
  declare -A used_mib=()
  declare -A total_mib=()

  while IFS=',' read -r index free used total; do
    index="$(trim_spaces "${index}")"
    free="$(trim_spaces "${free}")"
    used="$(trim_spaces "${used}")"
    total="$(trim_spaces "${total}")"

    if [ -n "${requested[${index}]:-}" ]; then
      free_mib["${index}"]="${free}"
      used_mib["${index}"]="${used}"
      total_mib["${index}"]="${total}"
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,memory.used,memory.total \
      --format=csv,noheader,nounits
  )

  for gpu in "${requested_gpus[@]}"; do
    gpu="$(trim_spaces "${gpu}")"
    if [ -z "${free_mib[${gpu}]:-}" ]; then
      log "GPU ${gpu} was not found by nvidia-smi"
      selected_ref=""
      return 1
    fi

    if [ "${free_mib[${gpu}]}" -ge "${min_free_mib}" ]; then
      eligible+=("${gpu}")
    fi
  done

  log "$(describe_status free_mib used_mib total_mib); need >= ${min_free_mib} MiB free"

  if [ "${mode}" = "all" ]; then
    if [ "${#eligible[@]}" -eq "${#requested_gpus[@]}" ]; then
      selected_ref="${gpu_list}"
      return 0
    fi
  else
    if [ "${#eligible[@]}" -gt 0 ]; then
      selected_ref="${eligible[0]}"
      return 0
    fi
  fi

  selected_ref=""
  return 1
}

log "watching GPU(s) ${gpu_list}; mode=${mode}; min_free=${min_free_mib} MiB; interval=${interval_seconds}s; checks=${required_checks}"

while true; do
  if check_vram selected_gpus; then
    passing_checks=$((passing_checks + 1))
    log "VRAM check passed (${passing_checks}/${required_checks}) for GPU(s) ${selected_gpus}"

    if [ "${passing_checks}" -ge "${required_checks}" ]; then
      if [ "${set_cuda_visible_devices}" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES="${selected_gpus}"
        log "launching with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}: $*"
      else
        log "launching without changing CUDA_VISIBLE_DEVICES: $*"
      fi
      exec "$@"
    fi
  else
    passing_checks=0
    if [ "${check_once}" -eq 1 ]; then
      log "VRAM is not enough; exiting because --once was set"
      exit 1
    fi
  fi

  if [ "${max_wait_seconds}" -gt 0 ]; then
    now=$(date +%s)
    elapsed=$((now - start_time))
    if [ "${elapsed}" -ge "${max_wait_seconds}" ]; then
      log "timed out after ${elapsed}s waiting for enough VRAM"
      exit 1
    fi
  fi

  sleep "${interval_seconds}"
done
