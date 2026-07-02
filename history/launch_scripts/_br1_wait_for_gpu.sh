#!/usr/bin/env bash
# Wait until at least one GPU has >= MIN_FREE_MB free memory, then print its index.
set -euo pipefail
MIN_FREE_MB=${MIN_FREE_MB:-8000}
POLL=${POLL:-60}
while true; do
  if nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v m="$MIN_FREE_MB" '$1>=m{found=1} END{exit !found}'; then
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v m="$MIN_FREE_MB" '$2>=m{print $1; exit}'
    exit 0
  fi
  sleep "$POLL"
done
