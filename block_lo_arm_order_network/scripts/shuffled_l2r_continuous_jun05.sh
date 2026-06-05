#!/bin/bash
# shuffled-L2R control baseline (continuous), 6.8-gate arm #4.
#
# Trains AR along the data's SHUFFLED layout (model_ascending), i.e. follows the
# permuted order we feed the model WITHOUT recovering the original L2R. Same
# clean_perm / data / hyperparams as the random + ours serial chain, so all four
# gate arms (ori-L2R / random / ours / shuffled-L2R) share one protocol.
# train_objective is read from the val_model_order column.
#
# Scheduling: GPU0 is owned by the random->ours serial chain tonight, so we do NOT
# touch it (grabbing it would collide in the ARM A->B gap). We wait for GPU1 (the
# collaborator's card) to free, then run there. Poll-and-confirm to avoid grabbing
# a momentary eval-time memory dip.
set -u
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
GPU=1
OUT=probe_results/shuffled_l2r_continuous_jun05
COMMON="--data-source continuous --max-steps 50000 --lr-decay-steps 50000 \
 --batch-size 64 --grad-accum 2 --lr 1e-3 --min-lr 1e-4 \
 --eval-interval 1000 --stream-eval-windows 2000 \
 --save-steps 0,1000,5000,10000,20000,30000,40000,50000 --log-interval 50"

gpu_used_mib () {
  nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

echo "[$(date)] waiting for GPU$GPU to free (behind collaborator job)..."
while :; do
  u=$(gpu_used_mib)
  if [ -n "$u" ] && [ "$u" -lt 2000 ]; then
    sleep 90                       # confirm it stays free (not a transient dip)
    u2=$(gpu_used_mib)
    if [ -n "$u2" ] && [ "$u2" -lt 2000 ]; then break; fi
  fi
  sleep 120
done
echo "[$(date)] GPU$GPU free (used=${u} MiB) -> launching shuffled_l2r"

rm -rf "$OUT"
python3 train_clean_aogpt.py --run-kind shuffled_l2r --device cuda:$GPU --output-dir "$OUT" $COMMON
echo "[$(date)] shuffled_l2r exited rc=$?"
echo "[$(date)] === shuffled-L2R DONE; gate arm #4 written to $OUT/eval_curve.tsv (col val_model_order) ==="
