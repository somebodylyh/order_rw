#!/bin/bash
# Wait for eps015 (PID 1026437) to finish, then launch top_k8_tau05 on GPU0

EPS015_PID=1026437
OUTPUT_DIR="probe_results/clean_method_graph_rw_a09_from20k_topk8_tau05"

echo "[$(date)] Waiting for eps015 PID $EPS015_PID to finish..."

while kill -0 $EPS015_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] eps015 finished. Launching top_k8_tau05..."

cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network

mkdir -p "$OUTPUT_DIR"

python -u train_method_from_base.py \
    --run-kind graph_rw \
    --resume-ckpt probe_results/clean_base_random_perm/ckpt_step20000.pt \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 40000 \
    --refresh-interval 2000 \
    --refresh-n-chunks 200 \
    --refresh-ema-beta 0.9 \
    --refresh-data-source eval \
    --alpha-start 0.0 \
    --alpha-target 0.9 \
    --alpha-warmup-steps 10000 \
    --rw-top-k 8 \
    --tau-start 0.5 \
    --tau-step 0.5 \
    --epsilon-uniform 0.0 \
    --eval-interval 1000 \
    --log-interval 10 \
    --save-steps "20000,25000,30000,35000,40000" \
    --device cuda:0 \
    > "$OUTPUT_DIR/stdout.log" 2>&1 &

echo "[$(date)] top_k8 launched, PID: $!"
