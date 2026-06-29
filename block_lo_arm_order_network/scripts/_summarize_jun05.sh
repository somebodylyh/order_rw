#!/bin/bash
cd /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network
best () {  # $1=dir $2=col(field idx in eval_curve)  -> "value@step"
  awk -F'\t' -v c="$2" 'NR>1 && $c!="" {print $c"\t"$1}' "$1/eval_curve.tsv" 2>/dev/null | sort -g | head -1 | awk '{printf "%.4f@%s",$1,$2}'
}
echo "================ OVERNIGHT SUMMARY $(date) ================"
for d in l2r_continuous_jun05 random_baseline_continuous_jun05 gbeta_from5000_continuous_jun05; do
  P=probe_results/$d
  if [ -f "$P/eval_curve.tsv" ]; then
    last=$(tail -1 "$P/eval_curve.tsv" | cut -f1)
    echo "--- $d  (last step=$last) ---"
    echo "   best ori_l2r       = $(best "$P" 5)"
    echo "   best train_obj      = $(best "$P" 4)"
    echo "   best beta_order(g_β)= $(best "$P" 10)"
    echo "   best rw_order       = $(best "$P" 9)"
  else
    echo "--- $d : (no eval_curve yet) ---"
  fi
done
echo "=========================================================="
