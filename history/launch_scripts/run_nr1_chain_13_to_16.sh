#!/bin/bash
# Sequential chain: Tasks 13 → 14 → 15 → 16.
# Each task aborts on first failure; downstream tasks are skipped.
# Run only after Task 12 smoke has PASSED (val tau >= 0.5).
set -euo pipefail

cd /home/admin/lyuyuhuan/order_lyu

mkdir -p analyses/neural_readout_nr1_2026-05-28

echo ""
echo "############################################################"
echo "## Task 13: full 10k dataset + train + §5.1 hard gates    ##"
echo "############################################################"
bash scripts/run_nr1_task13_full.sh

echo ""
echo "############################################################"
echo "## Task 14: 4 MVP ablations on full 10k                   ##"
echo "############################################################"
bash scripts/run_nr1_task14_ablations.sh

echo ""
echo "############################################################"
echo "## Task 15: cross-ckpt diagnostic (10k..60k)              ##"
echo "############################################################"
bash scripts/run_nr1_task15_crossckpt.sh

echo ""
echo "############################################################"
echo "## Task 16: aggregate REPORT.md                           ##"
echo "############################################################"
bash scripts/run_nr1_task16_report.sh

echo ""
echo "############################################################"
echo "## NR-1 chain Tasks 13-16 DONE                            ##"
echo "############################################################"
