#!/bin/bash
# ============================================================================
# cleanup_plan.sh — DRY-RUN repo cleanup (Phase 1: archive .sh launch history)
#
#   Review first. By default this ONLY PRINTS the commands (does nothing).
#   To actually apply:   bash cleanup_plan.sh --apply
#
# Scope of THIS phase (safe, minimal move):
#   - create history/ + scripts/launch/ + scripts/diagnostics/ scaffolding
#   - git mv one-off / dated *.sh launch scripts -> history/launch_scripts/
#   - keep reusable baseline/canonical launchers -> scripts/launch/
#   NO .py is moved (imports could break — that awaits the per-file audit).
#   NO deletions. Run on a DEDICATED cleanup branch, not the feature branch.
# ============================================================================
set -euo pipefail
APPLY="${1:-}"
run() { echo "+ $*"; if [ "$APPLY" = "--apply" ]; then "$@"; fi; }

# --- scaffolding -----------------------------------------------------------
run mkdir -p history/launch_scripts
run mkdir -p scripts/launch
run mkdir -p scripts/diagnostics   # (py moved here in Phase 2, after per-file audit)

# --- KEEP as reusable launchers (move to scripts/launch/) ------------------
# Baseline + canonical-protocol launchers collaborators re-use.
run git mv scripts/run_l2r_baselines_s2_s42_60k.sh      scripts/launch/
run git mv scripts/run_baselines_full_chain_60k.sh      scripts/launch/
run git mv scripts/run_extend_baselines_50k60k.sh       scripts/launch/
run git mv scripts/run_cdl_teacher_seed123.sh           scripts/launch/
run git mv scripts/run_frozen_gbeta_aligned_20260625.sh scripts/launch/   # canonical frozen_beta protocol

# --- ARCHIVE: one-off queued runs (queue_*) --------------------------------
for f in \
  queue_317M_continuous_after_pipeline.sh \
  queue_cdl_seed123_missing_20k40k.sh \
  queue_cdl_seed123_reverse_from20k.sh \
  queue_extend_l2r_seed123_50k60k.sh \
  queue_head_gated_gbeta_feedback.sh \
  queue_head_gated_gbeta_joint.sh \
  queue_l2r_seed123_51000_smoke.sh \
  queue_l2r_seed123_decide_after_51000_smoke.sh \
  queue_l2r_seed123_fresh60k.sh \
  queue_l2r_seed123_low_lr_after_closed_loop.sh \
  queue_l2r_strict65_scan.sh \
  queue_model_frame_feedback_diag.sh \
  queue_random_b1_headscan_after_gpu1.sh \
  queue_seed123_closed_loop_overnight.sh \
  queue_shuffled_ar_model_frame_diag.sh \
; do run git mv "scripts/$f" history/launch_scripts/; done

# --- ARCHIVE: dated overnight/aligned suites (2026*) -----------------------
for f in \
  overnight_20260625_from5k_aligned_suite.sh \
  overnight_from40k_orientation_20260622.sh \
  run_overnight_20260610_seed2_multistart.sh \
  run_overnight_20260611_cdl_multistart.sh \
  run_overnight_p0_serial.sh \
  run_compile_5k_ablation_20260626.sh \
  run_direct_policy_aligned_20260625.sh \
  run_from10k_gbeta_attention_maps_20260623.sh \
  run_gbeta_loss_comparison_20260625.sh \
  run_topk_cluster_teacher_match_20260626.sh \
; do run git mv "scripts/$f" history/launch_scripts/; done

# --- ARCHIVE: closed experiment lines (NR-1, BR-1, CT8, round2, image) -----
for f in \
  run_nr1_chain_13_to_16.sh run_nr1_orchestrator.sh run_nr1_task12_smoke.sh \
  run_nr1_task13_full.sh run_nr1_task14_ablations.sh run_nr1_task15_crossckpt.sh \
  run_nr1_task16_report.sh \
  run_br1_phase0_smoke.sh run_br1_phase1_full.sh run_br1_phase1_smoke.sh \
  run_br1_phase2_frozen.sh _br1_wait_for_gpu.sh \
  run_ct8_all.sh \
  run_round2_e3_5arm.sh run_round2_long_seed.sh \
  run_round2_sample_quality_common_orders.sh run_round2_sample_quality_seed42.sh \
  run_round2_seed123_minimal.sh run_round2_seed_minimal_param.sh \
  run_e2_vq_negative_minimal.sh run_e3large_fixed_negative.sh \
  run_imagelarge_grw_5arm.sh run_imagenet32_continuous_round2_minimal.sh \
  wait_and_diagnose_e3_control.sh \
; do run git mv "scripts/$f" history/launch_scripts/; done

# --- ARCHIVE: misc one-off launches/diagnostic shells ----------------------
for f in \
  check_large_model_progress.sh post_grw_diagnose.sh \
  run_block_len_variation.sh run_handoff_circuit_trajectory.sh \
  run_head_gated_audition.sh run_head_gated_gbeta_distill_full.sh \
  run_head_gated_gbeta_distill_smoke.sh \
  run_frozen_beta_b1_seed2.sh run_frozen_beta_multi_start.sh \
  run_large_model_random_baseline.sh \
  run_per_head_order_scan_ladder.sh run_per_head_order_scan_ladder_b1.sh \
  run_seed42_multistart.sh run_shuffle_granularity_chain.sh \
; do run git mv "scripts/$f" history/launch_scripts/; done

# ============================================================================
# NOT in this phase (proposed in cleanup_report.md, require per-file audit):
#   - analyses/plot_*.py, validate_*, eval_*, scan_*, diag_*, summarize_*   -> scripts/diagnostics/
#   - closed-line .py (image train_vq*/train_imagenet*, plot_old_gbeta_*)   -> history/
#   - src/order/ unified interface + configs/baselines/*                    -> dedicated refactor
# Do NOT move any .py until an exhaustive import-reference check confirms it
# is unreferenced by train_clean_aogpt.py / any config module.
# ============================================================================
echo "DRY-RUN complete. Re-run with --apply to execute (on a dedicated branch)."
