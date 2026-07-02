#!/bin/bash
# Phase 2 .py cleanup — DRY-RUN PROPOSAL. All git mv are COMMENTED OUT.
# Nothing here moves a .py automatically. Review, uncomment case-by-case,
# run on branch chore/cleanup-phase2-python-audit. NO rm. NO moving
# UNKNOWN / MAINLINE / CONFIG / LAUNCH_HELPER / import_risk>=MED.
set -euo pipefail
echo "This is a PROPOSAL. Uncomment individual git mv lines after confirming."

mkdir -p history/python/one_off_checks
mkdir -p analyses/diagnostics scripts/diagnostics

# ---- Phase 2A: HISTORY one-off smokes (0 refs of any kind) ----
#   scripts/run_frozen_gbeta_hook_smoke.py: 0 refs, one-off smoke
# git mv scripts/run_frozen_gbeta_hook_smoke.py history/python/one_off_checks/
#   scripts/run_p6_b2_smoke.py: 0 refs, one-off smoke
# git mv scripts/run_p6_b2_smoke.py history/python/one_off_checks/

# ---- Phase 2B: DIAGNOSTIC with 0 sh/doc refs (CONFIRM not a live figure first) ----
#   analyses/diag_hook_vs_offline_b.py: diag-name
# git mv analyses/diag_hook_vs_offline_b.py analyses/diagnostics/
#   analyses/eval_frozen_readout_cross_ckpt.py: diag-name
# git mv analyses/eval_frozen_readout_cross_ckpt.py analyses/diagnostics/
#   analyses/plot_attention_heatmaps.py: diag-name
# git mv analyses/plot_attention_heatmaps.py analyses/diagnostics/
#   analyses/plot_multistart_separate_seeds.py: diag-name
# git mv analyses/plot_multistart_separate_seeds.py analyses/diagnostics/
#   analyses/plot_old_gbeta_comparison.py: diag-name
# git mv analyses/plot_old_gbeta_comparison.py analyses/diagnostics/
#   analyses/plot_old_gbeta_per_seed.py: diag-name
# git mv analyses/plot_old_gbeta_per_seed.py analyses/diagnostics/
#   analyses/plot_wall_clock_overlay.py: diag-name
# git mv analyses/plot_wall_clock_overlay.py analyses/diagnostics/
#   analyses/scan_collab_32heads.py: diag-name
# git mv analyses/scan_collab_32heads.py analyses/diagnostics/
#   analyses/summarize_head_gated_drift.py: diag-name
# git mv analyses/summarize_head_gated_drift.py analyses/diagnostics/
#   analyses/validate_gbeta_unsupervised_selector.py: diag-name
# git mv analyses/validate_gbeta_unsupervised_selector.py analyses/diagnostics/
#   analyses/validate_selector_random_ckpts.py: diag-name
# git mv analyses/validate_selector_random_ckpts.py analyses/diagnostics/
#   block_lo_arm_order_network/analyses/diag_content_teacher.py: diag-name
# git mv block_lo_arm_order_network/analyses/diag_content_teacher.py scripts/diagnostics/
#   block_lo_arm_order_network/analyses/scan_content_heads.py: diag-name
# git mv block_lo_arm_order_network/analyses/scan_content_heads.py scripts/diagnostics/
#   block_lo_arm_order_network/diagnose_10k_patch8x8.py: diag-name; image-line
# git mv block_lo_arm_order_network/diagnose_10k_patch8x8.py scripts/diagnostics/
#   block_lo_arm_order_network/diagnose_batch_vs_per_sample.py: diag-name
# git mv block_lo_arm_order_network/diagnose_batch_vs_per_sample.py scripts/diagnostics/
#   block_lo_arm_order_network/diagnose_labels.py: diag-name
# git mv block_lo_arm_order_network/diagnose_labels.py scripts/diagnostics/
#   block_lo_arm_order_network/diagnose_patch_aggregation.py: diag-name; image-line
# git mv block_lo_arm_order_network/diagnose_patch_aggregation.py scripts/diagnostics/
#   block_lo_arm_order_network/diagnose_reward_signal.py: diag-name
# git mv block_lo_arm_order_network/diagnose_reward_signal.py scripts/diagnostics/
#   block_lo_arm_order_network/eval_alignment.py: diag-name
# git mv block_lo_arm_order_network/eval_alignment.py scripts/diagnostics/
#   block_lo_arm_order_network/eval_cotrain_per_step.py: diag-name
# git mv block_lo_arm_order_network/eval_cotrain_per_step.py scripts/diagnostics/
#   block_lo_arm_order_network/eval_on.py: diag-name
# git mv block_lo_arm_order_network/eval_on.py scripts/diagnostics/
#   block_lo_arm_order_network/eval_on64_entropy.py: diag-name
# git mv block_lo_arm_order_network/eval_on64_entropy.py scripts/diagnostics/
#   block_lo_arm_order_network/eval_order_sensitivity.py: diag-name
# git mv block_lo_arm_order_network/eval_order_sensitivity.py scripts/diagnostics/
#   block_lo_arm_order_network/eval_structural.py: diag-name
# git mv block_lo_arm_order_network/eval_structural.py scripts/diagnostics/
#   block_lo_arm_order_network/plot_training_curves.py: diag-name
# git mv block_lo_arm_order_network/plot_training_curves.py scripts/diagnostics/
#   block_lo_arm_order_network/scripts/scan_image_heads.py: diag-name; image-line
# git mv block_lo_arm_order_network/scripts/scan_image_heads.py scripts/diagnostics/
#   block_lo_arm_order_network/scripts/summarize_phase2_text.py: diag-name
# git mv block_lo_arm_order_network/scripts/summarize_phase2_text.py scripts/diagnostics/
#   scripts/diag_br1_lightweight_b.py: diag-name
# git mv scripts/diag_br1_lightweight_b.py scripts/diagnostics/
#   scripts/diagnose_attention_graphs.py: diag-name
# git mv scripts/diagnose_attention_graphs.py scripts/diagnostics/
#   scripts/diagnose_none_separated_65.py: diag-name
# git mv scripts/diagnose_none_separated_65.py scripts/diagnostics/
#   scripts/inspect_l0_B_heads.py: diag-name
# git mv scripts/inspect_l0_B_heads.py scripts/diagnostics/
#   scripts/inspect_l0_B_match_previous.py: diag-name
# git mv scripts/inspect_l0_B_match_previous.py scripts/diagnostics/
#   scripts/inspect_l0_attention_heads.py: diag-name
# git mv scripts/inspect_l0_attention_heads.py scripts/diagnostics/
#   scripts/plot_collaborator_attention_maps.py: diag-name
# git mv scripts/plot_collaborator_attention_maps.py scripts/diagnostics/
#   scripts/plot_collaborator_b1_fixed_order_relabel.py: diag-name
# git mv scripts/plot_collaborator_b1_fixed_order_relabel.py scripts/diagnostics/
#   scripts/plot_collaborator_b1_pure_axis_relabel.py: diag-name
# git mv scripts/plot_collaborator_b1_pure_axis_relabel.py scripts/diagnostics/
#   scripts/plot_collaborator_b1_two_panel.py: diag-name
# git mv scripts/plot_collaborator_b1_two_panel.py scripts/diagnostics/
#   scripts/plot_image_story_figures.py: diag-name; image-line
# git mv scripts/plot_image_story_figures.py scripts/diagnostics/
#   scripts/plot_l0h5_attention_across_ckpts.py: diag-name
# git mv scripts/plot_l0h5_attention_across_ckpts.py scripts/diagnostics/
#   scripts/plot_multisample_token_attention.py: diag-name
# git mv scripts/plot_multisample_token_attention.py scripts/diagnostics/
#   scripts/scan_all_heads_across_ckpts.py: diag-name
# git mv scripts/scan_all_heads_across_ckpts.py scripts/diagnostics/
#   scripts/summarize_e3large_fixed_negative.py: diag-name
# git mv scripts/summarize_e3large_fixed_negative.py scripts/diagnostics/
#   scripts/summarize_imagenet32_continuous_round2.py: diag-name; image-line
# git mv scripts/summarize_imagenet32_continuous_round2.py scripts/diagnostics/
#   scripts/summarize_seed123_minimal.py: diag-name
# git mv scripts/summarize_seed123_minimal.py scripts/diagnostics/
#   scripts/summarize_seed_minimal_param.py: diag-name
# git mv scripts/summarize_seed_minimal_param.py scripts/diagnostics/
#   scripts/verify_batched_extractor.py: diag-name
# git mv scripts/verify_batched_extractor.py scripts/diagnostics/
#   scripts/verify_cdl_evolution.py: diag-name
# git mv scripts/verify_cdl_evolution.py scripts/diagnostics/

# ---- NOT proposed (protected / needs human review): 297 MAINLINE, 14 CONFIG, 16 LAUNCH_HELPER, 115 UNKNOWN, 26 referenced DIAGNOSTIC ----
echo "dry-run: review phase2_py_reference_audit.md; uncomment moves individually."
