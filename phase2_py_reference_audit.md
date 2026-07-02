# Phase 2 Python Reference Audit

**Branch:** `chore/cleanup-phase1-launch-scripts` @ `dcdc2e9`  **Scope:** 517 tracked `.py` in `analyses/ scripts/ block_lo_arm_order_network/` (nested repos + gitignored artifacts excluded).

**READ-ONLY.** No `.py` moved, nothing deleted, no logic changed. Companion: `phase2_py_classification.csv` (machine-readable), `phase2_py_cleanup_plan.sh` (dry-run; all `git mv` COMMENTED OUT).

## Method
Built the import graph over all tracked `.py` (every identifier on an `import`/`from` line → conservative reverse index) plus `.sh`/`.md`/`.yaml` string references. **Rule: any file imported by other `.py` is MAINLINE and is NOT a move candidate.** Config files under `configs/` are exec'd (not imported) → labeled CONFIG and protected. `__init__.py` = MAINLINE.

## Summary

| label | count | meaning | action |
|---|---|---|---|
| MAINLINE | 297 | imported by code / tests / entrypoint closure | KEEP |
| CONFIG | 14 | exec'd config modules (configs/) | KEEP |
| LAUNCH_HELPER | 16 | referenced only by a launcher .sh | KEEP (launcher dep) |
| DIAGNOSTIC | 73 | figure/metric/analysis, not imported | -> diagnostics/ (Phase 2B, confirm) |
| HISTORY | 2 | one-off/dated, 0 refs | -> history/python/ (Phase 2A) |
| UNKNOWN | 115 | 0 imports, standalone CLI — needs human eyes | HUMAN REVIEW (do not move) |

**Total: 517.**

## ⚠ Key caveat
An import-graph audit cannot safely classify **standalone CLI scripts** (0 imports but legitimate analysis/experiment entrypoints). These land in **UNKNOWN (115)** and MUST be opened individually before any move — many are real diagnostics (e.g. `block_granularity_scan`, `causal_layer_ablation`, `p6_cdl_sanity`, `phase1_5_*`, `wall_clock_saving`) whose names don't match the diagnostic prefix regex. Do NOT archive UNKNOWN in bulk.

## HISTORY candidates (safe: 0 refs of any kind) — Phase 2A
- `scripts/run_frozen_gbeta_hook_smoke.py`
- `scripts/run_p6_b2_smoke.py`

## DIAGNOSTIC (move-proposals to diagnostics/ — Phase 2B, confirm not used for a live paper figure)

**47 with 0 sh/doc refs (safer):**
- `analyses/diag_hook_vs_offline_b.py`
- `analyses/eval_frozen_readout_cross_ckpt.py`
- `analyses/plot_attention_heatmaps.py`
- `analyses/plot_multistart_separate_seeds.py`
- `analyses/plot_old_gbeta_comparison.py`
- `analyses/plot_old_gbeta_per_seed.py`
- `analyses/plot_wall_clock_overlay.py`
- `analyses/scan_collab_32heads.py`
- `analyses/summarize_head_gated_drift.py`
- `analyses/validate_gbeta_unsupervised_selector.py`
- `analyses/validate_selector_random_ckpts.py`
- `block_lo_arm_order_network/analyses/diag_content_teacher.py`
- `block_lo_arm_order_network/analyses/scan_content_heads.py`
- `block_lo_arm_order_network/diagnose_10k_patch8x8.py`
- `block_lo_arm_order_network/diagnose_batch_vs_per_sample.py`
- `block_lo_arm_order_network/diagnose_labels.py`
- `block_lo_arm_order_network/diagnose_patch_aggregation.py`
- `block_lo_arm_order_network/diagnose_reward_signal.py`
- `block_lo_arm_order_network/eval_alignment.py`
- `block_lo_arm_order_network/eval_cotrain_per_step.py`
- `block_lo_arm_order_network/eval_on.py`
- `block_lo_arm_order_network/eval_on64_entropy.py`
- `block_lo_arm_order_network/eval_order_sensitivity.py`
- `block_lo_arm_order_network/eval_structural.py`
- `block_lo_arm_order_network/plot_training_curves.py`
- `block_lo_arm_order_network/scripts/scan_image_heads.py`
- `block_lo_arm_order_network/scripts/summarize_phase2_text.py`
- `scripts/diag_br1_lightweight_b.py`
- `scripts/diagnose_attention_graphs.py`
- `scripts/diagnose_none_separated_65.py`
- `scripts/inspect_l0_B_heads.py`
- `scripts/inspect_l0_B_match_previous.py`
- `scripts/inspect_l0_attention_heads.py`
- `scripts/plot_collaborator_attention_maps.py`
- `scripts/plot_collaborator_b1_fixed_order_relabel.py`
- `scripts/plot_collaborator_b1_pure_axis_relabel.py`
- `scripts/plot_collaborator_b1_two_panel.py`
- `scripts/plot_image_story_figures.py`
- `scripts/plot_l0h5_attention_across_ckpts.py`
- `scripts/plot_multisample_token_attention.py`
- `scripts/scan_all_heads_across_ckpts.py`
- `scripts/summarize_e3large_fixed_negative.py`
- `scripts/summarize_imagenet32_continuous_round2.py`
- `scripts/summarize_seed123_minimal.py`
- `scripts/summarize_seed_minimal_param.py`
- `scripts/verify_batched_extractor.py`
- `scripts/verify_cdl_evolution.py`

**26 still referenced by .sh/docs (KEEP until launcher/doc updated):**
- `analyses/characterize_l0_dynamic_gbeta.py`  (py:0;doc:4)
- `analyses/diag_model_frame_feedback.py`  (py:0;sh:1;doc:2)
- `analyses/diag_shuffled_ar_model_frame_order.py`  (py:0;sh:1;doc:1)
- `analyses/diag_shuffled_l2r_cdl.py`  (py:0;doc:4)
- `analyses/diagnose_readout_collapse.py`  (py:0;doc:1)
- `analyses/eval_head_gated_audition.py`  (py:0;sh:1;doc:2)
- `analyses/eval_head_gated_gbeta.py`  (py:0;sh:1;doc:2)
- `analyses/plot_B_diagnostics.py`  (py:0;doc:1)
- `analyses/plot_attn_diagnostic_heatmap.py`  (py:0;doc:1)
- `analyses/plot_attn_map_b1.py`  (py:0;doc:2)
- `analyses/plot_b_heatmaps.py`  (py:0;doc:1)
- `analyses/plot_cdl_gbeta_comprehensive.py`  (py:0;doc:1)
- `analyses/plot_handoff_path_patch.py`  (py:0;doc:1)
- `analyses/plot_head_gated_gbeta.py`  (py:0;doc:2)
- `analyses/scan_seed1_10k_heads.py`  (py:0;doc:1)
- `analyses/validate_selector_cross_ckpt.py`  (py:0;doc:1)
- `analyses/validate_unsupervised_selector.py`  (py:0;doc:1)
- `block_lo_arm_order_network/diag_a_diff.py`  (py:0;sh:1;doc:1)
- `block_lo_arm_order_network/eval_reranker.py`  (py:0;doc:1)
- `scripts/diag_br1_head_layer_scan.py`  (py:0;doc:2)
- `scripts/summarize_e2_vq_negative_minimal.py`  (py:0;sh:1)
- `scripts/summarize_long1_e3.py`  (py:0;doc:1)
- `scripts/summarize_round2_e3.py`  (py:0;doc:1)
- `scripts/summarize_sample_quality_common_orders.py`  (py:0;sh:1)
- `scripts/summarize_topk_cluster_teacher_match.py`  (py:0;sh:1;doc:2)
- `scripts/vis_order_animation.py`  (py:0;doc:1)

## Files that MUST NOT move yet
- all **MAINLINE** (297) + **CONFIG** (14) + **LAUNCH_HELPER** (16): imported / exec'd / launcher deps.
- all **UNKNOWN** (115): standalone CLI, human review first.
- any file with `import_risk` HIGH/MED in the CSV.

## Dynamic-import / config-string risks
- `configs/*.py` are loaded by the training entrypoint via a config PATH (exec/dynamic), not `import` — they show 0 import refs but are essential. Labeled CONFIG, protected.
- `run-kind` / `--rw-policy` / `--pg-update` are STRING dispatch in `train_clean_aogpt.py` (not dynamic file import), so no hidden .py file loads by name — but confirm no `importlib`/`getattr(module, name)` before moving any UNKNOWN. (scripts/train_nodewise_gbeta is imported by path in train_clean_aogpt → MAINLINE.)

## UNKNOWN (human review) — full list
- `analyses/block_granularity_scan.py`
- `analyses/build_gbeta_headset_dataset.py`
- `analyses/causal_layer_ablation.py`
- `analyses/cdl_teacher_ablation_image.py`
- `analyses/cdl_teacher_ablation_image_viz.py`
- `analyses/graph_diversity_20260521/analyze_diversity.py`
- `analyses/graph_diversity_20260521/extract_per_sample_gB.py`
- `analyses/handoff_signature.py`
- `analyses/head_stability_seed2.py`
- `analyses/label_free_pipeline.py`
- `analyses/order_eval_ladder.py`
- `analyses/p5_ood_layout.py`
- `analyses/p6_cdl_sanity.py`
- `analyses/p6_cdl_sanity_v2.py`
- `analyses/p6_h_order_evolution.py`
- `analyses/p6_what_h_encodes.py`
- `analyses/p7_reveal_policy.py`
- `analyses/phase1_5_aggregation_knee.py`
- `analyses/phase1_5_per_head.py`
- `analyses/run_gbeta_sanity_save_json.py`
- `analyses/smoke_head_gated_gbeta.py`
- `analyses/wall_clock_saving.py`
- `block_lo_arm_order_network/batch_readout/cheap_teacher.py`
- `block_lo_arm_order_network/cem_readout_search.py`
- `block_lo_arm_order_network/cheap_layer_screen_image.py`
- `block_lo_arm_order_network/compare_modes.py`
- `block_lo_arm_order_network/consensus_order_diagnostic.py`
- `block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/prepare.py`
- `block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/download_full_imagenet64.py`
- `block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/download_smoke_subset.py`
- `block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/fill_missing.py`
- `block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/prepare.py`
- `block_lo_arm_order_network/data/Imagenet64VQ_f4_800k/rearrange_to_patches.py`
- `block_lo_arm_order_network/debug_coordinate_system.py`
- `block_lo_arm_order_network/debug_mlp_steps.py`
- `block_lo_arm_order_network/drift_eval.py`
- `block_lo_arm_order_network/extract_A32_from_on32_aogpt.py`
- `block_lo_arm_order_network/extract_A32_nn_from_N64.py`
- `block_lo_arm_order_network/extract_A_ar_ckpt.py`
- `block_lo_arm_order_network/extract_train_A32.py`
- `block_lo_arm_order_network/generate_nn_paths.py`
- `block_lo_arm_order_network/graph_regime_diagnostic.py`
- `block_lo_arm_order_network/hill_climb_batched.py`
- `block_lo_arm_order_network/m_step_train.py`
- `block_lo_arm_order_network/nn_paths_gpu.py`
- `block_lo_arm_order_network/nn_paths_mp.py`
- `block_lo_arm_order_network/nn_paths_n64.py`
- `block_lo_arm_order_network/p0_extract_delta_attention.py`
- `block_lo_arm_order_network/p0_n16_direct.py`
- `block_lo_arm_order_network/p0_permuted_ckpt.py`
- `block_lo_arm_order_network/p0_route_a.py`
- `block_lo_arm_order_network/probe_adjacent.py`
- `block_lo_arm_order_network/readout_fitness.py`
- `block_lo_arm_order_network/run_experiments.py`
- `block_lo_arm_order_network/run_full10k.py`
- `block_lo_arm_order_network/run_image_large_grw_pipeline.py`
- `block_lo_arm_order_network/run_long7k.py`
- `block_lo_arm_order_network/run_multi_seed_short.py`
- `block_lo_arm_order_network/run_phase1_diagnostic.py`
- `block_lo_arm_order_network/sample_level_locality_diagnostic.py`
- `block_lo_arm_order_network/scripts/aggregate_frozen_beta_curves.py`
- `block_lo_arm_order_network/scripts/analyze_image_locality.py`
- `block_lo_arm_order_network/scripts/calibrate_position_only.py`
- `block_lo_arm_order_network/scripts/calibrate_position_only_fine.py`
- `block_lo_arm_order_network/scripts/image/aggregate_image_ci.py`
- `block_lo_arm_order_network/scripts/run_alternating_from0.py`
- `block_lo_arm_order_network/scripts/run_phase2_chain.py`
- `block_lo_arm_order_network/scripts/run_position_only.py`
- `block_lo_arm_order_network/scripts/run_text_phase2_smoke_preflight.py`
- `block_lo_arm_order_network/scripts/text/audit_text_runs.py`
- `block_lo_arm_order_network/stage_b_sanity_check.py`
- `block_lo_arm_order_network/structural_probes.py`
- `block_lo_arm_order_network/sweep_tau.py`
- `block_lo_arm_order_network/tokenize_wikitext103.py`
- `block_lo_arm_order_network/train_aogpt_with_on.py`
- `block_lo_arm_order_network/train_clean_base.py`
- `block_lo_arm_order_network/train_crossattn_on.py`
- `block_lo_arm_order_network/train_gnn_on.py`
- `block_lo_arm_order_network/train_joint.py`
- `block_lo_arm_order_network/train_learned_policy.py`
- `block_lo_arm_order_network/train_on32.py`
- `block_lo_arm_order_network/train_on_mixed_v2.py`
- `block_lo_arm_order_network/train_reranker_q.py`
- `block_lo_arm_order_network/train_reranker_struct.py`
- `scripts/_print_mlp_order.py`
- `scripts/ablate_graph_policy_from_ckpt.py`
- `scripts/aggregate_l0h5_evolution.py`
- `scripts/aggregate_multiseed_minimal.py`
- `scripts/compact_sample_quality_summary.py`
- `scripts/compare_ckpt_weights.py`
- `scripts/content_relocation_diagnostic.py`
- `scripts/debug_eval_original_objective_on_10k.py`
- `scripts/debug_reproduce_ckpt_val_loss.py`
- `scripts/extract_random_substrate_A.py`
- `scripts/fid_3ckpt_50k.py`
- `scripts/fid_alt_vq64.py`
- `scripts/final_results_tables.py`
- `scripts/label_free_selector_audit.py`
- `scripts/old_new_ckpt_matched_audit.py`
- `scripts/phase0_residual_gate.py`
- `scripts/run_attn_order_phase0_sanity.py`
- `scripts/run_attn_order_phase1_5_rollout_diagnostic.py`
- `scripts/run_frozen_gbeta_20k_comparison.py`
- `scripts/run_frozen_gbeta_60k_overnight.py`
- `scripts/run_frozen_gbeta_60k_v2.py`
- `scripts/run_frozen_gbeta_comparison.py`
- `scripts/run_p6_b2_full.py`
- `scripts/run_p6_hres_replication.py`
- `scripts/run_per_head_order_scan_ladder_b1_fast.py`
- `scripts/search_none_separated_65_heads.py`
- `scripts/show_readout_orders.py`
- `scripts/spotcheck_strict_lf_cleanbase.py`
- `scripts/train_vq64_alternating.py`
- `scripts/train_vq64_fixed_baseline.py`
- `scripts/train_vq64_fixed_order.py`

## Recommended minimal move plan
- **Phase 2A (now, safe):** move the 2 HISTORY one-off smokes to `history/python/one_off_checks/`. Nothing else auto-moves.
- **Phase 2B (after per-file confirm):** the 47 zero-ref DIAGNOSTIC → `analyses/diagnostics/` or `scripts/diagnostics/` (keep dir-of-origin). Confirm none back a live paper figure.
- **Phase 2C (defer, high-risk):** old order-policy / head-selection / gβ variants sit in UNKNOWN — resolve case-by-case, likely after the Phase-3 `src/order/` unified interface.
- Do the actual moves on a NEW branch `chore/cleanup-phase2-python-audit`, one small commit per sub-phase, so each is revertible.
