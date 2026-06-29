# Appendix File Trace

| Item | Source path | Type | Notes |
|---|---|---|---|
| strict 65-node report | `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md` | markdown | Main strict LF result summary. |
| strict protocol definition | `reports/strict_65node_discovery_ckpt_verification_20260617/01_protocol_definition.md` | markdown | Protocol description. |
| strict all-head sweep | `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv` | TSV | Source for L0H1-L0H4 strong pass and L0H7 fail rows. |
| strict all-head sweep JSON | `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.json` | JSON | Same result in JSON. |
| strict A_with_none model-frame | `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/A_with_none_lh_mean_MODEL_FRAME.npy` | NPY | Can generate L0H1-L0H4 B65 heatmaps. |
| oracle-remapped sweep | `reports/collaborator_ckpt_b1_scan_20260616/none_separated_65_head_method_search/all_head_methods_none_separated_65.tsv` | TSV | Oracle-remapped comparison. |
| clean-base ladder result | `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv` | TSV | 9-step clean-base stability table. |
| clean-base strict sweep A data | `reports/strict_65node_discovery_ckpt_verification_20260617/strict_lf_cleanbase_sweep/step*/A_with_none_lh_mean_MODEL_FRAME.npy` | NPY | Per-step strict LF data. |
| 65-node diagnostic JSON | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/none_separated_65_diagnostic.json` | JSON | L0H7 fail diagnostic and controls. |
| B65 NPY | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated.npy` | NPY | Heatmap data for L0H7. |
| B65 heatmap PNG | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated_heatmap.png` | PNG | L0H7 fail heatmap. |
| loss-aligned A maps | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/loss_aligned_L0H7_A_maps.png` | PNG | L0H7 loss-aligned A view. |
| B1 summary | `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md` | markdown | Defines current B1 result-bearing convention as `none_mode=predictor`. |
| B1 signal summary | `reports/b1_attention_extraction_summary_20260615/03_b1_signal_summary.md` | markdown | B1 diagnostic results. |
| extraction comparison | `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md` | markdown | B1 content-only vs loss-aligned AR + None-sep comparison. |
| B1 content-only source | `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json` | JSON | Source for 0.655 vs 1.000 table. |
| destroyed controls summary | `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md` | markdown | Control interpretation. |
| destroyed controls TSV | `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv` | TSV | Control values. |
| B1 model-vs-phys figure | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_b1_image1_style_step60000_L0H0_model_vs_phys.png` | PNG | Coordinate remap / B1 diagnostic figure. |
| token model-frame map | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.png` | PNG | Token-level reveal/model frame. |
| token original-L2R map | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.png` | PNG | Token-level original label remap. |
| loss-aligned extraction code | `block_lo_arm_order_network/per_head_order_scan.py` | Python | Contains `_attn_to_A_block_loss_aligned_with_none_vec` and related aggregation functions. |
| strict LF search code | `scripts/search_strict_label_free_65.py` | Python | Builds model-frame strict LF graph and controls. |
| oracle-remapped search code | `scripts/search_none_separated_65_heads.py` | Python | Oracle-remapped counterpart. |
| None-separated graph code | `block_lo_arm_order_network/none_separated_block_graph.py` | Python | `build_none_separated_B`, rollout, controls, gate helpers. |
| strict graph tests | `block_lo_arm_order_network/tests/test_none_separated_block_graph.py` | Python test | Tests B65 construction, rollout, controls, gates. |
| extraction tests | `block_lo_arm_order_network/tests/test_per_head_scan_b0.py` | Python test | Tests B0/B1/predictor/loss-aligned extraction behavior. |
| strict raw test | `block_lo_arm_order_network/tests/test_search_raw_none_separated_65_heads.py` | Python test | Tests raw loss-aligned labels. |

## Source Missing

| Requested item | Status |
|---|---|
| `memory/` | SOURCE_MISSING |
| `reports/loss_aligned_b1_extraction_*` | SOURCE_MISSING |
| `reports/none_anchor_leak_audit_*` | SOURCE_MISSING |
| `reports/content_anchored_none_diagnostic_*` | SOURCE_MISSING |
| strict 65-node graph schematic | NEEDS_FIGURE |
| L0H1-L0H4 unified strict heatmap PNG | NEEDS_FIGURE; data exists in `A_with_none_lh_mean_MODEL_FRAME.npy` |

