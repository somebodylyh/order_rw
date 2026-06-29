# Attention Figures Package 20260617

这个目录把 attention summary 里嵌入的所有图片复制成独立文件，方便上传 Notion 或放进汇报。

## Boss update main figures

- `00_main_strict65_step10000_L1H4_model_vs_physical_none_separated.png` — Strict 65-node 10k L1H4 model-order vs physical-order, None node separated
- `01_main_token_level_orientation_token_attention_model_frame.png` — Token attention model frame
- `02_main_token_level_ori_l2r_token_attention_original_l2r.png` — Token attention original L2R
- `03_main_l0h7_fail_b65_none_separated_l0h7_b65_none_separated_heatmap.png` — L0H7 B65 None-separated heatmap
- `07_main_b1_predictor_vs_physical_b1_model_frame_vs_physical_frame_heatmap.png` — B1 model-frame vs physical-frame heatmap
- `08_main_collaborator_b1_l0h3_collaborator_b1_l0h3_avg_model_to_physical.png` — Collaborator B1 L0H3 avg model-to-physical
- `09_main_b1_l0h2_model_vs_phys_b1_l0h2_step30000_model_vs_phys.png` — B1 L0H2 step30000 model-vs-phys
- `13_main_collaborator_attention_summary_collaborator_attention_maps_summary.png` — Collaborator attention maps summary

## Optional / context figures

- `04_optional_l0h7_loss_aligned_a_maps.png` — L0H7 loss-aligned A maps.
- `05_optional_l0h7_b_content.png` — L0H7 B content.
- `06_optional_clean_base_strict_a_b65_l0h0.png` — Clean-base strict A/B65 L0H0.
- `10_optional_content_anchor_then_none.png` — Content-anchor then None. Caveat: content-only / anchored diagnostic; do not present as full strict label-free discovery
- `11_optional_content_only_l1h3_fast_model_vs_phys.png` — Content-only L1H3 fast model-vs-phys. Caveat: content-only / anchored diagnostic; do not present as full strict label-free discovery
- `12_optional_content_only_l1h3_model_vs_phys.png` — Content-only L1H3 model-vs-phys. Caveat: content-only / anchored diagnostic; do not present as full strict label-free discovery
- `14_optional_b0_b1_before_after_remap.png` — B0/B1 before-after remap. Caveat: B0/B1 or relabel diagnostic; use with protocol caveat
- `15_optional_b1_fixed_order_axis_relabel.png` — B1 fixed-order axis relabel. Caveat: B0/B1 or relabel diagnostic; use with protocol caveat
- `16_optional_b1_pure_axis_relabel.png` — B1 pure axis relabel. Caveat: B0/B1 or relabel diagnostic; use with protocol caveat
- `17_optional_legacy_method_overview.png` — Legacy method overview. Caveat: legacy controller/training figure, not strict 65-node attention evidence
- `18_optional_legacy_g_beta_sanity_bar.png` — Legacy g_beta sanity bar. Caveat: legacy controller/training figure, not strict 65-node attention evidence
- `19_optional_legacy_catch_up_curve.png` — Legacy catch-up curve. Caveat: legacy controller/training figure, not strict 65-node attention evidence
- `20_optional_legacy_multistart_comparison.png` — Legacy multistart comparison. Caveat: legacy controller/training figure, not strict 65-node attention evidence
- `21_optional_cdl_tau_heatmap_3_panel.png` — CDL tau heatmap 3-panel.

## Manifest

`manifest.tsv` contains order, copied filename, original source path, role, and caveat.
