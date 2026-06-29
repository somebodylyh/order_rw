# Figures Index

## Figure / Data Index

| Figure | Path | Related protocol | What it shows | Use in boss update? | Caveat |
|---|---|---|---|---|---|
| B1 model-frame vs physical-frame heatmap | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_b1_image1_style_step60000_L0H0_model_vs_phys.png` | B1/model-vs-physical remap | Same attention probes in model frame vs original/physical frame; illustrates coordinate remap issue. | Yes, optional | B1 diagnostic, not strict 65-node teacher; [None] handling differs from final strict protocol. |
| Token attention model frame | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.png` | token-level reveal/model frame | Shows causal model-frame attention; future region is zero. | Yes, if explaining "not bidirectional" | Token-level, not block-level B65. |
| Token attention original L2R | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.png` | token attention remapped to original L2R | Shows why original L2R can look two-sided after random reveal remap. | Yes, paired with model-frame token map | Do not interpret as bidirectional model attention. |
| L0H7 B65 None-separated heatmap | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated_heatmap.png` | loss-aligned AR + None-separated 65-node | Heatmap for fail example L0H7. | Yes | Fail example; useful only as contrast. |
| L0H7 A maps | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/loss_aligned_L0H7_A_maps.png` | loss-aligned extraction | A maps for L0H7 under loss-aligned extraction. | Optional | L0H7 is fail case; not positive evidence. |
| L0H7 B content | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/loss_aligned_L0H7_B_content.png` | loss-aligned content graph | Content-only B view for L0H7. | Optional | Fail/caveat figure. |
| Clean-base strict A/B65 L0H0 | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_none65_step60000_L0H0_strict_A_B65.png` | strict 65-node clean-base | Shows strict A/B65 style for clean-base L0H0 at 60k. | Yes, if a positive heatmap is needed now | Clean-base L0H0, not collaborator L0H1-L0H4. |
| Collaborator B1 L0H3 avg model-to-physical | `reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_avg_model_to_physical_nomask.png` | B1 model-vs-physical | B1 remap illustration for collaborator L0H3. | Optional | B1 diagnostic, not strict 65-node. |
| B1 L0H2 step30000 model-vs-phys | `analyses/figures/attn_map_b1/b1_L0_H2_step30000_model_vs_phys.png` | B1 model-vs-phys | Earlier B1 model/physical heatmap. | Optional | Older figure; verify exact run before using as main evidence. |
| Content-anchor then None | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_L1H3_content_anchor_then_none.png` | content anchor / None diagnostic | Content anchor then None diagnostic for clean-base L1H3. | Optional | Not final strict protocol; use only to explain why anchor matters. |
| Content-only L1H3 fast model-vs-phys | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_content_only_L1H3_step5000_fast_model_vs_phys.png` | content-only diagnostic | Content-only model vs physical remap. | Optional | Demonstrates order-axis / anchor issue, not final proof. |
| Content-only L1H3 model-vs-phys | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_content_only_L1H3_step5000_model_vs_phys.png` | content-only diagnostic | Content-only graph heatmap. | Optional | Same caveat. |
| Collaborator attention maps summary | `reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_ckpt_attention_maps_M20.png` | mixed B0/B1 diagnostics | Panel of collaborator attention-derived maps. | Optional | Mixed protocols; must label panels carefully. |
| B0/B1 before-after remap | `reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_before_after_remap.png` | B1/predictor remap | Before/after remap illustration. | Optional | Not final strict 65-node. |
| B1 fixed-order axis relabel | `reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_fixed_order_axis_relabel.png` | B1 axis relabel | Axis relabel effect. | Optional | Coordinate demonstration only. |
| B1 pure axis relabel | `reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_pure_axis_relabel.png` | B1 axis relabel | Pure relabel of B1 map. | Optional | Coordinate demonstration only. |
| Legacy method overview | `analyses/figures/fig1_method_overview.png` | controller/training | Method overview. | Optional | Mainly controller/training, not strict attention protocol. |
| Legacy g_beta sanity bar | `analyses/figures/fig2_gbeta_sanity_bar.png` | legacy B0 controller | g_beta sanity. | Optional | Not strict 65-node teacher. |
| Legacy catch-up curve | `analyses/figures/fig3_catchup_curve.png` | training acceleration | Training catch-up. | Optional | Not attention protocol main evidence. |
| Legacy multistart comparison | `analyses/figures/fig4_multistart_comparison.png` | training acceleration | Multi-start training comparison. | Optional | Not attention protocol main evidence. |
| CDL tau heatmap 3-panel | `analyses/attention_diagnostic_20260609/cdl_tau_heatmap_3panel.png` | older attention diagnostic | Older CDL tau heatmap. | Optional | Older protocol; not final strict 65-node. |

## Heatmap Data Files

| Data | Path | Notes |
|---|---|---|
| L0H7 B65 data | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated.npy` | Corresponds to `B65_none_separated_heatmap.png`. |
| L0H7 diagnostic JSON | `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/none_separated_65_diagnostic.json` | Contains L0H7 fail metrics and controls. |
| strict collaborator A_with_none | `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/A_with_none_lh_mean_MODEL_FRAME.npy` | Data can generate L0H1-L0H4 strict heatmaps; no existing unified PNG found. |
| clean-base strict sweep A_with_none | `reports/strict_65node_discovery_ckpt_verification_20260617/strict_lf_cleanbase_sweep/step*/A_with_none_lh_mean_MODEL_FRAME.npy` | Per-step strict data. |
| token model frame npy | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.npy` | Token-level reveal/model frame. |
| token original L2R npy | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.npy` | Token-level original-label remap. |

## Boss Update Must-Have Figures

1. strict 65-node graph schematic: NEEDS_FIGURE. Draw node0=None/BOS, nodes1..64=physical blocks, rollout arrow from None, no manual None->phys0 fold.
2. L0H1-L0H4 strong-pass attention/heatmap: NEEDS_FIGURE. Existing data exists in `strict_label_free_65_search/A_with_none_lh_mean_MODEL_FRAME.npy`, but no unified heatmap PNG was found.
3. L0H7 fail heatmap: use `B65_none_separated_heatmap.png`.
4. Extraction frame comparison table: use `03_key_results.md` table; no figure required.
5. Gate distribution table/bar: use `03_key_results.md`; NEEDS_FIGURE only if a bar chart is desired.

## Optional Figures

- B1 predictor/model-vs-physical heatmaps for coordinate explanation.
- Token-level model-frame vs original-L2R pair for "not bidirectional; coordinate remap" explanation.
- Content-only heatmaps for anchor-loss explanation.
- Legacy `g_beta` sanity bar and training catch-up curves only if the update also touches controller/training.

