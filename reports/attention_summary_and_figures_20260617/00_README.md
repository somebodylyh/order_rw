# Attention Summary and Figures 2026-06-17

This package summarizes the last two days of attention extraction / block-level order-discovery discussion.

Scope:

- Focus: attention protocol, heatmaps, block-level graph construction, strict 65-node label-free discovery.
- Not focus: training acceleration, except where needed as a boundary.
- No training was run for this package.
- No training code was modified.

## Generated Files

| File | Purpose |
|---|---|
| `01_attention_story_summary.md` | Chinese narrative of the attention protocol discussion. |
| `02_protocol_definitions.md` | Protocol definitions and `inv_perm` boundary. |
| `03_key_results.md` | Key numeric results with source paths. |
| `04_figures_index.md` | Figure/data index, boss-update suitability, caveats. |
| `05_recommended_boss_slides.md` | 7-slide attention-focused boss update outline. |
| `06_claim_boundaries.md` | Current attention claim boundary. |
| `07_open_questions.md` | Open questions and suggested actions. |
| `08_appendix_file_trace.md` | Source-file trace for reports, data, scripts, and tests. |

## Primary Inputs Read

- `reports/b1_attention_extraction_summary_20260615/`
- `reports/strict_65node_discovery_ckpt_verification_20260617/`
- `reports/collaborator_ckpt_b1_scan_20260616/`
- `analyses/`
- `analyses/figures/`
- `probe_results/`
- `block_lo_arm_order_network/`
- `scripts/`
- `docs/`

## Missing Requested Inputs

The following requested paths/globs were not present and were not guessed:

- `memory/`
- `reports/loss_aligned_b1_extraction_*`
- `reports/none_anchor_leak_audit_*`
- `reports/content_anchored_none_diagnostic_*`

Related loss-aligned / None-separated materials were found under `reports/collaborator_ckpt_b1_scan_20260616/`.

## Main Conclusion

Attention-order discovery should now be described through the strict 65-node None-separated protocol: node 0 is independent None/BOS, nodes 1..64 are physical content blocks, rollout starts from None, and the first content block is selected by graph edges/readout. Under this protocol, selected early heads recover L2R; the result is head-specific and extraction-frame-dependent.

