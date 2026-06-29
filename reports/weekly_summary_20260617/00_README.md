# Weekly Summary 2026-06-17

This directory is a weekly reporting and evidence-entry package for the AO-GPT / Block-LO-ARM / order-discovery work.

## Generated Files

| File | Purpose |
|---|---|
| `01_executive_summary.md` | One-page Chinese executive summary for quick reporting. |
| `02_timeline_of_work.md` | Timeline of this week's protocol cleanup and verification work. |
| `03_main_technical_findings.md` | Main technical findings and current interpretation. |
| `04_protocol_evolution.md` | Protocol evolution table from B0/B1 to strict 65-node LF. |
| `05_experiment_results_summary.md` | Compact tables of key results. |
| `06_claims_lock_current.md` | Current can-claim / cannot-claim lock. |
| `07_open_risks_and_boundaries.md` | Resolved, manageable, and blocking risks. |
| `08_next_week_plan.md` | Prioritized next-week plan. |
| `09_boss_update_version.md` | 1-2 page boss update draft in Chinese. |
| `10_appendix_key_tables.md` | Traceability appendix: paths, formulas, definitions, gates. |

## Primary Sources Read

- `reports/evidence_package_20260613_verified/`
- `reports/b1_attention_extraction_summary_20260615/`
- `reports/strict_65node_discovery_ckpt_verification_20260617/`
- `reports/wednesday_boss_update_20260615/`
- `reports/collaborator_ckpt_b1_scan_20260616/`
- `block_lo_arm_order_network/`
- `scripts/`
- `docs/`
- `analyses/`
- `analyses/figures/`
- `probe_results/`

## Missing Requested Inputs

These requested paths or glob groups were not present at the top level and were not guessed:

- `memory/`
- `reports/loss_aligned_b1_extraction_*`
- `reports/none_anchor_leak_audit_*`
- `reports/content_anchored_none_diagnostic_*`

Related loss-aligned / None-separated materials were found under:

- `reports/collaborator_ckpt_b1_scan_20260616/loss_aligned_M20_b8_seed0.json`
- `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/`
- `reports/collaborator_ckpt_b1_scan_20260616/none_separated_65_head_method_search/`
- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/`

## Main Thread

This week's story is not a single experiment. It is a protocol cleanup:

1. Old B0/B1 attention extraction was audited and separated into diagnostic vs controller paths.
2. Predictor-frame, AR next-token shift, and None-anchor issues were clarified.
3. The discovery protocol was tightened to a strict 65-node graph: node 0 is None/BOS and nodes 1..64 are physical content blocks.
4. Selected early heads recover L2R from None under this stricter protocol.
5. Legacy frozen `g_beta` / hook acceleration remains valid as legacy B0 controller evidence, but is not yet a strict 65-node teacher result.

