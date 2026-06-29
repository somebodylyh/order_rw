# Wednesday Boss Update Package

Date: 2026-06-15

Scope: text-side AO-GPT / Block-LO-ARM / frozen g_beta evidence update for a Wednesday boss meeting. This package is a read-only synthesis of existing text-side evidence; it does not introduce new experiments, new training, or code changes.

Primary inputs:

- `reports/evidence_package_20260613_verified/05_FINAL_CLAIMS_LOCK.md`
- `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`
- `reports/evidence_package_20260613_verified/06_PAPER_READY_TEXT_SUMMARY.md`
- `reports/evidence_package_20260613_verified/08_NEXT_STEPS_AFTER_VERIFICATION.md`
- `reports/b1_attention_extraction_summary_20260615/00_README.md`
- `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md`
- `reports/b1_attention_extraction_summary_20260615/03_b1_signal_summary.md`
- `reports/b1_attention_extraction_summary_20260615/05_impact_on_existing_claims.md`
- `reports/b1_attention_extraction_summary_20260615/07_boss_update_summary.md`
- `reports/b1_attention_extraction_summary_20260615/08_unresolved_questions.md`

## Meeting Positioning

Title:

> Text-side AO-GPT order-controller evidence update

Main story:

1. Random-order AO-GPT develops sparse order-bearing attention heads.
2. We distill selected-head block attention graphs into a frozen order controller `g_beta`.
3. The controller specializes training toward a canonical reveal order and reduces steps to a target canonical-order loss.
4. B1/predictor-aligned diagnostics show that the attention-order signal is not a legacy B0 extraction artifact.

## Protocol Positioning

Current verified evidence has two layers:

| Layer | Role in Wednesday story | Evidence status |
|---|---|---|
| B0 legacy controller path | Current verified `g_beta` sanity and frozen-hook acceleration evidence | Complete for two matched seed groups |
| B1/predictor-aligned diagnostic | Collaborator-aligned attention-signal robustness check | Complete for clean-base ladder and latest continuous seed124 tracking |
| B1 controller path | Target migration direction if we replace all B0 programs | In progress / not yet verified in this evidence package |

Use this wording:

> B1 is the target collaborator-aligned protocol replacing B0. At this evidence checkpoint, B1 has already validated the attention-signal diagnosis, while the completed frozen-g_beta acceleration evidence still comes from the legacy B0 hook path until B1 controller replacement runs finish.

Do not say:

- B1 hook acceleration is established.
- Existing frozen-hook acceleration already used B1.
- B0 was wrong.

## Generated Files

- `01_one_page_executive_summary.md`
- `02_full_storyline.md`
- `03_slide_outline.md`
- `04_speaker_notes.md`
- `05_key_tables_for_boss.md`
- `06_figures_to_show.md`
- `07_expected_questions_and_answers.md`
- `08_claims_and_boundaries.md`
- `09_next_steps_and_decisions.md`
- `10_backup_appendix.md`

## Top Verified Numbers

| Number | Value | Source |
|---|---:|---|
| B0 clean-base strong heads | 9/32 heads with `|tau|>0.9` | `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json` |
| B1 clean-base ladder late signal | `L0H0 tau=1.000000` at 10k/50k/60k; `|tau|>0.9` = 7/32 at 10k and 6/32 at 50k/60k | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json` |
| B1 continuous seed124 final signal | `L0H4 tau=0.957589`, `mean_pairwise_tau=0.930060`, final `|tau|>0.9` = 1/32 | `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` |
| Legacy B0 `g_beta` real-vs-destroyed sanity | real B `tau_vs_l2r=0.967500`; Gaussian `-0.003353`; entry-shuffled `0.016488`; row/col shuffled `0.012748`; Gaussian pairwise `0.000313` | `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json` |
| Legacy B0 frozen-hook acceleration | recovery 86.0/83.0/58.7% for seed123; 106.1/101.5/66.9% for seed42; conservative step saving 33-42% | `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md` and listed `eval_curve.tsv` files |

