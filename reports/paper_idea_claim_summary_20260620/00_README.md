# Paper Idea, Claim Map & Experiment Summary

**Generated**: 2026-06-20
**Purpose**: 面向论文构思和老板汇报的系统总结。不是周报，是论文级别的 idea/claim map。

## File Index

| File | Content |
|------|---------|
| `01_core_idea.md` | Core idea in English and Chinese, two-layer framing (mechanism vs acceleration) |
| `02_positioning_and_novelty.md` | Novelty relative to existing work, novelty table, conservative novelty statement |
| `03_method_overview.md` | Method overview: AO-GPT training, attention graph extraction, CDL teacher, g_β controller |
| `04_experiment_design.md` | Experiment groups E1–E6 with objectives, protocols, and metrics |
| `05_current_results_by_claim.md` | Results organized by claim, with evidence strength, source paths, and caveats |
| `06_paper_outline.md` | Paper structure, title candidates (~5), abstract skeleton |
| `07_claims_lock.md` | CAN CLAIM / CANNOT CLAIM / BEST WORDING for abstract, intro, slide, limitations |
| `08_risks_and_missing_experiments.md` | Risk grading (green/yellow/red), missing experiments |
| `09_recommended_next_experiments.md` | Priority-ranked next experiments (P0, P1, P2) |
| `10_boss_summary.md` | One-page executive summary for boss presentation |
| `11_paper_abstract_drafts.md` | 2-3 abstract drafts with different emphases |
| `12_appendix_traceability.md` | Source path → claim mapping, all key numbers with file references |

## Key Source Directories

- `reports/evidence_package_20260613_verified/` — Legacy B0 controller verification
- `reports/strict_65node_discovery_ckpt_verification_20260617/` — Strict 65-node discovery
- `reports/b1_attention_extraction_summary_20260615/` — B1 protocol diagnosis
- `reports/attention_summary_and_figures_20260617/` — Attention summary and figures
- `reports/weekly_summary_20260617/` — Weekly summary package
- `reports/wednesday_boss_update_20260615/` — Boss meeting package
- `block_lo_arm_order_network/probe_results/` — All eval_curve.tsv and experiment outputs
- `block_lo_arm_order_network/batch_readout/logs/` — All g_β phase15 reports
- `analyses/` — Analysis scripts and wall_clock_data.tsv

## Critical Constraints (Do Not Violate)

1. Do not conflate strict 65-node discovery with legacy B0 hook acceleration.
2. Do not say "all heads discover L2R."
3. Do not say training loss is block-level — it is still token-level AR.
4. Do not say content-only 64-node proves content-driven adjacency under fixed permutation.
5. Distinguish nanogpt convention vs block_lo_arm_order_network convention for block_perm direction.
6. Do not restore deprecated numbers: seed42 random baseline = 3.466 (not 3.354); strong heads ≠ 28/32.
7. All key numbers must have source paths or be marked SOURCE_MISSING.
