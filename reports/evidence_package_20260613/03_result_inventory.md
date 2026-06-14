# 03 — Result Inventory

> All files relevant to the text-side evidence chain.
> Trust levels: HIGH = raw log/json/CSV; MEDIUM = script-generated summary; LOW = human-written memory.

---

## Primary Data (HIGH trust)

### Eval Curves (eval_curve.tsv)

| File | Key numbers | Notes |
|------|------|------|
| `probe_results/l2r_continuous_seed123/eval_curve.tsv` | ori-L2R ref @50k = 3.305 | Continuous-stream L2R training |
| `probe_results/random_baseline_continuous_jun08_seed2/eval_curve.tsv` | random @50k = 3.445 | Seed2 baseline, continuous stream |
| `probe_results/frozen_beta_random_baseline_jun05/eval_curve.tsv` | random @50k = 3.354 | Seed42 baseline (note: unusually close to L2R) |
| `probe_results/frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` | @50k = 3.325 | Seed2 L0H2 frozen_β from10k |
| `probe_results/frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` | @50k = 3.329 | Seed2 L0H2 frozen_β from20k |
| `probe_results/frozen_beta_seed2_from40k_l0h2/eval_curve.tsv` | @50k = 3.363 | Seed2 L0H2 frozen_β from40k |
| `probe_results/cdl_teacher_seed2_from10k_l0h2/eval_curve.tsv` | @50k = 3.284 | CDL teacher, beats L2R ref |
| `probe_results/cdl_teacher_seed2_from20k_l0h2/eval_curve.tsv` | @50k = 3.308 | CDL teacher from20k |
| `probe_results/cdl_teacher_seed2_from40k_l0h2/eval_curve.tsv` | @50k = 3.353 | CDL teacher from40k |
| `probe_results/frozen_beta_random_jun05_from10k/eval_curve.tsv` | @50k = 3.334 | Seed42 L0H4 frozen_β from10k |
| `probe_results/frozen_beta_random_jun05_from20k/eval_curve.tsv` | @50k = 3.339 | Seed42 L0H4 frozen_β from20k |
| `probe_results/frozen_beta_random_jun05_from40k/eval_curve.tsv` | @50k = 3.383 | Seed42 L0H4 frozen_β from40k (worse than random) |

### Head Scans (head_scan_*.json)

| File | Key numbers | Notes |
|------|------|------|
| `probe_results/clean_base_random_perm/head_scan_10k.json` | L0H0 τ=+1.0, L0H6 τ=+0.997 | 64-block clean_base, step10k |
| `probe_results/shuffle_gran_32/head_scan_10k.json` | L0H0 τ=+1.0, L2H0 τ=−0.938 | 32-group shuffle training |
| `probe_results/shuffle_gran_128/head_scan_10k.json` | L1H2 τ=+1.0, L2H3 τ=−0.938 | 128-group shuffle training |
| `probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json` | L0H10 τ=+0.955 | 317M model, step5000 |

### Config Files (config.json)

| File | Key fields |
|------|------|
| `probe_results/shuffle_gran_128/config.json` | shuffle_granularity=128, num_blocks=64, block_len=4 |
| `probe_results/shuffle_gran_32/config.json` | shuffle_granularity=32, num_blocks=64, block_len=4 |

### Granularity Scans

| File | Key numbers |
|------|------|
| `analyses/block_granularity_scan_results/scan_step5000_M100.json` | 32blk τ=0.9997, 64blk τ=0.9999, 128blk τ=1.000 |

---

## Summary Documents (MEDIUM trust)

| File | Content | Trust note |
|------|------|------|
| `CLAIMS_LOCK_20260613.md` | Can/Cannot/Need claims | Numbers verified against eval_curves |
| `TABLES_20260613.md` | Recovery, Step Saving, Sanity tables | Numbers verified |
| `memory/daily-summary-20260613.md` | Daily work summary | Cross-checked with primary data |
| `memory/shuffle-granularity-20260613.md` | Shuffle granularity per-head results | Matches head_scan JSONs |
| `memory/block-granularity-robust-20260613.md` | Block aggregation robustness | Matches scan JSON |
| `memory/gbeta-input-sanity-20260611.md` | g_β mechanism sanity | Primary script output |
| `memory/cdl-serial-multi-start-seed2-20260612.md` | CDL teacher multi-start | Verified against eval_curves |

### Strategy / Guidance (LOW trust — human-written, not data)

| File | Content |
|------|------|
| `memory/strategy-lock-evidence-20260613.md` | Strategic guidance from user |
| `docs/superpowers/specs/2026-05-30-quick-head-selector-design.md` | Head selector design spec |

---

## Figures (HIGH: machine-generated from primary data)

| File | Content | Generated |
|------|------|------|
| `analyses/figures/fig1_method_overview.png` | Method pipeline diagram | 2026-06-13 |
| `analyses/figures/fig2_gbeta_sanity_bar.png` | g_β sanity bar chart | 2026-06-13 |
| `analyses/figures/fig3_catchup_curve.png` | Catch-up curve | 2026-06-13 |
| `analyses/figures/fig4_multistart_comparison.png` | Multi-start comparison | 2026-06-13 |
| `analyses/figures/shuffle_gran128_B_heatmaps.png` | B matrix heatmaps | 2026-06-13 |
| `probe_results/frozen_beta_multi_start_comparison.png` | Older multi-start plot | Pre-2026-06-13 |

---

## Analysis Scripts (HIGH: code, MEDIUM: output interpretation)

| File | Purpose |
|------|------|
| `analyses/block_granularity_scan.py` | Aggregation granularity scan |
| `analyses/gbeta_input_sanity_final.py` | g_β mechanism sanity tests |
| `analyses/plot_frozen_beta_multi_start.py` | Multi-start comparison plots |
| `block_lo_arm_order_network/per_head_order_scan.py` | Per-head τ extraction |
| `block_lo_arm_order_network/neural_readout/extract_b.py` | B matrix extraction |
| `scripts/run_shuffle_granularity_chain.sh` | Shuffle granularity training chain |
