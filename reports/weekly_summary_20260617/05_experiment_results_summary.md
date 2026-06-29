# Experiment Results Summary

## Table A - Legacy frozen `g_beta` training acceleration

These are legacy controller path / B0 results, not strict 65-node teacher results.

| Seed group | Resume | Random @50k | L2R ref @50k | frozen_beta @50k | Recovery | Step Saving | Protocol |
|---|---:|---:|---:|---:|---:|---:|---|
| seed-123 L0H2 | from10k | 3.445 | 3.305 | 3.325 | 86.0% | 41.7% | legacy B0 controller |
| seed-123 L0H2 | from20k | 3.445 | 3.305 | 3.329 | 83.0% | 35.7% | legacy B0 controller |
| seed-123 L0H2 | from40k | 3.445 | 3.305 | 3.363 | 58.7% | 3.6%* | legacy B0 controller |
| seed-42 L0H4 | from10k | 3.466 | 3.341 | 3.334 | 106.1% | 39.3% | legacy B0 controller |
| seed-42 L0H4 | from20k | 3.466 | 3.341 | 3.339 | 101.5% | 33.3% | legacy B0 controller |
| seed-42 L0H4 | from40k | 3.466 | 3.341 | 3.383 | 66.9% | 3.6%* | legacy B0 controller |

*from40k starts below the threshold, so step saving is not a meaningful speed comparison.

Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.

## Table B - `g_beta` sanity

| Input B | tau / metric | Verdict |
|---|---:|---|
| Real selected-head B | `tau_vs_l2r=+0.9675` | reads structure |
| Gaussian random B | `tau_vs_l2r=-0.0034` | no fixed prior |
| Gaussian family | `mean_pairwise_tau=0.0003` | no constant order |
| Entry-shuffled B | `tau_vs_l2r=+0.0165` | destroyed structure collapses |
| Row/col shuffled B | `tau_vs_l2r=+0.0127` | destroyed structure collapses |
| zero B | `margin=0`, `tau=1.0` | tie artifact |

Source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`.

## Table C - Strict 65-node collaborator @50k strong heads

Primary rows shown with `L` readout; `C-D+L` also strong-passes for L0H1-L0H4. L0H7 is included as the fail example.

| Head | Method | tau | first | phys0_rank | p4 | p8 | destroyed |tau| | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L0H1 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0498 | strong_pass |
| L0H2 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0531 | strong_pass |
| L0H3 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0501 | strong_pass |
| L0H4 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0554 | strong_pass |
| L0H7 | C-D+L | 0.292 | 45 | 20 | 0 | 0 | 0.0825 | fail |

Source: `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`.

## Table D - Clean-base strict LF ladder

| Step range | strong heads / rows | best/stable heads | Interpretation |
|---|---:|---|---|
| 0/1k | 0 | - | signal not yet emerged |
| 5k-60k | 10-16 rows per checkpoint | stable L1H0-L1H4; late L0H0/L2H0/L2H4 | stable discovery phenomenon, drifting head identity |

Per-step source: `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv`.

## Table E - Extraction frame comparison

| Head | B1 predictor content-only | loss-aligned AR + None-separated | Interpretation |
|---|---:|---:|---|
| L0H1 | tau=0.655, best cyclic=1.000, starts at 6 | tau=1.000 | axis exists; anchor recovered only with None-sep |
| L0H2 | tau=0.655, best cyclic=1.000, starts at 6 | tau=1.000 | axis exists; anchor recovered only with None-sep |
| L0H3 | tau=1.000, best cyclic=1.000, starts at 0 | tau=1.000 | robust in both frames |
| L0H4 | tau=0.655, best cyclic=1.000, starts at 6 | tau=1.000 | axis exists; anchor recovered only with None-sep |

Source: `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json`; `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`.

## Table F - Destroyed controls

| Control | mean |tau| | Interpretation |
|---|---:|---|
| entry-shuffled | near random; collaborator strong rows roughly 0.05-0.062; spotcheck TSV aggregate 0.061 | destroys edge structure |
| label-permuted | near random; summary reports clean ladder <=0.07; spotcheck TSV aggregate 0.081 | destroys content-label alignment |

Source notes:

- Collaborator strong rows: `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md`.
- Spotcheck aggregate: `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv`.
- The robust claim is "near random", not a fragile exact threshold.

