# Current Runnable Aggregate — 2026-06-20

Generated from existing local JSON/TSV files only. No GPU training/diagnostic was launched because both 4090s were already occupied at 98–99% utilization.

## A. Direct answer to the reviewer gap

**Already have, but under-explained:** strict-65 protocol definitions, destroyed controls, full all-head JSON/TSV tables, clean-base 9-step ladder, B1 predictor 3-seed ladder, 317M strict-label-free diagnostic, matched eval curves for frozen_beta/CDL/random/L2R groups, and the seed-mismatch audit.
**Can promote immediately into paper tables:** all-head success-rate tables, matched-group training curves, seed123 CDL teacher run, seed124 supplementary CDL/frozen_beta runs, and the 317M negative result.
**Still genuinely missing:** cross-permutation / position-memory control, label-free head audition with held-out scoring, strict-65 teacher distilled into g_beta and hooked end-to-end, proper variance/significance tests, and a matched L2R reference for seed124.

## B. Strict-65 / label-free artifacts

| Artifact | protocol note | rows | strong | weak | best gate | best L/H/method | best tau | destroyed | max +tau | max abs(tau) |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step0/all_head_methods_none_separated_65.json` | none-separated | 96 | 0 | 0 | fail | L0H2 C-D+L | 0.191 | 0.063 | 0.191 | 0.191 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step1000/all_head_methods_none_separated_65.json` | none-separated | 96 | 0 | 0 | fail | L2H4 none_edge | 0.142 | 0.069 | 0.142 | 0.209 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step10000/all_head_methods_none_separated_65.json` | none-separated | 96 | 10 | 2 | strong_pass | L1H2 L | 1.000 | 0.050 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step20000/all_head_methods_none_separated_65.json` | none-separated | 96 | 16 | 1 | strong_pass | L1H4 L | 1.000 | 0.052 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step30000/all_head_methods_none_separated_65.json` | none-separated | 96 | 15 | 4 | strong_pass | L0H1 L | 1.000 | 0.053 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step40000/all_head_methods_none_separated_65.json` | none-separated | 96 | 15 | 1 | strong_pass | L1H2 L | 1.000 | 0.055 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step5000/all_head_methods_none_separated_65.json` | none-separated | 96 | 10 | 2 | strong_pass | L1H2 L | 1.000 | 0.050 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step50000/all_head_methods_none_separated_65.json` | none-separated | 96 | 14 | 1 | strong_pass | L0H0 L | 1.000 | 0.057 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/step60000/all_head_methods_none_separated_65.json` | none-separated | 96 | 16 | 2 | strong_pass | L0H0 L | 1.000 | 0.054 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/none_separated_65_head_method_search/all_head_methods_none_separated_65.json` | phys-remapped/oracle-ish | 256 | 20 | 11 | strong_pass | L0H3 L | 1.000 | 0.064 | 1.000 | 1.000 |
| `collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.json` | strict_label_free | 256 | 20 | 11 | strong_pass | L0H1 L | 1.000 | 0.050 | 1.000 | 1.000 |
| `large_317M_strict_65_scan/all_head_methods_strict_label_free_65.json` | strict_label_free | 512 | 0 | 0 | fail | L9H15 L | 0.134 | 0.042 | 0.291 | 0.291 |
| `random_bl_65node_scan/step0/all_head_methods_none_separated_65.json` | none-separated | 64 | 0 | 0 | fail | L2H6 C-D+L | 0.168 | 0.064 | 0.188 | 0.188 |
| `random_bl_65node_scan/step0_phys/all_head_methods_none_separated_65.json` | phys-remapped/oracle-ish | 64 | 0 | 0 | fail | L1H3 L | 0.145 | 0.063 | 0.328 | 0.328 |
| `random_bl_65node_scan/step10000/all_head_methods_none_separated_65.json` | none-separated | 64 | 0 | 0 | fail | L1H5 C-D+L | 0.035 | 0.077 | 0.185 | 0.185 |
| `random_bl_65node_scan/step10000_phys/all_head_methods_none_separated_65.json` | phys-remapped/oracle-ish | 64 | 8 | 3 | strong_pass | L0H0 L | 1.000 | 0.060 | 1.000 | 1.000 |
| `random_bl_65node_scan/step30000/all_head_methods_none_separated_65.json` | none-separated | 64 | 0 | 0 | fail | L1H7 L | 0.042 | 0.055 | 0.157 | 0.157 |
| `random_bl_65node_scan/step30000_phys/all_head_methods_none_separated_65.json` | phys-remapped/oracle-ish | 64 | 12 | 2 | strong_pass | L0H6 L | 1.000 | 0.057 | 1.000 | 1.000 |
| `random_bl_65node_scan/step5000/all_head_methods_none_separated_65.json` | none-separated | 64 | 0 | 0 | fail | L0H2 C-D+L | 0.039 | 0.059 | 0.169 | 0.200 |
| `random_bl_65node_scan/step5000_phys/all_head_methods_none_separated_65.json` | phys-remapped/oracle-ish | 64 | 1 | 0 | strong_pass | L0H3 C-D+L | 0.996 | 0.070 | 0.996 | 1.000 |
| `random_bl_65node_scan/step60000/all_head_methods_none_separated_65.json` | none-separated | 64 | 0 | 0 | fail | L1H5 C-D+L | 0.039 | 0.079 | 0.162 | 0.162 |
| `random_bl_65node_scan/step60000_phys/all_head_methods_none_separated_65.json` | phys-remapped/oracle-ish | 64 | 10 | 2 | strong_pass | L0H6 L | 1.000 | 0.055 | 1.000 | 1.000 |

Key caveat: the `*_phys` and non-strict `none_separated_65_head_method_search` rows are useful diagnostics but are not the strongest answer to oracle-selection criticism. For that, use `strict_label_free_65_search`; for 317M, current strict label-free scan is negative: 0 strong / 0 weak, best tau only 0.134 by combined-score row and max positive tau 0.291.

## C. B1 predictor 3-seed ladder

This is a predictor/none-mode diagnostic, not a replacement for strict-65 label-free discovery. It is still useful as evidence that the B1 pipeline has multi-seed stability.

| step | none_mode | strong abs(tau)>=0.9 per seed | best abs heads | heavy baseline tau |
|---:|---|---|---|---|
| 0 | predictor | s0:26, s1:28, s2:28 | s0:L1H1 +0.952, s1:L1H1 +0.955, s2:L1H1 +0.950 | s0:+1.000, s1:+1.000, s2:+1.000 |
| 1000 | predictor | s0:10, s1:9, s2:9 | s0:L0H6 +0.993, s1:L0H6 +0.989, s2:L0H6 +0.989 | s0:+0.998, s1:+0.999, s2:+0.998 |
| 5000 | predictor | s0:7, s1:7, s2:7 | s0:L0H0 +1.000, s1:L0H0 +1.000, s2:L0H0 +1.000 | s0:+1.000, s1:+1.000, s2:+0.998 |
| 10000 | predictor | s0:7, s1:7, s2:7 | s0:L0H0 +1.000, s1:L0H0 +1.000, s2:L0H0 +1.000 | s0:+1.000, s1:+1.000, s2:+1.000 |
| 20000 | predictor | s0:7, s1:7, s2:7 | s0:L0H0 +1.000, s1:L0H0 +1.000, s2:L0H0 +1.000 | s0:+1.000, s1:+1.000, s2:+1.000 |
| 30000 | predictor | s0:6, s1:6, s2:6 | s0:L0H0 +1.000, s1:L0H0 +1.000, s2:L0H0 +1.000 | s0:+1.000, s1:+1.000, s2:+1.000 |
| 40000 | predictor | s0:6, s1:6, s2:6 | s0:L0H0 +1.000, s1:L0H0 +1.000, s2:L0H0 +1.000 | s0:+1.000, s1:+1.000, s2:+1.000 |
| 50000 | predictor | s0:6, s1:6, s2:6 | s0:L0H0 +1.000, s1:L0H0 +1.000, s2:L0H0 +1.000 | s0:+1.000, s1:+1.000, s2:+1.000 |
| 60000 | predictor | s0:6, s1:6, s2:6 | s0:L0H0 +1.000, s1:L0H0 +1.000, s2:L0H0 +1.000 | s0:+1.000, s1:+1.000, s2:+1.000 |

## D. Matched training curves available now

| group | run | @50k val_ori_l2r | final step | final val_ori_l2r | recovery@50k vs matched refs | T<=3.47 |
|---|---|---:|---:|---:|---:|---|
| seed42 | `random_baseline_continuous_jun05` | 3.466 | 50000 | 3.466 | 0.0 | 47000 (3.470) |
| seed42 | `l2r_continuous_jun05` | 3.341 | 50000 | 3.341 | 100.0 | 19000 (3.467) |
| seed42 | `frozen_beta_random_jun05_from10k` | 3.334 | 60000 | 3.336 | 106.1 | 25500 (3.466) |
| seed42 | `frozen_beta_random_jun05_from20k` | 3.339 | 60000 | 3.339 | 101.5 | 28000 (3.467) |
| seed42 | `frozen_beta_random_jun05_from40k` | 3.383 | 60000 | 3.370 | 66.9 | 41500 (3.464) |
| seed123 | `random_baseline_continuous_jun08_seed2` | 3.445 | 50000 | 3.445 | 0.0 | 42000 (3.467) |
| seed123 | `l2r_continuous_seed123` | 3.305 | 50000 | 3.305 | 100.0 | 15000 (3.465) |
| seed123 | `frozen_beta_seed2_from10k_l0h2` | 3.325 | 60000 | 3.331 | 86.0 | 24500 (3.469) |
| seed123 | `frozen_beta_seed2_from20k_l0h2` | 3.329 | 60000 | 3.332 | 83.0 | 27000 (3.463) |
| seed123 | `frozen_beta_seed2_from40k_l0h2` | 3.363 | 60000 | 3.351 | 58.7 | 40500 (3.464) |
| seed123 | `cdl_teacher_seed123_from10k_l0h2` | 3.342 | 60000 | 3.345 | 74.1 | 27500 (3.467) |
| seed2 | `random_baseline_continuous_jun11_seed2_headtrack` | 3.413 | 50000 | 3.413 | 0.0 | 38000 (3.466) |
| seed2 | `l2r_continuous_seed2` | 3.370 | 60000 | 3.364 | 100.0 | 26000 (3.469) |
| seed2 | `cdl_teacher_seed2_from10k_l0h2` | 3.284 | 60000 | 3.288 | 302.2 | 22000 (3.467) |
| seed2 | `cdl_teacher_seed2_from20k_l0h2` | 3.308 | 60000 | 3.307 | 245.0 | 26500 (3.467) |
| seed2 | `cdl_teacher_seed2_from40k_l0h2` | 3.353 | 60000 | 3.341 | 138.6 | 40500 (3.463) |
| seed2 | `frozen_beta_b1_seed2_from20000` | 3.329 | 60000 | 3.332 | 196.5 | 28000 (3.469) |
| seed2 | `frozen_beta_b1_seed2_from40000` | 3.378 | 60000 | 3.366 | 81.6 | 41500 (3.470) |
| seed124 | `random_baseline_b1_headscan_seed124` | 3.478 | 60000 | 3.477 | 0.0 | no |
| seed124 | `cdl_teacher_seed124_from10k_l0h0_b1` | 3.322 | 60000 | 3.335 | nan | 20000 (3.469) |
| seed124 | `cdl_teacher_from20k_seed124` | 3.328 | 60000 | 3.339 | nan | 27000 (3.459) |
| seed124 | `cdl_teacher_from40k_seed124` | 3.383 | 60000 | 3.369 | nan | 42500 (3.461) |
| seed124 | `frozen_beta_from20k_seed124` | 3.357 | 60000 | 3.356 | nan | 30500 (3.467) |
| seed124 | `frozen_beta_from40k_seed124` | 3.910 | 60000 | 4.209 | nan | no |

## E. Claim edits implied by current data

- Replace `selected matched run` with: `matched seed2 and seed123 CDL-teacher evidence, plus seed124 supplementary runs pending matched L2R`.
- Replace `significantly improves` with: `improves in matched-run evidence; statistical significance remains untested`.
- State 317M honestly as `diagnostic negative under current strict-label-free scan`, not scale support.
- Use strict label-free collaborator result as the cleanest answer to oracle-remapping, but still acknowledge head selection/audition needs held-out or label-free selection.
- Keep legacy g_beta acceleration separate until strict-65 teacher is distilled and hooked.
