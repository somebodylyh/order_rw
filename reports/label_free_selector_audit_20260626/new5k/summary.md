# Label-Free Selector Audit

Selection score uses graph-intrinsic features only. Post-hoc tau/gate columns are not used for ranking.

- A tensor: `reports/5k_signal_provenance_audit_20260626/old_diag_jun25_5k/A_with_none_lh_mean.npy`
- Oracle JSON for post-hoc only: `reports/5k_signal_provenance_audit_20260626/old_diag_jun25_5k/all_head_methods_none_separated_65.json`
- Methods: `C-D+L, L`
- Control seeds: `[0, 1, 2, 3, 4, 5, 6, 7]`

## Top Label-Free Candidates

| rank | candidate | LF score | gate | delta_margin | entropy | topk_mass | sink | col_mass | posthoc_tau | posthoc_gate |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | L2H6 C-D+L | 5.409 | True | 0.01676 | 0.757 | 0.441 | 0.031 | 0.017 | -0.707 | fail |
| 2 | L2H6 L | 4.424 | True | 0.01502 | 0.757 | 0.441 | 0.031 | 0.017 | -0.701 | fail |
| 3 | L0H2 C-D+L | 3.462 | True | 0.008763 | 0.724 | 0.487 | 0.031 | 0.017 | -0.115 | fail |
| 4 | L0H2 L | 3.136 | True | 0.007271 | 0.724 | 0.487 | 0.031 | 0.017 | -0.121 | fail |
| 5 | L2H2 C-D+L | 2.620 | True | 0.005046 | 0.887 | 0.280 | 0.031 | 0.016 | 0.501 | fail |
| 6 | L2H2 L | 2.526 | True | 0.004532 | 0.887 | 0.280 | 0.031 | 0.016 | 0.501 | fail |
| 7 | L3H2 L | 2.518 | True | 0.001743 | 0.901 | 0.303 | 0.031 | 0.016 | -0.164 | fail |
| 8 | L3H3 C-D+L | 2.367 | True | 0.003764 | 0.922 | 0.237 | 0.031 | 0.016 | 0.707 | fail |

## Top Structure-Only Candidates

This ablation uses sharpness/top-k mass/non-sink/tie features only; it does not use destroyed-control margin.

| rank | candidate | structure score | entropy | topk_mass | sink | col_mass | posthoc_tau | posthoc_gate |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | L1H7 C-D+L | 1.118 | 0.393 | 0.782 | 0.031 | 0.022 | 1.000 | strong_pass |
| 2 | L1H7 L | 1.118 | 0.393 | 0.782 | 0.031 | 0.022 | 1.000 | strong_pass |
| 3 | L1H0 C-D+L | 0.889 | 0.473 | 0.705 | 0.031 | 0.018 | 1.000 | strong_pass |
| 4 | L1H0 L | 0.889 | 0.473 | 0.705 | 0.031 | 0.018 | 1.000 | strong_pass |
| 5 | L0H5 C-D+L | 0.683 | 0.617 | 0.677 | 0.031 | 0.023 | -1.000 | fail |
| 6 | L0H5 L | 0.683 | 0.617 | 0.677 | 0.031 | 0.023 | -0.957 | fail |
| 7 | L1H2 C-D+L | 0.678 | 0.573 | 0.651 | 0.031 | 0.020 | -1.000 | fail |
| 8 | L1H2 L | 0.678 | 0.573 | 0.651 | 0.031 | 0.020 | -1.000 | fail |

## Post-Hoc Summary

- Top-8 oracle strong-pass hits: 0
- Structure-only top-8 oracle strong-pass hits: 4
- Best selected candidate: L2H6 C-D+L
- Best structure-only candidate: L1H7 C-D+L

Reminder: oracle metrics in this section are revealed after selection.
