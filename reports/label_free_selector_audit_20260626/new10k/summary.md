# Label-Free Selector Audit

Selection score uses graph-intrinsic features only. Post-hoc tau/gate columns are not used for ranking.

- A tensor: `reports/10k_signal_carrier_layer_multiseed_20260626/seed123_new/A_with_none_lh_mean.npy`
- Oracle JSON for post-hoc only: `reports/10k_signal_carrier_layer_multiseed_20260626/seed123_new/all_head_methods_none_separated_65.json`
- Methods: `C-D+L, L`
- Control seeds: `[0, 1, 2, 3, 4, 5, 6, 7]`

## Top Label-Free Candidates

| rank | candidate | LF score | gate | delta_margin | entropy | topk_mass | sink | col_mass | posthoc_tau | posthoc_gate |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | L2H6 C-D+L | 5.945 | True | 0.01845 | 0.728 | 0.470 | 0.031 | 0.016 | -0.707 | fail |
| 2 | L2H6 L | 4.839 | True | 0.0176 | 0.728 | 0.470 | 0.031 | 0.016 | -0.701 | fail |
| 3 | L0H2 C-D+L | 3.449 | True | 0.009187 | 0.702 | 0.496 | 0.031 | 0.017 | -0.130 | fail |
| 4 | L0H2 L | 3.192 | True | 0.008893 | 0.702 | 0.496 | 0.031 | 0.017 | -0.134 | fail |
| 5 | L2H2 C-D+L | 2.680 | True | 0.005117 | 0.885 | 0.282 | 0.031 | 0.016 | 0.501 | fail |
| 6 | L2H2 L | 2.590 | True | 0.00475 | 0.885 | 0.282 | 0.031 | 0.016 | 0.501 | fail |
| 7 | L3H3 C-D+L | 2.132 | True | 0.001761 | 0.924 | 0.234 | 0.031 | 0.017 | 0.703 | fail |
| 8 | L3H3 L | 1.988 | True | 0.0005912 | 0.924 | 0.234 | 0.031 | 0.017 | -0.505 | fail |

## Top Structure-Only Candidates

This ablation uses sharpness/top-k mass/non-sink/tie features only; it does not use destroyed-control margin.

| rank | candidate | structure score | entropy | topk_mass | sink | col_mass | posthoc_tau | posthoc_gate |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | L1H7 C-D+L | 1.567 | 0.199 | 0.907 | 0.031 | 0.017 | 1.000 | strong_pass |
| 2 | L1H7 L | 1.567 | 0.199 | 0.907 | 0.031 | 0.017 | 1.000 | strong_pass |
| 3 | L1H0 C-D+L | 1.026 | 0.424 | 0.749 | 0.031 | 0.018 | 1.000 | strong_pass |
| 4 | L1H0 L | 1.026 | 0.424 | 0.749 | 0.031 | 0.018 | 1.000 | strong_pass |
| 5 | L1H2 C-D+L | 1.019 | 0.497 | 0.783 | 0.031 | 0.020 | -1.000 | fail |
| 6 | L1H2 L | 1.019 | 0.497 | 0.783 | 0.031 | 0.020 | -1.000 | fail |
| 7 | L0H5 C-D+L | 0.850 | 0.535 | 0.718 | 0.031 | 0.020 | -1.000 | fail |
| 8 | L0H5 L | 0.850 | 0.535 | 0.718 | 0.031 | 0.020 | -1.000 | fail |

## Post-Hoc Summary

- Top-8 oracle strong-pass hits: 0
- Structure-only top-8 oracle strong-pass hits: 4
- Best selected candidate: L2H6 C-D+L
- Best structure-only candidate: L1H7 C-D+L

Reminder: oracle metrics in this section are revealed after selection.
