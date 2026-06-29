# Figures To Show

Existing figure scan:

- `analyses/figures/fig1_method_overview.png`
- `analyses/figures/fig2_gbeta_sanity_bar.png`
- `analyses/figures/fig3_catchup_curve.png`
- `analyses/figures/fig4_multistart_comparison.png`
- `analyses/figures/attn_map_b1/b1_L0_H2_step30000_model_vs_phys.png`
- `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/l0h4_signal_curve.png`
- Other auxiliary figures in `analyses/figures/` and `block_lo_arm_order_network/probe_results/`

## Recommended Figures

| Figure | Path | Use in slide | What it proves | Caveat |
|---|---|---|---|---|
| Method overview | `analyses/figures/fig1_method_overview.png` | Slide 4 | Shows `B^{l,h} -> CDL -> g_beta -> frozen hook` pipeline | Need visual check that labels say B0/B1 correctly if used after B1 update |
| `g_beta` sanity bar | `analyses/figures/fig2_gbeta_sanity_bar.png` | Slide 6 | Real B high tau, destroyed/random B near zero | Should be labeled legacy B0 controller path |
| Catch-up curve | `analyses/figures/fig3_catchup_curve.png` | Slide 7 | Frozen `g_beta` reaches target loss earlier than random baseline | Needs visual check against corrected seed42 baseline 3.466 and verified 33-42% range |
| Multi-start comparison | `analyses/figures/fig4_multistart_comparison.png` | Slide 7 | Shows from10k/from20k/from40k behavior | Needs visual check against verified recovery table; do not use if old numbers remain |
| B1 model-vs-physical attention map | `analyses/figures/attn_map_b1/b1_L0_H2_step30000_model_vs_phys.png` | Slide 8 or appendix | Helps explain predictor-frame vs physical-remapped visualization | Visual only; not primary scalar evidence; head is L0H2 step30000, not final seed124 L0H4 |
| B1 L0H4 signal curve | `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/l0h4_signal_curve.png` | Slide 5 or Slide 8 | Shows latest continuous B1 L0H4 signal stability over training | Continuous seed124, not seed123/seed42 frozen hook group |
| L0H2 B heatmap | `analyses/figures/L0H2_B_heatmap_30k.png` | Backup | Shows selected-head graph structure visually | Protocol label must be checked before main slide use |
| Head signal emergence | `analyses/figures/head_signal_emergence.png` | Backup | Shows order signal emergence | Could contain older B0 framing; verify numbers before main deck |
| Frozen beta curves | `block_lo_arm_order_network/probe_results/frozen_beta_curves.png` | Backup | Training curve overview | Check if built from verified table; otherwise use `fig3`/`fig4` |

## Figure Guidance

- Use `fig1_method_overview.png`, `fig2_gbeta_sanity_bar.png`, `fig3_catchup_curve.png`, and `fig4_multistart_comparison.png` as the four core figures only after visually confirming they match the verified numbers.
- If a figure contains old "28/32 strong heads", seed42 random baseline `3.354`, or any B0/B1 mixed label, mark it `Needs regeneration`.
- For B1, prefer the scalar table plus `l0h4_signal_curve.png`; the attention map is useful but not the main proof.

