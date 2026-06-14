# Figures Index

> All figures for the evidence package. Quality check: title, axis labels, ori-L2R reference (not upper bound), legend clarity.

---

## Core Figures (Paper/Report Ready)

| Figure | Path | What it shows | Paper use? | Quality |
|------|------|------|:--:|------|
| fig1_method_overview | `analyses/figures/fig1_method_overview.png` | B→CDL→g_β→frozen hook pipeline | ✅ Section 3 (Method) | ✅ Clear |
| fig2_gbeta_sanity_bar | `analyses/figures/fig2_gbeta_sanity_bar.png` | Real vs Gaussian/Shuffled/Destroyed B → g_β τ | ✅ Section 5 (Mechanism) | ⚠️ Add numeric labels on bars; add "tie-breaking" annotation for zero B |
| fig3_catchup_curve | `analyses/figures/fig3_catchup_curve.png` | ori-L2R / random / frozen_β on same axes | ✅ Section 6 (Results) | ✅ Clear; ylim 3.2–4.0; ori-L2R labeled "reference" |
| fig4_multistart_comparison | `analyses/figures/fig4_multistart_comparison.png` | seed2 + seed42 × from10k/20k/40k | ✅ Section 6 (Results) | ✅ Clear; ylim 3.2–4.0; separate subplots per seed |

---

## Supplementary Figures

| Figure | Path | What it shows | Paper use? | Quality |
|------|------|------|:--:|------|
| shuffle_gran128_B_heatmaps | `analyses/figures/shuffle_gran128_B_heatmaps.png` | 6 strongest heads' B matrices (128-block training) | 🔶 Appendix | ✅ Clear; 2×3 grid; each head labeled with τ |
| head_signal_emergence | `analyses/figures/head_signal_emergence.png` | Head signal emergence over training | 🔶 Appendix | ⚠️ Verify data source and step range |
| seed2_attention_heatmaps | `analyses/figures/seed2_attention_heatmaps.png` | Seed2 key heads' B matrices across checkpoints | 🔶 Appendix | ⚠️ Check step labels; old figure from earlier seed2 analysis |
| frozen_beta_multi_start_comparison | `probe_results/frozen_beta_multi_start_comparison.png` | Older multi-start overview | ❌ Superseded by fig4 | ⚠️ May mix seeds/heads; use fig4 instead |
| L0H2_B_heatmap_30k | `analyses/figures/L0H2_B_heatmap_30k.png` | L0H2 B matrix at 30k | 🔶 Appendix (single-head B visualization) | ✅ Single head detail |
| L0H2_B_model_vs_phys | `analyses/figures/L0H2_B_model_vs_phys.png` | Model vs physical coordinate comparison | 🔶 Protocol illustration | ✅ Protocol validation |
| seed2_B_diagnostics | `analyses/figures/seed2_B_diagnostics.png` | B matrix diagnostics for seed2 | 🔶 Appendix | ⚠️ Verify axis labels |

---

## Diagnostic Figures (Not for Paper)

| Figure | Path | What it shows |
|------|------|------|
| cdl_tau_heatmap_3panel | `analyses/attention_diagnostic_20260609/cdl_tau_heatmap_3panel.png` | 3-panel CDL τ heatmap |
| wall_clock_overlay | `analyses/attention_diagnostic_20260609/wall_clock_overlay.png` | Wall clock step saving overlay |
| scatter_pca_umap | `analyses/graph_diversity_20260521/scatter_pca_umap.png` | Per-sample gB PCA/UMAP |
| violin_per_metric | `analyses/graph_diversity_20260521/violin_per_metric.png` | Per-metric violin plots |

---

## Quality Checklist for Core Figures

| Check | fig1 | fig2 | fig3 | fig4 |
|------|:--:|:--:|:--:|:--:|
| Title accurate | ✅ | ✅ | ✅ | ✅ |
| Axis labels clear | ✅ | ✅ | ✅ | ✅ |
| ori-L2R = "reference" (not upper bound) | — | — | ✅ | ✅ |
| Step/seed consistent | — | — | ✅ | ✅ |
| Legend distinguishes seeds/heads | — | ✅ | ✅ | ✅ |
| Colorblind-friendly | ⚠️ | ✅ | ⚠️ | ⚠️ |
| 300+ DPI for paper | ❌ (150) | ❌ (150) | ❌ (150) | ❌ (150) |
| Vector format available | ❌ | ❌ | ❌ | ❌ |

**Action items**:
- [ ] Regenerate core figures at 300 DPI
- [ ] Add vector format (PDF/SVG) for paper submission
- [ ] Check colorblind accessibility (red-green in fig3/fig4)
- [ ] Add "tie-breaking artifact" annotation to zero B bar in fig2
