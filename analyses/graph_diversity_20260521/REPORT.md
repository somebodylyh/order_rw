> # ⚠️ CORRECTED 2026-05-21 — read this banner first
>
> **The original "per-sample g(B) ≈ random on locality" conclusion below is DEPRECATED.**
> It was an **aggregation artifact**: `extract_per_sample_gB.py` pooled the 256-token
> attention to 64 blocks with `block_len=4` contiguous grouping, which scrambles the 2D
> spatial structure of the patch2x2 setup (E3). On a single global A, contiguous-4 gives
> p_nbr≤1 = 0.047 (≈random) vs the correct spatial `token_to_patch_indices` mapping = 0.984.
> Fix: e3_ctrl_small now uses the spatial mapping; text (1D) keeps contiguous-4 (correct);
> e2 uses no aggregation (unaffected). See `phase1_5_20260521/AGGREGATION_BUG_FINDING.md`.
>
> **Corrected per-sample E3-control-small (300 samples, spatial aggregation):**
> p_nbr≤1 = **0.552 ± 0.062** (NOT ~0.07), locality_score = **0.433 ± 0.063** (NOT ~0),
> argmax_dist = **3.02 ± 0.33** (NOT ~5.4/random), directionality = 0.379.
> Aggregation knee (`phase1_5_20260521/aggregation_knee.tsv`): n=1→0.46, n=5→0.83,
> n=30→0.96, n=50→0.97, global 0.984. **Knee ≈ n=5, saturates ~30–50.**
>
> **Corrected verdicts:**
> 1. Per-sample block-B is NOT random on locality — single-sample already carries
>    non-trivial locality; a modest batch (n≈10–30) reaches ~95% of global.
> 2. **Case A still holds (cleaner): all 12 g(B) metrics ratio<0.3, only e2_small↔e2_large
>    cross the boundary (1/300 ≈ 0.3%). And now the LOCALITY metrics also separate setups
>    (p_nbr≤1 ratio 0.127, locality_score 0.170) — separation is no longer driven only by
>    noise-amplified metrics.**
> 3. CEM/controller input: **batch / rolling-EMA block-B is sufficient** (NOT full-population
>    global). per-sample = signal-with-noise; batch B = recommended main input; global B =
>    stable offline upper reference. Good news for a future batch-level / rolling
>    attention-conditioned controller (AttnTNT).
>
> Everything below the line is the ORIGINAL (deprecated) report, kept for history.
>
> ---

# Graph Diversity Analysis: Within vs Between Setup Variance

**Date**: 2026-05-21  
**Question**: Does within-setup per-sample variance of g(B) cross setup boundaries?

---

## Executive Summary

**Verdict: Case A (with an important caveat)**

In the full 12-dimensional g(B) space, **setup label dominates overwhelmingly** — virtually zero samples cross setup boundaries (6/1200 = 0.5%, only between E2_small↔E2_large which are the same data at different model scale). 10 of 12 metrics have ratio(within_std / between_dist) < 0.3.

**However, the separation is driven by "noise-amplified" metrics (readiness_strength, argmax_dist) — not by the locality metrics (p_nbr_le1, locality_score) that define the structural regime at the global level.** At per-sample level, ALL setups are indistinguishable on locality metrics — including E3-control-small which has p_nbr_le1=0.95 at the population level.

---

## Key Finding: Per-Sample vs Global g(B)

| Metric | E3_ctrl_small GLOBAL | E3_ctrl_small PER-SAMPLE (mean±std) | Interpretation |
|---|---|---|---|
| readiness_strength | 2.49 | 16.39 ± 1.43 | Per-sample noise inflates readiness 6.6× |
| argmax_dist | 1.25 | 5.36 ± 0.29 | Global: neighbors dominate. Per-sample: near random (5.33) |
| p_nbr_le1 | 0.953 | 0.060 ± 0.024 | Global: 95% nearest-neighbor. Per-sample: 6% ≈ random |
| locality_score | 0.766 | 0.019 ± 0.030 | Global: strongly local. Per-sample: zero |
| directionality | 0.500 | 0.251 ± 0.029 | Global: multi-directional. Per-sample: random-like |

| Metric | Text GLOBAL | Text PER-SAMPLE (mean±std) | Interpretation |
|---|---|---|---|
| readiness_strength | 11.85 | 23.11 ± 3.16 | Noise doubles it |
| argmax_dist | 1.00 | 23.09 ± 1.43 | Global: perfect seq neighbor. Per-sample: random |
| p_nbr_le1 | 1.000 | 0.030 ± 0.015 | Global: 100%. Per-sample: 3% ≈ random |
| locality_score | 0.954 | 0.005 ± 0.018 | Global: near-perfect. Per-sample: zero |
| directionality | 0.984 | 0.530 ± 0.023 | Only metric partially preserved per-sample |

**Conclusion**: The spatial/sequential structure that characterizes each setup is a **population-level statistical effect**, not a per-sample property. Each individual sample's attention is near-random; the regime-defining structure emerges only through averaging.

---

## Ratio Analysis (within_std / between_distance)

| Metric | avg_within_std | avg_between_dist | Ratio | Verdict |
|---|---|---|---|---|
| readiness_strength | 2.38 | 10.15 | 0.23 | A (setup dominates) |
| asymmetry | 0.03 | 0.43 | 0.07 | A (setup dominates) |
| row_entropy | 0.004 | 0.04 | 0.09 | A (setup dominates) |
| top1_mass | 0.002 | 0.01 | 0.22 | A (setup dominates) |
| top4_mass | 0.005 | 0.03 | 0.14 | A (setup dominates) |
| argmax_dist | 0.58 | 8.89 | 0.07 | A (setup dominates) |
| **p_nbr_le1** | **0.024** | **0.015** | **1.60** | **B (within matters)** |
| **locality_score** | **0.028** | **0.009** | **3.00** | **B (within matters)** |
| local_greedy_dist | 0.48 | 8.58 | 0.06 | A (setup dominates) |
| directionality | 0.027 | 0.14 | 0.19 | A (setup dominates) |
| out_degree_std | 0.017 | 0.20 | 0.08 | A (setup dominates) |
| in_degree_std | 0.004 | 0.06 | 0.07 | A (setup dominates) |

The two "Case B" metrics (p_nbr_le1, locality_score) are NOT showing meaningful within-setup diversity crossing boundaries. They simply have **no between-setup separation at per-sample level** — all setups cluster at the random baseline value (~0.05 and ~0.02 respectively).

---

## Boundary Cases

| From → To | N crossing (of 300) | % |
|---|---|---|
| e2_small → e2_large | 5 | 1.7% |
| e2_large → e2_small | 1 | 0.3% |
| All other pairs | 0 | 0.0% |

Only E2_small ↔ E2_large (same data, different model scale) have any overlap at all.

---

## PCA

- PC1 explains 40.6%, PC2 explains 29.9% (70.5% total in 2D)
- Visual inspection shows complete cluster separation with no overlap between text, e3_ctrl_small, e2_small/large, and controls
- E2_small and E2_large slightly overlap (consistent with boundary case analysis)
- See: `scatter_pca_umap.png`

---

## What This Means for Paper Framing

1. **Per-sample g(B) diversity is real** — samples within a setup do vary (non-zero std on all metrics). A "learned controller" conditioned on per-sample g(B) would see different inputs across samples.

2. **BUT: the per-sample diversity is WITHIN the same regime** — it never crosses setup boundaries. There is no "text sample that looks like an image sample" in g(B) space.

3. **The critical implication**: The structural regime (proximity_dominant, readiness_dominant, uniform_noisy) is a **population-level property that requires averaging**. A controller operating on single-sample g(B) would NOT see the regime-defining metrics (p_nbr_le1, locality_score are all near zero per-sample). It would instead see readiness_strength and argmax_dist, which DO separate setups — but these capture noise-amplified scale properties, not the "interesting" structural information.

4. **For Paper 1 capstone figure** ("w* varies smoothly in g(B) space across setups"):
   - The figure IS empirically justified — setups separate cleanly with near-zero overlap
   - BUT the separation axis is NOT the locality/proximity axis that defines regimes at global level
   - A controller would need to operate on batched/rolling-average g(B), not single-sample g(B), to see the locality structure
   - Alternatively: the controller uses the per-sample metrics that DO discriminate (readiness_strength, directionality) as proxy for regime identity

---

## Caveats

1. **Sampling protocol**: M=3 random orders per sample, last 4 layers, all heads averaged. More orders would reduce per-sample noise but increase compute cost.
2. **Layer/head aggregation**: Averaging all heads/layers may wash out per-sample structure that exists in specific heads. A per-head analysis could reveal more.
3. **Block aggregation**: E3-control-small uses 4-token→1-block averaging, which smooths attention. Token-level analysis would be noisier.
4. **Text model is Graph-RW trained** (step 50k, α-schedule). Its attention may differ from a random-order baseline text model.
5. **E3-control-small checkpoint**: This is the final baseline (random-order trained). A graph-RW trained E3 model might show different per-sample structure.

---

## Files Produced

```
analyses/graph_diversity_20260521/
├── extract_per_sample_gB.py          # Extraction script
├── analyze_diversity.py              # Analysis script
├── e3_ctrl_small_per_sample_gB.tsv   # 300 samples × 12 dims
├── text_per_sample_gB.tsv            # 300 samples × 12 dims
├── e2_small_per_sample_gB.tsv        # 300 samples × 12 dims
├── e2_large_per_sample_gB.tsv        # 300 samples × 12 dims
├── random_uniform_per_sample_gB.tsv  # 300 synthetic controls
├── shuffled_rows_per_sample_gB.tsv   # 300 synthetic controls
├── all_setups_combined.tsv           # Merged (1800 rows)
├── within_setup_stats.tsv            # Per-setup per-metric summary
├── between_setup_distances.tsv       # Pairwise L2 distances
├── ratio_within_over_between.tsv     # Core verdict table
├── pca_2d.tsv                        # PCA coordinates
├── boundary_cases.json               # Cross-boundary samples
├── scatter_pca_umap.png              # 2D scatter (PCA only, UMAP unavailable)
├── violin_per_metric.png             # Violin plots per metric
└── REPORT.md                         # This report
```
