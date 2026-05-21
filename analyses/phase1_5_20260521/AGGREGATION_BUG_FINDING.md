# Phase 1.5 — Critical finding: per-sample g(B) aggregation bug

**Date:** 2026-05-21. Found while running the aggregation-knee experiment (1.5.1) as a
cross-check against the graph-diversity REPORT.

## The bug
`analyses/graph_diversity_20260521/extract_per_sample_gB.py` aggregates the 256-token
attention to a 64-block matrix using **`block_len=4` contiguous pooling** (block k = tokens
[4k..4k+3]). For the patch2x2 setups (E3-control-small, E3-large) this is the WRONG
token→block mapping. The correct mapping is the spatial one
(`diagnose_e3_control_dual_level.aggregate_token_to_block` / `token_to_patch_indices`,
16×16 token grid → 8×8 patch grid), which is how the reference `A_block_8x8.npy`
(P(d≤1)=0.984) was produced.

## Decisive test (same global A_global, two aggregations)
| aggregation | p_nbr≤1 | argmax_dist |
|---|--:|--:|
| token_to_patch_indices (spatial, CORRECT) | **0.984** | 1.156 |
| block_len=4 contiguous (used by extract_per_sample_gB) | 0.047 | 5.406 |

block_len=4 scrambles spatial structure even on the GLOBAL A (0.047 ≈ random). So the
per-sample locality metrics in the diversity REPORT for E3/E3-large are an **aggregation
artifact**, not evidence that single-sample attention lacks locality.

## Corrected per-sample picture (1.5.1 knee, CORRECT aggregation, E3-control-small)
| n images averaged | P(nbr≤1) | locality_score | argmax_dist |
|--:|--:|--:|--:|
| 1 | 0.461 ± 0.053 | 0.346 | 3.490 |
| 5 | 0.834 ± 0.044 | 0.657 | 1.829 |
| 30 | 0.959 ± 0.016 | 0.761 | 1.277 |
| 50 | 0.970 | 0.770 | 1.227 |
| global (500) | 0.984 | 0.783 | 1.156 |

- **Single-sample block-B is NOT random** — p_nbr≤1 = 0.46 (random ≈ 0.047). Locality is
  already substantially present in one sample at block level.
- **Knee ≈ n=5** (reaches 80% of global); saturates by n≈30–50.

## Implications (overturns part of the REPORT)
1. The REPORT's headline "per-sample g(B) ≈ random on locality; locality only emerges after
   population averaging" is an artifact of the wrong aggregation for E3/E3-large.
2. The REPORT's per-sample table corrupts ALL metrics for patch2x2 setups (block_len=4 and
   token_to_patch pool different token sets → different B → all g(B) features differ), so the
   Case-A separation / PCA used corrupted features for E3/E3-large and must be re-validated.
3. CEM/controller input: per-sample block-B already carries locality; a **modest batch
   (n≈10–30) reaches ~0.95** of global. So "batch-averaged B" is sufficient — full population
   is NOT required, and per-sample is not useless. This weakens the "must use population-level
   B" conclusion to "a small rolling/batch average suffices."

## Recommended fix + re-run
- Fix `extract_per_sample_gB.py` to use the spatial `token_to_patch_indices` aggregation for
  patch2x2 setups (E2 seq64 is single-token / 8×8 already, no 2×2 pooling — unaffected).
- Re-run the diversity analysis; re-check Case-A separation and the per-sample-vs-global
  comparison with the corrected B.
- E2 genuinely near-random at global too (P(d≤1)=0.047), so its no-structure verdict stands.
