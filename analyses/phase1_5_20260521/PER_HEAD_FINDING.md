# Phase 1.5.2 — per-(layer,head) locality (E3-control-small, CORRECT spatial aggregation)

**Date:** 2026-05-21. Script: `analyses/phase1_5_per_head.py` (200 imgs × 3 passes,
spatial token_to_patch_indices pooling). Answers Q2: stable locality head?

> ⚠️ The old tool `block_lo_arm_order_network/layer_head_locality_scan.py` has the SAME
> block_len=4 contiguous aggregation bug AND hardcodes N_LAYERS=8 (E3 is l4). Its prior
> result (`probe_results_image_large/layer_head_scan/...step20000.tsv`, all heads ≈random)
> is DEPRECATED. This focused script supersedes it for E3.

## Q2 ANSWER: YES — locality is head-concentrated (not emergent-only-after-averaging)

5/32 heads have locality_score > 0.5, **all in layer 0**:

| layer | head | p_nbr≤1 | locality_score |
|--:|--:|--:|--:|
| 0 | 4 | 0.953 | 0.798 |
| 0 | 3 | 0.984 | 0.783 |
| 0 | 7 | 0.984 | 0.783 |
| 0 | 2 | 0.891 | 0.745 |
| 0 | 6 | 0.703 | 0.581 |
| 0 | 5 | 0.531 | 0.399 |
| (layers 1–3) | … | ≤0.45 | ≤0.37 (near-random) |
| GLOBAL all-head | | 0.984 | 0.783 |

- Locality lives in **layer-0 heads 2–7**. Layers 1–3 are near-random.
- The all-head GLOBAL locality (0.783) ≈ the layer-0 local heads — the near-uniform
  non-local heads do not dilute it after averaging/normalization.

## Implication for CEM input (Step C)
Two viable inputs, both better than the deprecated "must be full-population global":
- **head-selected / head-weighted B** (layer-0 local heads): sharpest signal, potentially
  usable at small batch or even per-sample (those heads are individually local).
- **all-head batch B** (n≈10–30 per the knee): simpler, also reaches global locality.

Recommend CEM be able to take either; default to all-head batch B, with head-selected B
as an ablation that may sharpen the proximity regime.
