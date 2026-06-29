# P2 — Physical-Order Signal Source: Results (B / B+ / C)

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating` · **Compute:** no new training (CPU)
**Code:** `analyses/physical_signal_source.py`, `analyses/plot_physical_signal_source.py`
**Outputs:** `runs/physical_signal_source/seed{2,42,123}/{source.json,source.csv,source_metrics.png}`
**Foundation:** `analyses/canonical_order_README.md` (the canonical physical readout; the L0
carrier emerges, real, above the destroyed floor). This file answers **what its *source* is**.

## Question (recap)

The L0 physical-order carrier recovers the original block order. Under the **fixed** training
layout this is isomorphic at the order level (`τ_physical=1 ⟺ σ_model=block_perm`), so the
order cannot tell apart:
- **B** — a content-free fixed-layout slot→physical position map;
- **C** — content-dependent recovery;
- **B+** — a fixed map with a content-modulation residual.

We measure whether the carrier's **attention/B graph** depends on **content**, holding layout
and reveal orders fixed and varying only the text — never the (tautological) order output.

## Headline: **B+ in all three seeds** — a dominant fixed-layout map with a content residual

| seed | carrier (L0) | slot-only R² (raw; null = −0.08) | content/noise-floor ratio | relayout (anchor→mean) | verdict |
|------|--------------|----------------------------------|---------------------------|------------------------|---------|
| 2   | {2,3,4,5} | 0.96 – 0.99 | 7 – 19 | 1.00 → 0.13 (collapse) | **B+** |
| 42  | {2}       | 0.87        | 2.2    | 0.96 → 0.14 (collapse) | **B+** |
| 123 | {1,2,3,4} | 0.85 – 1.00 | 3 – 19 | 1.00 → 0.13 (collapse) | **B+** |

Three independent lines of evidence, consistent across seeds:

1. **Dominant fixed-layout map (B base).** A **content-free slot-pair mean table**
   (`B_hat[i,j] = mean over training texts`, no content features) predicts each held-out
   text's carrier B65 with **raw R² ≈ 0.85–0.99**, far above the content-randomized null
   (R² ≈ −0.08). So most of the carrier's attention pattern is a *fixed function of slot
   position* — the learned inverse of the fixed CleanPermutation.
2. **Real but secondary content modulation (the "+").** Between-text variance, **above the
   within-text reveal-split sampling-noise floor**, is a genuine content-dependent residual:
   `content_variance / noise_floor ≈ 2–19`. So it is **not pure B** — content measurably
   modulates the carrier — but this residual is small relative to the fixed-map base that
   the R² shows.
3. **No cross-layout generalization (consistent with B, not C).** Under OOD relayout the
   carrier τ **collapses 1.0 → ~0.13** (near the destroyed floor). A content-dependent
   recovery that generalized would survive; the collapse is consistent with a fixed-layout
   map (the model only ever saw one layout).

## Interpretation (the corrected, sober claim)

> **The L0 "physical-order recovery" is primarily the model inverting its own fixed input
> permutation — a learned slot→physical position map — with a smaller content-dependent
> modulation on top (B+). It is NOT primarily content-driven semantic order recovery (C),
> and NOT a pure position scaffold (B).**

This finalizes the earlier corrections: the canonical readout (P1) established the signal is
real and emergent (not an init artifact, not the L1 slot scaffold); P2 now shows its *source*
is a dominant fixed-layout map + content residual, retiring the premature "content-bound"
claim while keeping the real, non-trivial phenomenon.

## Method notes / caveats

- **Order output cannot disambiguate** here (proven: `τ_physical=1 ⟺ σ_model=block_perm`); the
  whole verdict rests on attention/B content-dependence, calibrated against synthetic
  content-invariant (variance≈0) and content-randomized anchors, on the valid B65 edge support
  with L1 row-normalization.
- **Raw vs normalized R².** Row-normalized R² has a high randomized null (≈0.68), so it is not
  strong B evidence on its own; the verdict uses the **raw** R² (null ≈ −0.08). Both are
  reported in `source.csv`.
- **Sampling-noise floor.** `content_variance` subtracts a within-text reveal-split floor
  (per text, two disjoint halves of `n_reveals=32`), so the content residual is above sampling
  noise — not a low-`n_reveals` artifact (the Task-1 finding that prompted this control).
- **seed42** is the single-head, late-emerging carrier (L0{2}); its content ratio (2.2) is the
  weakest, R² (0.87) the lowest — reported separately, same B+ direction.
- **Single training layout** is the core limitation: relayout collapse cannot *refute* C, only
  support B. Truly separating "learned fixed map" from "could-be-content-if-trained-multi-layout"
  needs **multi-layout training** (deferred).

## P3′ fork (pre-registered next step) — verdict is B+

P3′ (causal verification, separate spec) targets **separating the fixed-layout base map from
the content-modulation residual** on the L0 carrier:
- **base map:** intervene on the positional / slot / QK-geometry path → expect the bulk of
  physical τ to collapse (the map lives there);
- **content residual:** content corruption / block replacement at fixed slots → expect a
  smaller, head-specific change (the modulation);
- avoid output-ablation→same-head-readout (degenerate for an attention-map metric).
A decisive strengthening of the content side would be **multi-layout training** + re-running P2.
