# Phase 2.3 CEM dry-run — report

**Date:** 2026-05-21. Script: `block_lo_arm_order_network/cem_readout_search.py --mode sanity`.
Behavioral gate = primary; term-attribution = descriptive; term-ablation = causal necessity.

## Gate result: ALL 3 behavioral tests PASS

| test | behavioral metric | pass criterion | result |
|---|---|---|---|
| 1 proximity (E3) | mean_manh = 2.08 | < 3.0 | **PASS** |
| 2 no-false-structure (random B, γ_d disabled) | mean_manh = 5.27 (E3 cross-check 3.30) | > 4.5 | **PASS** |
| 3 readiness (text) | directionality = 1.00 | > 0.7 | **PASS** |

The unified readout family + CEM loop recover the correct readout BEHAVIOR for each regime.

## Term-attribution is non-identifiable (DESCRIPTIVE only, NOT a gate)

The family is over-parameterized (multiple terms produce similar behavior; terms are
correlated — like multicollinearity). Dominant-weight is therefore unreliable:
- E3 best_w dominant: γ_d 0.36 > β_dep 0.20 > ρ 0.18 > β_sup 0.14 > γ_B 0.13
- text best_w dominant: γ_B 0.27 > ρ 0.25 > β_sup 0.20 > γ_d 0.17 > β_dep 0.11

We do NOT gate on "which weight is biggest." We use term-ablation (causal necessity) instead.

## Term-ablation (knock each term to 0; Δ behavior = causal necessity)

**Test 1 — E3 proximity (mean_manh; higher after knockout ⇒ term necessary for locality):**
| knockout | mean_manh | reading |
|---|--:|---|
| full best_w | 2.08 | — |
| β_sup=0 | 1.95 | not necessary (slightly helps) |
| β_dep=0 | 1.98 | not necessary |
| ρ=0 | 2.11 | not necessary |
| **γ_B=0** | **2.03** | **B-local term NOT necessary for proximity** |
| **γ_d=0** | **3.42** | **distance term IS necessary** |
| fallback=1 | 5.28 | random (sanity) |

**Headline (negative, important):** under the locality surrogate, E3 proximity is driven by
the graph-INDEPENDENT spatial distance term (γ_d), NOT by the attention graph B (γ_B). B
contributes ~nothing to mean_manh beyond pure distance. This confirms the long-standing
concern (Bcov_distance_only ≈ Bcov_balanced on manh). **B's structural value cannot be shown
with a locality objective — it requires task-loss fitness (Phase 3).**

**Test 3 — text readiness (directionality; lower after knockout ⇒ term necessary):**
| knockout | directionality | reading |
|---|--:|---|
| full best_w | 1.00 | — |
| **ρ=0** | **0.74** | **readiness term is the most necessary single term** |
| γ_B=0 | 0.97 | B-local barely necessary |
| others | 0.96–1.00 | not necessary |
| fallback=1 | 0.06 | random (sanity) |

The ablation RECOVERS "readiness drives text" causally (ρ necessary), even though ρ is not
the largest weight — so ablation is the right diagnostic, not dominant-weight.

**Test 2 — random B:** every knockout stays ~5.3 (random); no term induces structure → no
false structure. Confirmed.

## Implication for Phase 3 (sharpened, not blocked)
- Phase 3 fitness MUST be task-loss (frozen / short-continuation), NOT locality.
- The decisive comparison is whether B-driven orders (γ_B path) beat distance/space-filling
  orders (γ_d path / Hilbert) on generation loss. The locality surrogate cannot answer this
  (proximity is distance-achievable). So Phase-3 oracle should include a γ_d-disabled or
  Bcov-vs-Hilbert contrast to isolate B's contribution.
- Behavioral gate passed → machinery is sound. Phase 3 remains HELD pending review + the
  other Phase-3 launch conditions (Task 1.2 TSV + spot-check + audit completion).
