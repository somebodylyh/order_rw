# Round-2 readout: Bcov_balanced — FROZEN definition

**Date frozen:** 2026-05-20. Locked before launching Round-2 so that the primary comparison
(Bcov_balanced vs Hilbert) genuinely tests a B-guided readout, not generic nearest-neighbor
spatial filling. Implementation: `block_lo_arm_order_network/readout_order_diagnostic.py`,
function `b_coverage_order(B, gamma_B, gamma_d, start)`.

## Score (per reveal step t)

Among unvisited candidates U, with `last` = previously revealed block:
```
bn(v) = minmax_normalize_over_U( B[last, v] )        # attention-graph local edge, [0,1]
dn(v) = minmax_normalize_over_U( manhattan(last, v) ) # spatial distance on 8x8 grid, [0,1]
score(v) = gamma_B * bn(v) - gamma_d * dn(v)
next = argmax_{v in U} score(v)                       # deterministic (diagnostic)
```
Per-step min-max normalization makes B and distance commensurable (raw B ~0.008, distance
1–14). Deterministic argmax for reproducible diagnostics; a tau-softmax stochastic variant
may be added for training-time regularization but is NOT part of the frozen definition.

## Frozen weights
- **Bcov_balanced (the main Round-2 method): `gamma_B = 1.0, gamma_d = 1.0`.**
- Reference variants kept for the diagnostic table only:
  - `Bcov_distance_only` (B tie-break): gamma_B = 0.01, gamma_d = 1.0
  - `Bcov_Bdominant_safety`: gamma_B = 1.0, gamma_d = 0.2

## Diagnostic — Bcov_balanced is distinct from both Hilbert and nearest-neighbor

| readout | mean_manh | P(d≤1) | same_q |
|---|--:|--:|--:|
| hilbert | 1.000 | 1.000 | 0.952 |
| Bcov_distance_only (nn / B tie-break) | 1.130 | 0.946 | 0.866 |
| **Bcov_balanced (gamma_B=gamma_d=1)** | **2.407** | **0.556** | **0.690** |
| Bcov_Bdominant_safety | 3.063 | 0.513 | 0.639 |

Position-wise disagreement: Bcov_balanced vs Hilbert 0.979; vs distance_only 0.941.

**Evidence B materially drives selection (not just tie-break):** the order statistics respond
monotonically to gamma_B — turning B on (gamma_B 0.01 → 1.0) moves mean_manh 1.13 → 2.41 and
P(d≤1) 0.95 → 0.56. If B were ignored, varying gamma_B would not change the order. (Note: the
disagreement metric alone saturates at ~0.96–0.98 for any two distinct 64-orders — even
Hilbert vs nn is 0.961 — so the gamma_B→manh response, not disagreement, is the real proof.)

Bcov_balanced is thus genuinely B-guided: clearly different from generic spatial filling
(manh 2.41 ≫ Hilbert 1.0), still far from random (P(d≤1) 0.56 ≫ 0.04).

## Round-2 continuation matrix (NOT yet launched)
```
cont_random           baseline
cont_v1_graph_rw      failed v1 readout, control (≈ random)
cont_hilbert          generic spatial-filling, locality upper (manh 1.0)
cont_Bcov_balanced    B-guided main method (manh 2.41, B-driven)
cont_raster           specialization upper reference
```
- **Primary:** Bcov_balanced vs Hilbert — does B's directed structure add anything over a
  generic local curve?
- Secondary: hilbert vs random (generic spatial-filling effect); Bcov_balanced vs
  v1_graph_rw (improved-readout effect); raster vs hilbert/Bcov (specialization upper ref).
- Decide by val loss / cross-order robustness / sample quality — locality is a diagnostic,
  not the objective.
- Negative control `cont_shuffled_B_coverage` added only AFTER Bcov_balanced shows an effect.
