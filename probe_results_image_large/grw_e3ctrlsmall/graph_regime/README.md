# Structure-adaptive readout — graph diagnostics → regime (Stage A + B)

`block_lo_arm_order_network/graph_regime_diagnostic.py`. Unified pipeline:
`B → g(B) [diagnostics] → regime → suggested readout`. Modality-specific behavior emerges
from graph structure, not from manually assigning algorithms to data types.

## Regime table (2026-05-20)

| graph | s_readiness | locality_score | directionality | regime → readout |
|---|--:|--:|--:|---|
| text_clean_method | 11.85 | 0.95 | 0.984 | **readiness_dominant** → readiness Graph-RW (v3, high ρ) |
| E3_ctrl_small_image | 2.49 | 0.77 | 0.500 | **proximity_dominant** → coverage readout (Bcov_balanced / Hilbert); single-step RW insufficient |
| E2_singletoken_vq_image | 7.75 | 0.04 | 0.23 | uniform_noisy → random fallback |
| E2_large_image | 3.41 | 0.00 | 0.27 | uniform_noisy → random fallback |
| E3_shuffled_cols (ctrl) | 2.99 | 0.02 | 0.25 | uniform_noisy → random fallback |
| random_B (ctrl) | 5.08 | 0.04 | 0.25 | uniform_noisy → random fallback |

## Two-stage rule (Stage B)

1. **Structured?** `locality_score ≥ 0.5 AND P(nbr≤1) ≥ 0.4` (strongest edges point to a
   topological neighbor).
2. If structured, split by **directionality** (concentration of argmax-edge directions):
   - `directionality ≥ 0.7` → one dominant direction → **readiness_dominant** (causal/L2R,
     e.g. text attends to i−1). Use readiness-guided Graph-RW.
   - else → directions spread over neighbors → **proximity_dominant** (2D spatial, e.g.
     image 4-neighbor). Use coverage readout (single-step Graph-RW is insufficient).
3. Else if `row_entropy ≥ 0.93 AND top1_mass ≤ 0.05` → **uniform_noisy** → random fallback,
   do not over-trust B.
4. Else → **weak_mixed** → random-mixed continuation, validate before trusting B.

## Why directionality is the unifier
`s_readiness` alone is NOT sufficient: E3 image B has s=2.49 (high) yet is proximity, not
text — so a readiness-only rule misclassifies it (the original Stage-1 trap). The real
discriminator is that in 1D (text) spatial-locality and readiness/causal structure are
entangled (the previous token is both the nearest neighbor and the prerequisite), whereas in
2D (image) proximity is multi-directional. `directionality` (1D directional ⇒ readiness;
multi-directional ⇒ proximity) separates them, and reproduces every Stage-1 observation:
text readiness_only τ=0.887; E3 P(d≤1)=0.984 with RW manh 4.95 (proximity but RW fails);
E2 P(d≤1)=0.047 (no structure).

## Paper framing
We view order readout as a graph-diagnostic problem: characterize B's structural regime
(readiness strength, asymmetry, entropy, locality, directionality), then choose readout
parameters/regime accordingly. Text graphs activate a readiness-dominant regime, image
patch-block graphs a proximity-dominant regime, near-uniform VQ graphs a no-structure
fallback — a unified framework where modality-specific behavior emerges from graph structure
rather than hand-assigned per data type. Current version is rule-based; a learned
graph-to-policy hypernetwork `w = f_φ(g(B))` is a natural follow-up.

## Next: Round-2 (Step C) validates the prediction
The diagnostic predicts E3 image B needs a proximity/coverage regime. Round-2 (random /
v1_graph_rw / hilbert / Bcov_balanced / raster) tests whether that prediction yields a
training benefit. Not yet launched.
