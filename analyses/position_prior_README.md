# Pillar ⑤-core+ — Position-Prior Decomposition & Content-Binding (Results)

> **🔴 CORRECTION (2026-06-28, after review): the "artifact" headline below is
> OVER-CLAIMED.** This entire analysis used **identity `probe_orders`** (reveal order =
> model-slot order). Under identity reveal + a causal mask, the C-D+L readout recovers
> ascending order *near-tautologically*, so the step-0 τ≈0.77 and the floor τ=1.0 are
> artifacts **of the identity-probe protocol**, not statements about the *learned* order
> signal. Decisive check: seed2 L0H2 step-0 τ = **0.72 under identity probe but −0.03
> under random probe**. Under random reveal (the protocol of
> `analyses/figures/head_signal_emergence.png`), the order signal **genuinely emerges**
> (τ ≈ 0 → 1.0 at ~step 1800), consistent with Spec A's "emergence at ~2k". So: the order
> signal is **not** an artifact; only the identity-probe step-0 baseline is. The
> frame-sanity result (slot-indexed, not content-indexed) still holds, but **Part 2's
> slot-scaffold / τ_content≈0 must be re-run under a non-tautological (random-probe)
> protocol** before it can be trusted. **Read the sections below as identity-probe
> measurements, not as the line-level conclusion.**

**Date:** 2026-06-28 · **Branch:** `attn-order-alternating` · **Compute:** no new training (CPU)
**Code:** `analyses/position_prior_decomp.py`, `analyses/plot_position_prior.py`
**Outputs:** `runs/position_prior/seed{2,42,123}/{part1.json,part2.json,*.csv,*.png}`

## Headline (the line-reframing result)

**The "order signal" this whole project studied is a positional model-slot *scaffold*,
not a content-bound order representation — and at init it is essentially the
causal-mask + C-D+L readout artifact.** Across all three seeds:

1. A **pure uniform-causal attention** (no learned weights) rolled out through the real
   readout gives **τ = 1.00** — the C-D+L rollout reconstructs ascending model-slot order
   from *any* causally-masked attention.
2. The **untrained (step-0) model** reads τ ≈ 0.77–0.80, and **removing the position
   embeddings raises τ toward the 1.0 floor** (zero-both ≈ 0.85–0.90) — so random-init
   PEs only *add noise* to an already-near-perfect causal readout; they do not create the
   order.
3. Even the **converged carrier (step 10000)** is **pure slot-bound**: under OOD content
   relocation across 8 layouts, the carrier order follows the **slot** (τ_pos ≈ 0.97) and
   carries **~zero content** (τ_content ≈ 0.01).

## Part 1 — step-0 decomposition (3 seeds)

| seed | floor uniform-causal | random-B null | full | zero_wpe | zero_wtpe | zero_both | frame model_slot / physical / perm_base |
|------|----------------------|---------------|------|----------|-----------|-----------|------------------------------------------|
| 2   | 1.00 | 0.01 | 0.77 | 0.86 | 0.77 | **0.88** | 0.78 / 0.00 / −0.01 |
| 42  | 1.00 | 0.01 | 0.80 | 0.88 | 0.80 | **0.90** | 0.77 / −0.01 / −0.01 |
| 123 | 1.00 | 0.01 | 0.78 | 0.85 | 0.79 | **0.85** | 0.77 / −0.02 / −0.01 |

- **Floor = 1.0:** causal mask + C-D+L readout alone perfectly recovers ascending
  model-slot order. The order metric is, at root, a causal-sequence reconstructor.
- **Δ_wpe < 0, Δ_wtpe ≈ 0, Δ_both < 0:** removing `wpe` (token-side slot PE) *raises* τ;
  `wtpe` (target/AdaLN conditioning) has ~no effect at init. So the only PE contribution
  at init is `wpe` adding noise that pulls the readout slightly *below* the pure-causal
  ceiling — not a positive order contribution.
- **Frame sanity passes cleanly:** τ_model_slot ≈ 0.77 ≫ τ_physical ≈ 0.00 ≈
  τ_perm_baseline. The prior is firmly a **model-slot** prior, divorced from the
  (shuffled) physical content — exactly as the data-shuffle premise predicted.

## Part 2 — content vs position binding (3 seeds, OOD relayout, K=8)

| seed | step | anchor τ_pos | relay τ_pos | relay τ_content | valid | verdict |
|------|------|--------------|-------------|-----------------|-------|---------|
| 2   | 0 | 0.76 | 0.76 | 0.02 | no\* | invalid |
| 2   | 2000 | 0.99 | 0.98 | 0.02 | yes | **slot-scaffold** |
| 2   | 10000 | 0.96 | 0.97 | 0.01 | yes | **slot-scaffold** |
| 42  | 0 | 0.77 | 0.77 | 0.03 | yes | **slot-scaffold** |
| 42  | 2000 | 0.76 | 0.78 | 0.01 | yes | **slot-scaffold** |
| 42  | 10000 | 0.75 | 0.76 | 0.01 | yes | **slot-scaffold** |
| 123 | 0 | 0.77 | 0.76 | 0.01 | no\* | invalid |
| 123 | 2000 | 0.66 | 0.64 | 0.02 | no\* | invalid |
| 123 | 10000 | 0.97 | 0.97 | 0.01 | yes | **slot-scaffold** |

\* "invalid" = anchor τ_pos below the strong-carrier gate (0.95) at that step — the strong
L1 carrier isn't established yet pre-/mid-pruning (consistent with Spec A's event at
~1400–1600); seed42's weak tier uses the 0.60 gate and passes throughout.

**Every valid checkpoint reads `slot-scaffold`**: `τ_content ≈ 0.01` across all 8 layouts
means relocating content to a different slot does **not** move the carrier's order with the
content. The carrier reads slot position, not content identity — even after full training.

## Interpretation & reframe

- **Spec A reframe (triggered):** the "diffuse order template present from init" is
  largely a **measurement/architecture artifact** — the C-D+L readout reconstructs
  sequential order from causal masking (floor τ=1.0), and the untrained model is a noised
  version of that floor. The winner-take-all "selection" Spec A characterized is selection
  among *positional* readout heads, not the birth of a content-order representation.
- **③ + A + ⑤ unified:** there is no L0→L1 handoff (③); the carrier is a seed-locked,
  layer-local redundant set selected by contingent symmetry-breaking (A); and that carrier
  is a **positional slot scaffold** carrying no cross-layout content order (⑤). The order
  signal is positional structure that attention aligns to, read out by an order-recovering
  metric — *not* a learned content-dependent ordering.

## Caveats (honest boundaries)

- **OOD by construction:** the model trained on a single layout, so it had no pressure to
  be content-bound across layouts; `τ_content ≈ 0` is therefore *consistent with* a
  slot-scaffold but does not by itself prove the model could never encode content order
  under multi-layout training. Combined with Part 1 (pure causal mask already gives τ=1.0),
  the slot-scaffold reading is nonetheless strongly supported.
- **Metric-shaped:** "order" here is whatever the C-D+L none-separated rollout recovers;
  Part 1 shows that is dominated by causal-sequence reconstruction. A different readout
  could surface different structure — but that would be a different claim, not this one.
- **Per-seed, single run; 3 seeds is a floor.** Effects are large and consistent here, so
  the direction is solid.

## Implication for next steps

This weakens the motivation for Spec B (contingency of *which* positional head wins) as a
headline result — the thing being selected is a positional scaffold, so "which slot-reader
wins" is a less load-bearing question than whether any content-order exists (it does not,
here). A more decisive follow-up would be **multi-layout training** (give the model a
reason to encode content order) and re-running this content-relocation diagnostic — that
is the experiment that could turn a slot scaffold into, or rule out, a content-bound order
representation.
