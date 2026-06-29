# P3′ — Causal Probe of the L0 Global Physical-Order Carrier

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating` · **Compute:** CPU (no training)
**Code:** `analyses/p3prime_causal_verify.py`, `analyses/plot_p3prime_causal.py`
**Spec:** `docs/superpowers/specs/2026-06-29-p3prime-causal-verification-design.md`
**Outputs:** `runs/p3prime_causal/seed{2,42,123}/{p3prime.json,p3prime.csv,p3prime_metrics.png}`
**Formal run:** `M=12`, `n_reveals=16`
**Foundations:** `analyses/canonical_order_README.md` (canonical readout) · `analyses/physical_signal_source_README.md` (P2 B+).

> **P3′ verdict: B+ gets partial causal support, not a clean confirmation.**
> The order signal (τ) is causally position-path dependent in every carrier and seed. But the
> "fixed base-map" half is not carrier-selective (null heads collapse too — the position ablation
> is a global intervention), and the content residual is detectable in only one head of one seed.

## Metric correction (read this first)

The first implementation had two bugs in the P3′-B block, now fixed (commit history):
1. **Self-referential R².** `r2_abl = slot_only_r2(B_abl)` scored the ablated map against *its own*
   slot-mean table, so a degenerate-but-cross-text-consistent ablated pattern scored high R² —
   it could not detect that the fixed map was disrupted. Fixed: `r2_abl` now scores the ablated B
   against the **clean** fixed-map table (`_base_map_cross_r2`), so a drop means the clean map no
   longer explains the attention.
2. **Wrong τ source.** `run_seed`'s inline B-block read `tau_abl` from the A0 **self-QK patch**
   (uniform-attention floor ≈0.037 for all heads) instead of the **position ablation**. Fixed: both
   `b_position_ablation` and `run_seed` now share `_b_per_head_row` and use the position-ablation τ.

All numbers below are post-fix.

## What P3′ tests

P2 found a dominant fixed slot→physical table plus a smaller content residual in the carrier B65
(the **L0 global physical-order carrier**; verdict **B+**). P3′ asks whether controlled
interventions recover the two causal paths. It uses
Q/K, QK-score, and input-to-QK interventions only; output/OV ablation on the same head's τ is
degenerate. **P3′ does not attempt to convert B+ into C** — that needs multi-layout training (P4).

**Red lines.** Output/OV/`c_proj` ablation read on the same head's τ is degenerate (the readout is
the head's own attention map). Self-QK patching of the read head (A0) is calibration only and is
**not treated as load-bearing evidence**. The load-bearing tests are P3′-B (position-path) and
P3′-C (content), with the verdict from the joint τ + clean-predict R² criterion.

## Seed-level result (post-fix)

| seed | L0 carrier | P3′-B (τ→ / clean-pred R²→) | base-map heads | P3′-C residual/floor | classifier verdict |
|------|------------|------------------------------|----------------|----------------------|--------------------|
| 2   | {2,3,4,5} | τ 0.9→0.07 all; R² e.g. H2 0.97→0.60, H5 0.99→0.43 | 2/4 collapse | 0.42–0.99, **0/4 driven** | **mixed/departure** |
| 42  | {2}       | τ 0.84→0.07; R² 0.86→0.35 | 1/1 collapse | 0.12, 0/1 driven | **mixed/departure** |
| 123 | {1,2,3,4} | τ 0.5–1.0→0.06 all; R² H3 0.98→0.08, H2 0.91→0.34 | **4/4 collapse** | H2 **1.465 driven**; others <0.6 | **B+ confirmed** |

`classify_p3prime` requires a majority of carrier heads to show joint τ+R² base-map collapse AND at
least one content-driven head. Only **seed123** meets this bar.

## The selectivity caveat (why "B+ confirmed" is not clean)

The null-head control shows the position ablation is **not carrier-specific**:

| seed | null heads | null R² none→abl |
|------|-----------|-------------------|
| 2   | {7,1} | 0.96→0.74, 0.99→0.71 |
| 42  | {0,5} | 1.00→0.30, 0.94→−7.11 |
| 123 | {6,7} | 0.83→0.19, 0.93→0.69 |

Zeroing wpe/wtpe degrades the fixed-map R² of **null** heads too. So "fixed-map R² collapses under
position ablation" is a **global** effect of removing position embeddings, not a carrier-localised
mechanism. The carrier-specific half of the evidence is therefore the **τ collapse** (null heads
have τ≈0 and cannot lose it); the R² half is confounded by the bluntness of the intervention.

## Conclusion (three layers, decreasing certainty)

1. **Robust:** the canonical order signal (τ) is **causally position-path dependent** — across every
   carrier and seed, zeroing position embeddings collapses τ from 0.5–1.0 to ≈0.06. The order
   recovery lives in the position/QK geometry.
2. **Not established:** the P2 "fixed slot→physical **base map**" cannot be cleanly localised to a
   carrier-specific position mechanism, because the position ablation is global (null heads collapse
   too). The B+ base-map decomposition is **not causally confirmed**.
3. **Near-absent:** the content-modulation residual ("+") moves above the within-text sampling floor
   in only **1 head of 1 seed** (seed123 H2, ratio 1.465). Content sensitivity is essentially a null
   across seeds.

**Net:** P3′ gives B+ partial causal support (position-path carries the order signal) but does
**not** confirm the carrier-specific base map or the content residual. This **reinforces the
conservative line framing**: a fixed-layout, position-path-driven global physical-order signal —
**not** content-driven recovery. The decisive content test remains **multi-layout training (P4)**,
with the cheap pre-step being an existing-ckpt unseen-permutation OOD evaluation (τ_test_physical vs
τ_train_map).

## What can / cannot be claimed

- **Can:** order readout is QK/position-path sensitive (A0, D); same-head OV/output is degenerate;
  multi-head L0 carriers are weakly-coupled parallel copies (A1 off-diagonal ≈0).
- **Cannot:** a clean modular position/content causal decomposition; a carrier-specific base-map
  locus; a cross-seed content residual; upgrading P2 from correlational B+ to a confirmed B+.
