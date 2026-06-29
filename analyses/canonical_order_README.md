# Canonical Order Readout — Frame Correction & Sanity (foundation)

**Date:** 2026-06-28 · **Branch:** `attn-order-alternating`

This is the **load-bearing foundation** for the order-recovery line. It fixes the readout
口径 that the recent ③/A/⑤ arc got wrong, and pins the four sanities that separate the
**real** physical-order recovery from a **slot-order scaffold artifact**. Read this before
running any further order experiment.

## What we actually want to measure

> **Can attention recover the physical / original block order of the (shuffled) text?**
> i.e. recover `physical order` — the original-text block order before the layout was
> permuted — **NOT** `model slot order` (the input slot index 0→1→2→…).

Because the data layout is shuffled, the model only ever sees model slots. Recovering the
*physical* order could in principle use **content** (the strong claim, C) or just a learned
**fixed-layout slot→physical map** (the weaker claim, B). Distinguishing these is P2; do not
assume "content" yet.

## The two readouts (and why one is confounded)

| readout | reveal order | frame / scoring | what it measures |
|---|---|---|---|
| **model-frame** (③/A/⑤ used this) | **identity** | σ_model vs `arange` (model slot) | **slot/reveal-order scaffold** — under identity reveal + a causal mask the rollout ascends near-tautologically |
| **canonical** (correct) | **random** | strict 65-node, σ_model → posthoc-inv → vs physical L2R | **physical-order recovery** — does attention reconstruct the original block order under *any* reveal |

**Only the canonical physical readout answers the paper's question.** Use it as the **sole
primary metric**. Mandatory naming from now on: **slot-order carrier** vs **physical-order
carrier** — never a bare "order carrier".

## The four sanities (all verified tonight)

### S1 — identity-reveal model-frame τ is confounded (high even when physical τ ≈ 0)
Same head, seed2 **L0H2 at step 0 (untrained)**: model-frame identity τ = **0.72**, but
canonical random-reveal physical τ = **−0.03**. The model-frame "signal" at init is the
identity+causal tautology, not order recovery.

### S2 — a slot-order scaffold is a causal-mask/readout artifact
A **uniform-causal attention** (no learned weights), read in the model frame under identity
reveal, rolls out to τ = **1.0**. So "model-frame τ high" can be produced by the causal mask
+ readout alone — it is not evidence of learned order. (This is what the ⑤ "slot-scaffold"
finding actually measured.)

### S3 — physical-order recovery genuinely EMERGES (canonical, 3 seeds)
Canonical per-(layer,head) physical τ on `runs/handoff_overnight`, K=3 sampling seeds,
method C-D+L, destroyed-floor anchored:

| seed | physical-order carrier (L0) | best \|τ\|: step0 → step10k | emergence crossing |
|------|------------------------------|------------------------------|--------------------|
| 2   | L0 {2,3,4,5} | 0.16 → **1.00** | step **1000→2000** |
| 42  | L0 {2}       | 0.13 → **1.00** | step **1000→2000** |
| 123 | L0 {1,2,3,4} | 0.15 → **1.00** | step **2000→3000** |

Absent at init (≈ destroyed floor), τ→1.0 around step ~2000, in a **redundant L0** head set.
This matches `analyses/figures/head_signal_emergence.png` (~1800) and Spec A's ~1400–2000.

### S4 — the canonical signal is real and not an inv-construction artifact (but its *source* is NOT yet content)
For the L0 carrier (seed2 L0H2, τ=1.0):
- **destroyed / entry-shuffled floor** ≈ **0.05–0.07** ⇒ τ=1.0 is far above null.
- **strict-LF (model-frame build + posthoc inv) ≡ oracle-remapped (inv in construction)**:
  both give L0 = **1.00** on handoff_overnight AND on `overnight_20260625_random_baseline`
  ⇒ the signal is **not** an inv_perm construction cheat.
- ⚠️ **`content_label_permutation_control` collapse (1.0→0.08) does NOT prove content-binding.**
  It permutes the B65 graph's *node labels* posthoc (`B[ix_(perm,perm)]`) — *any* order-recovering
  graph loses τ-vs-`arange` under relabeling. It is a graph-structure / null control, and it
  **cannot distinguish** a fixed-layout slot→physical map from genuine content recovery.

**Corrected claim:** *The canonical readout reveals a real, emergent global physical-order-aligned
signal in L0 — above the null floor and not an inv_perm-construction artifact. Whether its source
is **content** (C) or a **fixed-layout slot→physical position map** (B) is UNRESOLVED and is the
subject of P2 (signal-source disambiguation). Do not call it "content-bound" yet.*

## Layer split (the same ckpt read two ways)

seed2 step10000, per-layer best |τ|:

| layer | canonical (physical, random) | model-frame (identity) |
|-------|------------------------------|------------------------|
| **L0** | **1.00** (4 strong) | 0.80 |
| **L1** | 0.56 (0 strong) | **1.00** |
| L2 | 0.61 | 0.61 |
| L3 | 1.00 (1 strong) | 0.77 |

⇒ **two distinct mechanisms** in the same model:

| mechanism | name | layer | readout | at init? | content-bound? | paper role |
|-----------|------|-------|---------|----------|----------------|------------|
| reveal/slot following | **slot-order scaffold** | L1 | model-frame / identity | yes | no | confound / control |
| original-order reconstruction | **L0 global physical-order carrier** | L0 | canonical / random | no (emerges ~2k) | **B+ (P2): fixed-layout map + content modulation** | **main claim** |

## Status of the prior pillars

- **③ path-patching / A emergence / ⑤ binding** characterized the **L1 slot-order scaffold**
  under the confounded readout. Their *physical-order* conclusions (L1 carrier, "artifact",
  "slot-scaffold", "handoff fails") are **retracted as physical-order claims** and demoted to
  an **appendix methodology warning**: *identity-reveal model-frame τ mistakes a slot-order
  scaffold for order recovery.*
- The main phenomenon — **AOGPT attention develops a real, emergent global physical-order-aligned
  signal in L0** — **holds** under the canonical readout. Its *source* (content C vs fixed-layout
  map B) is the open P2 question.

## A measurement-error caution (for collaborators)

Two errors that produce a false "no signal":
1. **Averaging signed τ across heads** — L0 contains both +1.0 (pro-L2R) and −1.0 (anti-L2R)
   heads; the mean cancels to ~0. Report the **per-head gate / best |τ|**, never the mean.
2. **Running a collaborator-wired scan on a different model** — `search_none_separated_65_heads.py`
   defaults to the collaborator ckpt + `_physical_chunks_to_model` + `--perm-orientation
   model_to_phys`; pointed at a local model without fixing the chunk/perm wiring it mis-scores
   and reports FAIL. `overnight_20260625_random_baseline@60k` reads **τ=1.0, 8 strong** under
   the correctly-wired `analyses/canonical_reanalysis.canonical_scan` — the signal is there.

## Next (canonical main line — do NOT continue the old ③/A/⑤ route)

P1 physical-order emergence trajectory (per-layer / L0-per-head τ vs step, 3 seeds) ·
P2 systematic content-dependence controls · P3 **Pillar ③′**: causal verification on the **L0
physical carrier** via **QK/input ablation or content corruption** (not output-ablation, which
is degenerate for an attention-map readout) · P4 two-mechanism comparison table (above).
Deferred: Spec B (slot-reader contingency) → appendix; multi-layout training → later.
