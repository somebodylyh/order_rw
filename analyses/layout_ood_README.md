# Existing-ckpt Unseen-Permutation OOD Evaluation (P4-lite pre-step)

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating` · **Compute:** CPU (no training)
**Code:** `analyses/layout_ood_eval.py` · **Outputs:** `runs/layout_ood/seed{2,42,123}.json`
**Run:** K=8 unseen perms, M=12, n_reveals=16
**Foundations:** `analyses/physical_signal_source_README.md` (P2 B+, single-number relayout 1.0→0.13) ·
`analyses/p3prime_causal_README.md` (P3′ partial support).

> **Verdict: lookup — all three seeds.** The existing fixed-layout ckpt does NOT zero-shot
> generalize to unseen permutations. Fed a new layout, it keeps emitting the **training** slot→
> physical order, not the new physical order.

## Question

Without retraining, lay the evaluation data out under unseen block permutations and ask whether the
existing ckpt still recovers the *new* physical order (content/context recovery) or keeps emitting
its *training* slot→physical map (a fixed lookup).

## Diagnostic (the train-map probe)

Same forward pass, two `inv_perm` labelings of the SAME attention:
- `tau_test_physical` = σ_model scored under the **new** layout's inv → vs physical L2R.
- `tau_train_map`     = σ_model scored under the **training** inv → vs physical L2R.

The carrier head is fixed to the anchor's strongest-|τ| head (no head-switching confound). This
generalises P2's single relayout number with the *where did the order go* diagnostic, and — unlike
P3′-B — has **no global-intervention confound** (the model is untouched; we only relayout + relabel).

| signature | tau_test_physical | tau_train_map | meaning |
|-----------|-------------------|---------------|---------|
| **lookup** | low | high | model emits the training-layout order regardless of new content/layout |
| content_recovery | high | low | model recovers the new physical order from content |
| degraded | low | low | attention degraded into noise under new content |

## Result (K=8 unseen perms)

| seed | carrier | anchor τ | test_physical mean±std | train_map mean±std | interpretation |
|------|---------|----------|------------------------|--------------------|----------------|
| 2   | L0H2 | 1.000 | 0.078 ± 0.056 | **1.000 ± 0.000** | **lookup** |
| 42  | L0H2 | 1.000 | 0.078 ± 0.056 | **1.000 ± 0.000** | **lookup** |
| 123 | L0H1 | 1.000 | 0.078 ± 0.027 | **0.941 ± 0.101** | **lookup** |

Under every unseen permutation the carrier still rolls out the **training** physical order
(train_map ≈ 1.0; seed2/42 *exactly* 1.0 across all 8 perms — std 0), while its score against the
**new** layout's true physical order collapses to the floor (≈0.08). The model is executing a learned
training-layout slot→physical map, not re-deriving order from content.

## Interpretation

This is the decisive, confound-free version of P2's relayout collapse: the learned global
physical-order signal is **bound to the training layout** and **does not zero-shot generalize**. It
**reinforces the conservative line framing** — a fixed-layout, position-path-driven global
physical-order signal, **not** content-driven recovery (consistent with P3′'s "partial support").

## Why this gates P4-lite

Because the fixed-layout ckpt provably does not generalize zero-shot, the next step —
**P4-lite K-layout training** (sample one of K perms per sample, no layout_id conditioning; evaluate
on held-out perms) — is well-motivated: it removes the single-layout lookup shortcut and tests
whether layout diversity induces any generalizable / content-dependent order signal. The held-out
`τ_test_physical` there is the same metric used here.
