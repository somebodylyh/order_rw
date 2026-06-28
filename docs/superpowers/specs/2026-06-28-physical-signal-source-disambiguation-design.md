# P2 — Global Physical-Order Signal Source Disambiguation — design

**Date:** 2026-06-28
**Status:** design (approved direction, mechanism-level)
**Predecessors:** `2026-06-28-canonical-reanalysis-design.md` (canonical 65-node readout),
`analyses/canonical_order_README.md` (frame-correction foundation),
`runs/physical_emergence/` (P1 emergence trajectory).
**Branch:** `attn-order-alternating`. **Compute:** no new training; CPU forward on existing
ckpts; GPU only as accelerator. **P3′ (causal verification) is a separate later spec**, forked
by this spec's verdict.

## The question

The canonical readout shows a **real, emergent global physical-order-aligned signal** in the
L0 carrier (τ_physical → 1.0, above the destroyed floor, not an inv-construction artifact).
**Its source is unresolved:**

- **B — fixed-layout slot→physical map:** the model learned, for the *fixed* training layout,
  a content-free positional rule "model slot s → physical rank r". Attention at slot s is a
  function of s alone.
- **C — content-dependent recovery:** the model uses block *content* to infer each block's
  original rank; attention at slot s depends on the content at s.
- **B+ — map with content modulation:** mostly the fixed map, with measurable content-dependent
  residual.

## The decisive insight (why order/τ cannot disambiguate)

Under the **fixed training layout**, B and C are **isomorphic at the order level**:
`τ_physical = τ(inv_perm[σ_model], arange) = 1.0` ⟺ **`σ_model` is exactly the layout map
`block_perm`**. So *any* head with τ_physical=1.0 has `σ_model` = the slot→physical lookup,
whether it got there via slot position (B) or content (C). Therefore:

> **An order-level "lookup-similarity" test is tautologically 1.0 on the carrier and gives
> false B evidence. Disambiguation must be at the attention/mechanism level: does the
> attention that produces the order depend on content?**

(The earlier E2 "order lookup similarity" is **removed** for this reason; its intent is
replaced by E1's attention-level slot-only predictability.)

## Components

### E0 — slot-scaffold negative control (A already excluded)
Report the model-frame identity-reveal τ (L1 slot scaffold) vs canonical physical τ (L0) — the
contrast already established. Confirms the L1 slot scaffold is **not** the target L0 signal;
keeps A out of the B/C question.

### E1 — Same-layout content sensitivity **[MAIN — the only verdict driver]**

Hold the **training layout fixed** and a **shared set of reveal orders fixed**; vary the
**text content** across `M` natural samples. For each carrier head compute, on its per-text
B65 (the loss-aligned canonical extraction, batch-meaned over the shared reveals):

1. **τ_physical per sample + carrier-validity gate** — *validity only* (confirm the carrier is
   active on each text; `τ≈1`). **Never used to judge B/C**, but used to **gate**: a (text,head)
   enters the variance/R² analysis only if its canonical τ_physical clears a carrier-validity
   threshold on that text (strong heads ≥0.9; seed42's single weak head reported separately with
   a lower/soft threshold, never hard-excluded). Otherwise a text where the carrier simply isn't
   active would inflate attention variance and masquerade as C. **Report metrics both
   all-sample and carrier-valid-only.**
2. **Cross-text attention/B variance (primary B/C metric):** per causal edge,
   `Var_text B_text[i,j]`, computed **only on the valid canonical B65 edge support** (the edges
   the readout uses — None→content and content→content; **exclude structurally-invalid /
   always-zero edges** so that masked zeros do not deflate variance toward B). **Row-normalize**
   each valid row by its **L1 mass**: `B_norm[i,:] = B[i,:] / (Σ_{valid j}|B[i,j]| + ε)`.
   **Report both raw and row-normalized variance** as a robustness check (agreement = stable).
   **Low → B; high → C/B+.**
3. **Pairwise cross-text similarity:** mean over text pairs of `sim(B_text_a, B_text_b)` on the
   valid-edge support (row-normalized Frobenius/Spearman). **≈1 → B; lower → C/B+.**
4. **Slot-only predictor held-out R² (strong modeling B-evidence):** the strongest *content-
   free fixed-layout* predictor is the per-edge mean over training texts,
   `B_hat[i,j] = mean_{train texts} B[i,j]`. **The train/held-out split is over *text samples*,
   not over edges** — the predictor sees **no activations from held-out texts** (any edge split
   would leak content and void the "content-free" meaning). Predict B on held-out texts using
   only slot-pair identity; report `R²_slot_only` on the valid-edge support (raw and
   row-normalized). **High R² → a content-free slot-pair table explains the carrier = B; low R²
   with τ still high → content-dependent = C/B+.**

   Predictor complexity is fixed deliberately at the **mean-table** level (one free value per
   edge, no content features) — strong enough not to under-estimate B, but it *cannot* absorb
   content variation, so a high held-out R² is genuine B evidence.

**Secondary (supportive, not primary):**
- **Block-swap / cross-sample block replacement** — "content perturbation at fixed evaluation
  labels": same layout, swap/replace block content within or across samples; does the carrier's
  attention/B move? (Block-swap > random tokens because it stays natural-text.)
- **Random-token stress** — severe OOD content destruction; **stress control only**, never
  primary, because corruption can globally break the circuit (a τ collapse here is not C
  evidence).

**Cross-check matrix (guards against artifacts):**
| variance | R²_slot_only | reading |
|---|---|---|
| low | high | **B** (fixed map) |
| high | low | **C / B+** (content-driven) |
| high | high | **inspect residual** — could be insufficient row-normalization, or a global strength shift with fixed pattern shape, or structured content modulation; do **not** auto-judge |
| low | low | predictor / metric / valid-edge-mask error → debug before interpreting |

### E3 — Relayout OOD diagnostic **[supporting / appendix, not verdict]**
Reuse `position_prior_decomp.relayout_chunks`: training-layout **anchor** + `K` random
relayouts; report anchor τ_physical, relayout-mean τ_physical, drop. **Interpretation
(pre-registered, asymmetric):** relayout **survival → strong evidence for C**; relayout
**collapse → only *consistent with* B, not decisive** (the model trained on one layout, so a
content-dependent model that never learned to generalize would also collapse). Does **not**
override E1.

## Verdict (per seed, pre-registered)

**Calibrate "low/high" against synthetic baselines, not absolute numbers:** build a
**content-invariant** synthetic B (identical across texts → variance≈0, R²≈1) and a
**content-randomized** synthetic B (variance high, R² low). A real carrier's metrics are read
*relative to* these two anchors (and to a shuffled-text baseline). Avoid hard B/C calls when a
metric sits near the boundary between the anchors. Suggested operational guides (refined in the
plan, not hard-coded): held-out R² ≥ 0.8 *and* clearly above the shuffled-text baseline for B;
variance substantially above the content-invariant anchor for C/B+.

- **B (fixed-layout map):** τ_physical high; cross-text variance **near the content-invariant
  anchor (low)**; slot-only R² **high**; perturbations weak except severe OOD.
- **B+ (map with content modulation):** τ high; R² high **but incomplete**; measurable
  content-dependent residual variance; perturbations modulate but don't destroy.
- **C (content-dependent recovery):** τ high; cross-text variance **high in structured ways**;
  slot-only R² **low/moderate**; content perturbations drive attention/order changes (not a mere
  circuit break); relayout may survive.
- **mixed/inconclusive:** carrier unstable, τ drops broadly, or metrics conflict.

Report **per-seed first**. seed42 is a **single-head, late-emerging (step~5000) physical
carrier** (L0{2}) — analyze it separately as the single-head case; do not force the multi-head
redundancy framing of seed2/123 onto it.

## Architecture

- `analyses/physical_signal_source.py`: reuses `canonical_reanalysis` (the canonical
  extraction `_attn_to_A_block_loss_aligned_with_none_vec` + `build_none_separated_B` + rollout/
  metrics), but drives it with **fixed layout + shared reveal orders + varied text** (E1) and
  **relayout** (E3, via `position_prior_decomp.relayout_chunks`). Per-carrier-head B65 tensors
  across texts; variance / similarity / slot-only-R² readouts; block-swap perturbation helper.
- Carrier heads come from the frozen P1 config (seed2 L0{2,3,4,5}, seed42 L0{2}, seed123
  L0{1,2,3,4}).
- Outputs `runs/physical_signal_source/seed{2,42,123}/` (per-head metrics + verdict) + figures
  (cross-text B variance heatmap, similarity, R²) + `analyses/physical_signal_source_README.md`.

## P3′ fork (pre-registered, written into the report)

- **verdict B** → P3′ tests whether L0 heads implement the slot→physical map: intervene on the
  **positional / slot / QK-geometry** path (not content).
- **verdict C** → P3′ tests content-feature reliance: **content embedding / block-replacement /
  semantic-neighbourhood** causal interventions.
- **verdict B+** → P3′ separates the base map from the content modulation.
- All P3′ interventions avoid output-ablation→same-head-readout (degenerate for an
  attention-map metric; the Pillar-③ limitation).

## Testing (TDD)

- **τ_physical=1.0 ⟹ σ_model = block_perm** (the degeneracy) — unit test on a synthetic carrier
  B confirming order-level lookup similarity is tautological (documents *why* E2 was removed).
- **slot-only predictor**: on synthetic content-invariant B's (identical across texts), held-out
  R² ≈ 1 and variance ≈ 0; on content-randomized B's, R² low / variance high.
- **row normalization** leaves a content-invariant B's variance at 0 (no scale leakage).
- **relayout anchor round-trip** (reuse the `position_prior_decomp` guard): layout_0 = training
  reproduces the input.

## Risks

- **OOD confound in perturbations** — random-token corruption can break the circuit globally; it
  is stress-only, never primary; block-swap (natural text) is preferred; the cross-text *natural*
  variance (E1) is the clean primary.
- **Normalization** — entropy/scale shifts across texts can inflate variance; row-normalize and
  cross-check with R² (the matrix above).
- **Single-layout limitation** — E3 relayout can only support C (survival), not refute it; E1
  carries the verdict.
- **Per-seed, K-sampling variance** — reuse control_seeds / multi-sample means from the canonical
  pipeline; 3 seeds is a floor.
