# Early-ckpt Attention/Hidden Representation Divergence (text)

**Date**: 2026-05-27
**Branch**: attn-order-alternating
**Status**: design approved, pending spec review

## Motivation

The three hidden-state lines ([[hidden_residual_diag_line]], [[hidden_graph_diag_line]],
[[causal_hidden_probe_line]]) all closed CLEAN NEGATIVE / NULL at the **30k (mature)**
checkpoint: hidden state carries no order signal beyond the static attention graph `B_A`
(C-D+L readout) plus position. In particular at 30k both `B_A` and `B_H` (hidden cosine
graph) are ≈ position **and** ≈ each other (`corr(B_A,B_H)` high, `corr(B_H,B_pos)`
0.62–0.76).

**Hypothesis under test**: maybe this collapse is a property of the *mature* model. Early
in training, attention and hidden representations may be **divergent** — hidden could
transiently carry structure that attention does not (and that is not merely position),
before training collapses them onto the same representation. If so, the hidden lines were
closed prematurely by only probing 30k.

This is a **frozen diagnostic, no training**. Text only (the only ckpt set with a full
sweep). Image deferred (only 3 usable ckpts).

## Definitions of "divergence" (two independent signals)

Both are computed from the existing `run_hidden_graph_diag` substrate:

- **(a) mutual divergence**: `corr(B_A, B_H)` is **low early → rises** toward ~0.6–0.76 by
  30k+. A low-then-high curve = "attention and hidden start as different representations,
  then collapse together."
- **(b) hidden-independent structure**: `B_H_resid` (= `B_H` with position residualized
  out) has **real structure early** (sharpness z-score > 2 vs shuffle null) → becomes pure
  noise late. This is the stronger signal: hidden carrying non-position structure that
  attention lacks.

Either signal firing at some early step is grounds for Stage 2. Both flat/absent =
hypothesis falsified.

## Stage 1 — static-graph divergence curve (cheap, reuses existing code)

Sweep **9 text checkpoints**: step 0 / 1k / 5k / 10k / 20k / 30k / 40k / 50k / 60k from
`clean_base_random_perm/ckpt_step{S}.pt`.

For each ckpt, call the existing `run_hidden_graph_diag.analyze()` path which already builds
`B_A` (attention collapse graph via `extract_A_matrices` + `build_B_set`), `B_H`
(`HG.cosine_graph` of full-context block hidden), `B_pos` (`PG.text_position_graph`), and
the `corr` dict (`BH_vs_BA`, `BA_vs_Bpos`, `BH_vs_Bpos`, `BHresid_vs_BA`,
`BHresid_vs_Bpos`). Also collect `B_H_resid` sharpness via
`graph_structure_metrics.structure_vs_null` (z-score vs shuffle).

**Output**: `probe_results/attn_hidden_divergence/divergence_curve.tsv` — one row per ckpt,
columns: `step, corr_BA_BH, corr_BA_Bpos, corr_BH_Bpos, corr_BHresid_BA, BH_resid_sharpness_z`
(plus B_A error bars, below). Plus a curve plot `divergence_curve.png` and a short
`report.md`.

### Noise guardrail (CRITICAL)

`extract_A_matrices` (`train_clean_aogpt.py`) samples random block orders with an
**unseeded `torch.randperm`** → `B_A` has cross-run sampling variance. A spurious dip in
`corr(B_A,B_H)` could be sampling noise rather than a real early divergence — which would
falsely trigger Stage 2.

Mitigation: at each ckpt, **repeat `B_A` extraction R≥3 times** (different random orders),
compute `corr(B_A,B_H)` for each, and report **mean ± std** as error bars on the curve.
A divergence claim requires the early/late difference to exceed the B_A sampling band.
`B_H` (full-context cosine) is deterministic given the order and needs no repeats. Use a
decent `n_chunks` (≥8) per extraction so each `B_A` is itself reasonably converged.

## Stage 2 — gated causal probe at the divergence step

**Gate**: Stage 1 must show a *clear* early divergence beyond the B_A noise band — either
signal (a) `corr(B_A,B_H)` notably below its 30k value at some early step, or signal (b)
`B_H_resid` sharpness z > 2 at some early step. Pick the step with the strongest signal as
the **divergence step** `S*`.

- **If the gate fires**: re-run the CT8 causal probe (`run_causal_hidden_probe.py`, Path X
  MAIN) at `ckpt_step{S*}.pt`. Question: does per-step causal hidden add usable order signal
  there (γ>0 beats γ0 under the 6-gate WIN criterion)? This tests whether the early
  divergence is *useful* (translates to an order that lowers NLL) or merely a transient
  geometric artifact. Within-run γ0-vs-γ>0 comparison is robust to B_A absolute drift.
- **If the gate does NOT fire** (corr flat-high across all steps, B_H_resid noise
  throughout): **STOP. Hypothesis falsified.** This is itself a clean, reportable result:
  attention and hidden are the same representation from initialization onward, strengthening
  the three-line NEGATIVE verdict (the collapse is not a maturity artifact).

## Components / reuse

- **No new diagnostic math.** Stage 1 driver = a thin sweep wrapper over the existing
  `scripts/run_hidden_graph_diag.py:analyze()` + `extract_A_matrices` repeat loop + TSV/plot
  writer. New file: `scripts/run_attn_hidden_divergence.py`.
- Stage 2 = existing `scripts/run_causal_hidden_probe.py` invoked at `S*` (no code change,
  just a ckpt arg).
- ckpts: `block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step*.pt`
  (full 0–60k sweep already on disk).

## Testing

- TDD: tiny-model smoke for the sweep wrapper — assert (1) `divergence_curve.tsv` has one
  row per ckpt with all columns populated, (2) the R-repeat B_A loop produces a non-zero std
  column (sampling variance is captured), (3) at the same ckpt the wrapper's `corr` values
  match a direct `analyze()` call (no wiring drift). Reuse the tiny BlockAOGPT fixture from
  the hidden-graph / causal-hidden test suites.

## Out of scope (YAGNI)

- Image sweep (only 3 ckpts; revisit if text shows divergence).
- Model-attn (qk) score in the probe (CT9-gated, untouched here).
- Any training / controller work — this is purely diagnostic.
