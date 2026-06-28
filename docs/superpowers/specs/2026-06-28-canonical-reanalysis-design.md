# Canonical Physical-Frame Re-analysis of the Order-Emergence Line — design

**Date:** 2026-06-28
**Status:** design (approved direction, sync-with-existing)
**Predecessors:** `2026-06-27-handoff-causal-path-patching-design.md` (③),
`2026-06-27-emergence-characterization-design.md` (A),
`2026-06-28-position-prior-decomposition-design.md` (⑤). All three used the
**model-frame + identity-reveal** readout, now shown to be tautological at init and
to identify a different carrier (L1) than the project's canonical metric (L0).
**Branch:** `attn-order-alternating`. **Compute:** no GPU (11 existing ckpts/seed); GPU
only if coarse timing proves insufficient.

## Motivation (why redo)

The recent ③/A/⑤ arc read order via `build_model_frame_strict65` (model-frame, NO
inv_perm) on the trajectory's **identity** `probe_orders`. Under identity reveal + a
causal mask, the C-D+L rollout reconstructs ascending model-slot order **near-
tautologically** — so step-0 τ≈0.77 and the τ=1.0 "carrier" are artifacts of that
protocol, and the carrier it picks (seed2 = L1) differs from the project's canonical
metric.

The project's **canonical, sealed** readout is **physical-frame + random reveal**: per
sample, a *random* reveal order; single-head attention → physical block-agg via
`inv_perm` (so blocks are in original-text coordinates) → `B=Aᵀ`, zero diagonal →
batch-mean → teacher-CDL σ → `τ_vs_l2r` vs `arange(N)` (physical). Decisive contrast on
seed2 (canonical `per_head_order_scan`): **step0 max|τ|=0.07 → step10000 max|τ|=0.96,
carriers L0H4/H2/H5** — i.e. the order signal is **absent at init and genuinely emerges**,
recovering the original-text block order (which, under a shuffled layout, *requires
content*), matching `analyses/figures/head_signal_emergence.png` and the early L0H2/L0H5
work. **This is the real finding; the model-frame arc characterized a metric artifact.**

## Principle: SYNC with the sealed pipeline (do not reinvent)

The project already has the canonical machinery; the redo **reuses it**, pointed at the
`runs/handoff_overnight/seed{2,42,123}` checkpoints:

- `scripts/diag_l0h5_evolution_scan.py` — per (ckpt, sampling-seed) **full (L,H)** scan
  (random reveal seeded `seed+i`, physical block-agg, `B=Aᵀ`, teacher-CDL, heavy
  baseline). Outputs `scan_step{S}_seed{D}.json` with
  `per_head_layer_sorted_by_abs_tau_vs_l2r` + `heavy_baseline`.
- `scripts/scan_all_heads_across_ckpts.py` — multi-ckpt orchestrator producing a
  32-head × steps τ matrix heatmap + per-step winner / per-layer best.
- `scripts/aggregate_l0h5_evolution.py` — aggregates over sampling seeds → per-step
  `modal_winner`, `winner_per_seed`, τ mean±std, layer0 fraction, winner-drift (Q2).

**none_mode = `b0`** (the sealed-canonical B0 variant), pinned throughout; the scan
scripts also support `old`/`predictor` but the redo fixes `b0` and records it.

## Variance control

The extraction's per-sample reveal RNG (`seed+i`) makes a single scan noisy
(documented). Run each (seed, ckpt) at **K=3 sampling seeds**, aggregate mean±std per
(layer,head) (the `aggregate_l0h5_evolution.py` pattern). Carriers and emergence timing
are read from the **aggregated** curves, never a single scan.

## Components

### C0/C1 — Canonical emergence + carrier baseline (foundation)

Run the canonical full-(L,H) scan on **3 seeds × 11 ckpts** (`step0,1000,…,10000`) ×
**K=3 sampling seeds**, `none_mode=b0`. Adapt `scan_all_heads_across_ckpts.py`'s ckpt
list to be parametrized per seed (it currently hardcodes the old clean_base trajectory).
Aggregate per seed:

- **Carrier (localization):** per-step `modal_winner` and per-layer best |τ|; the
  step-10000 modal winner = the **real carrier** (expect L0; confirm per seed, compare
  to the model-frame L1).
- **Emergence (strength + timing):** carrier-head `τ_vs_l2r` mean±std vs step; the coarse
  (1000-step) interval where τ crosses from ~0 to high (the real emergence window).
- **Heavy baseline** τ as the multi-head reference (already in the scan).

**Anchor:** reproduce the seed2 contrast (step0 ≈ 0, step10000 ≈ 0.9+ on an L0 head)
under `b0` before trusting the rest.

Output: `runs/canonical_reanalysis/seed{2,42,123}/heads_tau.json` + 32×11 heatmap +
emergence curve; cross-seed carrier/timing table.

### C2 — ③ re-assessment (load-bearing / handoff on the real carrier)

The real carrier is single-layer (L0). The "L0→L1 handoff" question dissolves; the
meaningful tests on the **canonical** readout are:

- **Load-bearing:** mean-ablate the real carrier head set (reuse
  `path_patch_handoff.mean_ablation_prehook` on `attn.c_proj`), re-run the canonical scan
  on the ablated model, measure the carrier/global `τ_vs_l2r` collapse vs a null-head
  ablation.
- **Cross-layer structure:** does ablating L0 carrier change later-layer (L1/L2/L3)
  canonical τ? (Is there any L0→later propagation in the *physical* frame, or is order
  read entirely in L0?)

Reuses the ablation hook; **swaps the readout to the canonical scan**.

### C3 — A re-assessment (emergence shape, coarse)

From the C1 aggregated curves: is emergence a clean single rise (~0→0.9) or the
model-frame dip-then-sharpen? Single carrier locked from the first non-zero step, or
winner-drift before locking (Q2)? Contingency/early-bias at 1000-step resolution is
**coarse**; if the crossing is hidden inside `0→1000`, flag that fine timing needs a GPU
re-train with online canonical logging (out of scope for v1).

### C4 — ⑤ re-assessment (content vs position binding — correction)

Under the canonical physical-frame readout, `τ_vs_l2r ≈ 0.9+` at convergence **is**
recovery of the original-text block order. Because the layout is shuffled, recovering it
requires content/structure, so the carrier is **content/structure-bound**, not the
"slot scaffold" the model-frame ⑤ wrongly concluded. C4 states this correction with the
canonical numbers and the identity-vs-random / model-vs-physical contrast already
measured (identity-model τ=1.0 tautology vs random-physical τ=0.96 real). No new
relocation run needed for the headline; an optional canonical relayout confirmation is
deferred.

## Architecture

- Thin orchestrator `analyses/canonical_reanalysis.py` that (a) parametrizes the
  existing scan over `runs/handoff_overnight/seed{2,42,123}` ckpts × K sampling seeds
  (`none_mode=b0`), (b) aggregates (reusing the `aggregate_l0h5_evolution.py` logic,
  generalized from hardcoded L0H5 to the modal winner), (c) drives C2 ablation via the
  canonical scan. Plots reuse the existing heatmap/curve code.
- Outputs under `runs/canonical_reanalysis/`; a `analyses/canonical_reanalysis_README.md`
  that **restates ③/A/⑤ corrected** and supersedes the model-frame conclusions.

## Decision criteria

- **Carrier:** the step-10000 aggregated modal winner per seed (expect L0); report
  whether all 3 seeds agree and how it differs from the model-frame L1.
- **Emergence:** τ rises from within-noise (~|τ|<0.2, anchored by step0) to a stable high
  value; report the coarse crossing interval and heavy-baseline reference.
- **③:** carrier ablation drops canonical τ beyond null (load-bearing); cross-layer
  effect reported.
- **⑤:** convergence τ_vs_l2r (physical) is high ⇒ content/structure recovery; correct
  the slot-scaffold claim.
- Per-seed first; K=3 sampling seeds for variance; 3 training seeds is a floor.

## Risks

- **none_mode choice** (`b0` vs `old`/`predictor`) shifts absolute τ; pinned to `b0`
  (sealed) and anchored against the seed2 0.07→0.96 contrast.
- **Coarse timing** — 11 ckpts give 1000-step resolution; emergence crossing may sit
  inside one interval → GPU fine run deferred, flagged not hidden.
- **Sampling variance** — K=3 sampling seeds + mean±std; never decide on a single scan.
- **Existing-script fidelity** — adapt the sealed scripts minimally (ckpt list + winner
  generalization); do not silently change the extraction math.
- **Scope** — C0–C4 in one spec (user choice); C2 is the heaviest (readout swap into the
  ablation path).
