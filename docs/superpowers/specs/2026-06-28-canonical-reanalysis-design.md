# Canonical 65-node Re-analysis of the Order-Emergence Line — design

**Date:** 2026-06-28
**Status:** design (approved direction; synced to the sealed strict-65-node protocol)
**Predecessors:** ③ `2026-06-27-handoff-causal-path-patching-design.md`,
A `2026-06-27-emergence-characterization-design.md`,
⑤ `2026-06-28-position-prior-decomposition-design.md` — all three used a model-frame
readout **without the posthoc-inv scoring**, making step-0 τ tautological (identity
reveal) and picking the wrong carrier (L1 vs the canonical L0).
**Branch:** `attn-order-alternating`. **Compute:** no GPU (11 ckpts/seed); GPU only if
coarse timing is insufficient.

## The canonical protocol to sync with (already sealed)

`reports/strict_65node_discovery_ckpt_verification_20260617/` defines the **strict
65-node None-separated discovery protocol**, and `scripts/search_none_separated_65_heads.py`
implements it. The redo **reuses that tool**; it does not invent a readout.

Protocol (from `01_protocol_definition.md`):
- **65 nodes**: node 0 = `[None]`/BOS (separate, not folded into a content block);
  nodes 1..64 = content blocks. `B65 = build_none_separated_B(A)` (`diag=0`).
- **Loss-aligned AR frame** `attn[:T,:T]`: query axis = target block, key axis = source
  block with node-0 reserved for `[None]`. Extraction
  `_attn_to_A_block_loss_aligned_with_none_vec`.
- **`inv_perm` for scoring (posthoc), not avoided**: the order is rolled out, then scored
  against **physical L2R**. (Strict-LF model-frame and oracle-remapped are proven
  *equivalent*, `05_extraction_frame_comparison.md`: gate distribution identical, Δ=0.000
  — permutation-equivariance. The search tool uses the oracle-remapped path.)
- **Random reveal** orders per sample (seeded `seed+i`).
- **Methods**: `("C-D+L","L","C-D","C+L","C","none_edge",…)`; **`L` is the primary
  strong-pass method** in the existing sweep (not C-D+L). Report L + C-D+L + none_edge.
- **Gate**: `classify_gate_status` → `strong_pass` (τ=1.0, first_block=0, phys0_rank=0,
  prefix@4=4, prefix@8=8) / `weak_pass` / `fail`.
- **Destroyed controls** (the null + the content-dependence test): `entry_shuffled_control`
  + `content_label_permutation_control`, 5 seeds → mean |τ| ≈ 0.05–0.07 (random floor).

Authoritative `clean_base` sweep (`strict_65node_ckpt_sweep.tsv`) shows the **real
emergence**: best_tau **0.19 @step0 → 0.14 @1000 → 1.0 @5000+**, strong heads **L0H1–H4
(method L)**, destroyed floor ≈0.05. This is the metric the redo runs on
`runs/handoff_overnight`.

## Why this corrects ③/A/⑤

⑤/handoff used `build_model_frame_strict65` (same 65-node *extraction*) **but scored τ vs
`arange` in the model frame under identity reveal — skipping the posthoc-inv physical
scoring and using identity instead of random reveal** → step-0 τ≈0.77 (tautology) and an
L1 "carrier". The canonical tool restores: random reveal + posthoc-inv physical scoring +
method L + gate/destroyed controls → order **absent at init, emerges to τ=1.0 in L0**.

## Components (C0–C4, one spec per user choice)

### C0/C1 — Canonical emergence + carrier baseline (foundation)

Run `search_none_separated_65_heads.py` (or the existing sweep driver that produced
`strict_65node_ckpt_sweep.tsv`) on **3 seeds × 11 ckpts** (`step0,1000,…,10000`),
**M=8, batch_size=8, methods=L/C-D+L/none_edge, control_seeds=5**, random reveal. Per
seed, per step record: **gate distribution** (#strong/#weak/#fail), **best_head /
best_method / best_tau**, and **destroyed mean |τ|**.

Outputs per seed: `strict65_sweep.tsv` (step × {gate counts, best_head, best_tau,
destroyed_floor}) + a `32 heads × 11 steps` strong-pass/τ heatmap.

Read off:
- **Real carrier**: the strong-pass heads at step 10000 (expect **L0**; record per seed
  and contrast with the model-frame L1).
- **Real emergence timing**: the coarse interval where best_tau crosses the destroyed
  floor toward 1.0 / strong_pass count rises from 0.
- **Anchor**: reproduce the seed2 contrast (step0 best_tau ≈ destroyed floor; step10000
  best_tau = 1.0, L0 strong heads) before trusting the rest.

### C2 — ③ re-assessment (load-bearing on the real carrier, canonical readout)

The real carrier is single-layer L0 → "L0→L1 handoff" dissolves. On the **canonical
readout**: mean-ablate the L0 strong-pass head set (reuse
`path_patch_handoff.mean_ablation_prehook` on `attn.c_proj`), re-run
`search_none_separated_65_heads.py`, and measure the **strong_pass count / best_tau
collapse** vs a null-head ablation. Also: does ablating L0 change any later-layer
strong passes (cross-layer structure in the physical frame)?

### C3 — A re-assessment (emergence shape, coarse)

From the C1 sweep: is emergence a single clean rise (floor→1.0) and at which interval;
does the strong-pass head set drift (Q2 winner-drift) before locking; how many seeds
agree on L0. 1000-step resolution is coarse — if the crossing hides inside `0→1000` or
`1000→5000`, flag that fine timing needs a GPU re-train with online canonical logging
(deferred).

### C4 — ⑤ re-assessment (content vs position — correction, partly built-in)

Under the canonical tool, a strong-pass `τ=1.0` **is** recovery of the physical
(original-text) block order via posthoc inv — which under a shuffled layout requires
content. The tool's **`content_label_permutation_control`** already tests content
dependence: if permuting content-node labels collapses τ to the destroyed floor, the
signal is **content/label-dependent**, not a positional slot artifact. C4 reports this
control at the converged carrier and **corrects ⑤'s "slot-scaffold" to content/structure-
bound** with the canonical numbers.

## Architecture

- Thin orchestrator `analyses/canonical_reanalysis.py`: loop
  `search_none_separated_65_heads.py` over `runs/handoff_overnight/seed{2,42,123}` ckpts
  (or reuse the existing sweep driver), aggregate per-seed sweep TSV + heatmap; drive C2
  ablation by registering the mean-ablation hook before the canonical scan.
- Reuse `none_separated_block_graph` (rollout/gate/controls), `per_head_order_scan`
  extraction, existing plotting.
- Outputs `runs/canonical_reanalysis/`; `analyses/canonical_reanalysis_README.md` that
  **restates ③/A/⑤ corrected** and supersedes the model-frame conclusions.

## Decision criteria

- **Carrier** = step-10000 strong-pass heads per seed (expect L0); report cross-seed
  agreement + contrast with model-frame L1.
- **Emergence** = best_tau rises from the destroyed floor (~0.05) to 1.0; strong_pass
  count 0 → N; report the coarse crossing interval.
- **③** = L0-carrier ablation collapses strong_pass/best_tau beyond null; cross-layer
  effect reported.
- **⑤** = converged strong-pass τ=1.0 + `content_label_permutation_control` collapse ⇒
  content/structure-bound; slot-scaffold claim corrected.
- Per-seed first; control_seeds=5 + M-averaging for variance; 3 training seeds is a floor.

## Risks

- **Protocol fidelity** — pin method `L` primary, 65-node None-separated, posthoc-inv
  scoring, random reveal, destroyed controls; anchor against the clean_base sweep numbers
  (0.19→1.0, L0 strong heads, floor 0.05). Do not silently alter the extraction math.
- **Coarse timing** — 11 ckpts = 1000-step resolution; crossing may sit inside an
  interval → GPU fine run deferred, flagged.
- **Sampling variance** — control_seeds=5 + M-averaging; never decide on a single scan.
- **Scope** — C0–C4 in one spec (user choice); C2 (readout swap into the ablation path)
  is heaviest.
