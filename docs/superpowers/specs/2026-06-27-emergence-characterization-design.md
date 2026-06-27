# Order-Signal Emergence Characterization (Spec A) — design

**Date:** 2026-06-27
**Status:** design (approved direction)
**Predecessors:** `2026-06-26-order-signal-handoff-circuit-design.md` (Pillars ①②),
`2026-06-27-handoff-causal-path-patching-design.md` (Pillar ③, falsified handoff)
**Branch:** `attn-order-alternating`
**Compute:** **no GPU** (existing trajectories + cheap re-extraction from saved ckpts).
GPU is a fallback only if a no-GPU readout proves impossible.

## Context & Motivation

Pillar ③ falsified the cross-layer **handoff** hypothesis: the L1 carrier is robust to
L0 ablation, the L0→L1-QK path is below null, and the carrier is a redundantly
distributed but **layer-local, seed-locked** set. The mechanism question therefore
shifts from "how is order handed off L0→L1→…" to:

> **How is the carrier layer/head set *selected*?**

The existing every-200-step trajectory already shows the shape of the answer (two
phases, three seeds):

- **Phase A (step 0–~1200): diffuse alignment.** All layers, all heads reach
  max|τ|≈0.9–1.0, `strong_pass=8` everywhere — a generic position/order signal is
  *everywhere*.
- **Phase B (step ~1200–2000): winner-take-all pruning / symmetry-breaking.** Around
  step 1200–1400 `strong_pass` collapses 8→2–5; the order signal is pruned out of most
  heads and concentrates onto one **seed-specific** layer's set (seed2/123→L1,
  seed42→L0; L2 dies first in seed42).

So the carrier **survives a pruning event**, it is not gradually **built**. Spec A turns
this from an eyeball description into quantified results and sets up Spec B (contingency).

**Pre-checked data facts that shape this spec:**
- `runs/handoff_overnight/seed{2,42,123}/attention_trajectory/raw/step_*/tau_table.npz`
  exists at **every 200 steps** (51 files): per-(layer,head) signed τ for methods
  `C-D+L`,`L`. → A1/A2 read directly, **no re-extraction**.
- `B_model.pt` saved per step is **L0-only** (`B_all[0]` alias); the per-head **B
  matrices** for other layers are *not* saved. → A4 must **re-extract** the winning
  layer's B from checkpoints via `extract_all_layer_B` (Pillar-3 pipeline).
- Checkpoints exist at **every 1000 steps** only. → A4 brackets the event with
  pre=1000 / post=2000 / converged=10000; **transition-step B (1200–1400) is
  unavailable without a new run — a documented limitation.**
- `eval_curve.tsv` (per 200 steps) has loss columns + `lr`. LR config:
  `warmup_iters=0, lr=1e-3, lr_decay_steps=50000, max_steps=10000` → **LR is ~flat over
  the pruning window** (step200 already 9.9996e-4). A3 will almost certainly find no
  LR alignment → intrinsic phase transition (kept as a low-cost, explicitly
  correlational readout).

## Scope

Characterize the emergence mechanism from existing artifacts only, per seed (report
per-seed first). Four components; **A1/A2 are the load-bearing mainline, A3/A4 are
low-cost lite additions that must not dominate.** Full representation-content analysis
is left to Pillar ⑤; Spec B (GPU contingency) is a separate later spec, forked by A2's
outcome.

## Components

### A1 — Pruning-event detection & timing (mainline)

From the 51 per-step `tau_table.npz`, compute three concentration metrics over step
(avoid relying on a single 0.95-thresholded count):

1. **strong_head_count(layer, step):** number of heads with `|τ|≥0.95` per layer.
2. **τ-mass entropy(step):** Shannon entropy of `softmax(|τ|)` over all 32 (layer,head)
   cells — high = diffuse, low = concentrated.
3. **layer concentration(step):** per layer, `sum_h |τ[layer,h]|` (and its share of the
   total), plus the top-1 layer's share.

Method `C-D+L`, signed τ read but metrics use `|τ|`. Output per seed:
**pruning onset / midpoint / completion** (defined on the entropy curve: onset = last
step before entropy begins its monotone drop; completion = first step entropy
re-plateaus; midpoint = steepest-descent step), **winning layer**, **winning head set**
(final `|τ|≥0.95` members at step 10000).

### A2 — Winner predictability / predetermination (mainline; sets Spec B direction)

Trace the final carrier heads backward through early steps. Readouts:

1. **early→final membership AUC:** at each early step `s∈{0,200,600,1000}`, rank heads
   by `|τ[s]|`; treat final carrier membership (step-10000 `|τ|≥0.95`) as the label;
   report ROC-AUC of the early-`|τ|` score. AUC≈1 → winners already lead early;
   AUC≈0.5 → not predictable early.
2. **early→final Spearman:** Spearman ρ between `|τ[s]|` (per head) and `|τ[10000]|`,
   per early step.
3. **winner vs pruned τ trajectories:** per-head τ-vs-step curves, winners vs pruned,
   for the winning layer (plot + the step at which their paths separate).
4. **Layer-level lead (the decisive question):** does the final **winning layer**
   already have the **highest `sum_h|τ|`** *before* pruning (step ≤ 1000)? Report, per
   seed, the winning layer's `sum|τ|` rank among layers at steps {200,600,1000}.

**Verdict rule (pre-registered):** if early `|τ|` rank predicts final membership
(AUC high, ρ high, winning layer already leads) → **early-bias / initialization-locked**;
else (paths indistinguishable until step ~1200) → **dynamically contingent
symmetry-breaking**. This verdict selects Spec B's design (see fork below).

### A3 — Schedule/loss overlay (lite, correlational only)

Overlay the A1 event timing (onset/midpoint/completion) against:
- the `lr` column of `eval_curve.tsv` (expected flat → no alignment);
- the loss columns (`val_train_objective`, `val_model_order`) — does the pruning window
  coincide with a loss inflection/plateau?

Output: one overlay plot per seed + a one-line classification per seed
{`aligns-with-LR` | `aligns-with-loss-transition` | `intrinsic (no alignment)`}.
**Stated as temporal association, not causal evidence.**

### A4 — Carrier B-structure across the event (⑤-lite)

Re-extract the **winning layer's** per-head B (via `run_clean` →
`extract_all_layer_B`, Pillar-3 pipeline) from checkpoints at **{1000 (pre),
2000 (post), 10000 (converged)}**, batch-mean over `bs_mean=16` probes. For each
winning carrier head, render: B heatmap, rolled-out order (`rollout_by_method`,
`C-D+L`), `τ_vs_l2r`. Classify the emergence shape:

- **Pattern A (sharpening):** the diffuse-phase (pre=1000) B is already weak-L2R-like
  and merely sharpens post-pruning.
- **Pattern B (structural switch):** the pre-pruning B is unstructured/different and
  becomes L2R-like only post-pruning.

**Limitation (documented):** pre=1000 is just before onset and post=2000 just after
completion, so the event is bracketed but the *transition* (1200–1400) B is not
observed (no ckpt). Leave full representation content to Pillar ⑤.

## Architecture

One analysis module `analyses/emergence_characterization.py` (pure functions + a
per-seed driver) and one plotting module `analyses/plot_emergence.py`. Reuses:
- `path_patch_handoff.load_model_and_chunks_seed`, `make_probe_batch`, `run_clean`,
  `tau_table_from_attn` (A4 re-extraction);
- `attention_trajectory.extract_all_layer_B`;
- `batch_readout.order_tau_readout` / `none_separated_block_graph.rollout_by_method`
  (A4 rollout);
- direct `np.load` of `tau_table.npz` (A1/A2) and `eval_curve.tsv` (A3).

**Units (one responsibility each):**
- `load_tau_trajectory(seed) -> (steps, tau[step,L,H,method])` — stack the 51 npz.
- `concentration_metrics(tau_traj)` → A1 metrics + event timing.
- `winner_predictability(tau_traj, winning_layer, carrier_heads)` → A2 readouts + verdict.
- `schedule_loss_overlay(seed, event)` → A3 classification.
- `carrier_b_structure(seed, winning_layer, carrier_heads, ckpt_steps)` → A4 B/rollout/τ.
- `run_seed_emergence(seed) -> summary dict` + table/JSON writers; plots in the plot module.

Outputs per seed under `runs/emergence/seed{2,42,123}/`:
`concentration.csv`, `predictability.json`, `schedule_overlay.png`,
`b_structure/`(heatmaps), `summary.json`; plus a cross-seed
`analyses/emergence_README.md` with the per-seed verdicts and the Spec B fork.

## Spec B fork (pre-registered, written into the report)

- **A2 = early-bias** → Spec B tests initialization/early-lock stability (perturb init
  / early head activations / graft a very-early ckpt; "do tiny initial differences
  decide the winner?").
- **A2 = contingent** → Spec B tests pruning-window contingency (branch from
  pre-pruning ckpts with different dataloader order / micro-noise; "does randomness in
  the pruning window decide the winner?").
- **A3 = aligns-with-LR** → Spec B adds an LR/warmup-intervention arm.
- **A4 = sharpening** → Pillar ⑤ asks what latent order template is already present
  early; **= switch** → Pillar ⑤ asks what representational transition creates the
  final order code.

## Testing (TDD)

Pure functions unit-tested on synthetic inputs before touching real data:
- `concentration_metrics` on a hand-built tau_traj where a known layer wins at a known
  step returns that step as midpoint and that layer as winner; entropy is high pre /
  low post.
- `winner_predictability` on a synthetic case where winners lead from step 0 returns
  AUC≈1, ρ≈1; on a case where winners are tied until late returns AUC≈0.5.
- A4 re-extraction regression: winning-layer τ at step 10000 from re-extracted B
  matches the saved `tau_table.npz` (within sampling tolerance) — same anchor style as
  Pillar-3 Task 2.
- Event-timing is deterministic given a fixed tau_traj (no RNG in A1/A2).

## Risks

- **Single-run-per-seed:** each seed is one trajectory; A1/A2 describe *these* runs.
  The contingency question (is the winner reproducible?) is explicitly deferred to
  Spec B — A2 only measures *predictability-within-the-observed-run*, not reproducibility.
- **τ sampling variance** in A4 re-extraction: use `bs_mean=16` + the multi-batch mean
  pattern from Pillar-3; A1/A2 use the saved batch-mean tau_tables as-is.
- **Transition B gap** (A4): bracketed not resolved; documented, GPU fallback only if
  the pre/post bracket is judged insufficient.
- **Over-reading A3:** correlational only; LR is flat so expect intrinsic, but a loss
  coincidence must not be phrased causally.
