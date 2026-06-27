# Position-Prior Decomposition & Content-Binding (Pillar ⑤-core+) — design

**Date:** 2026-06-28
**Status:** design (approved direction, hardened)
**Predecessors:** `2026-06-27-emergence-characterization-design.md` (Spec A: the order
template is present diffusely from init; the carrier is *selected*, not constructed),
`2026-06-27-handoff-causal-path-patching-design.md` (Pillar ③: no handoff)
**Branch:** `attn-order-alternating`
**Compute:** no new training. CPU forward re-extraction from saved ckpts
(`ckpt_step{0,1000,…,10000}.pt` exist for all seeds); GPU only as an accelerator.

## Motivation

Spec A found the order/L2R signal is already present at **step 0 (untrained)**:
`τ_vs_l2r ≈ 0.8` model-wide while `val_train_objective ≈ ln(vocab)` (random). So the
"order signal" is not learned-from-scratch — it is a **prior**. Two questions follow,
and both are sharpened by the data-shuffle premise:

**Data-shuffle premise (central constraint).** The data layout is permuted: physical
block `p` is placed at model slot `s = layout_perm(p)` (a block originally at physical
position 7 may be at model slot 35). The model **only indexes model slots**; physical
position is permuted away and never fed in. `forward_fn` shuffles each token *and its
position embedding* by the reveal `orders` together, conditions AdaLN on the target
position embedding `wtpe`, and applies a causal mask. The trajectory τ uses **identity
probe_orders** and compares the rolled-out `σ_model` to `np.arange` — i.e. the
**model-slot** order. Therefore any step-0 "position prior" is necessarily a prior over
**model-slot order**, divorced from the (shuffled) content.

**Q1 — Why is there a position prior, and is it even real?** Decompose `τ(step0)` into a
measurement/architecture **floor** vs a genuine **position-embedding** contribution. If
the floor explains most of `0.8`, the diffuse-prior story in Spec A is partly a
metric/architecture artifact and must be reframed.

**Q2 — Is the converged carrier position-bound (a pure slot scaffold) or content-bound
(a real learned order representation)?** Distinguish via an OOD content-relocation
diagnostic (the tool flagged in [[confound_decomposition_20260621]]).

## Scope

Two parts, per seed (2/42/123), no training. Part 1 decomposes the step-0 prior; Part 2
probes content vs position binding across training. `inv_perm` is used **only** for
posthoc frame translation, never in construction (per [[inv_perm_strict_boundary]]).

## Part 1 — Step-0 model-slot prior decomposition

Read out per-(layer,head) `τ_vs_l2r` (method `C-D+L`, model-slot frame) under a ladder
of controls. **Report an ablation table, not a forced additive decomposition** (wpe/wtpe
interact; the floors are different *kinds* of baselines).

### 1A — Floor baselines (three layers)

| control | what it isolates |
|---|---|
| **random/destroyed-B null** | rollout/readout chance baseline (no structure) |
| **synthetic uniform-causal attention** | pure mask+readout theoretical floor: `att[i,j]=1/(i+1)` for `j≤i`, fed through `build_model_frame_strict65 → C-D+L rollout` |
| **real step-0 zero-both-PE** | real architecture floor: the actual step-0 model with `wpe` *and* `wtpe` zeroed (random QK geometry + RMSNorm/qk-norm + causal mask, learned positional conditioning removed) |

**Mask-convention requirement:** the synthetic uniform-causal baseline **must match the
model's actual causal mask convention**, especially whether the diagonal/self edge
(`j=i`) is allowed. The plan's first step verifies the model's mask (the formula above
includes the diagonal); if the model forbids self-attention the synthetic must too,
else the floor is biased.

### 1B — PE contribution (4 arms, hook-based)

At `ckpt_step0`, run forward with: **full / zero-wpe / zero-wtpe / zero-both**. Zeroing
is via a reversible `forward_hook` (or context manager) on the `wpe`/`wtpe` embedding
modules returning zeros — **never a permanent parameter edit**; a bit-identical restore
test guards it. Read per-(layer,head) τ, especially the eventual carrier-candidate
heads.

**wpe and wtpe are semantically distinct and reported separately:**
- `wpe` = **token-side** slot positional embedding (tells each token which model slot it
  occupies);
- `wtpe` = **target/AdaLN-conditioning** slot embedding (tells the block which slot is
  being predicted).

Report columns `τ_full, τ_zero_wpe, τ_zero_wtpe, τ_zero_both, Δ_wpe=full−zero_wpe,
Δ_wtpe=full−zero_wtpe, Δ_both=full−zero_both` — **`Δ_wpe + Δ_wtpe ≠ Δ_both` (not
additive).** If `Δ_wpe` is small but `Δ_wtpe` large, the conclusion is not "PE is
useless" but **"the slot prior comes mainly from target-position conditioning."**

**Optional robustness arm (only if zeroing causes a scale/OOD artifact):**
`permute_wpe_indices` / `permute_wtpe_indices` (shuffle which slot gets which PE — keeps
activation scale in-distribution).

### Interpretation (Part 1)

- `uniform_causal` already `τ≈0.7–0.8` and `zero_both ≈ full` → the prior is **mostly a
  metric/mask artifact** → **reframe Spec A** (diffuse signal is partly artifact).
- `zero_both` low, `full` high → learned **PE/target conditioning** supplies the slot
  prior.
- `zero_both` > `uniform_causal` → the real step-0 **architecture** (random QK,
  RMSNorm/qk-norm) itself reinforces slot order beyond the bare mask.

Decomposition reported as:
`τ_full_step0` vs `τ_zero_both (real arch floor)` vs `τ_uniform_causal (theoretical
floor)` vs `τ_random_B (null)`; `full − zero_both` = learned PE/target-PE contribution
(upper estimate).

### 1C — Frame sanity

**1C-1 (main, no forward change):** from the *same* forward/B, evaluate the rolled-out
order in two frames — `τ_model_slot = τ(σ_model, arange)` and
`τ_physical = τ(inv_perm[σ_model], arange)`. Also report the
**permutation-only baseline** `τ_perm_baseline = τ(training_inv_perm[arange(N)],
arange(N))` (the τ induced by the fixed CleanPermutation alone). **Expected:
`τ_model_slot ≫ τ_physical`, and `τ_physical ≈ τ_perm_baseline`** (i.e. when
`σ_model ≈ arange`, the physical-frame τ is explained by the permutation itself, *not* by
learned physical-order recovery). The key claim is the inequality, **not** that
`τ_physical = 0`.

**1C-2 (optional):** probe_orders robustness — identity vs random probe_orders;
explicitly a *probe-order* robustness check, not the frame sanity. Off the main line.

## Part 2 — OOD content-relocation diagnostic

Same text, same ckpt, fed under **K=8 layouts**; ask whether the carrier's order readout
follows the **slot** (position-bound) or the **physical block content** (content-bound).
**Explicitly OOD**: the model trained on one layout; re-layout is out-of-distribution.

### Layouts (frozen, reproducible)

- `layout_0 = training/CleanPermutation` (the in-distribution **anchor**).
- `layout_1..7` = fixed-seed random relayout permutations.
- Saved to `layouts.json`: `{layout_id, perm, inv_perm, is_training_layout, rng_seed}`.
- (Expandable to K=16 if the first round is borderline; K=8 is the v1 floor.)

Relayout mechanics: recover physical-frame blocks (un-permute the model-frame chunk by
the training `inv_perm`), then re-permute by `layout_k` to build the model-frame input;
run; translate `σ_model` back to physical via `inv_perm_k` for the content score.

### Checkpoints

`step0` (untrained slot prior) · `step2000` (post-pruning carrier — connects to Spec A's
event) · `step10000` (converged carrier).

### Scores (per layout `π_k`, carrier heads)

Rollout gives the model-slot order `σ_k_model`; `σ_k_phys = inv_perm_k[σ_k_model]`.

- **position-bound:** `τ_pos(k) = τ(σ_k_model, arange(N))` — high ⇒ order keys on slot
  regardless of which content sits there.
- **content-bound:** `τ_content(k) = τ(σ_k_phys, arange(N))` — high ⇒ rolled-out order
  matches the physical-block identity order (follows content).
- **anchor + drop (always reported):** `τ` at `layout_0`, `relayout_mean τ`, and
  `relayout_drop = relayout_mean − anchor`. If relayout τ collapses, Part 2 is **OOD
  inconclusive** (the circuit broke, not a binding result).

**Anchor-validity gate (before interpreting any binding score):** require the carrier
readout to be *valid on the training layout* — `τ_pos(anchor)` high for the intended
carrier heads. If the anchor τ is already low, the checkpoint/head extraction failed or
the carrier is not active, so binding is marked **invalid/inconclusive** and not
interpreted (this is distinct from an OOD break, where the anchor is high but relayout
collapses). Tier-specific expectation (not a hard gate): strong-carrier seeds (2/123) →
`τ_pos(anchor) ≥ 0.95` at step10000; seed42 weak tier → weak-carrier range, e.g.
`≥ 0.60`.
- **secondary (optional):** cross-layout pairwise stability
  `stability_slot = mean_{i<j} τ(σ_i_model, σ_j_model)`,
  `stability_phys = mean_{i<j} τ(σ_i_phys, σ_j_phys)`.

Report mean `τ_pos` / `τ_content` over relayouts with CIs over layouts and samples.

### Interpretation (Part 2)

- `τ_pos` high, `τ_content` low → **slot scaffold** (order is positional, survives OOD).
- `τ_content` high → **content-bound** order that generalizes across layouts (rare but
  strong — a real learned content-order representation).
- both low, or anchor high but relayout τ collapses → **OOD break / inconclusive**.

Run at step0 (expect position-bound≈1, content-bound≈0) → step2000 → step10000 to see
whether (and when) any content binding appears relative to the pruning event.

## Architecture

- `analyses/position_prior_decomp.py` — pure functions + a reversible PE-ablation hook +
  a layout-relocation helper. Reuses `path_patch_handoff.load_model_and_chunks_seed`,
  `run_clean`, `make_probe_batch`; `attention_trajectory.extract_all_layer_B`;
  `batch_readout.l0_strict65.build_model_frame_strict65`;
  `none_separated_block_graph.rollout_by_method`; `order_tau_readout.per_head_tau`;
  `clean_training_protocol` (CleanPermutation, block↔token expansion) for relayout.
- `analyses/plot_position_prior.py` — floor/ablation bar charts (Part 1), τ_pos vs
  τ_content across ckpts (Part 2).
- Outputs `runs/position_prior/seed{2,42,123}/`: `part1_ablation.csv`, `layouts.json`,
  `part2_binding.csv`, `summary.json`, figures; cross-seed
  `analyses/position_prior_README.md`.

## Units

- `synthetic_uniform_causal_attn(S,H,T)` → causal-uniform attention stack.
- `pe_ablation(model, which)` context manager (`which ∈ {wpe,wtpe,both,none}`),
  bit-identical on `none`.
- `tau_table_under(model, chunks, ablation) -> tau[L,H]`.
- `make_layouts(seed, K) -> list[dict]` (anchor + fixed random), persisted.
- `relayout_chunks(chunks, training_inv_perm, layout_k)` → model-frame input.
- `binding_scores(model, ckpt, layouts, carrier_heads)` → `τ_pos`, `τ_content`, drops.

## Testing (TDD)

- `synthetic_uniform_causal_attn` fed to the rollout yields a **known** ascending τ
  (regression-pinned value), confirming the mask/readout floor is computed correctly.
- `pe_ablation(none)` is **bit-identical** to a plain `run_clean` (τ Δ ≡ 0); restoration
  after the context manager leaves weights unchanged.
- Frame sanity: on a synthetic σ with a known permutation, `τ_model_slot` and
  `τ_physical = τ(inv_perm[σ])` match hand-computed values.
- Relayout: `relayout_chunks` with `layout_0 = training` reproduces the original chunks
  (identity round-trip). At `layout_0`, `τ_content` is the **hand-computed physical-frame
  translation** of `σ_model` (apply `training_inv_perm` then Kendall-τ vs `arange`); it
  **need not equal `τ_pos`** (they coincide only if the training layout ≈ identity). A
  low anchor `τ_content` under a slot-bound model is expected, set by the CleanPermutation
  — not a bug.

## Risks

- **PE-zero scale artifact** — zeroing PE shifts activation scale (OOD); mitigated by the
  optional permute-PE robustness arm and by reading the *direction* of the change, not
  absolute τ.
- **Part 2 OOD break** — relayout may simply break the circuit; the anchor + relayout_drop
  make this detectable and reported as inconclusive rather than over-read.
- **inv_perm misuse** — used only for posthoc σ_model→σ_phys translation, never in
  constructing B (regression-guarded by the relayout round-trip test).
- **Floor is not additive** — wpe/wtpe interact; results are an ablation table, not a
  linear decomposition.
- **Single run per seed** — per-seed first; 3 seeds is a floor.

## Why this matters

Part 1 decides whether the line's "order signal" is, at root, a **metric/mask artifact**,
a **learned positional scaffold**, or both. Part 2 decides whether training ever turns
that scaffold into a **content-bound order representation**. Either outcome reshapes the
paper's central claim, so this is run before any further GPU work (Spec B contingency).
