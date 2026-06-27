# Order-Signal Handoff Circuit — Pillar ③: Causal Handoff Verification (path patching)

**Date:** 2026-06-27
**Status:** design (approved direction, hardened)
**Predecessor:** `2026-06-26-order-signal-handoff-circuit-design.md` (Pillars ①②, online recording)
**Branch:** `attn-order-alternating`

## Context & Motivation

Pillars ①② (overnight 3-seed × 10k trajectory, `runs/handoff_overnight/`) produced
a clear but *non-trivial* picture of where the order signal lives:

| seed  | dominant carrier layer | shape |
|-------|------------------------|-------|
| seed2  | **L1** (4-head strong set {H0,H3,H5,H7}, all τ=1.0, from step ~2000) | L0 stable ~0.80 |
| seed42 | **L0** (no strong head anywhere; weak ~0.85) | L1/L2/L3 decay to weak |
| seed123 | **L1** (4-head strong set {H0,H5,H6,H7}; H0 τ=0.96, others τ=1.0; from step ~3000) | L3 secondary ~0.80 |

Two facts drive this design:

1. **No clean temporal handoff.** The carrier does not visibly *move* L0→L1 over
   training. Instead all layers pass at step 0–1000 (`strong_pass=8`), then an early
   reorganization (step ~1000→2000) collapses the signal onto a single carrier
   *layer* that is then **seed-locked**. So the object Pillar ③ tests is not a
   moving handoff but a *static causal dependency* between heads/layers in the
   converged ckpt.
2. **The carrier is multi-head redundant.** In the L1-carrier seeds, the order
   signal is replicated across a **4-head strong set** (seed2 {H0,H3,H5,H7} all τ=1.0;
   seed123 {H0,H5,H6,H7} with H0 τ=0.96 and the other three τ=1.0 — all ≥ the 0.95
   strong threshold). This is not a one-head/one-mechanism circuit; it looks like a
   **redundant / distributed carrier subspace**. Single-head ablation will therefore
   easily produce **false negatives** — absence of effect from one head is evidence
   of redundancy, not absence of causality.

**Why τ (not NLL) is the causal readout.** Training is any-order (random reveal),
so the objective is by construction ~order-insensitive — ablating a carrier head and
measuring `model_order` NLL would likely show ~0 change *because the behavioral
metric is order-agnostic by design*, not because the head is unimportant. The order
signal lives in the attention's model-frame B structure (→ `tau_vs_l2r`), not in the
loss (consistent with the line's framing: emergent intrinsic-structure alignment, not
behavioral necessity). **All causal readouts are τ-based.**

## Scope

Causal verification of the converged-checkpoint circuit, in three staged
interventions, **per seed** (report per-seed first, aggregate second; 3 seeds is a
floor). Pillars ④ (freeze/graft training-dynamics inheritance) and ⑤ (representation
content) remain later specs. The **onset-checkpoint** dynamics experiment is an
**optional appendix here, not main line** — the main line proves the static circuit
on the stable (step 10000) checkpoint.

## Stage 0 — Fixed carrier-set definition (no runtime selection)

Carrier sets are read **once** from the step-10000 `tau_table.npz` (method `C-D+L`,
signed τ, `bs_mean=16`) and **written into config**. Pillar ③ is causal *verification*,
not re-discovery — runtime head selection would let the interpretation口径 drift.

Three tiers per (seed, layer), by `|τ|`:

- **strong carrier:** `|τ| ≥ 0.95`
- **weak carrier:** `0.60 ≤ |τ| < 0.95`
- **null head:** smallest-`|τ|` same-layer heads, **excluding any strong/weak carrier**
  (take the 3 smallest). Null is defined by exclusion, never by a bare "low τ" — so
  e.g. seed42 L1 H2 (τ=0.70) is a *weak carrier* and can never be used as null.

Naming convention (avoid "carrier" overload): **L1 strong carrier sets are the
primary intervention targets; L0 weak carriers are treated as candidate upstream
sources.** No L0 head clears 0.95 in any seed, so L0 is always a *weak upstream
source*, never a strong carrier.

### Frozen carrier-set config (from step-10000 τ table, method C-D+L)

```
seed2:
  L0  strong: []                         weak(src): [H1,H6,H7]        null: [H5,H2,H3]
  L1  strong: [H0,H3,H5,H7]              weak: [H1,H2]                null: [H4,H6]
  L2  strong: []                         weak: [H3]                   null: [H1,H6,H5]
  L3  strong: []                         weak: [H5]                   null: [H7,H1,H6]
seed42:
  L0  strong: []                         weak(src): [H0,H3,H5,H6,H7]  null: [H4,H1,H2]
  L1  strong: []                         weak: [H2,H6]                null: [H4,H1,H3]
  L2  strong: []                         weak: [H3,H5]                null: [H7,H0,H6]
  L3  strong: []                         weak: [H2]                   null: [H3,H4,H0]
seed123:
  L0  strong: []                         weak(src): [H6,H7]           null: [H3,H4,H1]
  L1  strong: [H0,H5,H6,H7]              weak: [H1,H4]                null: [H2,H3]
  L2  strong: []                         weak: [H4]                   null: [H0,H7,H3]
  L3  strong: []                         weak: [H0,H1,H3,H4]          null: [H5,H7,H2]
```

**seed42 has no strong carrier anywhere** — it is the natural negative/contrast seed
(no strong L1 carrier set to inherit a handoff). seed2/seed123 each have a
**4-head strong L1 carrier set** — the redundancy this design is built to probe.

## Stage 1 — Load-bearing carrier-set ablation (mean-ablation of whole head)

**Intervention.** In a forward pass over the fixed probe set, replace a source head
set's per-example output slice (the head's `hs` columns of `y`, *before* `c_proj`)
with its **position-wise batch mean** (mean-ablation — stays in-distribution; not
zero-ablation). Precisely, only the **batch** dimension is averaged; the token/block
**position** structure is preserved:

```
y[:, pos, head_slice]  ←  mean_over_batch( y[:, pos, head_slice] )   for every pos
```

The position/block dimension is **not** averaged out (doing so would erase positional
information, making the intervention too strong and inflating the null). All other
heads/layers/MLPs unchanged. (A global position-collapsed mean is a stronger variant,
out of scope for the main experiment.)

**Readout.** Re-extract attention of all layers **downstream of the ablated layer**,
rebuild model-frame B65 per head → `none_separated` rollout (`C-D+L`) → record:

1. downstream **per-(layer,head) τ_vs_l2r** (signed) and its Δ from clean;
2. **global consensus order** τ_vs_l2r and its Δ;
3. **carrier multiplicity collapse** — per downstream layer, the number of heads with
   `|τ| ≥ 0.95` after intervention vs before (e.g. seed2 L1 4→? ; the count, not just Δτ).

**Two sub-interventions (the minimal handoff evidence chain):**

- **Stage 1a — L1 strong carrier set is a source.** Ablate the L1 strong carrier set;
  read out L2/L3 + global. Tests whether L1 feeds the later layers / global order.
- **Stage 1b — L1 carrier depends on L0 upstream source.** Ablate the L0 weak-source
  set; read out L1 (+L2/L3/global). Tests whether the L1 carrier *structure itself*
  depends on upstream L0 heads. **Without 1b we only show L1 is load-bearing, not that
  L0→L1 is a handoff.**

**Redundancy granularity (the core of this pillar).** For the L1 strong carrier set,
run the ablation at four granularities. We **expect** an approximate monotone trend
(within CI), but it is **reported, not used as a hard exclusion criterion** — τ
sampling noise can locally violate it (e.g. LOO 0.21 vs full 0.19):

```
single-head ablation  ≤  leave-one-out ablation  ≤  full-set ablation   (expected, within CI)
(ablate 1)               (ablate all but 1)          (ablate all 4)
```

**Controls.** Same-layer **null-head** ablation (matched count) as the null Δτ
distribution. seed42 (no strong L1 set) as the cross-seed contrast.

### Interpretation table (written *before* running)

| result pattern | interpretation |
|---|---|
| single-head effective, set stronger | single head partially load-bearing; set more complete |
| single-head null, set effective | L1 carrier is **redundant / distributed** |
| both single and set null | L1 τ may be **epiphenomenal**, or ablation site / readout wrong |
| non-carrier (null) ablation equally effective | intervention too coarse or readout too noisy → de-noise |

## Stage 2 — Path-restricted L0→L1 patching

Run **only if** Stage 1b shows L0 ablation hurts the L1 carrier structure.

**Precise intervention (standard path-patch / causal-tracing language).** For each L1
**destination** carrier head, compute its **Q and K** from a *corrupted* L1-input
residual in which **only the L0 source-head OV contribution is mean-ablated**; use
**clean** activations for the destination head's **V** input, for all other heads, for
MLPs, and for every non-target computation.

```
Q_dst, K_dst  ← corrupted residual  (L0 source-head OV contribution mean-ablated)
V_dst         ← clean residual
other heads, MLP, direct paths ← clean
```

This isolates the question: does the L0 source influence the L1 destination through
its **attention pattern / order-reading (QK)** path, rather than through V content or
other routes? The readout is the L1 destination carrier's **τ_vs_l2r Δ** (whether the
B structure collapses) — matching "is the order structure inherited via QK".

**Pre-registered effect-size ratio** (avoids post-hoc drift; magnitude-based because
collapse may show as a τ drop of either sign):

```
path_fraction = |Δτ_path-restricted_QK| / |Δτ_full_L0-source_ablation|
```

Report `path_fraction` as a **continuous value with a matched null-path control**
(patching a null-head OV → L1 dst QK). A `path_fraction` substantially above the
null-path CI supports that the L0→L1 QK route accounts for a meaningful part of the
full L0-ablation effect. We **do not hard-gate** on a fixed threshold at 3-seed scale
(a provisional `≥ 0.3 and above null CI` may be noted, but the continuous fraction +
null comparison is the operative report).

**Contrast (not a hard "no effect" claim).** Because seed42 lacks a strong L1 carrier
set, its L0→L1 path-restricted effects should be **substantially weaker, more diffuse,
or fail the load-bearing criterion** vs seed2/seed123 — *not* asserted to be exactly
zero (seed42 L1 H2 τ=0.70 may carry weak relay). Concretely:

- seed2/seed123: L0→L1 edge effect **concentrated on the L1 strong-carrier destinations**;
- seed42: no strong L1 destination set → edge effects weak / diffuse / sub-criterion.

## Decision criteria

The handoff (L0 → redundant L1 carrier → downstream/global order) is **supported** if:

1. **L1 carrier-set ablation** causes downstream/global Δτ **beyond the null-head
   distribution** (Stage 1a), with an **expected** redundancy trend single ≤ LOO ≤
   full-set (within CI; reported, not a hard gate).
2. **L0 weak-source ablation** causes **L1 carrier collapse** (multiplicity drop +
   Δτ beyond null) (Stage 1b).
3. **L0→L1 path-restricted QK patch** yields `path_fraction = |Δτ_path| /
   |Δτ_full_L0-ablation|` **substantially above the matched null-path control CI**
   (Stage 2) — reported as a continuous fraction, not hard-gated.
4. **seed42** lacks the same coherent L0→L1 edge effect (cross-seed contrast).

Report **per-seed first**; cross-seed aggregation is secondary (random-order runs
diverge; 3 seeds is a floor, not a law).

## Noise control

- τ uses **`bs_mean=16`**; emit **CIs over multiple probe batches** (random-order τ has
  real sampling variance — never decide on a single extraction).
- All "effect" judgments are **clean-vs-intervened Δ on the same probe batch** (paired),
  compared against the matched null-head ablation distribution.

## Outputs

- `analyses/path_patch_handoff.py` — pure intervention + readout (mean-ablation hook,
  path-restricted QK patch, τ readout reused from the trajectory pipeline).
- Per-seed Δτ tables. **Main table schema:**

  ```
  seed | stage | intervention | target layer | target heads | mean τ before |
       | mean τ after | #strong before | #strong after | Δ global τ
  ```

  rows for: Stage1a (single / LOO / full-set / null), Stage1b (L0-source / null),
  Stage2 (per L1 dst head edge), seed42 contrast.
- One **src-ablation × downstream-layer Δτ heatmap** per seed.
- (Optional appendix) onset-ckpt run of Stages 1–2 to test whether the causal edge
  appears concurrently with the rise of the L1 τ signal (handoff *emergence*).

## Testing (TDD)

Pure functions unit-tested on synthetic inputs before any real ckpt run:

1. **Mean-ablation hook** replaces exactly the target head's `y` slice with its batch
   mean and leaves all other slices bit-identical.
2. **Path-restricted patch** changes only the destination head's Q/K (V bit-identical;
   other heads / MLP / direct paths bit-identical).
3. **Bit-identical anchor:** when the ablation source equals the clean activation
   (mean == per-example, or empty ablation set), Δτ ≡ 0.
4. **Hook locality:** ablating layer ℓ leaves all activations/attention from layers
   `< ℓ` **bit-identical** (guards against hooking the wrong site).
5. **Redundancy sanity** on seed2/seed123 L1 strong set (real ckpt smoke): the
   leave-one-out ordering holds — `single-head ≤ leave-one-out ≤ full-set` Δτ.

## Risks

- **False-negative from redundancy** — the central risk; mitigated by the
  set/single/LOO granularity ladder and the multiplicity-collapse readout.
- **Composition approximation** under RMSNorm qk-norm / AdaLN — Stage 2 path-restricted
  patch is the causal cross-check the Pillar ② weight-based score deferred to.
- **τ sampling noise** under random-order — `bs_mean=16` + multi-batch CIs + paired Δ.
- **Carrier-set staleness** — frozen from a single step-10000 table; documented and
  read back from the ckpt's seed for integrity.
