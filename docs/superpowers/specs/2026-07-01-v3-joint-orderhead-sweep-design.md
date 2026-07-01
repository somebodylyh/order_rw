# V3 Joint OrderHead Sweep — Design (2026-07-01)

Branch `p5-direct-nll-routing`. Builds on the completed feasibility work
(`docs/superpowers/specs/2026-07-01-orderhead-internalization-feasibility-design.md`,
commits `cdd250a..12eea25`) and the P7 findings
(`analyses/P7_CONTEXT_ORDER_FINDINGS.md`, `analyses/P7_POLICY_SUMMARY.md`).

## Core question (headline)

The internal OrderHead loop is feasible (bit-match ✅, gradient routing airtight,
joint smoke runs healthy). The V3 sweep asks:

> Does joint AO-GPT + OrderHead training **manufacture** a **group-level** reveal-
> order signal that frozen B/H probes could not recover, and does that signal
> improve the fixed-order validation PPL / training efficiency?

**Phase-1 tests group-level signal manufacturing at m=16.** Per-sample (m=1) is a
Phase-2 stretch, run only if the group-level gate passes. Wording matters: Phase-1
is a *group-order* experiment, not a per-sample one.

## Why group-level, not per-sample (the dilemma this resolves)

The order line is dilemma-locked:

- The only headroom over true L2R is **per-sample** (oracle +0.19 nat @10k, +0.06
  @20k); a global order structurally cannot reach it.
- But **frozen per-sample B is too noisy** (consensus τ=0.31); no per-block feature
  learns σ\* held-out (P7 null). The feasibility per-sample smoke reproduced this
  (τ_consensus 0.10, Δ_probe worse than L2R by +0.35→+0.42).
- The **global** gβ readout has a robust learnable signal (τ=0.99) but its consensus
  order is ~0.5 nat **worse** than L2R.

So the thing that can beat L2R has no learnable signal, and the thing with a signal
can't beat L2R. V3's only escape is **co-adaptation manufacturing** signal: with the
backbone unfrozen, B_φ can evolve to become readable. Group size **m is the
per-sample↔global dial** (m=1 per-sample, m=64 global; P7's old protocol was m=64).
**m=16** sits in the middle: more per-sample-ish than global (4 distinct orders per
64-batch → captures between-group variation), but with 16 samples per group to
average out B noise and bound PG variance. It is the most favorable point to detect
manufactured non-global signal.

## Phase-0 — pre-fixes (hard blockers before any run)

Carried from the feasibility final review. All three must land and be tested before
Phase-1 training starts.

1. **`OrderHeadModule.device`** — currently a plain string attribute that does not
   follow `.to("cuda")`. Derive device from a module parameter (or add a `.to()`
   override), so GPU placement is correct.
2. **Backbone `.train()`** — the co-adapt loop must explicitly set the backbone to
   `.train()` (it loads via `.eval()`). Irrelevant only if dropout=0; set it
   explicitly regardless.
3. **Group-level credit (m=16)** — replace the feasibility smoke's batch-scalar
   advantage with per-group reward (see next section). This is the central Phase-1
   mechanism, not a cosmetic fix.

## Group credit — explicit (do not regress to batch-scalar)

Batch of B=64, group size m=16, G=4 groups. Each group g gets ONE OrderHead order
σ_g from its group-mean readout B_g, broadcast to the group's 16 samples
x_{g,1..16}. The LM forward returns **per-sample** loss via
`forward_fn(..., return_token_loss=True)` → ℓ_i; the group reward is the group mean:

```
ℓ_g = (1/m) Σ_{i∈g} ℓ_i
A_g = b_g − ℓ_g                      # advantage, b_g detached
L_PG = −(1/G) Σ_g stopgrad(A_g) · log P_θ(σ_g | B_g) − β·H
```

**Baseline: prefer a per-group EMA** `b_g ← α·b_g + (1−α)·ℓ_g` (each of the G group
slots keeps its own running mean across steps). **Fallback: a single scalar EMA** is
acceptable for Phase-1 implementation simplicity — the spec permits it but flags
per-group EMA as preferred (lower-variance advantage).

Gradient routing is unchanged from feasibility: `B` is detached before the OrderHead
(PG updates θ only), the LM loss (grad-enabled `forward_fn`) updates φ only. The two
graphs are disjoint under a shared optimizer.

## Phase-1 — arms (1 seed, 10k → 30k)

All arms continue from the **same 10k parent ckpt** (seed123 lineage); the **only
variable is the order policy**. No `random` arm in Phase-1 — historical lower
baselines already establish random is worse; the question is whether joint-group
beats frozen-gβ / L2R.

| arm | backbone | OrderHead | purpose |
|---|---|---|---|
| L2R | train | none (fixed L2R) | hand-designed baseline |
| frozen-gβ | train | frozen global | Phase-1 curriculum baseline (**primary control**) |
| **joint-group m=16** | train | train, per-group reward | **main scientific arm** |
| joint-batch-level m=64 | train | train, global reward | distinguishes "just global co-adapt" from group signal |

**Start:** 10k parent (headroom is largest there; at 20k L2R is near-optimal so
there is almost nothing to manufacture). **Train:** 10k→30k (20k steps).

## Metrics

### Mechanism — group-level probe (the headline measurement)

On a fixed held-out set, group the samples (m=16), compute each group's OrderHead
order σ_g, and measure:

```
Δ_probe^group = E_g[ ℓ(X_g, σ_g) − ℓ(X_g, σ_L2R) ]
```

where ℓ(X_g, σ_g) is the group order applied to the group's held-out samples,
mean NLL. Δ_probe^group < 0 means the manufactured group order beats L2R.

**Guards against false positives (a Δ improvement that is really just reverting to
the global/L2R order, not new signal):**
- `τ(σ_g, σ_L2R)` and `τ(σ_g, σ_frozen-gβ)` — if Δ improves only because σ_g moved
  toward L2R (τ→1), that is not signal manufacturing.
- **real-B vs shuffle/zero/noise-B** — the group order must beat its own shuffled/
  zeroed-B controls; otherwise the "signal" is a position shortcut, not read from B.
- `τ_consensus` across group orders (are groups producing *different* orders?).

Per-sample Δ_probe is deferred to the Phase-2 stretch arm (it does not match the
Phase-1 group-order training objective).

### Payoff — training efficiency (fixed-order eval)

- Fixed-order `val_ori_l2r_block` PPL, evaluated identically across all arms.
- Full **10k→30k trajectory**: final value, best checkpoint, AUC/trend, and
  compute-matched step-to-threshold vs frozen-gβ / L2R.

**Primary comparison:** joint-group **vs** frozen-gβ **vs** L2R.

## Staged gate — trajectory-based, two-tier (decided at 30k)

Evaluate every **15k / 20k / 25k / 30k** (plus the 10k start). The gate is **decided
at 30k using the full 10k–30k trajectory** — final value, best checkpoint, AUC/trend
— never a single 30k point (co-adaptation may show mid-run life that the L2R
attractor later masks).

**Tier 1 — Hard health gate (all required):**
- no NaN; entropy healthy (no collapse, no explosion); OrderHead params move;
- joint-group PPL not catastrophically worse than frozen-gβ:
  `PPL_joint ≤ PPL_frozen-gβ + ε`.

**Tier 2 — Life-sign gate (any one, under the real-B guard):**
1. **Mechanism life:** Δ_probe^group vs frozen-gβ improves at the best checkpoint,
   AND real-B beats shuffle/zero (not a position shortcut, not merely τ→L2R).
2. **Payoff life:** compute-matched fixed-order PPL / AUC / step-to-threshold matches
   or beats frozen-gβ over the trajectory.

**PASS** (Tier 1 ∧ Tier 2) → Phase-2: extend to 60k, add per-sample m=1 stretch,
add multi-seed on the surviving arms. **FAIL** (Tier 1 fails, or Tier 2 shows no
life across the trajectory) → stop and document the null:

> Even internal joint co-adaptation with group-level PL credit does not manufacture
> an exploitable non-global order signal in text; the AO-GPT order line remains
> dominated by the global/L2R curriculum.

This null is itself a clean, publishable closure of the order line.

## Outcome matrix (all four close cleanly)

- **Strong:** Δ_probe^group < 0 (real-B guarded) AND joint-group PPL/step-savings
  beats frozen-gβ/L2R → co-adaptation manufactures a group signal and it pays off.
- **Mechanism-only:** group probe beats L2R but PPL does not improve → readable
  signal exists, does not convert to training efficiency under this schedule.
- **Payoff-only:** PPL improves but no clean group-probe signal → trajectory benefit
  without a measurable manufactured order signal (interpret cautiously).
- **Null:** neither → the closure above.

## Phase-2 (deferred; fires only if the Phase-1 gate passes)

Extend joint-group to 60k; add **per-sample m=1 stretch** (with per-sample Δ_probe);
add **multi-seed** (≥3) on surviving arms for step-savings noise-floor; optional
random lower reference; the full confound-guarded probe harness. A separate spec.

## Reused infrastructure

- Feasibility modules: `analyses/order_head_module.py`
  (`OrderHeadModule`, `AOGPTWithOrderHead`), `analyses/v3_feasibility_smoke.py`
  (`run_smoke` loop skeleton, `denoising_metrics`).
- `analyses/p5_utility_controller`: `load_p5_ckpt`, `order_nll`, `N=64`, `BLOCK_LEN`.
- `analyses/p7_gbeta_policy`: `GBETA_CKPT`, `HEAD=(1,7)`, `NONE_MODE='strict65_model'`,
  `sample_pl`, the group-batch protocol (`group_nll`, group ids) — the m=16 grouping
  reuses this rather than reinventing it.
- Backbone per-sample loss: `AOGPT_block.forward(..., return_token_loss=True)`.
- 10k parent ckpt: `runs/handoff_overnight/seed123/ckpt_step10000.pt`.

## Testing

- Phase-0 unit tests: `OrderHeadModule` device follows `.to()`; group-reward
  assembly (per-sample loss → group mean → per-group advantage) on a fixed fixture;
  per-group EMA update; gradient routing still isolates φ/θ under group credit.
- Phase-1 is a training run, not a unit test — its "test" is the metrics/gate above,
  logged to a results JSON + fixed-order eval curve under `runs/v3_sweep/**`.
