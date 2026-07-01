# OrderHead Internalization — Feasibility Design (2026-07-01)

Branch context: `p5-direct-nll-routing`. Companion science:
`analyses/P7_CONTEXT_ORDER_FINDINGS.md`, `analyses/P7_POLICY_SUMMARY.md`.

## Purpose

Turn the external `gβ` order controller (currently an out-of-model hook: probe
forward → extract attention `B` → external MLP → order → feed back into training)
into an **internal model module** — an `OrderHead` inside the
`AOGPTWithOrderHead` wrapper / training graph — and verify, **at minimum cost,
before any GPU sweep**, that the resulting architecture is (a) engineering-feasible
and (b) scientifically worth scaling.

**Framing caveat (do not overstate):** this internalizes `gβ` as a model module
in the training graph. It is **not** yet a zero-extra-forward, single-forward
architecture where the current train forward's intermediate attention directly
sets that same batch's reveal order. D2 deliberately uses a **same-batch
probe-then-train two-forward path** (§Deliverable 2). The single-forward lagged
controller is a later engineering variant, out of scope here.

This spec is **feasibility only**. It deliberately does not run the full arm
matrix. It produces a go/no-go gate for a later, separate V3 sweep spec.

## Background (locked science)

- `B` (attention) is a global physical-order carrier; `H` (hidden) is a redundant
  position carrier — the H line is **closed negative**.
- A context-dependent per-sample reveal order **exists** (oracle beats true L2R by
  ~+0.19 nat, non-myopic, not easy-first) but is **search-only**: no per-block
  feature (B/H/content/one-step) predicts σ\* held-out (all lose to L2R).
- P7 established the **loss-trained policy mechanism**: converting `gβ` from
  deterministic `argsort` into a Plackett–Luce (PL) policy and training it with
  AO-NLL policy gradient (EMA baseline, **frozen AO-GPT**) reliably updates the
  MLP (−0.02…−0.08 nat), but does **not** beat L2R — the exploitable per-sample
  signal is absent from the frozen, position-dominated, batch-mean readout.

**The one untested cell** (motivating V3, not this feasibility spec's verdict):
if `OrderHead` is *part of the architecture* and the backbone is **not frozen**,
can joint co-adaptation *manufacture* an exploitable per-sample order signal that
frozen probes could not find? This feasibility work is the runway to that test.

## Scope decisions (locked with user)

| Decision | Choice |
|---|---|
| Staging | V1/V2 = engineering sanity; **V3 joint = the real bet** (deferred to sweep) |
| Co-adaptation gradient flow | **Implicit**: `s = g_θ(stopgrad(B_φ(x)))`. Backbone updates only from LM loss under sampled σ. No readability gradient → no representation collapse. |
| V3 headline metric | **signal-manufacturing probe** (per-sample order beats L2R held-out); step-saving/PPL is payoff — both deferred to sweep |
| Readout granularity | Feasibility smoke: **per-sample main line + batch-level control arm**. Per-sample is the headline-testable config; batch-level is an engineering-sanity control. |
| Per-sample order support | **Verified fact** (not a risk): the backbone *requires* per-sample `[B, T]` orders — `AOGPT_block.forward` asserts `idx.shape == orders.shape`, and any-order training already feeds a distinct randperm per batch row. Per-sample is the native path; batch-level broadcast is the special case. |
| **This spec** | engineering internalization + joint smoke (per-sample main + batch-level control) only |

## Deliverable 1 — Internal OrderHead + external-hook bit-match

**Goal:** prove `gβ` can live inside AO-GPT as a module and reproduce the
external hook's order. Engineering feasibility. CPU, minutes.

### Modules

1. **`OrderHeadModule`** (`nn.Module`) — wraps the deployed `gβ` MLP as a model
   component.
   - Input: attention-derived `B`. Output: node scores `z ∈ ℝ^N`.
   - Readout modes: `batch-mean` (one consensus `B` → one `z`, matches deployed
     `gβ`) and `per-sample` (per-text `B_φ(x)` → per-text `z`).
   - Order APIs: `argsort(z)` (deploy/deterministic) and PL sampling (train).
   - Init: load the deployed `gβ` params
     (`gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt`). The **deterministic**
     `argsort(z)` equals the deployed order (built-in anchor, no distillation).
     Low-temperature PL *concentrates near* this order but is **not required to
     exactly equal** it — bit-match (D1) therefore uses `argsort`, not PL sampling.
2. **`AOGPTWithOrderHead`** (wrapper `nn.Module`) — `backbone` + `order_head`.
   - `extract_B(batch, probe)`: single head **L1H7**, `none_mode='strict65_model'`,
     MODEL frame — reuse `extract_selected_head_A_for_batch`.
   - `compute_order_logits(batch, probe)`: `extract_B` → **detach** →
     `order_head(B)` → `z` (MODEL frame).
   - `sample_order(z, tau)`: PL sample → `(order_model, logp, entropy)`.
   - Frame remap `model → physical` before any `order_nll` (physical frame), via
     the existing `clean_perm` path used in `p7_gbeta_policy`.

### Bit-match gate (batch-mean config)

Compare the internal `AOGPTWithOrderHead` **deterministic `argsort(z)`** order
against the external `FrozenBetaHook.step` order, on the same 20k ckpt / same
probe seeds / same batch. Use `argsort`, not PL sampling (see OrderHead init).

**Frame normalization is explicit.** Both orders are compared in the **same
frame**: prefer raw MODEL frame for the direct `gβ` comparison, and additionally
verify the `model → physical` remap matches before any `order_nll`. Reporting τ
without pinning the frame makes a mismatch indistinguishable from an
implementation bug.

- order-agreement rate and Kendall τ between internal and external order.
- **PASS:** τ ≥ 0.999 (ideally exact match) across ≥3 probe seeds, in a pinned
  frame. Confirms the internalization is behavior-preserving, not
  re-implementation drift.

## Deliverable 2 — Joint smoke (per-sample main + batch-level control)

**Goal:** verify the joint co-adaptation loop is feasible and worth scaling.
Sign check, **not a verdict**. Single seed, short.

**Path shape (stated explicitly).** D2 uses a **same-batch probe-then-train
two-forward path**: one probe forward to extract `B`, then one train forward under
the OrderHead-produced order. This preserves correct credit assignment for σ on
the *same* batch, at the cost of an extra probe forward. It is an internal
OrderHead module but **not** the zero-extra-forward single-forward controller.

### Main arm — joint-per-sample

- Start: **10k parent** ckpt (the `gβ` anchor lineage), train **+5–10k steps**.
- Per-sample orders are `[B, N]` — the backbone's native format (verified; no
  special support needed).
- Loop per step (`forward_train_with_policy`):
  1. `z = order_head(stopgrad(extract_B(batch, probe)))` — per-sample `z ∈ ℝ^{B×N}`.
  2. `order, logp, ent = sample_order(z, tau)` — per-sample PL, `order ∈ [B, N]`.
  3. `lm_loss = backbone(batch, order=order).loss` (backbone trainable).
  4. `A = ema_baseline − lm_loss.detach()`; `L_PG = −A·logp − β·ent`.
  5. `loss = lm_loss + λ_PG · L_PG`; step optimizer.
- Reuse P7/P6 hyperparameters as defaults: PL `tau`, `β` (entropy), `ema_decay`,
  `adv_clip`, `lr` from `p7_gbeta_policy.run_gbeta_policy`; **cosine logits** (P6
  fix) to prevent softmax saturation / entropy collapse.

### Control arm — batch-level (subset-probed broadcast)

Runs alongside the main arm as an engineering-stability sanity, not a headline
metric. `B̄ = mean over a subset of m ∈ {4, 8} probed samples (or full-batch
mean)` → `order_head(B̄)` → **one** `z` → **one** σ broadcast to all rows
(`order ∈ [N]`, then broadcast). This matches the deployed `gβ` batch-mean regime
and the reliable Phase-1 evidence, so it is the low-risk stability reference. Also
serves as the **cost fallback** if per-sample probe forwards prove too expensive.

### Gradient attribution (explicit — must be asserted, not assumed)

- **LM loss updates backbone parameters φ.** `lm_loss` flows into the backbone.
- **PG loss updates OrderHead parameters θ.** No PG gradient reaches the backbone,
  because `B` is `stopgrad`-detached into the OrderHead **and** the advantage `A`
  is detached.
- Verify with grad logging (per smoke): `backbone.grad` from LM > 0;
  `order_head.grad` from PG > 0; `backbone.grad` from the PG term == 0 (the σ is a
  hard discrete order, so no reparameterized path exists either). A nonzero
  backbone-from-PG grad is a wiring bug and fails the smoke.

### Feasibility checks (sign checks — no scientific claim)

Checks 1–3 apply to **both** arms; check 4 is the **per-sample** headline sign.

1. **Runs:** end-to-end, no NaN/crash for the full smoke.
2. **PG bites:** OrderHead param delta > 0 (params actually move).
3. **Entropy healthy:** first-step / mean PL entropy stays in a sane band (does not
   collapse to 0 nor explode) — log per step; reuse cosine-logits regime.
4. **Denoising sign (per-sample arm only).** Two operational metrics, before vs
   after the smoke, on a fixed held-out set (M samples):
   - **Consensus concentration** — pairwise Kendall τ across per-sample orders
     `σ_i = argsort(g_θ(B_φ(x_i)))`:
     `τ_consensus = 2/(M(M−1)) · Σ_{i<j} τ(σ_i, σ_j)`, compared to the P7 frozen
     baseline `≈ 0.31`.
   - **Probe advantage vs L2R** — `Δ_probe = ℓ_φ(x, σ_OrderHead(x)) − ℓ_φ(x, σ_L2R)`,
     reported as the shift `Δ_probe^after − Δ_probe^before`.

   This is a **direction / sign** read, not a significance test. (Consensus τ is
   undefined/degenerate for the batch-level arm — one broadcast order makes it
   trivially 1.0 — so check 4 is per-sample only and is **not** a batch-level gate.)

## Explicitly deferred (fires only if feasibility passes)

Full arm matrix (L2R / random / frozen-gβ / joint-batch-mean / joint-per-sample),
multi-seed, 10k→60k, step-to-threshold / PPL curves, the **metric-driven**
batch-mean vs per-sample verdict (feasibility only runs both as *sign checks*, not
a scored comparison), full confound-guarded probe harness (real-B vs
shuffle/zero/noise-B, τ_to_phys, τ_to_origβ, order diversity, oracle). These
belong to the V3 sweep spec, written only after this gate passes.

## Feasibility gate → decision

- **Greenlight V3 sweep** if: D1 bit-match PASS **and** D2 loop runs healthy on
  both arms (checks 1–3) **and** the per-sample arm shows *any* denoising sign
  (check 4).
- **Diagnose/stop** otherwise, with the failure layer isolated:
  - D1 fails → internalization bug (not science).
  - D2 checks 1–3 fail → infra/stability bug. The batch-level control arm helps
    localize this: if batch-level is stable but per-sample is not, the issue is
    per-sample variance/wiring, not the joint loop itself.
  - Only check 4 is flat → record it and decide whether the per-sample denoising
    bet is worth the GPU sweep (per-sample signal is known-weak; a flat sign is an
    informative negative, not a wiring failure).

## Reused infrastructure

- `analyses/p5_utility_controller`: `load_p5_ckpt`, `order_nll`, `N`.
- `analyses/p7_gbeta_policy`: `gbeta_scores_batchmean` (batch-mean path),
  `sample_pl`, `run_gbeta_policy` (hyperparameters, batch-group protocol, MODEL→
  physical remap), `HEAD=L1H7`, `NONE_MODE='strict65_model'`.
- `batch_readout.integration_hook.FrozenBetaHook` (external order to bit-match).
- `batch_readout.hook_order_provider.extract_selected_head_A_for_batch`,
  `random_probe_token_orders` (B extraction: `extract_selected_head_A_for_batch`
  returns per-sample `A` of shape `(B, N, N)` — per-sample = use as-is;
  batch-level = mean over the subset/batch, as in `gbeta_scores_batchmean`).
- Backbone per-sample order support: `AOGPT_block.forward` (`mode=None` requires
  `idx.shape == orders.shape`) — `[B, T]` orders are native.

## Testing

- D1: unit test asserting internal-vs-external **`argsort`** order τ ≥ 0.999 in a
  pinned frame on a fixed ckpt/seed fixture.
- D2: smoke test asserting, for **both** arms, that the loop runs N steps, params
  move, entropy stays in band, and the gradient-attribution assertions hold
  (`backbone`-from-PG grad == 0). The per-sample arm additionally logs the check-4
  metrics (`τ_consensus`, `Δ_probe` before/after). All metrics land in a results
  JSON under `runs/v3_feasibility/**`.
