# OrderHead Internalization — Feasibility Design (2026-07-01)

Branch context: `p5-direct-nll-routing`. Companion science:
`analyses/P7_CONTEXT_ORDER_FINDINGS.md`, `analyses/P7_POLICY_SUMMARY.md`.

## Purpose

Turn the external `gβ` order controller (currently an out-of-model hook: probe
forward → extract attention `B` → external MLP → order → feed back into training)
into an **internal `OrderHead` module** that is part of AO-GPT's forward, and
verify — **at minimum cost, before any GPU sweep** — that the resulting
architecture is (a) engineering-feasible and (b) scientifically worth scaling.

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
| Readout granularity | V3 sweep runs **both** batch-mean and per-sample arms; **feasibility smoke uses per-sample** (the headline config) |
| **This spec** | engineering internalization + **single-arm** joint smoke only |

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
     (`gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt`). At low PL temperature,
     `PL(z) == argsort(z) == deployed order` — built-in anchor, no distillation.
2. **`AOGPTWithOrderHead`** (wrapper `nn.Module`) — `backbone` + `order_head`.
   - `extract_B(batch, probe)`: single head **L1H7**, `none_mode='strict65_model'`,
     MODEL frame — reuse `extract_selected_head_A_for_batch`.
   - `compute_order_logits(batch, probe)`: `extract_B` → **detach** →
     `order_head(B)` → `z` (MODEL frame).
   - `sample_order(z, tau)`: PL sample → `(order_model, logp, entropy)`.
   - Frame remap `model → physical` before any `order_nll` (physical frame), via
     the existing `clean_perm` path used in `p7_gbeta_policy`.

### Bit-match gate (batch-mean config)

Compare internal `AOGPTWithOrderHead` deterministic order against the external
`FrozenBetaHook.step` order, on the same 20k ckpt / same probe seeds / same batch:

- order-agreement rate and Kendall τ between internal and external order.
- **PASS:** τ ≥ 0.999 (ideally exact match) across ≥3 probe seeds. Confirms the
  internalization is behavior-preserving, not a re-implementation drift.

## Deliverable 2 — Single-arm joint smoke (per-sample, implicit)

**Goal:** verify the joint co-adaptation loop is feasible and worth scaling.
Sign check, **not a verdict**. Single seed, short.

### Config

- Arm: **joint-per-sample** (the V3 headline config).
- Start: **10k parent** ckpt (the `gβ` anchor lineage), train **+5–10k steps**.
- Loop per step (`forward_train_with_policy`):
  1. `z = order_head(stopgrad(extract_B(batch, probe)))` — per-sample.
  2. `order, logp, ent = sample_order(z, tau)`.
  3. `lm_loss = backbone(batch, order=order).loss` (backbone trainable).
  4. `A = ema_baseline − lm_loss.detach()`; `L_PG = −A·logp − β·ent`.
  5. `loss = lm_loss + λ_PG · L_PG`; update backbone (LM) + OrderHead (PG).
- Reuse P7/P6 hyperparameters as defaults: PL `tau`, `β` (entropy), `ema_decay`,
  `adv_clip`, `lr` from `p7_gbeta_policy.run_gbeta_policy`; **cosine logits** (P6
  fix) to prevent softmax saturation / entropy collapse.

### Feasibility checks (sign checks — no scientific claim)

1. **Runs:** end-to-end, no NaN/crash for the full smoke.
2. **PG bites:** OrderHead param delta > 0 (params actually move).
3. **Entropy healthy:** first-step / mean PL entropy stays in a sane band (does not
   collapse to 0 nor explode) — log per step; reuse cosine-logits regime.
4. **Denoising sign:** per-sample readout consensus τ **before vs after** the smoke
   (P7 frozen baseline = 0.31), and Δ_probe **direction** on a small held-out set
   `Δ_probe = ℓ(σ_OrderHead) − ℓ(σ_L2R)`. Looking for *any* movement toward
   readable/exploitable per-sample structure — direction, not significance.

## Explicitly deferred (fires only if feasibility passes)

Full arm matrix (L2R / random / frozen-gβ / joint-batch-mean / joint-per-sample),
multi-seed, 10k→60k, step-to-threshold / PPL curves, batch-mean vs per-sample
comparison, full confound-guarded probe harness (real-B vs shuffle/zero/noise-B,
τ_to_phys, τ_to_origβ, order diversity, oracle). These belong to the V3 sweep
spec, written only after this gate passes.

## Feasibility gate → decision

- **Greenlight V3 sweep** if: D1 bit-match PASS **and** D2 loop runs healthy
  (checks 1–3) **and** per-sample `B` shows *any* denoising sign (check 4).
- **Diagnose/stop** otherwise: if D1 fails it is an internalization bug (not
  science); if D2 checks 1–3 fail it is an infra/stability bug; if only check 4 is
  flat, record it and decide whether the per-sample denoising bet is worth the
  GPU sweep.

## Reused infrastructure

- `analyses/p5_utility_controller`: `load_p5_ckpt`, `order_nll`, `N`.
- `analyses/p7_gbeta_policy`: `gbeta_scores_batchmean` (batch-mean path),
  `sample_pl`, `run_gbeta_policy` (hyperparameters, batch-group protocol, MODEL→
  physical remap), `HEAD=L1H7`, `NONE_MODE='strict65_model'`.
- `batch_readout.integration_hook.FrozenBetaHook` (external order to bit-match).
- `batch_readout.hook_order_provider.extract_selected_head_A_for_batch`,
  `random_probe_token_orders` (B extraction; per-sample = skip the batch mean).

## Testing

- D1: unit test asserting internal-vs-external order τ ≥ 0.999 on a fixed
  ckpt/seed fixture.
- D2: smoke test asserting the loop runs N steps, params move, entropy stays in
  band, and the four feasibility metrics are logged to a results JSON under
  `runs/v3_feasibility/**`.
