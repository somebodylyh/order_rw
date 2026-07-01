# OrderHead CDL-Pretrain → Frozen Warmup → PG Unfreeze (Decoupled Init) — production path

**Date:** 2026-07-02 (rewritten after code verification; supersedes the v3_group_trainer draft)
**Branch:** `v3-joint-orderhead-phase1`
**Status:** design — awaiting user review before writing implementation plan

## Purpose

Decouple the OrderHead's **initialization** from its downstream **fine-tuning**,
on the production training path (`block_lo_arm_order_network/train_clean_aogpt.py`).

One gβ parameter set flows through three stages, and CDL and downstream PG never
meet in a single loss:

```
Stage 1  CDL pretrain (offline producer)     ── CDL appears ONLY here
   ↓ produces g_beta_best.pt (L0DynamicGBeta)
Stage 2  frozen curriculum warmup            ── gβ frozen, argsort, NO PG
   ↓ at --unfreeze-orderhead-at-step
Stage 3  AO-NLL PG unfreeze fine-tune         ── LM-NLL + PG only, NO CDL
```

Core principle (locked):

> **CDL initializes gβ ⇒ frozen argsort curriculum ⇒ LM-NLL PG fine-tuning.**
> CDL provides a readout initialization; once training starts, the controller is
> optimized **only** by AO-GPT downstream loss.

## Verified ground truth (why this rewrite exists)

The earlier draft targeted `analyses/v3_group_trainer.py` and assumed the
embedded gβ was a `NodewiseReadout` produced by `stage_bc_train_readout`. Code
verification refuted that. The corrected facts (all read from code):

- The production gβ is **`L0DynamicGBeta`** (`batch_readout/l0_dynamic_gbeta.py`),
  input `B_raw (batch, H=8, 65, 65)`, output `(scores (B,64), aux)`.
- It is deployed by `train_clean_aogpt.py --run-kind frozen_beta
  --batch-mean-probes 4`, which selects
  `FrozenGBetaModelFrameBlockProvider` (`batch_readout/frozen_gbeta_hook.py`):
  8-head strict65 B, batch-mean over N probe forwards, `none_mode='model'`,
  `scores, _ = model(B_t, apply_head_dropout=False)` then
  `sigma_model = scores.argsort(descending=True)` — all under `@torch.no_grad()`.
  ("Single head" refers to the upstream head-selection that builds the CDL
  teacher, **not** the gβ input, which is 8-head.)
- It is CDL-pretrained by a **two-step** offline pipeline:
  1. `analyses/build_l0_dynamic_gbeta_dataset.py::build_l0_dynamic_gbeta_dataset(
     ckpt_path, ...)` → `.npz` with `B_raw` + CDL dynamic teacher
     (`teacher_pairwise`, `teacher_consensus_order`, `teacher_ranks`, weights)
     via `build_dynamic_teacher` (the shared CDL machinery).
  2. `batch_readout/train_l0_dynamic_gbeta.py::train(dataset_path, ...,
     loss_type="pairwise_bce", heads=8)` → trains `L0DynamicGBeta`, selects by
     val pairwise_acc, saves `{"model_state_dict", "config"}` → `g_beta_best.pt`.
- `--loss-type` default is `pairwise_bce` (internally `soft_pairwise_bce_loss`);
  `listmle` / `rank_kl` also exist.
- The canonical source backbone is the 10k ckpt
  `block_lo_arm_order_network/probe_results/overnight_20260625_random_baseline/ckpt_step10000.pt`.

## Non-goals

- Do **not** reimplement CDL. Reuse `build_l0_dynamic_gbeta_dataset` +
  `train_l0_dynamic_gbeta` verbatim.
- Do **not** combine CDL and PG in a single loss, ever.
- Do **not** change existing `frozen_beta` behavior when the new
  `--unfreeze-orderhead-at-step` flag is absent (backward compatible default).
- `listmle` / `rank_kl` remain optional ablations, not the canonical pipeline.

---

## Stage 1 — CDL pretrain producer (offline, two steps)

Canonical: **do not** pick from the pool of old `g_beta_best.pt`. Produce gβ from
the *same* 10k parent as the Stage-2/3 backbone, so the causal chain is clean.

New thin orchestrator: `analyses/gbeta_cdl_pretrain.py`

```python
def pretrain_gbeta_cdl(
    ckpt_10k,               # default: overnight_20260625_random_baseline/ckpt_step10000.pt
    *,
    out_dir,                # writes <out_dir>/g_beta_best.pt (+ dataset .npz + metadata)
    loss_type="pairwise_bce",
    M=2000,
    heads=8,
    epochs=40,
    seed=0,
    device="cpu",
    dataset_path=None,      # if given, skip the build step and reuse this .npz
) -> str:                   # returns str(<out_dir>/g_beta_best.pt)
    ...
```

**Requirements (reuse, do not rewrite):**
1. Call `build_l0_dynamic_gbeta_dataset(ckpt_path=ckpt_10k, M=M, ...)` → `.npz`
   (skipped if `dataset_path` is supplied).
2. Call `train_l0_dynamic_gbeta.train(dataset_path=<npz>, out_dir=out_dir,
   loss_type=loss_type, heads=heads, epochs=epochs, seed=seed, device=device)`.
3. The produced `g_beta_best.pt` is already `FrozenGBetaModelFrameBlockProvider`-
   loadable (`{"model_state_dict", "config"}`) — no reformatting.
4. Write a metadata sidecar `<out_dir>/gbeta_provenance.json`:

```json
{
  "producer": "build_l0_dynamic_gbeta_dataset + train_l0_dynamic_gbeta",
  "source_ckpt": "<ckpt_10k>",
  "loss_type": "pairwise_bce",
  "M": 2000, "heads": 8, "epochs": 40, "seed": 0,
  "dataset_path": "<npz>"
}
```

**Claim locked:** *The CDL-pretrained OrderHead is produced by the exact
existing L0DynamicGBeta CDL pipeline; only orchestrated + provenance-tagged.*

---

## Stage 2 — frozen curriculum warmup (existing, unchanged)

Already supported: `train_clean_aogpt.py --run-kind frozen_beta
--frozen-beta-ckpt <g_beta_best.pt> --batch-mean-probes 4` with gβ frozen.
Order each step = `argsort(-gβ(B))` via `FrozenGBetaModelFrameBlockProvider`
(no_grad). Backbone trains on LM-NLL under that order.

New default wiring for the canonical pipeline (see CLI below): if
`--frozen-beta-ckpt` is omitted **and** Stage-1 is requested, the trainer runs
`pretrain_gbeta_cdl(...)` first and loads its output. `--frozen-beta-ckpt <path>`
remains a full override for reuse/debug/reproduction.

---

## Stage 3 — AO-NLL PG unfreeze fine-tune (new)

New flag `--unfreeze-orderhead-at-step INT` (default unset → pure Stage-2, fully
backward compatible). On a `frozen_beta` run, once `global_step >= unfreeze_at`:

**Phase A — `global_step < unfreeze_at` (frozen warmup):**
- gβ `requires_grad=False`, not in the optimizer.
- order = deterministic `argsort(-gβ(B))` — from the **same L0DynamicGBeta
  instance** that will be unfrozen (not a reload, not a different head mode).
- no PG term (`λ_PG = 0`).

**Phase B — `global_step >= unfreeze_at` (PG unfreeze):**
- gβ `requires_grad=True`; one-time optimizer param-group insertion.
- order policy switches argsort → **Plackett-Luce sampling**:
  `σ ~ PL(scores, τ)` sequentially — at each step sample the next block from
  `softmax(scores/τ)` over the remaining blocks, accumulating `logp` and
  `entropy`. τ→0 recovers argsort, so the switch is continuous.
- **grad-enabled gβ forward**: recompute `scores = gβ(B)` with grad, where `B` is
  the same batch-mean 8-head strict65 tensor but **detached** (no PG grad into
  the backbone). This is a new grad-enabled path parallel to the provider's
  `@torch.no_grad()` `model_frame_token_order`.
- reward = downstream **LM-NLL**; PG uses the v3 group-credit machinery
  (`analyses/v3_group_credit.py`: `group_ids_for`, `group_rewards`,
  `per_sample_loss`, `GroupEMA`) — reused, not rebuilt:
  `advantage = (baseline − ell_g).detach().clamp(±adv_clip)`,
  `L_PG = −(advantage · logp).mean() − β·entropy.mean()`.
- **CDL never appears.**

**Weight continuity (hard):** θ at the start of Phase B == θ at the end of
Phase A. Unfreeze is a `requires_grad` flip + param-group insertion, never a
reload/re-init.

**Optimizer param-group insertion (hard):** the backbone optimizer
(`model.configure_optimizers(...)`) is built without gβ; at exactly `unfreeze_at`,
once:

```python
gbeta_module.requires_grad_(True)
optimizer.add_param_group({
    "params": list(gbeta_module.parameters()),
    "lr": args.orderhead_lr, "weight_decay": 0.0, "is_orderhead": True,
})
```

Invariants (tested): before unfreeze, zero `is_orderhead` groups and no gβ param
`id()` in any group; after, exactly one `is_orderhead` group and each gβ param
`id()` in exactly one group (no double-insertion).

---

## Data flow (per step, Phase B)

```
idx ─► backbone.forward_fn(return_attentions) ─► attn ─► 8-head strict65 B
        (batch-mean over probes; none_mode='model')  ─(detach)─► B_det
B_det ─► gβ(B_det) [GRAD] ─► scores ─► PL sample(scores,τ) ─► σ, logp, entropy
σ ─► token_order ─► backbone.forward_fn(idx, token_order) ─► per-token CE ─► ell_i
ell_g = group_rewards(ell_i);  advantage = (baseline − ell_g).detach()
L_PG = −(advantage·logp).mean() − β·entropy.mean()
L_total = L_LM + λ_PG · L_PG      # L_LM updates backbone; L_PG updates gβ only
```

`λ_PG` (`--lam-pg`, default `1.0`): Phase A `L_PG` not constructed (≡ 0);
Phase B `λ_PG > 0`. Per-token CE reuses the `compute_token_ce` contract
(`logits, _ = forward_fn(idx, token_order); targets = idx.gather(1, token_order)`).

---

## CLI additions to `train_clean_aogpt.py`

- `--unfreeze-orderhead-at-step INT` (default: unset → Stage-2 only).
- `--orderhead-lr FLOAT` (default `3e-4`) — gβ param-group LR after unfreeze.
- `--lam-pg FLOAT` (default `1.0`) — PG weight λ_PG.
- `--pg-tau FLOAT` (default `1.0`) — PL temperature.
- `--pg-group-m INT` (default `16`) — group size for group-credit advantage.
- `--pg-beta FLOAT` (default `3e-3`) — entropy bonus.
- `--pg-adv-clip FLOAT` (default `0.3`).
- `--cdl-pretrain` (flag): if set and `--frozen-beta-ckpt` absent, run
  `pretrain_gbeta_cdl(...)` from `--cdl-source-ckpt` (default 10k ckpt) and use
  its output. `--frozen-beta-ckpt` always overrides (reuse/debug).

---

## Red lines (enforced by tests)

1. **No CDL after Stage 1** — the `train_clean_aogpt` loop imports/executes no
   CDL symbol (`build_dynamic_teacher`, `cdl_rollout_*`, `soft_pairwise_bce_loss`,
   `build_l0_dynamic_gbeta_dataset`, `train_l0_dynamic_gbeta`). CDL lives only in
   `pretrain_gbeta_cdl`.
2. **B detached** before the grad-enabled gβ forward (no PG grad into backbone).
3. **PG advantage detached.**
4. **PG cannot update the backbone** — assert `∂L_PG/∂backbone ≈ 0`.
5. **LM loss updates the backbone** (Phase A and B).
6. **No gβ optimizer group before unfreeze; exactly one after; no duplicate**
   (param-`id()` set invariant).
7. **Weight continuity** — no gβ reload/re-init at unfreeze.

### Observability

Run-level: `producer_ckpt`, `producer_stage_finished`,
`training_loop_cdl_enabled=false`. Per-step (Phase B): `pg_active`,
`orderhead_trainable`, `cdl_loss=null`, `cdl_calls=0`, `pg`, `entropy`,
`orderhead_grad_norm`, `backbone_grad_norm`. Evidence beats the counter:
`cdl_calls_before == cdl_calls_after` across the loop, plus the no-import rule.

---

## Testing (TDD)

**Producer (`pretrain_gbeta_cdl`)**
1. Loadability/shape: output `g_beta_best.pt` loads via
   `FrozenGBetaModelFrameBlockProvider`; `model(B_t)[0]` shape `(Bsz, 64)`, finite.
2. Learning sanity (primary `pairwise_acc`, secondary τ; soft smoke thresholds):
   on tiny M, **primary** `pairwise_acc(gβ, CDL) > 0.55`; **secondary**
   `τ(gβ, CDL) > 0.2`; passing either is acceptable, both reported. Hard `τ>0`
   rejected.
3. Provenance: `gbeta_provenance.json` records producer + source ckpt + loss_type.

**Schedule (`--unfreeze-orderhead-at-step`)**
4. Frozen phase: across Phase A, gβ param delta == 0, `pg == 0`, no `is_orderhead`
   group.
5. Unfreeze transition: after `unfreeze_at`, exactly one `is_orderhead` group; gβ
   param `id()`s unique across groups.
6. Learning after unfreeze: over Phase B, gβ param delta > 0.
7. Weight continuity: gβ state_dict identical across the unfreeze boundary.
8. No-CDL red line: `cdl_calls` non-increasing during the loop; no CDL import in
   the loop.
9. Grad routing: `∂L_PG/∂backbone ≈ 0`; `∂L_LM/∂backbone` normal.
10. Backward compatibility (interface, not bit-identical curves): with the flag
    absent, `frozen_beta` behaves exactly as today (no gβ group, argsort, gβ
    frozen). At most a fixed-seed 1-step smoke comparing `order`, `pg_active`,
    `param_groups`.

---

## Files touched

- **new** `analyses/gbeta_cdl_pretrain.py` — orchestrates
  `build_l0_dynamic_gbeta_dataset` + `train_l0_dynamic_gbeta`; provenance sidecar.
- **edit** `block_lo_arm_order_network/train_clean_aogpt.py`:
  - new CLI flags (above);
  - optional Stage-1 auto-produce when `--cdl-pretrain` and no `--frozen-beta-ckpt`;
  - a grad-enabled gβ scorer usable in the training step (parallel to the
    provider's no_grad `model_frame_token_order`);
  - Phase-A/Phase-B branch in the step: argsort vs PL+PG; one-time unfreeze +
    param-group insertion; `λ_PG`-weighted PG via `v3_group_credit`;
    run/step logging; the no-CDL-import rule.
- **new** tests under `block_lo_arm_order_network/tests/` (or `tests/`) for
  producer + schedule + red lines.

## Implementation plan task structure (for writing-plans)

- **Task 1 — CDL producer orchestrator:** `analyses/gbeta_cdl_pretrain.py` wrapping
  the two existing steps + provenance; tests 1–3.
- **Task 2 — grad-enabled gβ scorer + CLI:** add the grad path and the new flags;
  `--cdl-pretrain`/override resolution; smoke that a `frozen_beta` run still works
  unchanged (test 10).
- **Task 3 — freeze→unfreeze schedule:** Phase-A/B branch, one-time param-group
  insertion, argsort→PL switch, weight continuity; tests 4–7.
- **Task 4 — PG wiring + red lines:** `v3_group_credit` advantage + `λ_PG`;
  grad-routing guards, no-CDL invariants, observability fields; tests 8–9 + the
  red-line asserts.

## Open items resolved

- Target = production `train_clean_aogpt.py` (not `v3_group_trainer.py`).
- gβ = `L0DynamicGBeta`, 8-head batch-mean deploy via
  `FrozenGBetaModelFrameBlockProvider`.
- Stage-1 = `build_l0_dynamic_gbeta_dataset` + `train_l0_dynamic_gbeta`, loss
  `pairwise_bce`, from the 10k parent; default fresh-produce, `--frozen-beta-ckpt`
  override.
- Stage-3 reuses `v3_group_credit` + PL sampling (not rebuilt); `λ_PG` weighting;
  CDL only at init; weight continuity + single param-group insertion invariants.
