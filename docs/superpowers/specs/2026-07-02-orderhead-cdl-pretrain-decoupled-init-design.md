# OrderHead CDL-Pretrain → Frozen Warmup → PG Unfreeze (Decoupled Init)

**Date:** 2026-07-02
**Branch:** `v3-joint-orderhead-phase1`
**Status:** design — awaiting user review before writing implementation plan

## Purpose

Decouple OrderHead **initialization** from downstream AO-NLL **fine-tuning**.

Today the V3 trainer (`analyses/v3_group_trainer.py`) couples the OrderHead's
update signal to the LM-NLL policy-gradient path (`reward="lm"`), and gβ is
either fully frozen (`frozen_gbeta`) or trained by PG from step 0
(`joint_group`/`joint_batch`). There is no clean way to say: *first give the
OrderHead a stable CDL teacher initialization, then let AO-GPT loss take over,
with CDL and PG never mixed in the same loss.*

This spec introduces a single, clean three-part pipeline that shares **one** gβ
parameter tensor:

```
Stage 1  CDL pretrain (offline producer)   ── CDL appears ONLY here
   ↓ produces gβ ckpt (FrozenBetaHook / OrderHeadModule compatible)
Stage 2  frozen curriculum warmup          ── gβ frozen, argsort, NO PG
   ↓ at unfreeze_orderhead_at_step
Stage 3  AO-NLL PG unfreeze fine-tune       ── LM-NLL + PG only, NO CDL
```

The narrative claim this enables:

> CDL provides a readout **initialization**; after training starts, the
> controller is optimized **only** by AO-GPT downstream loss. The
> CDL-pretrained OrderHead is produced by the *same* readout-training path as
> the deployed gβ, only rewrapped for V3.

**Scope for this spec: `analyses/v3_group_trainer.py` + one new producer module.**
Wiring the internal MLP into the production `train_clean_aogpt` loop is an
explicit *later* step (validate here first). The producer and the embedded
OrderHead are designed to be reusable so they can later drop into production.

## Non-goals

- Do **not** reimplement CDL. Reuse the legacy producer verbatim.
- Do **not** touch the production `train_clean_aogpt` loop in this spec.
- Do **not** combine CDL and PG in a single loss anywhere, ever.
- Do **not** change the existing `frozen_gbeta` / `joint_*` arm semantics when
  `unfreeze_orderhead_at_step is None` (backward compatible default).

---

## Stage 1 — CDL pretrain producer

New module: `analyses/gbeta_cdl_pretrain.py`

```python
def pretrain_gbeta_cdl(
    ckpt_10k,
    *,
    out_ckpt,
    cluster_heads=None,   # default reproduces the deployed gβ head config
    M=1000,
    batch_size=8,
    n_reveal=4,
    epochs=40,
    lr=3e-4,
    seed=0,
    device="cpu",
) -> str:                 # returns str(out_ckpt)
    ...
```

**Requirements (reuse, do not rewrite):**

- Reuse `analyses.uniform_label_free_v1.stage_bc_train_readout` — the *original*
  producer of `reports/uniform_label_free_v1/nodewise_K1000.pt` (the deployed
  `GBETA_CKPT`). Its objective is the legacy path:
  `B → per-sample CDL soft pairwise Y → binary_cross_entropy_with_logits → gβ`.
- This module is a **thin wrapper**: call `stage_bc_train_readout(...)`, then
  ensure the result is written in `FrozenBetaHook` / `OrderHeadModule`-loadable
  format (the same format `nodewise_K1000.pt` already uses — compatibility is
  established by construction, since that ckpt was produced by this path and is
  loaded by `FrozenBetaHook`).
- Save a metadata sidecar recording provenance:

```json
{
  "producer": "stage_bc_train_readout",
  "source_ckpt": "<ckpt_10k>",
  "cluster_heads": [[1, 7]],
  "M": 1000,
  "n_reveal": 4,
  "epochs": 40,
  "lr": 3e-4,
  "seed": 0
}
```

**Claim locked by this design:** *CDL-pretrained OrderHead is produced by the
same readout-training path as the deployed gβ, only rewrapped for V3.*

If `cluster_heads is None`, the producer resolves the deployed head config
(single head `L1H7 = (1, 7)`, matching `p7_gbeta_policy.HEAD`) so a re-run
reproduces the deployed gβ rather than inventing a new cluster.

---

## Stage 2 / 3 — single-run freeze → unfreeze schedule

Extend `train_arm(...)` with three arguments (all default to current behavior):

```python
train_arm(
    ...,
    gbeta_init="load",               # "load" | "cdl_pretrain"
    gbeta_ckpt=None,                 # custom ckpt path (see constraint 2)
    unfreeze_orderhead_at_step=None, # None | int (local step)
    lam_pg=1.0,                      # PG loss weight λ_PG (new; see Data flow)
)
```

### `gbeta_init`

| value | behavior |
|-------|----------|
| `"load"` (default) | load `gbeta_ckpt`; if `gbeta_ckpt is None`, default to `GBETA_CKPT` (`nodewise_K1000.pt`). Custom paths ARE allowed so a previously-pretrained ckpt can be reused without re-running CDL. |
| `"cdl_pretrain"` | call `pretrain_gbeta_cdl(...)` → `out_ckpt`, then load `out_ckpt`. |

There is intentionally **no** separate `"path"` mode — `"load"` with a custom
`gbeta_ckpt` covers it.

### `unfreeze_orderhead_at_step`

`None` → preserve today's per-arm static behavior exactly (backward compatible).

`int` is valid only for an arm with `train_orderhead=True` (e.g. `joint_group`).
When set, Phase A suppresses the OrderHead exactly as `frozen_gbeta` does —
regardless of the arm's nominal `train_orderhead` — and Phase B restores the
arm's joint behavior. This gives a single run with two phases:

**Phase A — `local_step < unfreeze_at` (frozen curriculum warmup):**
- gβ `requires_grad = False`
- gβ params are **NOT** in any optimizer param group
- order policy: deterministic `σ = argsort(g_θ(B))` — **from the same OrderHead
  instance that will later be unfrozen**. `θ` is the Stage-1
  post-pretrain parameter set. Explicitly **not**: fixed L2R, the old default
  `GBETA_CKPT`, a re-loaded `frozen_gβ`, or a cached static order.
  > Phase A uses deterministic argsort from the same initialized OrderHead
  > instance that will later be unfrozen.
- **no PG term** (no PL sampling, no advantage, no entropy bonus); equivalently
  `λ_PG = 0` (see Data flow)
- backbone trains on LM-NLL as usual

**Phase B — `local_step >= unfreeze_at` (AO-NLL PG unfreeze):**
- gβ `requires_grad = True`
- order policy: PL sampling + PG (joint behavior)
- **CDL never appears**

**Weight continuity (hard):**

θ at the start of Phase B == θ at the end of Phase A. Unfreeze is a
`requires_grad` flip + optimizer param-group insertion, **never** a re-load or
re-init of the OrderHead. Formally: `θ_unfreeze_start = θ_after_frozen_warmup`.

**Optimizer param-group insertion (hard):**

`_build_optimizer` currently decides the OrderHead param group once at startup.
Change: when `unfreeze_orderhead_at_step is not None`, the backbone optimizer is
built **without** the OrderHead group; at exactly the unfreeze step the trainer
does, once:

```python
order_head.gbeta.requires_grad_(True)
optimizer.add_param_group({
    "params": list(order_head.gbeta.parameters()),
    "lr": float(lr_orderhead),
    "weight_decay": 0.0,
    "is_orderhead": True,
})
```

Invariants asserted in code/tests:
- **before** unfreeze: zero param groups with `is_orderhead=True`; no OrderHead
  param id appears in any `optimizer.param_groups`.
- **after** unfreeze: exactly one `is_orderhead=True` group; each OrderHead param
  id appears in exactly one param group (no double-insertion → no parameter
  optimized twice). Checked via a param-`id()` set.

---

## Data flow (per micro-step, Phase B)

Unchanged from today's joint path, restated to fix the red lines:

```
idx ─► backbone ─► A ─(detach)─► B = bmatrix_from_A(A) ─► gβ ─► scores
                                                              │
                                        PL sample(scores,τ) ──┤─► order, logp, entropy
                                                              │
order ─► token_order ─► backbone.forward ─► token_losses ─► ell_i
                                                              │
ell_g = group_rewards(ell_i);  advantage = (baseline - ell_g).detach()
pg = -(advantage * logp).mean() - β * entropy.mean()
total = lm_loss + λ_PG * pg   # lm_loss updates backbone; pg updates gβ only
```

**`λ_PG`:** introduce a new `lam_pg` argument to `train_arm`
(default `1.0` → preserves today's `lm_loss + pg`). Phase weighting:
- Phase A: `pg` is **not constructed** (≡ `λ_PG = 0`) — no PL sampling at all.
- Phase B: `λ_PG = lam_pg > 0`.

This is a *new* parameter (today's trainer hard-codes `lm_loss + pg`); it exists
so the PG term can be scaled/swept and so Phase A/B weighting is explicit.

---

## Red lines (all enforced by tests)

1. **No CDL after Stage 1** — the freeze/unfreeze training loop calls no CDL
   teacher, no CDL rollout, no CDL loss. Post-train objective is strictly
   `LM-NLL + PG`.
2. **B detached** before the OrderHead (no PG grad path into the backbone via B).
3. **PG advantage detached**.
4. **PG cannot update the backbone** — `pg_only_backbone_grad` guard stays.
5. **LM loss updates the backbone** (Phase A and Phase B).
6. **No OrderHead optimizer group before unfreeze.**
7. **No repeated param-group insertion** (param-`id()` set invariant).
8. **Weight continuity** — no OrderHead re-init/re-load at unfreeze.

### Observability

The `cdl_calls` process counter is a **guard**, but a process-level counter
alone is fragile (test processes, subprocesses, import reload). So config + logs
carry the **evidence**; the counter is a secondary check.

Run-level fields recorded in the result JSON:

```json
{
  "producer_ckpt": "<stage-1 ckpt path or null>",
  "producer_stage_finished": true,
  "training_loop_cdl_enabled": false
}
```

Each logged step records:

```json
{
  "cdl_loss": null,
  "cdl_teacher": null,
  "cdl_calls": 0,
  "pg_active": true,
  "orderhead_trainable": true
}
```

**No-import rule (hard):** the freeze/unfreeze training loop imports/executes no
CDL symbol — not `stage_bc_train_readout`, not `cdl_rollout_with_standardized_margin`,
not any CDL teacher/rollout/loss. CDL lives strictly inside `pretrain_gbeta_cdl`.

Evidence checked by tests during the `train_arm` loop:
- no `cdl_loss` key ever holds a finite value (always `null`)
- `cdl_calls_before == cdl_calls_after` across the whole loop
- the run's `reward == "lm"` and `training_loop_cdl_enabled == false`

---

## Testing (TDD)

### Producer tests (`gbeta_cdl_pretrain`)

1. **Loadability / shape:** `pretrain_gbeta_cdl(...)` on a tiny M produces a
   ckpt that `OrderHeadModule(out_ckpt)` loads; `scores(A, per_sample=False)`
   returns shape `[rows, 64]` with finite, grad-enabled values.
2. **Learning sanity (two metrics; pairwise_acc is primary — constraint 4):**
   on a small-M smoke, the produced gβ agrees with its CDL teacher above a
   random baseline. The CDL teacher is a pairwise soft teacher, so `pairwise_acc`
   is the **primary** metric (Kendall τ is unstable under small M / ties /
   near-L2R). Report **both**:
   - **Primary:** `pairwise_acc(gβ, CDL) > 0.55`
   - **Secondary:** `tau(gβ order, CDL order) > 0.2`

   Passing **either** is acceptable for a tiny smoke, but both are reported.
   Thresholds are smoke-level (small M is noisy); a hard `τ > 0` is explicitly
   rejected as too weak.
3. **Provenance:** metadata sidecar records `producer="stage_bc_train_readout"`
   and the source ckpt + hyperparameters.

### Schedule tests (`train_arm` with `unfreeze_orderhead_at_step`)

4. **Frozen phase invariants:** across Phase A, `orderhead_param_delta == 0`,
   `pg == 0`, and no `is_orderhead` group exists in the optimizer.
5. **Unfreeze transition:** after the unfreeze step, exactly one `is_orderhead`
   group exists; OrderHead param ids form a set with no duplicates across groups.
6. **Learning after unfreeze:** over Phase B, `orderhead_param_delta > 0`.
7. **Weight continuity:** OrderHead state_dict at the unfreeze boundary is
   bit-identical before/after the requires_grad flip (no re-init).
8. **No-CDL red line:** `cdl_calls` counter does not increase during the
   freeze/unfreeze loop; the trainer imports/executes no CDL symbol in the loop.
9. **Backbone grad routing:** `pg_only_backbone_grad ≈ 0` (PG does not reach
   backbone) — reuse the existing guard.
10. **Backward compatibility (interface, not bit-identical curves — constraint 5):**
    with `unfreeze_orderhead_at_step=None` (and `lam_pg=1.0`), assert *interface*
    compatibility, not an identical loss curve (randomness / optimizer ordering /
    added log keys make bit-identical curves unrealistic):
    - `frozen_gbeta`: no `is_orderhead` param group, no PG, argsort path.
    - `joint_*`: `is_orderhead` param group exists at startup, PG active from
      step 0.
    - all previously-required metrics / log keys still present.

    If a numeric check is wanted, use at most a fixed-seed **1-step** smoke
    comparing `order`, `pg_active`, and `param_groups` — not a full curve.

---

## Files touched

- **new** `analyses/gbeta_cdl_pretrain.py` — thin wrapper over
  `stage_bc_train_readout`, FrozenBetaHook-format output + metadata.
- **edit** `analyses/v3_group_trainer.py`:
  - `train_arm(..., gbeta_init, gbeta_ckpt, unfreeze_orderhead_at_step, lam_pg)`
  - `_build_optimizer` — support deferred OrderHead group when a schedule is set.
  - main loop — per-step frozen/unfrozen branch, one-time unfreeze insertion,
    `λ_PG`-weighted PG, `producer_ckpt`/`training_loop_cdl_enabled` run fields,
    `cdl_loss`/`cdl_teacher`/`cdl_calls`/`pg_active`/`orderhead_trainable`
    step logging, invariants, and the **no-CDL-import** rule in the loop.
- **new** tests under `tests/` for producer + schedule + red lines.

## Implementation plan task structure (for writing-plans)

- **Task 1 — CDL producer wrapper:** new `analyses/gbeta_cdl_pretrain.py`; call
  `stage_bc_train_readout`; save ckpt + metadata sidecar; tests for
  loadability / shape / provenance / pairwise sanity (primary) + τ (secondary).
- **Task 2 — trainer init path:** `gbeta_init` + custom `gbeta_ckpt`;
  `"cdl_pretrain"` runs producer then loads; `"load"` default/custom; test
  custom-ckpt reuse does not re-run pretrain.
- **Task 3 — freeze→unfreeze schedule:** `unfreeze_orderhead_at_step` + `lam_pg`;
  Phase A same OrderHead / frozen argsort / no PG / no optimizer group; Phase B
  requires_grad flip + one-time param-group insertion + PL+PG; tests for
  param-id invariant / weight continuity / delta before-after.
- **Task 4 — red-line tests and logging:** PG→backbone grad ≈ 0; LM→backbone
  grad normal; `cdl_calls` non-increasing + no-import; log fields
  (`pg_active`, `orderhead_trainable`, `cdl_loss=null`, `cdl_calls=0`,
  `training_loop_cdl_enabled=false`); backward-compat interface checks for
  `unfreeze_orderhead_at_step=None`.

## Open items resolved in this design

- Stage-1 mechanism: **reuse** `stage_bc_train_readout`.
- `gbeta_init` values + custom `gbeta_ckpt` reuse.
- Weight continuity guarantee across the unfreeze boundary.
- Deferred single param-group insertion + id-set invariant.
- CDL absence scoped to Stage 3 + observability fields + no-import rule.
- Two-metric producer smoke (pairwise_acc primary, τ secondary), soft thresholds.
- `λ_PG` (`lam_pg`) weighting; Phase A `λ_PG=0`, Phase B `λ_PG>0`.
- Backward compatibility asserted at the *interface* level, not bit-identical curves.
