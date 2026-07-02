# Port V3 frozen-gβ MLP order policy into chenhe train.py — design

**Date:** 2026-07-02
**Status:** design (reviewed; revised with reviewer points 1–12)
**Scope:** V3 **Stage 1 (CDL pretrain producer)** + **Stage 2 (frozen gβ order
deploy)** ported into the chenhe mainline trainer. **Stage 3 (PG unfreeze) is
OUT of scope** — deferred entirely until the frozen MLP is solid.

Related: retargets the entrypoint of
`2026-07-02-orderhead-cdl-pretrain-decoupled-init-design.md` (that spec targets
admin `train_clean_aogpt.py`; this one targets chenhe `train.py`). Baseline
decisions: `chenhe_rerun/experiments/chenhe_baseline_lock.md`.

---

## Purpose

Make our method — a frozen, CDL-pretrained MLP that reads the backbone's L0
attention and emits a block reveal order — run as a first-class order policy
**inside chenhe's mainline trainer** (`chenhe_rerun/train.py`,
`seq256/permute/block64`), same-table comparable with chenhe AR / Random.

Everything except the order policy is inherited verbatim from the chenhe
baseline. The MLP controller is **our methodology**; it is **not** aligned to
chenhe `attn_mlp_order_policy.py`.

## Verified ground truth (attention-extraction spike, 2026-07-02)

- chenhe AOGPT shares admin's lineage. `AOGPT.forward_fn(idx, orders,
  return_attentions=True)` → `(logits, loss, attentions)`; `attentions[0]` = L0
  per-head attention `(B, 8, 257, 257)`, None token at index 0, shuffled by
  `orders`.
- Shape chain runs on real data: `forward_fn → attn_l0 (B,8,257,257) →
  build_model_frame_strict65 → B (B,8,65,65) → L0DynamicGBeta → scores (B,64) →
  argsort → σ`. All asserts pass.
- Spike proved **shape/well-formedness only**, NOT frame-index semantics.
- gβ = `L0DynamicGBeta` (8-head, 65-node strict65). Block count matches (64).
- Baseline `dropout = 0` → probe forwards are deterministic and model-mode
  independent; no dropout-induced order jitter.

## Non-goals

- No Stage 3 / PG / unfreeze / param-group insertion / group credit.
- Do not reimplement CDL or the gβ model — port math-equivalent, tested.
- Do not reuse chenhe `attn_mlp_order_policy.py` as the method.
- Do not change chenhe baseline behavior when the new mode/flags are absent.
- No physical-frame `inv_perm` remap anywhere in training OR gβ supervision.

---

## Locked design facts

- **Baseline / frame:** `seq256/permute/block64`, `permute_seed=42`,
  `block_size=256`, `block_order_block_len=4` → 64 blocks. (baseline lock)
- **Metric semantics** (chenhe `estimate_loss()` @ `train.py:7708`, verified):
  - `val` = validation NLL under the **active training order policy**. For
    `GBetaFrozenOrder` this is the loss under the batch-mean gβ-selected
    model-frame order. It is a **policy/objective diagnostic** — useful to check
    the deployed policy is well-formed and optimizable — but **NOT the headline
    cross-method metric**, because each method may choose a different reveal
    order (a shorter/easier conditioning path can lower `val` without improving
    text-L2R generation NLL). *(reviewer point 2)*
  - `val_origin_l2r_loss` = NLL revealing blocks in `inverse_block_perm` order =
    **original text-L2R NLL**. Computed purely in chenhe's own frame → comparable
    across all runs. **Headline / main-table metric.**
  - `val_l2r_loss` = current-frame index-order (permuted) AR NLL — NOT text L2R.
  - `val_main_eval_loss` = segment-guided only; NOT our selection metric.
  - `val_origin_l2r_loss` is eval/report only; never drives training selection.

## 🔴 Hard constraints

1. **gβ frozen throughout** — `requires_grad=False`, never in the optimizer,
   forward under `@torch.no_grad()`. (No PG in scope.)
2. **Batch-mean B only (training AND eval).** gβ order is computed from a
   **batch-mean** strict65 `B` = mean over (batch samples × `batch_mean_probes`
   probe forwards). **One σ per batch/refresh**, broadcast to all samples.
   Per-sample gβ signal ≈ noise → inaccurate order. Eval must NOT degrade to
   single-sample. *(reviewer point 4)*
3. **Frame stays in chenhe's current model-frame.** The ported path produces
   model-frame block orders over indices `[0, 63]`; the training path and the gβ
   supervision target introduce **no** `inverse_block_perm` / physical remap.
   chenhe's `inverse_block_perm` is used only posthoc inside `val_origin_l2r_loss`.
4. **gβ supervision frame (Stage 1).** The CDL teacher used to pretrain gβ MUST
   emit **model-frame block orders over chenhe current-frame indices [0, 63]**.
   No `inverse_block_perm` / physical-frame conversion when constructing the gβ
   target. Physical-L2R may be computed separately as a diagnostic only, never as
   the gβ training target. *(reviewer point 3)*
5. **B-extraction mode parity.** Stage-1 producer and Stage-2 provider MUST use
   the same backbone mode for the attention probes, and probes are always
   `@torch.no_grad()` and never contribute to the optimization graph. (With
   `dropout=0` this is jitter-free; the parity constraint guards future configs.)
   *(reviewer point 4)*
6. **B-source parity.** Ported B == admin `FrozenGBetaModelFrameBlockProvider`
   convention: 8-head strict65, batch-mean probes, `none_mode='model'`, None at
   node 0, diagonal zero.
7. **Layout guard.** The V3 port supports **only** `seq256/permute/block64`
   (N=64, block_len=4, heads=8, strict65-65). It must **hard-fail** on any other
   layout until explicitly extended. *(reviewer point 10)*

---

## Architecture

Three parts, following chenhe's order-policy module pattern.

### Part A — ported math package `chenhe_rerun/orderhead_v3/`

**Math-equivalent ports** (not file-level copies) of the validated admin modules,
with **chenhe-local imports** and explicit tests. Constants are config-checked,
not silently global. *(reviewer points 10, 11)*

```
chenhe_rerun/orderhead_v3/
├── __init__.py
├── constants.py            # SEQ_LEN=256, N=64, BLOCK_LEN=4, HEADS=8, STRICT_NODES=65
├── l0_strict65.py          # build_model_frame_strict65
├── per_head_order_scan.py  # _attn_to_A_block_loss_aligned_with_none_model_vec
├── none_separated_block_graph.py
├── l0_dynamic_gbeta.py     # L0DynamicGBeta, normalize_strict65
├── cdl_teacher.py          # build_dynamic_teacher, teacher_scores
├── soft_pairwise.py        # gβ training losses
└── gbeta_provider.py       # Part C (below)
```

Every entry point asserts the layout guard (constraint 7).

### Part B — Stage-1 producer `chenhe_rerun/gbeta_cdl_pretrain.py`

Offline orchestrator (never runs inside the training loop):

```
pretrain_gbeta_cdl(parent_ckpt, out_dir, *, M, batch_mean_probes, heads=8,
                   epochs, seed, device) -> str  # path to g_beta_best.pt
```

1. **chenhe-native model+data load** (the one adaptation): load the chenhe parent
   ckpt + chenhe AOGPT + chenhe data windows. Replaces admin's
   `neural_readout.extract_b._load_model_and_chunks` (admin `clean_protocol` fmt).
2. Extract strict65 B over probes via chenhe `forward_fn` (Part A math), under the
   **mode fixed by constraint 5**.
3. `build_dynamic_teacher(...)` → CDL teacher emitting **model-frame** block
   orders over `[0,63]` (constraint 4).
4. Train `L0DynamicGBeta` (`loss_type='pairwise_bce'`), select by val
   `pairwise_acc` → save `{model_state_dict, config}` = `g_beta_best.pt`.
5. Write `gbeta_provenance.json`: producer, parent ckpt path + **content hash**,
   `seq_len, num_blocks, block_len, heads, permute_seed, none_mode='model',
   strict65=true, probe_mode, batch_mean_probes, M, loss_type, seed`.
   *(reviewer point 8)*

### Part C — Stage-2 frozen deploy in `chenhe_rerun/train.py`

**Provider** `orderhead_v3/gbeta_provider.py`: given `(model, idx_batch,
global_step)` → batch-mean strict65 B (chenhe `forward_fn`, `batch_mean_probes`,
`none_mode='model'`, mode per constraint 5) → frozen `gβ(B)` → `argsort(-scores)`
→ **one** model-frame block order → broadcast to `block_orders[B, 64]`. All
`@torch.no_grad()`.

**Provider load-time hard asserts** (constraint 6/7/8, *reviewer point 8*):
gβ config `seq_len==256, num_blocks==64, block_len==4, heads==8,
none_mode=='model', strict65==true, permute_seed==42`; and
`gbeta_provenance.parent_hash == hash(init_from_ckpt)` (mismatch → hard error;
path-only difference with equal hash is OK).

**Refresh / cache semantics** *(reviewer point 5)*:
- First call always computes a fresh σ.
- Training: recompute σ every `gbeta_refresh_every` optimizer steps using the
  current batch; reuse the cached σ in between.
- Eval: recompute σ for **every eval batch** by default (avoids evaluating val
  batches with a σ derived from a train batch). Optional flag can pin an eval σ.
- Train and eval caches are **separate**.

**train.py edits:**
- dispatch (~L7551): `if aogpt_train_mode == 'GBetaFrozenOrder': return
  provider(...)` (the active-policy path routes `val` through the provider).
- init (~L714): `--init-from-ckpt <parent>` + `--init-from-ckpt-mode
  {weights_only, full_state}` — see comparison protocol. *(reviewer point 9)*
- flags: `--gbeta-ckpt`, `--gbeta-batch-mean-probes` (4), `--gbeta-refresh-every`
  (1), `--gbeta-eval-refresh` (default per-eval-batch).
- config: `config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py`
  (method-only diff off `random.py`).

### Data flow (per training/eval step)

```
idx ─► forward_fn(idx, probe_k, return_attentions=True)   × batch_mean_probes  (no_grad)
     ─► attn_l0 (B,8,257,257) ─► build_model_frame_strict65 ─► B_k (B,8,65,65)
B = mean_k mean_batch B_k                                  # batch-mean
scores = gβ(B) [frozen] ─► σ = argsort(-scores)           # ONE model-frame order
block_orders = σ.broadcast(B,64) ─► chenhe reveals blocks ─► LM-NLL (backbone)
```

---

## Comparison protocol *(reviewer point 7)*

The **primary controlled comparison** for `GBetaFrozenOrder` is **continuation
from the same random parent checkpoint P** (default P=10k). Therefore:

- All continuation methods resume from the **same parent** and run the **same
  number of additional optimizer steps** (P→50k): Random-continue, AR-continue,
  GBetaFrozen-continue (and CDL-continue if available).
- From-scratch AR / Random curves (0→50k, e.g. tonight's runs) are reported
  **separately as reference curves**, NOT as the direct controlled comparison.
- **`--init-from-ckpt-mode` parity:** whatever mode the GBetaFrozen continuation
  uses (weights-only vs full optimizer/scheduler state), the Random/AR
  continuation baselines MUST use the **same** mode. Default: `full_state`
  (chenhe native resume semantics) so continuation curves are directly
  comparable; the plan may add a `weights_only` variant as an ablation.
  *(reviewer point 9)*

## Compute overhead reporting *(reviewer point 6)*

Each step does `batch_mean_probes` no_grad attention forwards + the real loss
forward (~5× forward at probes=4). We therefore report:

- **Primary method-comparison metric = step-based NLL** under chenhe's protocol.
- **Separately logged:** tokens/sec, seconds/step, probe forwards per optimizer
  step, effective `refresh_every`. This preempts "is the step saving eaten by 4
  probe forwards?" — wall-clock is a distinct axis from step-based NLL.

## Stage-1 prerequisite: the parent ckpt

The frozen-gβ run and the gβ pretrain both need a **chenhe random baseline
checkpoint at step P (default 10k)**. Tonight's B1 random run overwrites
`ckpt.pt` to 50k and keeps no 10k snapshot. Stage 1 requires either a dedicated
random-to-10k run saved as the parent (~17 min) or periodic snapshots. Explicit,
cheap prerequisite (not a method code change).

---

## Testing (TDD)

**Part A / B-extraction**
1. Shape/well-formedness (spike-green): strict65 B `(B,8,65,65)`, None col 0,
   diag 0, finite.
2. **Frame-index semantics (must-verify):** feed a controlled attention (known
   peak) through the ported path; assert block index `i` == chenhe current-frame
   block `i` (argsort reveals the intended chenhe blocks). Proves no inv_perm
   mismatch. *(reviewer point 5 for spec §, this is the port-correctness test)*
3. B-source parity vs the admin provider convention on the same batch/probes.
4. Layout guard: non-block64 config → hard-fail. *(reviewer point 10)*

**Part B producer**
5. `g_beta_best.pt` loads; `gβ(B)[0]` shape `(Bsz, 64)`, finite.
6. Learning sanity: `pairwise_acc(gβ, CDL teacher) > 0.55` (soft, tiny M),
   `τ(gβ order, teacher_consensus) > 0.2` secondary; both reported.
7. **Teacher-frame test:** the CDL supervision target is model-frame `[0,63]`
   (no inverse_block_perm applied). *(reviewer point 3)*
8. Provenance sidecar records producer + parent hash + all config fields.

**Part C integration**
9. gβ **frozen**: gβ param delta == 0 across training; gβ in no optimizer group.
10. **Batch-mean**: one σ per batch from batch-mean B; assert NO per-sample σ
    path (training and eval).
11. **Cache semantics**: first-call fresh; train reuses σ between refreshes; eval
    recomputes per eval batch; train/eval caches separate. *(reviewer point 5)*
12. **Mode parity**: producer and provider use the same probe mode; probes are
    no_grad. *(reviewer point 4)*
13. **Provenance binding**: provider hard-asserts config match + parent hash;
    mismatched gβ ckpt → error. *(reviewer point 8)*
14. Order-frame: provider returns model-frame orders; training path calls no
    inv_perm; `val_origin_l2r_loss` still uses `inverse_block_perm`.
15. Eval = model-selected-order loss: `val` under gβ batch-mean σ; no
    single-sample fallback.
16. Backward compatibility: with `GBetaFrozenOrder`/flags absent, chenhe train.py
    behaves exactly as today (fixed-seed 1-step smoke on `order`, param_groups).

---

## Files touched

- **new** `chenhe_rerun/orderhead_v3/` (package w/ `__init__.py`, `constants.py`,
  ported math, `gbeta_provider.py`).
- **new** `chenhe_rerun/gbeta_cdl_pretrain.py` — Stage-1 producer, chenhe-native
  model+data load.
- **edit** `chenhe_rerun/train.py` — `GBetaFrozenOrder` dispatch; `--init-from-ckpt`
  + `--init-from-ckpt-mode` + gβ flags; ensure eval active-policy path uses the
  batch-mean provider with per-eval-batch refresh.
- **new** `config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py`.
- **new** tests under `chenhe_rerun/tests/` for Parts A/B/C.
- **repo hygiene:** `.gitignore` for `chenhe_rerun/{out,wandb,logs}/`; work on a
  new branch `feat/v3-frozen-gbeta-chenhe-port` (NOT the cleanup branch).
  *(reviewer point 12)*

## Implementation task structure (for writing-plans)

- **Task 1 — port math package (Part A)**: chenhe-local imports, constants,
  layout guard; tests 1–4 (incl. frame-index semantics).
- **Task 2 — Stage-1 producer (Part B)**: chenhe-native model/data load,
  model-frame teacher, provenance; tests 5–8.
- **Task 3 — provider + dispatch + flags (Part C)**: batch-mean frozen path,
  cache semantics, mode parity, provenance binding, `--init-from-ckpt[-mode]`;
  tests 9–16 + backward-compat.
- **Task 4 — parent ckpt + method config + end-to-end smoke**: produce the 10k
  parent, `gbeta_frozen_warmup.py`, a short resume-from-parent smoke emitting
  `val` + `val_origin_l2r_loss` + overhead logs.

## Open items resolved

- Target = chenhe `train.py` (seq256/block64); frozen gβ only (Stage 3 deferred).
- gβ = `L0DynamicGBeta`, 8-head batch-mean; block count matches (64).
- Metric: headline `val_origin_l2r_loss`; `val` = diagnostic only.
- Batch-mean B in training AND eval; never per-sample; eval refreshes per batch.
- Frame: model-frame throughout, incl. gβ supervision target; `inverse_block_perm`
  only inside the metric; frame-index semantics is a dedicated test.
- Mode parity for probes; probes no_grad; dropout=0 → jitter-free.
- Comparison = continuation from shared parent P; from-scratch curves separate;
  init-from-ckpt mode parity across continuation methods.
- Overhead (tokens/sec, sec/step, probes/step) logged; step-based NLL primary.
- Provider hard-asserts gβ↔parent provenance binding.
- Layout guard hard-fails off block64.
