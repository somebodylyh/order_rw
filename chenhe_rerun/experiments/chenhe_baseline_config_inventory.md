# chenhe Baseline Config Inventory

Reference for the rerun. All paths relative to `chenhe_rerun/`.
See `chenhe_baseline_lock.md` for the locked field values.

> **Mainline = `seq256/permute/block64`** (64 blocks; chenhe-confirmed
> 2026-07-02). `seq384/permute/block96` is an ablation, not same-table with the
> mainline.

## Native baseline configs (seq256 mainline, block64)

| Config | `permute_data` | `aogpt_train_mode` | `main_eval_mode` | Purpose |
|--------|:--:|--------------------|------------------|---------|
| `config/WikiText103/seq256/permute/block64/ar.py` | True | `AR` | `AR` | **Permuted-AR baseline** (L2R in fixed permuted frame) |
| `config/WikiText103/seq256/permute/block64/random.py` | True | `Random` | `Random` | **Permuted-Random baseline** |
| `config/WikiText103/seq256/non_permute/block64/ar.py` | False | `AR` | `AR` | Non-permuted AR (diagnostic, different frame) |

`ar.py` vs `random.py` diff (the method-only template): `aogpt_train_mode`,
`main_eval_mode`, `out_dir`, `wandb_run_name` (+ `wandb_log` bookkeeping).

Launch (config-exec):
```bash
python train.py config/WikiText103/seq256/permute/block64/ar.py
python train.py config/WikiText103/seq256/permute/block64/random.py
```

## Order-policy dispatch modes already in `train.py`

Central registry: `train.py` ~L7551 (`aogpt_train_mode` → `block_orders[B,64]`).
Interface: given a batch `idx`, returns `(block_orders, ordered_units, label)`.

| `aogpt_train_mode` | Backing module | Notes |
|--------------------|----------------|-------|
| `AR` | inline (`arange`) | L2R in current frame |
| `Random` | inline (`sample_random_block_orders`) | |
| `FixedBlockOrder` | inline; needs `fixed_block_order` block ids | candidate for a manual-order ablation |
| `AttnMLPFrozenOrder` | `attn_mlp_order_policy.py` | chenhe's own MLP order policy — **NOT our method; do not reuse for V3** |
| `OnlineSpectralFixedHeadOrder` | `online_spectral_order_policy.py` | fixed-head spectral order |
| `OnlineSpectralOrderDistribution` | `online_spectral_order_policy.py` | spectral order distribution |

## Our method = V3 (ported into `train.py`)

The method under test is **V3** (3-stage OrderHead regime; spec
`docs/superpowers/specs/2026-07-02-orderhead-cdl-pretrain-decoupled-init-design.md`),
currently implemented in admin `train_clean_aogpt.py`. **Decision (2026-07-02):
port full V3 (Stage 2+3) into chenhe `train.py`.**

- **Stage 1** CDL pretrain (offline): produces `g_beta_best.pt`
  (`gβ = L0DynamicGBeta`, 8-head batch-mean, input `B_raw (B,8,65,65)` →
  `scores (B,64)`). Reuse `build_l0_dynamic_gbeta_dataset` +
  `train_l0_dynamic_gbeta` verbatim.
- **Stage 2** frozen warmup: new dispatch path — order = `argsort(-gβ(B))`,
  backbone trains on LM-NLL. gβ frozen.
- **Stage 3** PG unfreeze: loop surgery — gβ `requires_grad=True`, one-time
  optimizer param-group insertion, argsort→PL sampling, grad-enabled gβ forward
  on **detached** B, PG loss via `v3_group_credit` + `λ_PG`. CDL never reappears.

> ✅ **Block count matches**: V3 is native 64-block (65-node strict65); the
> mainline is 64-block. No dimension rebuild of gβ / CDL / B-extraction needed.
>
> ⚠️ **B-source + frame invariant (hard):** the ported Stage-2/3 B must be
> constructed identically to `FrozenGBetaModelFrameBlockProvider` (8-head
> strict65, batch-mean probes, `none_mode='model'`, model-frame), and PL orders
> are model-frame → reuse the provider's model-frame→token conversion, no
> physical remap. Frame mismatch is the #1 known failure mode (memory
> `frozen-gbeta-deploy-noneframe-bug`).
>
> ⚠️ **Attention-extraction bridge:** chenhe backbone attention interface vs
> admin's `extract_selected_head_A_for_batch` may differ — the port must produce
> the same strict65 8-head B from chenhe's AOGPT. Verify before wiring.
