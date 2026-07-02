# chenhe Baseline Lock

**Status:** LOCKED — established 2026-07-02
**Owner of reference system:** chenhe
**Reference config source:** `config/WikiText103/seq256/permute/block64/{ar,random}.py`

> **Mainline correction (2026-07-02):** chenhe confirmed the **mainline is
> `seq256/permute/block64`** (64 blocks). `seq384/permute/block96` is an
> **ablation**, NOT the mainline. Evidence: `docs/prompts/prompt_language_block_current_task.md`
> = "Language Block64 Current Task … current handoff for WikiText103
> `seq256/permute/block64`". This also matches admin's entire method line and V3
> native size (64 blocks / 65-node strict65) — no V3 rebuild needed.

This file defines the single reference training system. Every rerun experiment
inherits ALL fields below **verbatim**. Only method / order-policy fields may
differ between configs.

> Working tree: `chenhe_rerun/` is an admin-owned, code-only copy of
> `chenhe_nanogpt_learned_order/` (26G `Report/` excluded), writable/committable
> into `order_lyu`. Original `chenhe_nanogpt_learned_order/` stays read-only.

---

## Entry point & config style

| Item | Value |
|------|-------|
| Training entrypoint | `python train.py <config.py> [--override=...]` |
| Config loading | config-file **exec** (Python file assigns globals) |
| Reference baseline configs | `config/WikiText103/seq256/permute/block64/{ar,random}.py` |

## LOCKED fields (inherit verbatim — seq256 / block64 mainline)

### Data pipeline
| Field | Value |
|-------|-------|
| `dataset` | `wikitext103` |
| data access | `np.memmap` random access — `data/wikitext103/{train,val}.bin` (uint16), `meta.pkl` |
| `permute_data` | `True` |
| `permute_seed` | `42` |
| `permute_mode` | `block` |

### Sequence / block layout
| Field | Value |
|-------|-------|
| `block_size` | `256` |
| `block_order_block_len` | `4` |
| derived num blocks | `256 / 4 = 64` |

### Model config
| Field | Value |
|-------|-------|
| `model_type` | `aogpt` |
| `train_stage` | `standard` |
| `n_layer` | `4` |
| `n_head` | `8` |
| `n_embd` | `384` |
| `dropout` | `0` |

### Optimizer / schedule
| Field | Value |
|-------|-------|
| `learning_rate` | `1e-3` |
| `min_lr` | `1e-4` |
| `max_iters` | `50000` |
| `lr_decay_iters` | `50000` |
| `warmup_iters` | `0` |
| `beta2` | `0.99` |
| `batch_size` | `64` |
| `gradient_accumulation_steps` | `2` |

### Eval / logging cadence
| Field | Value |
|-------|-------|
| `eval_interval` | `250` |
| `eval_iters` | `200` |
| `log_interval` | `10` |

> Fields not set in the block64 base configs (`eval_batch_size`, `compile`,
> `order_impl`) take `train.py` defaults — do NOT introduce them in method
> configs unless the base does.

---

## Metric policy (RESOLVED 2026-07-02)

- **`val_origin_l2r_loss`** — NLL in the original (un-permuted) L2R frame.
- **`val_main_eval_loss`** — NLL in the current permuted frame under `main_eval_mode`.

| Use | Metric | Rationale |
|-----|--------|-----------|
| **Final headline / main-table judgment** (one metric for all methods) | `val_origin_l2r_loss` | Paper "L2R validation NLL"; cross-method comparable. |
| **Training-side decisions** (early-stop / selection / ckpt pick) | `val_main_eval_loss` (current frame) | chenhe Safety Note: under `permute_data=True` do NOT use `*_original` for training decisions. |

`val_origin_l2r_loss` is **eval/report only** — never drives training selection.
Do NOT put admin's `val_ori_l2r_block` in the main table.

---

## W&B

**Not inherited from chenhe** — we use our own W&B. Only constraint: run tags
must distinguish groups and not collide. (chenhe reference:
`wandb_project = 'AOGPT-order-block-64-final'`.)

---

## Allowed-to-change fields (method / order-policy ONLY)

- `aogpt_train_mode`, `main_eval_mode`, `generalization_eval_mode`
- order-policy module / controller checkpoint / refresh / ramp / head selection
- V3 flags (Stage-2/3): `--unfreeze-orderhead-at-step`, `--orderhead-lr`,
  `--lam-pg`, `--pg-tau`, `--pg-group-m`, `--pg-beta`, `--pg-adv-clip`,
  `--cdl-pretrain`, `--cdl-source-ckpt` (once ported into `train.py`)
- `out_dir` (bookkeeping), `wandb_*` (our own)

**Method-only pattern proof:** `ar.py` vs `random.py` differ in exactly
`aogpt_train_mode`, `main_eval_mode`, `out_dir`, `wandb_run_name` (+ `wandb_log`
bookkeeping). Every method config must diff in that same shape only.

---

**All fields above (except W&B and the allowed list) are LOCKED. Only method / order-policy fields may change.**
