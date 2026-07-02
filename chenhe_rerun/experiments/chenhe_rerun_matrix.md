# chenhe Rerun Matrix

All rows share the LOCKED baseline (`chenhe_baseline_lock.md`,
**seq256 / permute / block64**).
**Baseline system = chenhe `train.py`** for every row.
**Headline metric = `val_origin_l2r_loss`** (judgment); training-side selection
uses `val_main_eval_loss` (current frame).

## Baseline group (native chenhe, no method code)

| Exp | Config | `aogpt_train_mode` | Purpose |
|-----|--------|--------------------|---------|
| B0 | `.../seq256/permute/block64/ar.py` | `AR` | L2R reference anchor |
| B1 | `.../seq256/permute/block64/random.py` | `Random` | random-order baseline |
| B2 | `random_warmup_ar.py` *(to create)* | `Random`→`AR` schedule | random warmup → L2R (optional) |

## Method group = V3 (ported into `train.py`)

Our method is **V3** (3-stage OrderHead). The rows below are its canonical
configuration + ablations, all on the same chenhe baseline.

| Exp | Config *(to create)* | What | Dispatch / flags |
|-----|----------------------|------|------------------|
| V3-S2 | `gbeta_frozen_warmup.py` | Stage-2 only: frozen `argsort(-gβ(B))`, no PG | new `GBetaFrozenOrder` branch |
| V3-S3 | `gbeta_cdl_pretrain_pg.py` | Full 3-stage: CDL init → frozen warmup → PG unfreeze | `--cdl-pretrain`, `--unfreeze-orderhead-at-step`, `--lam-pg`, `--pg-tau`, … |
| (abl) | `manual_one_shot.py` | manual fixed order (cheap ablation) | existing `FixedBlockOrder` |

Stage-1 (CDL pretrain producer) is offline → produces `g_beta_best.pt` from the
canonical 10k parent; not a training row itself.

## Fixed columns (every row)

| Column | Value |
|--------|-------|
| Baseline system | chenhe `train.py` |
| Data pipeline | `permute_data=True`, memmap random access, `permute_seed=42`, block mode |
| seq / blocks | `block_size=256`, `block_order_block_len=4` → 64 blocks |
| Model | aogpt, 4L/8H/384d |
| Steps | `max_iters=50000` |
| Headline metric | `val_origin_l2r_loss` |
| Selection metric | `val_main_eval_loss` |
| W&B | our own project/tag (not chenhe's) |
| Output dir | `out/rerun/<exp>/...` |

## Status / dependencies

- **B0, B1**: native configs exist — runnable once data `.bin` present.
- **V3-S2 / V3-S3**: require porting V3 into `train.py` (new SDD). Block count
  already matches (64) — no gβ/CDL dimension rebuild.
- **B2, manual**: pure config / existing dispatch.
