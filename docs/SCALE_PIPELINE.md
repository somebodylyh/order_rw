# AMOR Scale-up Pipeline (12L/768 · OpenWebText 2.5B)

Scale test of the text AMOR result: does the own-order advantage (pgonly 3.47 vs random 3.94,
PG>frozen) hold with a ~12× bigger model + a large de-duplicated corpus?

**Scale axes**: model 4L/384 → **12L/768/8h** (~124M); data wikitext-103 118M → **OWT 2.5B** (no reuse).
**Unchanged** (so the entire gβ pipeline is drop-in): N=64 / seq256 / block4 / **n_head=8** /
stream + permute_data loading. Metric: **own-order val only** (wandb project `amor-order`).

> n_head stays 8 — `orderhead_v3.constants.assert_layout` hard-codes HEADS=8. 768/8 = head_dim 96.

## Cross-machine prerequisites
- `git pull` this branch (code only; no data/ckpt in git).
- `wandb login` once (login state is not in git).
- 1× ≥24GB GPU. If OOM on the 12L model: `--batch_size=32 --gradient_accumulation_steps=4`.
- Disk: OWT streaming peak ~5GB (2.5B × 2 bytes). No 54GB raw cache.
- All commands run from `chenhe_rerun/` cwd (paths are relative to it).

## Stage 1 — OpenWebText data (streaming, ~2.5B tokens, ~20-30min)
```bash
python -u chenhe_rerun/data/openwebtext/prepare_streaming.py
# → chenhe_rerun/data/openwebtext/{train.bin (~5GB), val.bin}
```
`dataset='openwebtext'` resolves to `data/openwebtext/` (no symlink needed; unlike the VQ line).
No meta.pkl needed — vocab_size=50304 is set in the config.

## Stage 2 — Scaled random backbone: warmup + baseline (12L/768/8h)
chenhe train.py only saves `ckpt.pt` (overwritten each eval — no periodic step-named ckpts),
so the 10k warmup is a **separate short run**, then continued for the baseline.
```bash
cd chenhe_rerun
# 2a. warmup → ckpt.pt = the 10k warmup snapshot (head-select + gβ read this)
python -u train.py config/openwebtext/scale_warmup10k_owt.py
#     → out/rerun_owt/scale12L768_warmup10k/ckpt.pt

# 2b. baseline = random continuation 10k→60k (the random arm)
python -u train.py config/openwebtext/scale_baseline_owt.py
#     → out/rerun_owt/scale12L768_baseline/ckpt.pt
```
LR decay is anchored to 60k in the warmup config so the 10k snapshot isn't over-annealed.

## Stage 3 — gβ pretrain on the 10k warmup (head-select + CDL + readout)
```bash
python -u chenhe_rerun/run_gbeta_owt_pretrain.py
# → out/rerun_owt/gbeta_owt_bm16/{g_beta_best.pt, gbeta_provenance.json}
# reports selected head + val_pairwise_acc (wikitext ref 98.44%; VQ image was 65-74%)
```

## Stage 4 — Deploy (frozen + pgonly), 10k→60k, PG@20k
```bash
cd chenhe_rerun
python -u train.py config/openwebtext/deploy_frozen_owt.py   # gβ frozen 10k->60k
python -u train.py config/openwebtext/deploy_pgonly_owt.py   # gβ frozen 10k->20k, PG 20k->60k
```
Both init from `ckpt_step10000.pt` (chenhe-native → no compat ckpt needed).

## Read the result (own-order val, wandb `amor-order`)
| arm | own-order val | tests |
|-----|---------------|-------|
| random (Stage 2 @ matched step) | baseline | — |
| frozen gβ | vs random | order-consistency co-adaptation |
| pgonly gβ | vs frozen | does PG convert (best_gap) |

**Key questions**: does the **own-order gap** (wikitext was 0.47) hold / grow / shrink at scale?
Does **PG > frozen** hold (text) or go inert (image)? Compare own-order val curves across arms.

## Files
- `chenhe_rerun/data/openwebtext/prepare_streaming.py` — OWT streaming tokenizer (2.5B cap)
- `chenhe_rerun/config/openwebtext/scale_warmup10k_owt.py` — 12L/768/8h random warmup → 10k
- `chenhe_rerun/config/openwebtext/scale_baseline_owt.py` — random continuation 10k→60k (baseline arm)
- `chenhe_rerun/run_gbeta_owt_pretrain.py` — gβ pretrain on the 10k warmup
- `chenhe_rerun/config/openwebtext/deploy_{frozen,pgonly}_owt.py` — gβ deploy arms 10k→60k

## Compute note
12L/768 @ 60k steps ≈ multi-hour per arm on one 24GB GPU; 4 runs (warmup+baseline+frozen+pgonly)
= a multi-GPU-day campaign. Run arms in parallel across machines/GPUs if available (wandb groups them).
