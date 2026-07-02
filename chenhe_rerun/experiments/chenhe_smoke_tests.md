# chenhe Rerun Smoke Tests

Dry-run / tiny-step commands only. **Do NOT launch full training here.**
Each smoke run checks: (1) starts, (2) loss finite, (3) eval emits
`val_origin_l2r_loss` + `val_main_eval_loss`, (4) checkpoint saves, (5) W&B tag
unique, (6) config diff is method-only.

## Prerequisite: data `.bin` not present

`data/wikitext103/` currently has only `prepare.py` — **no `train.bin` / `val.bin`
/ `meta.pkl`**. `train.py:644–685` requires them. Generate first:

```bash
cd chenhe_rerun
python data/wikitext103/prepare.py
```

(Needs wikitext103 raw + tokenizer; use the offline HF mirror per memory
`offline_hf` if network is restricted. This is the one hard blocker before any
smoke run.)

## Smoke command template

Tiny run: 1 step, eval every step, W&B off, no compile.

```bash
python train.py <config> \
  --max_iters=1 --eval_interval=1 --eval_iters=1 \
  --wandb_log=False --compile=False
```

## Baseline group

```bash
# B0 permuted-AR
python train.py config/WikiText103/seq256/permute/block64/ar.py \
  --max_iters=1 --eval_interval=1 --eval_iters=1 --wandb_log=False --compile=False

# B1 permuted-Random
python train.py config/WikiText103/seq256/permute/block64/random.py \
  --max_iters=1 --eval_interval=1 --eval_iters=1 --wandb_log=False --compile=False
```

## Method group (after batch-2 configs exist)

```bash
# B2 / M1 / M2 / M3 / M4 — same template, swap config path:
python train.py config/WikiText103/seq256/permute/block64/<method>.py \
  --max_iters=1 --eval_interval=1 --eval_iters=1 --wandb_log=False --compile=False
```

## Checks per run

- [ ] process starts, no import/dispatch error
- [ ] first-step loss is finite (not NaN/inf)
- [ ] eval log contains `val_origin_l2r_loss` AND `val_main_eval_loss`
- [ ] checkpoint written under `out_dir`
- [ ] W&B run tag unique (when wandb enabled)
- [ ] `diff <method>.py <base>.py` touches method/bookkeeping fields only
      (cross-check against `chenhe_method_config_diff.md`)
