# Running the frozen-gβ rerun on another machine

Everything below runs from **`chenhe_rerun/`** as the working directory
(`train.py` reads `data/` and `config/` by relative path).

## 1. Code

```bash
git clone https://github.com/somebodylyh/order_rw.git
cd order_rw
git checkout feat/v3-frozen-gbeta-chenhe-port
cd chenhe_rerun
```

The whole `chenhe_rerun/` tree (backbone `AOGPT_block.py`, `train.py`, configs,
`orderhead_v3/`, `gbeta_cdl_pretrain.py`) is committed. Artifacts are NOT
(`out/ wandb/ *.pt *.bin`); regenerate them below.

## 2. Python environment

Tested on **Python 3.8.5 + torch 2.4.1+cu121**. Install:

```bash
pip install torch numpy scipy wandb tiktoken datasets tqdm
```

Notes:
- `torch`: match the machine's CUDA (2.4.x used here). CPU-only works for tests.
- `wandb` only needed when `--wandb_log=True`; otherwise pass `--wandb_log=False`.
- `tiktoken` + `datasets` only needed for data prep (step 3).

## 3. Data (NOT in git — regenerate)

`data/wikitext103/{train,val}.bin` are gitignored (they were symlinks here).
Regenerate from HuggingFace (downloads wikitext-103-raw-v1, ~190MB):

```bash
python data/wikitext103/prepare.py
```

If the machine is offline / behind the GFW, use the mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com python data/wikitext103/prepare.py
```

This writes `data/wikitext103/train.bin` (~237MB) and `val.bin`. `meta.pkl` is
NOT required (train.py falls back to vocab_size=50304).

Alternatively `rsync` the two `.bin` files from this machine:
`/home/admin/lyuyuhuan/order-shakespeare/nanoGPT/data/wikitext103/{train,val}.bin`.

## 4. Sanity: run the tests

```bash
python -m pytest tests/test_orderhead_v3_strict65.py tests/test_orderhead_v3_gbeta.py \
  tests/test_gbeta_cdl_pretrain.py tests/test_gbeta_provider.py \
  tests/test_gbeta_frozen_e2e.py -q          # 11 tests, ~10s (CPU ok)
```

(The pre-existing `tests/test_aogpt_random_orders.py` fails — chenhe test rot,
unrelated to this work; ignore it.)

## 5. Baselines (seq256 / permute / block64 mainline)

```bash
python train.py config/WikiText103/seq256/permute/block64/random.py --wandb_project=<yours>
python train.py config/WikiText103/seq256/permute/block64/ar.py     --wandb_project=<yours>
```

Headline metric logged: `val_origin_l2r_loss` (original text-L2R NLL, comparable
across methods). `val` = active-policy loss (order-quality diagnostic).

## 6. Frozen-gβ method (three steps)

```bash
# (a) shared random parent P at step 10k — KEEP lr_decay_iters=50000 for lr continuity
python train.py config/WikiText103/seq256/permute/block64/random.py \
  --max_iters=10000 --out_dir=out/rerun/parent_random_10k --wandb_log=False

# (b) CDL-pretrain gβ from that parent (offline; ~mins on GPU)
python -c "from gbeta_cdl_pretrain import pretrain_gbeta_cdl; \
  print(pretrain_gbeta_cdl('out/rerun/parent_random_10k/ckpt.pt', \
        'out/rerun/gbeta_from_parent10k', M=2000, batch_mean_size=16, \
        epochs=40, device='cuda'))"

# (c) frozen-gβ continuation to 50k (resumes the parent, only order policy differs)
python train.py config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py \
  --wandb_project=<yours>
```

`gbeta_frozen_warmup.py` already points `init_from_ckpt` and `gbeta_ckpt` at the
outputs of (a) and (b). The provider hard-asserts the gβ was pretrained from that
exact parent (content hash), so mismatched ckpts error out.

## 7. What each machine-specific thing is

| Thing | In git? | How to get on new machine |
|-------|:-------:|---------------------------|
| code + configs | yes | clone + checkout |
| `train/val.bin` | no | `prepare.py` (step 3) or rsync |
| parent ckpt, gβ ckpt, run outputs | no | produced by step 6 |
| Python env / CUDA | no | step 2 |
| W&B project | n/a | pass `--wandb_project=<yours>` |
