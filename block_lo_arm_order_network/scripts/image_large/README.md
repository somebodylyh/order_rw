# Partner ImageNet-64 VQ-f4 800k — End-to-end Setup

Goal: reproduce ych's image-side AOGPT training (`l8h8e512`, vocab=8192, 256
VQ tokens per image, ImageNet-64) on our own machine, then plug in our
Graph-RW order policy as a continuation curriculum.

ych's `train.py` + `AOGPT_block.py` are reused **as-is** — there is no
"image branch" of the code. The only differences from text training are:
1. A new `data/Imagenet64VQ_f4_800k/{train.bin, val.bin, meta.pkl}` file
2. A config file with the partner architecture/data hyperparams
   (`configs/image_partner/imagenet64_l8h8e512_random_base.py`)

This document is the operational checklist.

---

## Step 0 — environment (~5 min, one-off)

```bash
pip install --upgrade diffusers==0.27.0 omegaconf einops pillow
# OR if diffusers cannot load CompVis/vq-f4 directly:
pip install taming-transformers omegaconf
```

`torch` and `numpy` are already in the base env.

---

## Step 1 — download the CompVis VQ-f4 VAE checkpoint (~330 MB, one-off)

Two acceptable sources:

### (a) Diffusers-format on HF (preferred — no extra deps)

```bash
huggingface-cli download CompVis/vq-f4 --local-dir ~/models/vq-f4
```

If this returns 404 / not-found, fall back to (b).

### (b) Original CompVis .ckpt + config.yaml (fallback)

```bash
mkdir -p ~/models/vq-f4
cd ~/models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
unzip vq-f4.zip   # produces model.ckpt and config.yaml
```

The `prepare.py` script will auto-detect which loader path applies.

---

## Step 2 — download ImageNet-64 (~14 GB train, ~600 MB val, one-off)

Official source (requires ImageNet credentials):

1. Sign up at <https://image-net.org/download.php> (academic agreement)
2. Download `Imagenet64_train.zip` and `Imagenet64_val.zip`
3. Unzip to:
   - `~/data/imagenet64/train/`  (10 batch files: `train_data_batch_1` ... `_10`)
   - `~/data/imagenet64/val/`    (1 batch file: `val_data`)

The batches are pickle files with key `'data'` (uint8 (N, 12288) flat) and
`'labels'`. `prepare.py` accepts both raw pickle and `.npz` wrappers.

---

## Step 3 — tokenize → produce 800k subset .bin files (~3-5 min on a 4090)

```bash
cd block_lo_arm_order_network
python data/Imagenet64VQ_f4_800k/prepare.py \
    --imagenet-train-dir ~/data/imagenet64/train \
    --imagenet-val-dir   ~/data/imagenet64/val \
    --vae-path           ~/models/vq-f4 \
    --batch-size         128 \
    --device             cuda
```

Outputs (deterministic; subset shuffled with seed=42, take first 800 000):

- `data/Imagenet64VQ_f4_800k/train.bin`  ≈ 410 MB  (800 000 × 256 × 2 B)
- `data/Imagenet64VQ_f4_800k/val.bin`    ≈  25 MB  ( 50 000 × 256 × 2 B)
- `data/Imagenet64VQ_f4_800k/meta.pkl`   small JSON with vocab_size + token shape

Quick smoke (no downloads, no real encoding — verifies the script runs):

```bash
python data/Imagenet64VQ_f4_800k/prepare.py --smoke
```

---

## Step 4 — wire into ych's train.py (~30 s)

Symlink the data into ych's repo (don't duplicate 410 MB):

```bash
mkdir -p ~/ych/nanogpt-learned-order/data
ln -s $(pwd)/data/Imagenet64VQ_f4_800k ~/ych/nanogpt-learned-order/data/
```

Run the partner baseline (uses ych's train.py + the partner config in this
repo):

```bash
cd ~/ych/nanogpt-learned-order
python train.py /home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/configs/image_partner/imagenet64_l8h8e512_random_base.py
```

Expected: `train.py` finds `data/Imagenet64VQ_f4_800k/{train.bin, val.bin,
meta.pkl}`, reads `vocab_size=8192` from `meta.pkl`, builds the l8h8e512
model, runs 50 000 iters of random-order block training. wandb / checkpoint
land under `out/base/permute/imagenet64_vq_f4_800k/...`.

---

## Step 5 — Graph-RW continuation (NOT YET WIRED)

Once Step 4 lands a baseline checkpoint, the Graph-RW continuation only
requires a small patch to ych's `train.py` to add a new `--graph-rw-policy`
branch in `_forward_with_active_training_policy(X)` (around line 432–512)
that calls `directed_graph_policy.sample_orders_batched_torch` from this
repo.

Patch / config files for that step will be added once the baseline result is
in hand. Do NOT run the patch before the baseline is verified.

---

## Cost summary (single 4090, partner-shared GPU 0)

| step | wall clock | network | disk |
|---|---|---|---|
| 0 install deps | < 5 min | 100 MB | < 50 MB |
| 1 download VAE | < 5 min | 330 MB | 330 MB |
| 2 download ImageNet-64 | ~30 min | 14 GB | 14 GB |
| 3 tokenize 800k subset | 3-5 min | — | 440 MB |
| 4 baseline train 50k iters | ~6 hrs | — | ~500 MB ckpt |

Total one-off setup ≈ 40 min + 14 GB disk; baseline run ≈ 6 hrs.

---

## Open questions to confirm with ych before running Step 4

1. Is `CompVis/vq-f4` (codebook=8192, f=4) the same VAE you used? If not, give
   us the HF id or ckpt path.
2. The `permute_data=True, permute_mode='block'` pre-shuffle uses a fixed
   random permutation per dataset instance — is `permute_seed=42` what you
   used too? (We can match exactly if you tell us the seed.)
3. Once Step 4 baseline reaches 50 k iters, we plan to fork the ckpt and run
   Graph-RW α=0.9 continuation for another 5 k iters. Any objection?
