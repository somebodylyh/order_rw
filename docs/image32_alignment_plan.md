# ImageNet32 Alignment Experiment Plan

**Date:** 2026-05-16
**Status:** Plan only — no training launched.
**Motivation:** Determine whether the toy `image32` (CIFAR continuous patch + MSE) success can be reproduced when one variable at a time is lifted toward the current `ImageNet64 VQ-f4` (`l8h8e512` + CE) setting that does *not* show spatial attention.

---

## 1. Why the current ImageNet64 result cannot refute the toy

We previously showed that the toy CIFAR continuous-patch baseline produces attention with clear 2D spatial locality (`mean_manhattan ≈ 2.4`, low readiness signal `s < 1.0`). The current ImageNet64 VQ-f4 `patch2x2` baseline shows the opposite (`manh ≈ 11.7`, `s ≈ 95`, near-global attention). These two settings differ on **at least seven independent axes**:

| Axis | Toy (E0) | Current large (E3) |
|---|---|---|
| Representation | Continuous RGB patch | Discrete VQ token id |
| Loss | MSE | CE |
| Native grid | 8×8 patches | 16×16 VQ tokens → 8×8 blocks of 2×2 |
| Model | `image_order/model_image_aogpt.py`, l4h8e256, cond_dim=128 | nanoGPT-style `model_AOGPT_..._cond_128_trunc_qknorm.py`, l8h8e512 |
| Data | CIFAR-10 (50k, 10 classes) | ImageNet (800k subset, 1000 classes) |
| Optim | lr=3e-4, batch=64, warmup=200, 10k steps | lr=1e-3, batch=256, warmup=0, 50k steps |
| Position encoding | 1D `wpe(N+1, n_embd)` | 1D `wpe(block_size+1, n_embd)` (see §4) |

A single negative observation at the end of a 7-variable change tells us nothing about which axis breaks spatial attention. We need minimal-change bridges.

---

## 2. Experiment matrix

| ID | Data | Repr / Loss | Grid | Model | Purpose |
|---|---|---|---|---|---|
| **E0** | CIFAR-10 32×32 | continuous patch, MSE | 8×8 (native) | l4h8e256 | Existing toy success (reference) |
| **E1** | **ImageNet32** (bilinear from ImageNet64) | continuous patch, MSE | 8×8 (native) | **l4h8e256** (toy-matched) | **Isolate dataset complexity** |
| **E2** | ImageNet32 VQ-f4 | discrete token, CE | 8×8 = 64 tokens (native) | l4h8e256 or l8h8e512 | Isolate VQ/CE objective |
| **E3** | ImageNet64 VQ-f4 (existing) | discrete token, CE | 16×16 → 2×2 blocks → 8×8 | l8h8e512 | Current negative/boundary setting |

**Diff between adjacent experiments — exactly one axis changes:**

- E0 → E1 : **dataset only** (CIFAR → ImageNet) ; representation/loss/model/grid all held.
- E1 → E2 : **representation/loss only** (continuous+MSE → VQ+CE) ; dataset/grid held.
- E2 → E3 : **scale + aggregation** (32×32 native → 64×64 with 2×2 block aggregation, plus model l4→l8 if both run small first).

If E1 shows locality, dataset complexity is **not** the killer.
If E1 has locality and E2 does not, **VQ/CE** is the killer.
If E1 already has no locality, the killer is in the data distribution or model/training defaults — not VQ/CE.

---

## 3. First-round budget

**Only run E1 in the first round.**

- Model: `l4h8e256` (match toy exactly).
- Data: ImageNet32 produced by **bilinear downsample of the existing `/tmp/imagenet64_800k/` raw `.npz`**. Do not introduce a new dataset source.
- Steps: 10k (match toy `cifar_patch_baseline10k.py`). Optional smoke at 1k–2k first.
- Optim: same as toy (`lr=3e-4`, `batch=64`, `warmup=200`).
- Eval: `val_random` + `val_raster` curves + `A_global` extraction + same diagnostic suite as CIFAR baseline (`mean_manh`, `P(d≤1)`, `P(d≤2)`, `same_q`, `same_s`, `readiness_signal`, `hub_score` on `random / shuffled_B / random_B` controls).

E2/E3 only get configured after E1 results are in.

---

## 4. Position-encoding sanity check (resolved)

Inspection result:

- `block_order_layout` is a **decorative** config field. `grep -rn block_order_layout` finds it only in `configs/image_large/*.py`; **zero usages outside configs**. The config file `imagenet64_l8h8e512_random_base.py:21-24` itself documents this.
- nanoGPT-style large model uses `wpe(block_size+1, n_embd)` — 1D absolute (`model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py:171`).
- Toy ImageAOGPT uses `wpe(n_patches+1, n_embd)` — 1D absolute (`image_order/model_image_aogpt.py:68`).

**Conclusion: both toy and large already use the same 1D absolute scheme. Position encoding is *not* an unaligned variable.** RoPE / 2D-absolute ablations are deferred to Round 3 and only get triggered if E2 fails with locality unrecovered.

Encoding ablation order (for later use, not Round 1): `1D absolute → 2D row+col absolute → 2D RoPE`.

---

## 5. Data preparation policy

- Source: `/tmp/imagenet64_800k/train_npz/*.npz` (626 batches, 800k images) and `/tmp/imagenet64_800k/val_npz/*.npz` (40 batches, 50k images), each `(N, 3, 64, 64) uint8`.
- Transform: per-image bilinear downsample 64→32 via `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)`.
- Storage: `/home/admin/lyuyuhuan/order_lyu/image_order/data/imagenet32/{train.npy, val.npy}` as `uint8 (N, 3, 32, 32)` (~2.5 GB train, ~150 MB val).
- `/tmp` is volatile; the downsample script must be re-runnable, but as long as `/tmp/imagenet64_800k/` still exists we can rebuild in <10 min.
- **Do not introduce a new dataset source.**

---

## 6. Code surface for E1

1. `image_order/data/imagenet32/build_imagenet32.py` — one-time bilinear downsample script. Reads `/tmp/imagenet64_800k/{train,val}_npz/`, writes `.npy` to `image_order/data/imagenet32/`. Supports `--limit-train N` for smoke (no full 800k required to get started).
2. `image_order/data_imagenet32_patches.py` — `ImageNet32Patches` Dataset, mirrors `CIFAR10Patches` interface (`.patches`, `.images`, `.labels` of same shape/dtype/normalization). Loads the `.npy` files via `np.load(..., mmap_mode='r')` to avoid 2.5 GB resident memory.
3. `image_order/train_image_random_imagenet32.py` — thin clone of `train_image_random.py` that swaps `CIFAR10Patches` → `ImageNet32Patches`. Same model, same optimizer, same evaluation. Same CLI surface.
4. `block_lo_arm_order_network/configs/image/imagenet32_patch_baseline10k.py` — documentation-style config matching `cifar_patch_baseline10k.py` (CLI invocation is the actual contract).

All four are written now; **none are executed**.

---

## 7. Open questions for Round 2 / Round 3 (not Round 1)

- E2 model capacity: do we run E2 at `l4h8e256` first or jump to `l8h8e512`? Decide after E1.
- For E2 we need a VQ-f4 tokenizer pass on 32×32 inputs (downsampled ImageNet32). Reuse `data/Imagenet64VQ_f4_800k/prepare.py` machinery; only the input dir changes.
- Whether to also fix optim/batch/warmup before declaring "VQ is the killer": E2 should match toy optim *first*, then run a `l8h8e512+lr1e-3+batch256` sweep only if locality is observed at the small/match-toy setting.

---

## 8. What this plan does NOT do

- Does not continue diagnostics on ImageNet64 E3 unless E1/E2 results need a cross-check.
- Does not change position encoding in Round 1.
- Does not change model size in E1.
- Does not launch any training.
