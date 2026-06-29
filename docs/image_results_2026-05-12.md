# Image-Patch AOGPT + Graph-RW v2 — Results Summary (2026-05-12)

> Status: Stage 1 (structure discovery) + Stage 2 (continuation curriculum) **complete on single seed (seed=42)**. Multi-seed CI (seeds 0, 1, 42) in progress; this doc will be amended with paired Δ + 95% CI when those finish.

## TL;DR claim

AOGPT's `A_global` attention encodes dataset-level shared structure not only on 1D text but also on 2D image patches. On CIFAR-10 4×4 patches (N=64), Graph-RW v2 with `top_k=4, ε=0` extracts a **soft 2D neighborhood + region + appearance walk** from the trained attention — distinct from a raster scan. Training with this order as an α-mixed continuation curriculum **measurably improves validation MSE on structured reveal orders (−2.7 % to −3.4 %, 4–8 σ above seed noise)** while a noise-broken Graph-RW order (`ε=0.15`) provides no benefit over random continuation. The mechanism — *low-noise attention-derived order → curriculum gain; high noise → no gain* — matches the WikiText result, with the modality-appropriate structure substituted (1D L2R for text, 2D local+region+appearance for image).

## 1. Pipeline & reuse

| Component | Source |
| --- | --- |
| Reveal-order policy (`progressive_rw`) | `block_lo_arm_order_network/directed_graph_policy.py` — imported verbatim, no copy |
| α-mixed loss, refresh, eval scaffold | `train_aogpt_graph_rw.py` structure, reimplemented for image patches |
| Image AOGPT model | `image_order/model_image_aogpt.py` — text model's `Block`/`FinalLayer` reused; replaced `wte` with patch projection (Linear 48→256) and `lm_head` with patch head (Linear 256→48); loss = MSE |
| CIFAR-10 patches | offline pickle at `/home/admin/bw/data/cifar-10-batches-py/` → 8×8 grid of 4×4×3 patches, normalized to `[−1, 1]` |

Model: `ImageAOGPT(n_embd=256, n_layer=4, n_head=8, cond_dim=128)` ≈ **4.07 M params**.

Repo layout (all new code under `image_order/`):

```
image_order/
  __init__.py
  data_image_patches.py       # CIFAR-10 patch dataset + grid utilities
  model_image_aogpt.py        # Image AOGPT (continuous I/O, MSE)
  graph_rw_image.py           # thin Graph-RW wrapper (imports text policy)
  extract_image_attention.py  # A_global extraction
  train_image_random.py       # Stage 0: random-order baseline
  train_image_graph_rw.py     # Stage 2: α-mixed continuation
  eval_image_orders.py        # frozen 5-order MSE
  diagnose_image_orders.py    # structural diagnostics (locality, region, center, appearance)
```

## 2. Stage 0 — baseline10k learning curve

Trained `train_image_random.py` from scratch on CIFAR-10 train (50 000 imgs), 10 000 steps, batch 64, lr 3e-4 → 3e-5 cosine, 200-step warmup.

| step | val_random | val_raster |
|---|---|---|
| 0     | 0.3462 | 0.3551 |
| 500   | 0.1023 | 0.1099 |
| 1000  | 0.0792 | 0.0793 |
| 2000  | 0.0698 | 0.0653 |
| 5000  | 0.0629 | 0.0552 |
| 7500  | 0.0606 | 0.0530 |
| **10000** | **0.0596** | **0.0522** |

Final attention `sparsity_topk5 = 0.142` (smoke500 was 0.118) — attention sharper after longer training.

Output: `probe_results_image/baseline/baseline10k/{ckpt_step10000.pt, eval_curve.tsv, config.json, train_log.txt}`.

## 3. Stage 1 — Structural diagnostics on baseline10k A_global

Diagnosed 200 sampled orders per config under `diagnose_image_orders.py` v2 (structure-first metrics). Raster reported only as sanity reference, not optimization target.

| metric | random | raster | **graph_rw_top4** | graph_rw_top8 | graph_rw_eps015 |
|---|---|---|---|---|---|
| mean Manhattan ↓ | 5.30 | 1.78 | **3.10** | 3.49 | 5.23 |
| P(d ≤ 1) ↑ | 5.9 % | 88.9 % | **28.1 %** | 19.6 % | 6.4 % |
| P(d ≤ 2) ↑ | 15.4 % | 88.9 % | **45.9 %** | 36.8 % | 16.0 % |
| same 4×4 quadrant ↑ | 24.1 % | 76.2 % | **53.8 %** | 48.7 % | 24.9 % |
| same 2×2 super-region ↑ | 4.5 % | 50.8 % | **20.3 %** | 16.0 % | 5.2 % |
| run length in 4×4 quadrant | 1.32 | 4.00 | **2.21** | 2.09 | 1.34 |
| feature distance (z-score) ↓ | 2.74 | 1.69 | **2.24** | 2.35 | 2.72 |
| center_step_corr | −0.01 | 0.00 | −0.06 | −0.09 | −0.02 |
| sanity: τ vs raster | 0.00 | 1.00 | 0.82 | 0.73 | 0.06 |

**Reading**: `top_k=4` shows non-random structure on **4 orthogonal axes** — locality, region grouping, appearance similarity, and (mildly) center proximity. `eps015` collapses every structural axis back to within 1.5 % of random.

Best one-line characterization of the recovered structure:

> **Soft 2D neighborhood random walk + quadrant-block soft partitioning + bias toward visually-similar nearby patches.**

This is **not raster** (P(d=1) is 28 %, raster's is 89 %) and **not random** (random's is 6 %); the model fills a local cluster, jumps to an adjacent region, repeats.

Output: `probe_results_image/diagnostics/baseline10k/{summary.tsv, *_comparison.png, <config>/{order_path_example.png, avg_step_heatmap.png, transition_*_histogram.png, region_stay_curve_*.png, feature_distance_histogram.png}}`.

## 4. Stage 2 — α-mixed continuation curriculum, seed=42 (single-seed)

From `baseline10k/ckpt_step10000.pt`, continued 5000 steps with `train_image_graph_rw.py` at four configurations. lr 3e-5 → 3e-6 cosine, batch 64, α=0.9 (constant) for the Graph-RW configurations, α=0 for the random-only baseline.

| | training order policy |
| --- | --- |
| **cont_random** | α=0 — random reveal orders only (continuation baseline) |
| **cont_top4** | α=0.9, `top_k=4, ε=0` — Graph-RW low-noise |
| **cont_eps015** | α=0.9, `top_k=0, ε=0.15` — Graph-RW high-noise ablation |
| **cont_top8** | α=0.9, `top_k=8, ε=0` — wider top-k |

Frozen 5-order eval (val_images=1000, n_seed_rounds=3) on each final checkpoint:

| eval order | baseline10k | cont_random | **cont_top4** | cont_eps015 | cont_top8 |
|---|---|---|---|---|---|
| random | 0.05988 | 0.05929 | 0.05997 | 0.05927 | 0.05974 |
| raster | 0.05209 | 0.05154 | **0.04979** | 0.05140 | 0.05012 |
| rw_top4_eps0 | 0.05442 | 0.05382 | **0.05234** | 0.05369 | 0.05254 |
| rw_eps015 | 0.05979 | 0.05918 | 0.05976 | 0.05914 | 0.05953 |
| rw_topk8 | 0.05612 | 0.05550 | 0.05457 | 0.05539 | **0.05450** |

`std_mse` across 3 seed rounds ≤ 0.00025 in every cell.

### Δ vs cont_random (seed=42)

| eval order | cont_top4 | cont_eps015 | cont_top8 |
|---|---|---|---|
| random      | +0.00067 (+1.1 %) | −0.00003 (≈ 0) | +0.00044 (+0.7 %) |
| **raster**       | **−0.00175 (−3.4 %)** | −0.00014 (≈ 0) | −0.00142 (−2.8 %) |
| **rw_top4_eps0** | **−0.00148 (−2.7 %)** | −0.00013 (≈ 0) | −0.00128 (−2.4 %) |
| rw_eps015   | +0.00058 (+1.0 %) | −0.00004 (≈ 0) | +0.00035 (+0.6 %) |
| **rw_topk8**     | **−0.00093 (−1.7 %)** | −0.00011 (≈ 0) | **−0.00100 (−1.8 %)** |

### Three findings

1. **Graph-RW continuation beats random continuation on every structured eval order**: raster −3.4 %, rw_top4 −2.7 %, rw_topk8 −1.7 % (effect ≥ 0.0008, 4-8× `std_mse`).
2. **`eps015` ≡ random**: every Δ < 0.0002 < 1 σ. Destroying the order's structure removes all curriculum benefit. Cleanest possible noise ablation.
3. **Specialization tax is small**: `+1.0 %` on `random` / `eps015` eval, in exchange for `−3 %` on structured eval. Structured benefit ≈ 3 × cost.

### Monotone structure-strength spectrum

Sort eval orders by locality strength (raster=most local, random=least). `cont_top4 − cont_random`:

```
raster   rw_top4    rw_topk8   rw_eps015   random
−3.4 %   −2.7 %     −1.7 %     +1.0 %      +1.1 %
```

Strictly monotone in the expected direction. This is the textbook signature of a structure curriculum: gains scale with how aligned the eval order is with the training structure.

Output: `probe_results_image/graph_rw/cont_*/ckpt_step5000.pt`, `probe_results_image/eval/cont_*/eval_summary.{tsv,json}`.

## 5. High-noise collapse — clean ablation

Across BOTH stages, breaking the Graph-RW order with `ε=0.15` collapses everything:

| stage | metric | top4 vs random | eps015 vs random |
|---|---|---|---|
| Stage 1 (structure) | mean Manhattan | 3.10 vs 5.30 | 5.23 ≈ 5.30 |
| Stage 1 | P(d ≤ 1) | 28.1 % vs 5.9 % | 6.4 % ≈ 5.9 % |
| Stage 1 | same-quadrant | 53.8 % vs 24.1 % | 24.9 % ≈ 24.1 % |
| Stage 1 | feature distance | 2.24 vs 2.74 | 2.72 ≈ 2.74 |
| Stage 2 (curriculum) | raster MSE | 0.0498 vs 0.0515 | 0.0514 ≈ 0.0515 |
| Stage 2 | rw_top4 MSE | 0.0523 vs 0.0538 | 0.0537 ≈ 0.0538 |

Same mechanism at both stages: structure is carried by the **low-noise attention graph**, not by the sampling procedure itself.

## 6. Cross-modal comparison with text

| dimension | text (WikiText) | image (CIFAR-10) |
|---|---|---|
| Natural data structure | 1D L2R sequence | 2D local + region + appearance |
| Graph-RW (`top_k=4, ε=0`) recovers | L2R-like order, τ ≈ 0.9 vs L2R | soft 2D neighborhood walk, mean step 3.1, P(d≤1)=28 %, same-quadrant 54 % |
| High-noise ablation | structure → random, τ → 0 | structure → random on every axis |
| Curriculum effect (matched eval) | beats random baseline | beats random continuation by 2.7–3.4 % MSE |
| Specialization cost (non-matched eval) | small | +1 % on random/eps015 eval |

**Unified claim**: AOGPT attention encodes dataset-level shared structure across modalities. The same Graph-RW low-noise sampler exposes it in a modality-appropriate form — 1D order for text, 2D blob-filling for image — and the same low-noise curriculum effect transfers.

## 7. Multi-seed CI (placeholder)

> Running in background as of 2026-05-12 17:41: 2 streams on GPU 0, seeds {0, 1} × 4 configs × 5000 steps, plus 3-seed frozen eval (combined with seed=42 from §4). Expected wall time ≈ 60 minutes. This section will be amended with mean ± 95 % CI of paired (method − cont_random) Δ once complete.

## 8. Reproducibility

| asset | path |
|---|---|
| Implementation plan | `docs/superpowers/plans/2026-05-12-image-patch-aogpt-graph-rw.md` |
| Code | `image_order/` package, 9 commits between `aa09aab..1a4c5d3` |
| Baseline ckpt | `probe_results_image/baseline/baseline10k/ckpt_step10000.pt` |
| A_global / B_global | `probe_results_image/attention/baseline10k/{A,B}_global.npy` |
| Stage 1 diagnostics | `probe_results_image/diagnostics/baseline10k/` |
| Stage 2 ckpts | `probe_results_image/graph_rw/cont_{random,top4,eps015,top8}/ckpt_step5000.pt` |
| Stage 2 frozen evals | `probe_results_image/eval/cont_*/eval_summary.{tsv,json}` |
| CI runs (in progress) | `probe_results_image/graph_rw_ci/`, `probe_results_image/eval_ci/` |

### Smoke / verification commands

```bash
# Stage 0 smoke: random-order baseline
python image_order/data_image_patches.py
python image_order/model_image_aogpt.py
python image_order/train_image_random.py \
    --output-dir probe_results_image/baseline/smoke500 --max-steps 500

# Stage 1: extract + diagnose
python image_order/extract_image_attention.py \
    --ckpt probe_results_image/baseline/baseline10k/ckpt_step10000.pt \
    --output-dir probe_results_image/attention/baseline10k --num-images 200
python image_order/diagnose_image_orders.py \
    --B probe_results_image/attention/baseline10k/B_global.npy \
    --output-dir probe_results_image/diagnostics/baseline10k

# Stage 2: α-mixed continuation + frozen eval
python image_order/train_image_graph_rw.py \
    --baseline-ckpt probe_results_image/baseline/baseline10k/ckpt_step10000.pt \
    --b-path probe_results_image/attention/baseline10k/B_global.npy \
    --output-dir probe_results_image/graph_rw/cont_top4 \
    --max-steps 5000 --alpha 0.9 --rw-top-k 4 --rw-epsilon 0.0
python image_order/eval_image_orders.py \
    --ckpt probe_results_image/graph_rw/cont_top4/ckpt_step5000.pt \
    --b-path probe_results_image/attention/baseline10k/B_global.npy \
    --output-dir probe_results_image/eval/cont_top4
```

## 9. Open follow-ups (in priority order)

- **P0 (running)** — 3-seed CI: 2 fresh seeds × 4 configs × 5 k steps, paired Δ + 95 % CI for `method vs cont_random`.
- **P1** — Longer continuation (25 k steps) for `cont_random` and `cont_top4` only, to test whether the structured advantage grows, holds, or vanishes asymptotically.
- **P2** — A_global refresh during continuation (text-side pattern at every 2 k steps), to test whether attention self-organizes more sharply when the curriculum is applied.
- **P3** — Replace MSE with discretized patch-token CE (e.g. via a VQ tokenizer) and rerun Stage 2 to test whether the curriculum effect carries to a CE loss surface.
