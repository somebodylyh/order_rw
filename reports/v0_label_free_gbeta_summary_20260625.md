# Label-Free L0 Multi-Head g_β Readout & Frozen Hook — Technical Summary

**Date**: 2026-06-25
**Checkpoint**: random_baseline_continuous_jun08_seed2, step 20000
**Model**: 4L/8H/d=384, Wikitext-103 continuous stream

---

## 1. Motivation

Random-reveal AO-GPT L0 attention heads carry strong order signal recoverable as physical L2R / layout path. Prior work relied on single-head diagnostics (e.g., L0H2) — vulnerable to hand-picked oracle criticism. This work builds a **fully label-free, model-frame, multi-head g_β readout** from raw L0 attention maps, and integrates it as a **frozen order provider** into the AO-GPT training loop.

**Boundaries**: Does NOT claim sample-dependent dynamic gate, beat L2R/layout path, or full online co-training.

---

## 2. v0 Pipeline Implementation

| # | Module | Purpose | Tests |
|---|--------|---------|:---:|
| 1 | `batch_readout/l0_strict65.py` | Model-frame strict-65 extraction. Node 0=None, nodes 1..64=content. Zero physical fields. | 24 |
| 2 | `batch_readout/label_free_cdl_teacher.py` | CDL consensus teacher: rollout, standardized margin, destroyed-gap, rank agreement, dynamic weights, soft pairwise Y. | 33 |
| 3 | `analyses/build_l0_dynamic_gbeta_dataset.py` | .npz dataset: B_raw + teacher fields + splits. `del clean_perm` after loading. Forbidden-fields audit. | 15 |
| 4 | `batch_readout/l0_dynamic_gbeta.py` | g_β model: 4-ch normalize (raw/prob/logz/rowz), shared block scorer, shared dynamic gate. Head-permutation equivariant. Gate excludes raw channel. | 27 |
| 5 | `batch_readout/soft_pairwise.py` | Soft pairwise BCE, accuracy (excl. diag/ties), per-head aux loss, entropy floor. | 19 |
| 6 | `batch_readout/train_l0_dynamic_gbeta.py` | Training loop. Checkpoint selection: val_loss_final only. No physical fields. | 13 |
| 7 | `analyses/eval_l0_dynamic_gbeta.py` | Sanity: destroyed / remove-top-α / uniform-α. | 10 |
| 8 | `analyses/characterize_l0_dynamic_gbeta.py` | Post-hoc τ vs layout_path / identity / L2R. **Only script that reads physical metadata.** | — |
| 9 | `batch_readout/frozen_gbeta_hook.py` | Model-frame frozen order provider (batch_mean_probes=4). Oracle baselines labeled separately. | 6 |
| **Total** | **9 modules** | | **180** |

**Label-free audit**: Method path has zero forbidden physical field usage (verified by `inspect.signature` + `grep`).

---

## 3. v0 Pretrain Results (M=2000, Layout A)

| Epoch | Train loss | Val loss | Train acc | Val acc |
|------|:---:|:---:|:---:|:---:|
| 1 | 0.627 | 0.597 | 0.678 | 0.709 |
| 40 | 0.286 | 0.297 | 0.996 | 0.997 |

**Sanity checks (val, N=200)**:

| Condition | Acc | Δ vs Normal |
|-----------|:---:|:---:|
| Normal | 0.997 | — |
| Destroyed | 0.519 | **−0.478** |
| Remove-top-α | 0.995 | −0.001 |
| Uniform-α | 0.974 | −0.023 |

**Gate**: H_α ≈ 1.56 (effective heads ≈ 4.8/8). Learned α: H2=0.27, H7=0.29, H1=0.16, H0=0.16, H6=0.12. Student uses auxiliary heads (H7/H0/H6) beyond teacher-preferred H1/H2.

**Conclusion**: g_β reads real attention structure (destroyed drop −0.478), not single-head oracle (remove-top drop −0.001), learned mixture > uniform (+0.023). Gate is stable global mixture, not sample-dependent dynamic.

---

## 4. Post-hoc Characterization (Step 8)

**Layout A, val N=200**:

| Condition | τ vs teacher | τ vs layout_path | τ vs identity | τ vs L2R |
|------|:---:|:---:|:---:|:---:|
| Normal | +0.984 | **+0.993** | +0.002 | +0.993 |
| Destroyed | +0.046 | +0.046 | −0.285 | +0.046 |

g_β label-free recovers the layout path (τ=+0.993), not identity successor (τ=+0.002), destroyed collapses to random (τ=+0.046). Supports: *semantically shaped, positionally compiled layout path*.

---

## 5. Cross-Layout Replication (Step 9)

Two orthogonal layouts: τ(Layout A path, Layout B path) ≈ **+0.006**.

| Metric | Layout A (seed=123) | Layout B (seed=42) |
|--------|:---:|:---:|
| Normal acc | 0.997 | 0.988 |
| Destroyed Δacc | −0.478 | −0.456 |
| Remove-top Δacc | −0.001 | −0.005 |
| Learned − Uniform | +0.023 | +0.018 |
| **τ vs own layout** | **+0.993** | **+0.980** |
| τ vs other layout | +0.074 | +0.053 |
| τ vs identity | +0.002 | −0.014 |

> *Across two nearly orthogonal fixed layouts, the label-free g_β readout recovers each layout's own path with τ > 0.98, while showing near-zero agreement with the other layout's path and identity order. This rules out both a trivial identity successor and a single-layout artifact.*

---

## 6. Frozen Hook Design (Step 10)

**Key interface constraint**: hook operates entirely in model-frame.

```
g_β(B) → scores → sigma_model → expand → model-frame token order
```

**NO inv_perm / clean_perm / block_perm** in method path. Physical mapping only in oracle baselines (explicitly labeled) or post-hoc characterization.

**Comparison groups**:

| Policy | Type | Physical fields? |
|--------|------|:---:|
| random | label-free baseline | No |
| layout_path | oracle | Yes (labeled) |
| single_head_CDL | diagnostic | No |
| **frozen_gbeta** | **method** | **No** |
| gbeta_destroyed | sanity | No |
| gbeta_uniform | ablation | No |

---

## 7. Critical Fix: Batch-Mean Probe Mismatch

**Root cause**: g_β trained on batch_mean=16 B matrices. Hook originally used single-probe B → gate collapsed to uniform (H_α=2.07), τ_vs_bp dropped to 0.42.

**Fix**: `batch_mean_probes=4` — 4 probe forward passes, average B matrices before g_β.

| | Before fix | After fix |
|---|:---:|:---:|
| Hook τ_vs_bp | 0.42 | **0.91** |
| Gate H_α | 2.07 (uniform) | 1.74 |
| Refresh time | 8ms | 27ms |
| frozen vs random (5k) | ≈ same | **−0.072** |

---

## 8. 20k Policy Comparison

**Config**: lr=3e-5, batch=8, batch_mean_probes=4, from step 20k checkpoint.

| Policy | val_l2r | Δrandom | τ_vs_bp |
|--------|:---:|:---:|:---:|
| layout_path | 3.283 | −0.089 | — |
| **frozen_gbeta** | **3.328** | **−0.045** | 0.89 |
| gbeta_uniform | 3.340 | −0.033 | 0.81 |
| single_head_CDL | 3.342 | −0.031 | — |
| gbeta_destroyed | 3.376 | +0.003 | — |
| random | 3.373 | — | — |

**All four core criteria passed**: frozen > random ✓, destroyed ≈ random ✓, frozen > uniform ✓, τ stable 0.89 ± 0.04.

---

## 9. 60k Overnight Results

**Config**: lr=1e-3→1e-4 cosine, batch=64, batch_mean_probes=4, fixed eval set (5 batches). Baseline val_l2r at step 20k = **3.967**.

| Policy | Final val_l2r | Δbaseline | τ_vs_bp |
|--------|:---:|:---:|:---:|
| **frozen_gbeta** | **3.222** | −0.746 | 0.73 |
| gbeta_uniform | 3.231 | −0.736 | 0.67 |
| gbeta_destroyed | 3.309 | −0.658 | — |

**Val_l2r curve (frozen_gbeta)**: 3.967 → 3.567 → 3.500 → 3.443 → 3.379 → 3.333 → 3.291 → 3.256 → 3.238 → 3.227 → 3.222

**Comparison context**: AR baseline (random-reveal, 60k from step 0) reached train_loss ≈ 3.61. Ori-L2R continuous baseline reached ~3.34 at 33k. Our frozen_gbeta reaches 3.222 (80k total: 20k baseline + 60k hook), which is in the expected range — neither anomalously low nor suspiciously high. The −0.746 improvement from the 20k baseline reflects 60k additional training steps on the data, with frozen_gbeta orders providing a small but consistent benefit over destroyed/uniform.

**Key observations**:
- τ_vs_bp drops from 0.89 (20k) → 0.73 (60k): attention distribution drift
- Frozen readout benefit is real but modest (frozen vs destroyed gap = 0.087)
- Sorting correct: frozen < uniform < destroyed
- Gap frozen vs uniform is small (0.009) — learned gate contribution marginal at 60k

---

## 10. CDL Teacher Diagnostic Clarification

**batch_mean_size is the dominant factor**, not text chunk selection:

| Probes | CDL_H2 τ | Gate H_α |
|:---:|:---:|:---:|
| 1 | **+0.08** | — |
| 4 | +0.65 | 1.74 |
| 16 | **+1.00** | 1.56 |

Text chunks (pretrain vs random stream) do not affect τ given sufficient probes. On pretrain dataset (batch_mean=16): L0H2=+1.00, consensus=+0.99, g_beta=+0.99 — all matching historical results. CDL teacher signal is **stable across checkpoints** (20k/55k/60k), not degrading.

---

## 11. Figure Proposals

### Figure 1: Pipeline Overview
Flowchart: raw L0 attn → strict-65 extraction → CDL teacher → soft pairwise Y → g_β (scorer + gate) → predicted order → frozen hook → AO-GPT training. Annotated: "label-free", "model-frame", "physical only post-hoc".

### Figure 2: Sanity Bar Plot (Dual Layout)
Grouped bars: normal / destroyed / remove-top / uniform accuracy. Two groups (Layout A, Layout B). Shows: destroyed collapses, remove-top barely drops, uniform slightly below learned.

### Figure 3: Cross-Layout τ Heatmap
2×2 matrix: rows = model trained on Layout A/B, cols = evaluated on Layout A/B path. Values: 0.993, 0.074, 0.053, 0.980. Diagonal >> off-diagonal.

### Figure 4: Probe Ablation
x = batch_mean_probes (1, 4, 16). y1 = τ_vs_bp (line). y2 = gate H_α (line). Shows: 1 probe fails, 4 probes recovers, 16 probes optimal.

### Figure 5: 20k Policy Comparison
Horizontal bar chart: 6 policies ranked by val_l2r. Color: baseline (grey), oracle (blue), method (green), sanity (red).

### Figure 6: 60k Drift Diagnostic
x = training step. y1 = τ_vs_bp (declining). y2 = val_l2r (declining). Shows: drift over 60k steps.

### Figure 7: Teacher Diagnostic Comparison
Grouped bars: L0H2 / L0H7 / consensus / g_beta τ on pretrain dataset vs training stream. Shows: teacher perfect on clean data, g_beta more robust on noisy data.

---

## 12. Conclusions

### Completed
- Label-free, model-frame, multi-head g_β readout
- Cross-layout replication (τ > 0.98 on own layout, τ ≈ 0.05 on other)
- Frozen hook integration with batch-mean fix
- 20k training benefit over random (Δ −0.045)
- 60k confirms benefit with drift limitation

### Not Claimed
- Sample-dependent dynamic gate (α_std is tiny — stable global mixture)
- Beats L2R / layout path
- Full online co-training

### Paper-Style Summary
> We present a fully label-free pipeline that learns a multi-head readout (g_β) from raw L0 attention maps of a random-reveal AO-GPT model. The readout recovers each layout's specific path (τ > 0.98), is not reducible to a single head (remove-top-α has negligible impact), and depends on real attention structure (destroyed τ ≈ 0). When integrated as a frozen order provider into AO-GPT training, g_β yields consistent improvement over random order at 20k steps, though long-run benefit is limited by attention distribution drift. All method paths are strictly free of physical-coordinate leakage.

### 中文汇报版
v0 成功构建 label-free multi-head g_β readout。双 layout 验证跨 layout 可复现（τ>0.98 vs own, τ≈0.05 vs other）。Frozen hook 接入训练 loop，20k 优于 random（Δ−0.045），60k 确认收益但暴露 frozen readout 分布漂移限制（τ 0.89→0.73）。Gate 学到稳定 global head mixture，非 sample-dependent dynamic。下一步：multi-checkpoint / drift-robust g_β pretrain。
