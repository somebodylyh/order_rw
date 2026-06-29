# Text Adaptive Residual Order Diagnostic

## Goal

判断 text 上是否存在**超出 L2R 的可学习 order residual**。

不是验证 MLP 能不能恢复 L2R；不是继续优化 v3；不是大规模 long training。

核心问题：
1. difficulty/confidence ranking 在 residualize position 后，是否还有跨样本一致结构？
2. 如果有，D 系 policy 能否产生 structured non-L2R deviation？
3. 这种 deviation 是否超过 matched random local-swap control？
4. 如果没有，立即停止 text hidden/difficulty，把重点转 image Phase2 / sudoku positive control。

## Background

当前 text from-0 MLP 线已经证明：
- 5k random warmup 后，global/refreshed B 上的 L2R-recovery signal 很强
- `teacher_tau_vs_l2r ≈ 0.98+`，`MLP rollout_tau ≈ 0.98+`
- `alt_from0_mlp_finetune` 已达到 `ori_l2r ≈ 3.482`
- text 上"恢复 L2R"已经不是问题

现在要问的是：**减掉 L2R/position 主轴之后，hidden state / difficulty / confidence 是否还提供一致的 adaptive residual signal？**

---

## Phase 0 — Residual Signal Gate (before training β)

这是最重要的前置 gate。**不要先训练 D policy。**

### Task 0.1: Select checkpoints

使用已有 text checkpoints：
- `alt_from0_mlp_finetune` @10k（或最接近的已有 ckpt）
- @20k（或最接近的已有 ckpt）
- @28k 或 @30k

不重跑训练。

### Task 0.2: Compute difficulty/confidence signals

对 held-out diagnostic subset 计算每个 block v 的信号：

**A. position:**
- `pos_v = v / N`

**B. difficulty:**
- `CE_v` = current model per-block CE
- normalized CE within sequence/block set

**C. confidence:**
- token entropy 或 block entropy
- `confidence_v = -mean entropy`
- 可选 top1-top2 margin

**D. hidden:**
- `h_v = mean final-layer hidden over block tokens`
- `hidden_norm`
- `cosine to last/selected/global mean`（if available）

注意：
- 这些都只用于 diagnostic
- 不用 validation 更新 β
- 不用未来 checkpoint
- CE/confidence 要标清是在哪个 probe/reveal state 下计算的（order-dependent/endogenous）

### Task 0.3: Residualize L2R position

对每个 signal `s_v`，先去掉 position 主轴：

```
s_v = a * pos_v + b + residual_v
```

或者用 Spearman residual / rank residual：

```
residual_rank(s) = rank(s) after regressing out rank(pos)
```

然后分析 residual 是否有可学习结构。

**指标：**

1. **residual across-sequence consistency:**
   - 每个位置/相对位置的 residual mean/std
   - sequence-to-sequence rank correlation
   - residual PCA / first component explained variance
   - residual sign consistency

2. **residual vs attention:**
   - `corr(residual_CE, attention readiness)`
   - `corr(residual_confidence, C-D+L score)`
   - `corr(residual_hidden, C-D+L score)`

3. **residual SNR:**
   - between-pattern variance / within-sequence variance
   - report a scalar `residual_SNR`

### Phase 0 Gate

**PASS** only if:
- residual signal is not pure per-sequence noise
- residual rank has cross-sequence consistency
- at least one of CE/confidence/hidden residual has interpretable structure
- residual signal is not fully explained by position

**FAIL** if:
- residuals are mostly random after removing L2R
- sequence-to-sequence residual correlation is near zero
- no stable pattern across checkpoints

**If FAIL:**
- stop text hidden/difficulty
- write conclusion: text has strong L2R-recovery signal but little learnable adaptive residual
- recommend pivot to image Phase2 / sudoku positive control

**Output:**
- `probe_results/attention_order_hidden_residual/PHASE0_RESIDUAL_GATE.md`
- `probe_results/attention_order_hidden_residual/residual_metrics.tsv`

---

## Phase 1 — D-Series Policies Only

**Do not run A/B/C as full experiments.**

Reason: A/B/C use `KL(C-D+L teacher || πβ)`, and the C-D+L teacher is near-L2R. There is no gradient pressure to use hidden/CE features → these variants will almost certainly become L2R clones. They may be included only as sanity baselines, not as main experiments.

**Only implement D-series policies if Phase 0 passes.**

### D_easy

Easy-first / ready-first:

```
r_v = s_CDL(v) - μ * CE_norm(v)
```

or with confidence:

```
r_v = s_CDL(v) + μ * confidence_norm(v)
```

Purpose: prefer easier / more confident blocks, but still anchored to attention structure.

### D_hard

Hard-first:

```
r_v = s_CDL(v) + μ * CE_norm(v)
```

Purpose: test whether hard-example ordering gives useful deviation.

Important: D_hard with high μ is likely the most dangerous — it can push order farthest from L2R. Track displacement carefully.

### D_conf_static

Static confidence-based teacher:

```
r_v = s_CDL(v) + μ * confidence_norm(v)
```

where confidence is computed once under a fixed probe state.

Purpose: bridge to confidence-based unmasking without dynamic recomputation.

### D_conf_dynamic

Dynamic confidence teacher:

During rollout, at each step t:
1. reveal current selected set S_t
2. compute model confidence/entropy for remaining blocks U_t
3. pick or score most-confident blocks
4. form a teacher distribution over U_t

This is closer to MaskGIT / MDM-style inference unmasking.

Purpose: make training-time order policy align with model's own inference-time readiness.

This is more expensive; include as optional setting if static confidence looks promising, or implement as a diagnostic-only trajectory first.

### Loss

```
Lβ = w_KL * KL(p_CDL || πβ)
   + λ_rank * KL(p_rank || πβ)
   + optional entropy target
```

But rank must be strong enough to actually leave L2R attractor.

**Suggested grid:**
- `w_KL ∈ {1.0, 0.3}`
- `λ_rank ∈ {1.0, 3.0}`
- `μ ∈ {0.3, 0.7}`

Do not use λ_rank=0.1 as a main setting; it is probably too weak.

---

## Phase 2 — Matched Random Local-Swap Controls

**This is mandatory.**

For every D policy rollout, construct a control: **L2R + random local swaps**

Match the D policy on:
- `tau_vs_l2r` or Kendall distance
- average displacement from L2R
- entropy / unique orders
- local swap rate (if possible)

Purpose: distinguish real adaptive signal from "any same-magnitude deviation from L2R acts as regularization."

If D policy does not outperform / differ meaningfully from matched random local-swap in offline metrics or short continuation, **do not claim adaptive signal.**

**Output:** `probe_results/attention_order_hidden_residual/matched_swap_controls.tsv`

---

## Phase 3 — Dead-Feature Logging

For every trained β, log whether hidden/CE/confidence features are actually used.

**Metrics:**
- first-layer weight norm on attention features
- first-layer weight norm on hidden features
- first-layer weight norm on CE/difficulty features
- first-layer weight norm on confidence features
- gradient norm by feature group (if available)
- ablation: zero hidden/CE/conf features and measure rollout change

If hidden/CE/conf weights collapse to ≈0:
- mark policy as L2R-clone regime
- do not proceed to continuation

**Output:** `probe_results/attention_order_hidden_residual/feature_usage.tsv`

---

## Phase 4 — Offline Rollout Diagnostic

For each D policy and matched-swap control, evaluate rollout on held-out subset.

**Metrics:**
- `tau_vs_l2r`
- `abs_tau`
- entropy
- unique order count
- average displacement from L2R
- local swap rate
- start node distribution
- selected-step vs CE correlation
- selected-step vs confidence correlation
- selected-step vs hidden_norm correlation
- early-selected block CE vs late-selected block CE
- rollout stability across samples/checkpoints
- distance from current MLP_CDL_source_start rollout

**Gate to continuation:**

Only continue if a D policy:
1. is structured, not random
2. is not pure L2R clone
3. uses hidden/CE/conf features nontrivially
4. has interpretable CE/confidence correlation
5. differs from matched random local-swap control

**If no policy passes:** stop text hidden/difficulty.

**Output:** `probe_results/attention_order_hidden_residual/OFFLINE_DIAGNOSTIC_REPORT.md`

---

## Phase 5 — Short Continuation Smoke (only if gate passes)

Only if Phase 4 passes.

Run **at most one** D policy for **1k–2k** continuation. Do not run 30k.

**Recommended starting point:**
- `alt_from0_mlp_finetune` @20k 或 @25k checkpoint
- compare against existing MLP curve over the same step window

**Primary metric:**
- `val_ori_l2r_block`

**Secondary:**
- order entropy
- CE/confidence correlation
- whether performance exceeds matched random-swap control

**If no clear improvement:** do not extend.

**Output:** `probe_results/attention_order_hidden_residual/short_smoke_<policy>/REPORT.md`

---

## Interpretation

### Outcome A: Residual gate fails

**Conclusion:** Text has strong L2R-recovery signal but little stable adaptive residual beyond L2R.

**Action:** Stop text hidden/difficulty. Move to image Phase2 or sudoku.

### Outcome B: Residual exists offline but D policy equals matched random swap

**Conclusion:** Deviation from L2R may just be entropy/noise regularization, not meaningful adaptive signal.

**Action:** Do not claim difficulty-aware order.

### Outcome C: D_conf or D_easy/D_hard passes offline and improves short continuation

**Conclusion:** There is adaptive signal beyond L2R; hidden/confidence/difficulty can improve order policy.

**Action:** Then consider longer run or integration into from-0 alternating.

---

## Notes

- Do not write this as "confirmatory negative." Keep language neutral: minimal diagnostic to test whether adaptive residual exists beyond L2R.
- Do not start training until Phase 0–4 reports are reviewed.
- All output goes to `probe_results/attention_order_hidden_residual/`.
- **No long training. No multi-seed sweep. No v3 baseline comparison.**
