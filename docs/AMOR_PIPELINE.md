# Anchor-Mean Order Router (AMOR) Pipeline

> **Status**: v1 locked (2026-07-07). PG-only, tau=0.05, teacher-refresh disabled.
> **Reference**: chenhe-baseline-sole-reference, anchor-router-v1

---

## 0. Context & Notation

| Symbol | Meaning | Value |
|--------|---------|-------|
| `N` | Content blocks per sequence | 64 |
| `BLOCK_LEN` | Tokens per block | 4 |
| `SEQ_LEN` | Total tokens | 256 |
| `B` | Microbatch size | 64 |
| `A` | Anchor subset size | 16 |
| `gβ` | NodewiseReadout: B(64×64) → scores(64) | ~0.3M params |
| `σ` | Block reveal order (permutation of 0..63) | `argsort(scores, descending)` |
| `σ_cache` | Lag-1 deployment order, persists across PG updates | `_last_sigma` |

---

## 1. 预训练阶段 (gβ Pretrain)

### 1.1 Backbone

在 WikiText-103 上训练一个 4-layer、8-head、d=384 的 AOGPT，**10k steps**，每步用随机 block order（`permute_mode='block'`, `permute_seed=42`）。产出 ckpt：`parent_random_10k/ckpt.pt`。

选择 10k 而非 0-step random init 的理由：(a) 10k 的 attention 已有稳定的 block-level structure（τ ≈ 1.0）；(b) 比 20k 更"通用"，gβ 读的是 early-training 信号而非过拟合到特定 order 分布的信号。

### 1.2 Stage A：Head Selection（无监督）

从 800 个随机 text window 中，对每个 (layer, head) pair：
- 做 `n_reveal=8` 次 random block-order probe（strict-65 协议）
- 每次 probe 提取 content B (64×64)，取 batch-mean over 4 probes/sample
- 对 800 个窗口打分：split-half reproducibility + cross-reveal agreement + pairwise confidence + triadic cycle consistency

选出的 canonical head：**L1H6**（layer 1, head 6）。

### 1.3 Stage B：NodewiseReadout Training

**数据构造**：从 backbone 抽取 24,000 个 per-sample B (64×64)，按 `batch_mean_size=16` 分组 → 1,500 个 batch-mean B (1,64,64)。

为什么是 **batch-mean of 16** 而非 per-sample？单个 64-token window 的 attention graph 噪声太大，不足以承载稳定的 block-order 结构。16 个独立 window 取平均后，噪声抵消、结构浮现。这是整个 pipeline 的核心分布假设。

**Teacher**：对每个 batch-mean B 运行 CDL（C−D+L）greedy rollout → hard pairwise target T (64×64)，其中 `T[i,j] = 1` 表示 block i 应先于 block j。

CDL rollout 每步选 `argmax_v [C(v) − D(v) + L(v)]`：
- `C(v)` = 已选节点到 v 的 mean attention
- `D(v)` = 剩余节点到 v 的 mean attention（惩罚被依赖的节点）
- `L(v)` = 上一个被选节点到 v 的 attention

**Loss**：pairwise binary cross-entropy with logits
```python
scores = readout(B)          # (batch, 64)
sd = scores[:, :, None] - scores[:, None, :]  # (batch, 64, 64)
loss = BCEWithLogits(sd, T)  # push sigmoid(sd[i,j]) → T[i,j]
```

**Architecture**：`NodewiseReadout` — 对每个 block i，concat outgoing row `B[i,:]` + incoming column `B[:,i]` → 128-dim 特征，过 2-layer Transformer encoder（d=64, 4 heads, GELU）→ Linear → scalar score。~0.3M 参数。

**结果**：val pairwise accuracy = **98.44%**。产出 ckpt：`gbeta_bm16_g1500/g_beta_best.pt`。

### 1.4 关键设计选择

| 选择 | 理由 |
|------|------|
| 无 ground-truth order | CDL teacher 是纯 heuristic（attention-only），不依赖任何外部 label |
| gβ 读 batch-mean B 而非 per-sample B | 部署时也是 batch-mean → 分布匹配 |
| gβ 输出 scores 而非直接输出 order | scores 可微、可被 PG 更新；argsort 将 scores 转为离散 order |
| 小 MLP（~0.3M） | 防止 gβ 本身成为 bottleneck；快（< 1ms 推理） |

---

## 2. 训练阶段（Anchor-Mean 路由部署）

### 2.1 初始化

从 `jul3_gbeta_frozen_20k` fork（parent backbone 续训至 20k step 的 random-order ckpt）。

每个 training step：

```
Step t:
  ┌─ [1] 取 microbatch X (64, 256)
  ├─ [2] Anchor forward: X[:16] 在 σ_cache 下前向 → 提取 L1H6 attention
  │        → build_none_separated_B → per-sample B (16, 64, 64)
  │        → mean(dim=0) → B_anchor_mean (1, 64, 64)
  ├─ [3] gβ(B_anchor_mean) → scores (64) → argsort → σ_batch (64,)
  ├─ [4] σ_batch 广播到全部 64 行 → expand_block_orders_to_token_orders
  ├─ [5] Full forward + backward (64 samples under σ_batch token order)
  ├─ [6] optimizer.step() (backbone only)
  └─ [7] if step % 100 == 0: PG update gβ (decoupled, no backbone grad)
```

### 2.2 σ_cache 机制

**因果约束**：当前 step 的 B 不能决定当前 step 的 order（否则循环依赖）。因此 B 必须在 **上一步的 deployment order**（σ_cache）下读出。

```
Step 0 (bootstrap):
  _last_sigma = None → random-reveal 探针 (4 probes × 16 anchor)
  → B_anchor_mean → gβ → σ_0 → 存入 _last_sigma

Step 1+ (steady state):
  用 _last_sigma (σ_{t-1}) 做 anchor forward → 提取 B
  → B_anchor_mean → gβ → σ_t → 更新 _last_sigma
```

**关键**：`_last_sigma` 在 `invalidate_cache()` 时不被清除。PG update 后 gβ 参数变了、`_train_sigma` 被清掉，但 `_last_sigma` 保持不变——部署 lag 不会被 PG 打断。

Eval 时：一个 step 的所有 eval batch 共享同一个 σ（cached by `global_step`），O(eval_iters) 次 strict65 变成 O(1) 次。

### 2.3 为什么 A=16 匹配 gβ 预训练分布

gβ 预训练时输入是 `batch_mean_size=16` 的 B_anchor_mean。部署时 `A=16` 恰好产生同样 batch-size 的 mean B。这不是巧合——是刻意对齐的。

更大的 A 在统计上更稳定但成本更高（anchor forward 与 A 成正比）。A=16 是"刚好匹配预训练分布"的最小值。

### 2.4 为什么 batch-mean B 而非 mean scores

两个选项：
- **(A) Mean B then score**: `mean(B_i) → gβ → scores` ← 我们用的
- **(B) Score then mean**: `gβ(B_i) → mean(scores_i)` ← 不做

选项 (A) 保留了 block 间的交互结构（B 矩阵的行/列 pattern），gβ 的 Transformer encoder 可以利用这些交互。选项 (B) 把每个 sample 的 scores 独立算出再平均，丢失了跨样本的结构信息。

### 2.5 为什么没有 EMA

第一版保持朴素。EMA 引入了额外的超参（decay rate）和 lag，对于验证"信号是否存在"的阶段是多余的。如果未来需要减少 noise，可以在 B_anchor_mean 层面加 EMA。

---

## 3. gβ Update：PG-only (Periodic Policy Gradient)

### 3.1 机制

每 100 步，用 RL-style 梯度更新 gβ（不影响 backbone）：

```
1. scores = gβ(B_anchor_mean)
2. 从 Plackett-Luce(scores/τ) 采样 k=4 个 candidate orders
3. 对每个 candidate + greedy order 做 no-grad full-batch forward → NLL loss
4. Leave-one-out advantage: A_i = mean(loss_{-i}) - loss_i
5. PG loss = -mean(A_i * log P_π(candidate_i) / N)
6. backward → clip grad → optimizer.step()
```

### 3.2 参数

| 参数 | 值 | 理由 |
|------|-----|------|
| `gbeta_pg_start_iter` | 20,000 | 从 fork point 开始，backbone 已有稳定 attention |
| `gbeta_pg_update_every` | 100 | 足够频繁但不至于 overhead 爆炸（~10% steps 有 PG spike） |
| `gbeta_pg_tau` | **0.05** | tau sweep 确认 winner，极冷 → candidates 极近 greedy |
| `gbeta_pg_k` | 4 | 太小无 diversity，太大 cost 线性增长 |
| `gbeta_pg_lr` | 3e-5 | 小 lr，gβ 已有 98.44% pretrain acc，只需微调 |
| `gbeta_pg_adv_clip` | 0.1 | 限制单步 PG 更新幅度 |

### 3.3 Teacher Refresh（已禁用）

v1 joint arm 包含 CDL teacher refresh（每 1000 步用 σ_cache-read B + BC nudge），但实验发现它 **hurt own-order val**：teacher Δ ≈ 0.5 完全压倒 PG Δ ≈ 0.01，把 gβ 推向 C−D+L heuristic 方向而非 NLL-optimal 方向。val 在第一个 refresh（step 21,000）时跳 +0.12。

**结论**：PG-only 是当前 best arm。Teacher refresh 保留作为消融选项。

---

## 4. 开销分析

### 4.1 Per-step overhead

正常 backbone forward（B=64, seq=256, 16,384 tokens）约占总 iter time 的 40–45%。

Anchor forward（A=16, seq=256, 4,096 tokens = 25% of full）加 attention extraction (NumPy CPU) + gβ readout + argsort：

| 组件 | 开销 |
|------|------|
| Anchor forward pass | ~10–12% |
| CPU B extraction + gβ + argsort | ~4–6% |
| **总 overhead** | **~16%** (185ms vs 159ms) |
| PG spike (每 100 步) | ~250ms（额外 4 次 full-forward） |

对比旧方案（non-anchor，每次读全部 64 samples 的 B）：anchor 把 graph-estimation 成本从 ~64 samples → 16 samples，节省 **4×**。

### 4.2 与完全 probe 全 batch 的对比

| 方案 | Graph-estimation forward 成本 | 相对 |
|------|------------------------------|------|
| Full probe (64-sample B extraction) | ~64 个 per-sample forward → 极慢 | 4× anchor |
| **Anchor (A=16)** | 16-sample forward | **1× (baseline)** |
| No probe (random order) | 0 | 0.84× anchor |

---

## 5. 结果总结

### 5.1 主实验结果

所有实验从 `jul3_gbeta_frozen_20k` fork，训练至 50k steps。

| 实验 | gβ | val loss @ 50k | Δ vs frozen |
|------|-----|---------------|-------------|
| Frozen anchor | frozen, σ_cache deploy | ~3.73 | — |
| Joint (PG + teacher) | PG+BC both on | ❌ val 跳 +0.12 | 反效 |
| **PG-only** (tau=0.05) | PG only, teacher disabled | **3.4678** | 🏆 **best** |

### 5.2 Tau Sweep（PG-only）

| tau | val loss @ 50k | best_gap 范围 | 结论 |
|-----|----------------|---------------|------|
| **0.05** | **3.4678** | ~0 | 🏆 极冷→接近 greedy，但最稳定 |
| 0.10 | 3.4854 | −0.047 to 0 | 能找到好 candidate，但不转化 |
| 0.20 | 3.5246 | −0.030 to 0 | 更差 |
| 0.50 | 3.5635 | −0.032 to 0 | 更差 |
| 1.00 | 3.5380 | 全正 | 纯噪声，比 greedy 还差 |

**tau 越低越好，单调。** 高温 PL 采样能找到 loss 更低的 candidate，但 PG 梯度无法有效利用这些发现——可能是因为这些"好"candidate 的 log-prob 在高温下太小、梯度信号太弱。

### 5.3 关键 metric：`gbeta_pg/best_gap`

`best_gap = min(loss_candidates) - loss_greedy`。负值表示 PL 采样找到了优于当前 gβ greedy 的 order。

- tau=0.05: 大部分步 best_gap ≈ 0（candidates ≈ greedy）
- tau=0.10: best_gap 可到 −0.047，说明存在优于 greedy 的 order
- 但这些"好发现"未转化为 lower val loss → 当前 PG 机制无法 capitalize 探索的收益

---

## 6. 已执行消融

| 消融 | 状态 | 结论 |
|------|------|------|
| Random order baseline | ✅ | 确认 gβ order 优于 random |
| Frozen vs PG-updated gβ | ✅ | PG-only > frozen (~0.26 nat) |
| σ_cache deploy vs random probe | ✅ | deployment mode 闭环读取更好 |
| Teacher refresh on/off | ✅ | OFF 更好（teacher hurts own-order val） |
| Tau sweep {0.05, 0.10, 0.20, 0.50, 1.00} | ✅ | 0.05 winner，单调 |
| A=16 anchor-size sensitivity | 待做 | A ∈ {4, 8, 32}？ |
| EMA on B_anchor_mean | 待做 | 可能平滑噪声 |
| Per-sample score-mean vs batch-mean B | 待做 | 验证 (A) > (B) |

---

## 7. 文件索引

| 文件 | 用途 |
|------|------|
| `orderhead_v3/gbeta_provider.py` | 部署时 block-order provider（σ_cache、anchor、deployment mode） |
| `orderhead_v3/readouts.py` | NodewiseReadout 架构（2-layer Transformer encoder） |
| `orderhead_v3/periodic_pg.py` | Plackett-Luce 采样 + LOO advantage |
| `orderhead_v3/teacher_refresh.py` | CDL teacher BC refresh（已禁用但保留） |
| `orderhead_v3/cdl_teacher.py` | CDL teacher 派生 + pairwise BCE loss |
| `orderhead_v3/none_separated_block_graph.py` | `build_none_separated_B` — attention → block graph |
| `orderhead_v3/l0_strict65.py` | Strict-65 B extraction 协议 |
| `orderhead_v3/constants.py` | N=64, BLOCK_LEN=4, SEQ_LEN=256 |
| `gbeta_cdl_pretrain.py` | Stage A+B 预训练（head selection + readout training） |
| `train.py:L7654-7680` | `GBetaFrozenOrder` dispatch + `_forward_with_explicit_block_orders` |
| `train.py:L7758-7781` | `_gbeta_provider` 构造 |
| `train.py:L12131-12196` | `_run_gbeta_periodic_pg` |
| `train.py:L12199-12243` | `_run_gbeta_teacher_refresh`（已禁用） |
| `config/.../gbeta_anchor_joint_pgonly_from20k.py` | 🏆 生产配置（PG-only, tau=0.05） |
| `config/.../gbeta_anchor_frozen_from20k.py` | Frozen anchor 对照 |
| `config/.../gbeta_anchor_joint_from20k.py` | Joint (PG + teacher) 对照 |
| `out/rerun/gbeta_bm16_g1500/` | gβ pretrain 产出（L1H6, val_acc=98.44%） |
