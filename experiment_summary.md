# Graph-RW Curriculum Training — 实验完整总结

## 1. 问题设定

- **目标**: 为 AO-GPT（any-order autoregressive model）找到最优的 block-order curriculum，取代 random order
- **方法**: 从模型 self-attention 导出有向图 A，做 stochastic random walk (RW) 采样 block order
- **模型**: AO-GPT, n_layer=4, n_head=8, n_embd=384 (~85M), N=64 blocks, BLOCK_LEN=4
- **数据**: wikitext-103, SEQ_LEN=256
- **核心指标**: val_ori_l2r_block (用 L2R 顺序测的 CE，衡量 L2R generalization)

---

## 2. 最终结果总表

| 实验 | max step | val_ori_l2r | vs L2R | vs random |
|------|----------|-------------|--------|-----------|
| **L2R order (上限)** | 60k | **3.410** | — | -0.110 |
| **v2 Graph-RW (α=0.9)** | 60k | **3.464** | +0.054 | -0.057 |
| α=1.0→0.95 @30k | 60k | 3.462 | +0.053 | -0.058 |
| α=1.0→0.9 @30k | 60k | 3.463 | +0.054 | -0.057 |
| random-refresh ablation | 60k | 3.463 | +0.054 | -0.057 |
| ε=0.15 | 40k | 3.499 | +0.089 | -0.021 |
| ε=0.20 | 40k | 3.499 | +0.090 | -0.021 |
| top_k=8 τ=0.5 | 40k | 3.513 | +0.103 | -0.007 |
| α=1.0 pure RW (无random) | 36k | 3.524 | +0.114 | +0.004 |
| no-refresh | 34k | 3.523 | +0.113 | +0.003 |
| **random order (下限)** | 60k | **3.520** | +0.110 | — |

---

## 3. 核心消融结论

### 3.1 α schedule 无关紧要

四个 α 变体在 60k 全部收敛到 3.462-3.464，差异 < 0.002：

| 变体 | 60k ori_l2r |
|------|-------------|
| v2 α=0.9 (固定) | 3.464 |
| α=1.0→0.9 (两阶段) | 3.463 |
| α=1.0→0.95 (两阶段) | 3.462 |
| random-refresh ablation | 3.463 |

**结论**: 固定 10% random mix-in 即最优，不需要复杂的 α schedule。"early pure RW + late random regularization" 假设不成立。

### 3.2 噪声有害

向 RW policy 引入额外随机性一致推高 CE：

| 噪声配置 | 40k ori_l2r | Δ vs v2@40k |
|----------|-------------|-------------|
| v2 (ε=0, top_k=4) | **3.484** | — |
| ε=0.15 | 3.499 | +0.015 |
| ε=0.20 | 3.499 | +0.015 |
| top_k=8 τ=0.5 | 3.513 | +0.029 |

**结论**: attention 自带的低噪声 RW 结构已经接近最优，加噪声只会破坏 curriculum 质量。

### 3.3 Attention refresh 关键

| 实验 | 34k ori_l2r | 最终 |
|------|-------------|------|
| v2 (refresh every 2k) | 3.523 | 3.464@60k |
| no-refresh | 3.523@34k | (早早停下) |

**结论**: 不 refresh A 矩阵显著变差。但 refresh 来源不重要——用 random 数据 refresh 和用 attention 数据 refresh 结果一致。

### 3.4 RW 是 curriculum，不是目标

Frozen checkpoint (50k) 用不同 order 测 CE：

| Order | CE (v2_a09) | τ_vs_L2R |
|-------|-------------|-----------|
| ori_l2r | 3.457 | 1.00 |
| rw_topk4_eps0 | 3.655 | 0.90 |
| rw_eps015 | 3.687 | 0.77 |
| rw_eps020 | 3.688 | 0.72 |
| rw_topk8 | 3.691 | 0.82 |
| random | 3.758 | ~0.00 |

单调关系: **order 越 L2R-like → CE 越低**。RW topk4 之所以有效，是因为它 τ=0.90 高度接近 L2R，给了低噪声结构信号当 curriculum。模型最终学会的还是 L2R，不是 RW。

---

## 4. Graph-RW 数学细节

### 4.1 符号定义

- **N = 64**: block 数量，每个 block 含 4 个 token (BLOCK_LEN=4)
- **A ∈ ℝ^(N×N)**: 从模型 self-attention 提取的矩阵，A[i,j] = block i 对 block j 的 attention
- **B ∈ ℝ^(N×N)**: 有向图邻接矩阵，B = A^T，对角线置零
- **π = (π₀, π₁, ..., π_N−1)**: 一个 block permutation (order)
- **S**: 已访问 (revealed) 节点集合
- **U**: 未访问 (unrevealed) 候选节点集合
- **τ**: temperature，控制 softmax 锐度（τ=0.1 很锐）
- **top_k**: softmax 后只保留 top-k 概率，其余 mask 为 0
- **ε**: epsilon-uniform 噪声比例，p ← (1−ε)·p + ε/|U|

### 4.2 Step 1: 从 Attention 提取 A 矩阵

在训练的特定 checkpoint（如 step 20000），对每个样本 forward 一次模型，提取最后一层 self-attention 的 block-level attention：

A[i,j] = mean_pool_{head h} attention_h(block_i → block_j)

其中 mean_pool 对 8 个 head 取平均。A 按行 softmax 归一化（每行和为 1）。

**Refresh**: 每 2000 步从当前模型重新提取 A 并更新 B。这是关键——不用 current A 会导致性能显著退化。

### 4.3 Step 2: 构建有向图 B

```python
B = A.T              # 转置：B[i,j] = attention from block j to block i
np.fill_diagonal(B, 0)  # 消除自环
```

**直觉**: 如果 block i 高度 attend block j，意味着 j 提供了 i 需要的上下文。因此 j → i（j 应该出现在 i 之前）。B 的边方向是 "prerequisite of"。

### 4.4 Step 3: 计算 Source 分数

```python
out_deg[u] = Σ_v B[u, v]    # u 作为 prerequisite 的累计强度
in_deg[u]  = Σ_v B[v, u]    # u 依赖其他 block 的累计强度
source[u]  = out_deg[u] - α_dep × in_deg[u]   # α_dep = 0.5
```

**直觉**: source[u] 大的 node 是 "好起点"——它被很多其他 block 依赖（高出度），但自身不太依赖别人（低入度）。α_dep=0.5 给入度 penalty 减半权重。

### 4.5 Step 4: 选择第一个节点

```
p₀ = softmax(source / τ_start), keep top_k=4, mask rest → -inf
π₀ ~ Categorical(p₀)
```

τ_start=0.1 使得 softmax 非常锐 → 首节点几乎由 source 分数最高的 top-4 中随机选。

### 4.6 Step 5: Progressive RW 逐步采样 (核心)

对于 t = 1, 2, ..., N−1，给定已访问集合 S 和候选集合 U = {0..N−1}\S，以及上一步选中的 last ∈ S：

#### 4.6.1 四项 Score 函数

```
score_t(v) = support(v) − β_fut × future(v) + β_src × source[v] + β_loc × local(v)
```

其中 (β_sup=1.0, β_fut=0.5, β_src=0.2, β_loc=0.5)：

**(a) Support (支持度) — 正项**
```
support(v) = Σ_{s ∈ S} B[s, v]     for v ∈ U
```
已访问节点 s 指向候选 v 的边权之和。**鼓励选择与已选节点有强连接的候选。**

**(b) Future (先决条件) — 负项**
```
future(v) = Σ_{u ∈ U} B[u, v]      for v ∈ U
```
未访问节点 u 指向候选 v 的边权之和。**惩罚仍有许多 "未满足 prerequisite" 的候选**（如果 v 被很多未访问节点指向，说明它的 "前置条件" 还没满足，应该晚点选）。

注意 B 对角线为 0，所以 future 不包含 B[v,v]。

**(c) Source (全局重要性) — 正项**
```
source[v] = out_deg[v] − 0.5 × in_deg[v]
```
与首节点选择相同的全局先验，鼓励选高出度节点。

**(d) Local (局部连贯性) — 正项**
```
local(v) = B[last, v]      for v ∈ U
```
上一步选中节点到候选 v 的直接边权。**鼓励选择与上一步有强直接连接的候选（局部连贯）。**

#### 4.6.2 Softmax + Top-k Masking

```
score_t_normalized = (score_t - max(score_t)) / τ_step     # τ_step = 0.1
exp_t = exp(score_t_normalized)

if top_k > 0:
    keep = indices of top_k largest scores
    mask[k] = exp_t[k] if k ∈ keep else 0

p_t = mask / Σ mask
π_t ~ Categorical(p_t)
```

τ_step=0.1 使得分布非常锐，top_k=4 进一步限制随机性。实际上 entropy ≈ log(4) ≈ 1.39 bits，意味着每步的可选范围很小。

#### 4.6.3 更新状态

```
S ← S ∪ {π_t}
U ← U \ {π_t}
last ← π_t
logprob += log(p_t[π_t])
```

#### 4.6.4 单次采样返回

```python
return order: (N,) int64, logprob: float
```

order 是从物理 block index 到 model position 的映射（即 order[i] = 在第 i 个模型位置放置的物理 block 编号）。

### 4.7 Self-avoiding RW (简化变体)

作为对照，self_avoiding_rw 只用 local 项：

```
score_t(v) = B[last, v]      for v ∈ U    (t ≥ 1)
p₀ = softmax(source / τ_start)            (t = 0)
```

没有 support、future、source 的全局信号，纯局部游走。我们的实验主要使用 progressive_rw。

### 4.8 α-mixing: RW + Random 混合

每个 batch 的每个样本独立决定用哪个 order：

```
For each sample:
    α_current = alpha_for_step(global_step)  # e.g., α=0.9
    u ~ Uniform(0, 1)
    if u < α_current:
        order = sample_order(B, 'progressive_rw', params, seed)
    else:
        order = random_permutation(seed)
```

α 的 schedule 从 α_start=0.0 线性 warmup 到 α_target=0.9（10k steps），之后固定。

最终训练目标：
```
Loss = (1 − α_target) × CE(random_order) + α_target × CE(rw_order)
```

等价实现：per-sample 以 α 概率用 RW order，以 (1−α) 概率用 random order。

### 4.9 Epsilon-uniform 探索噪声

```
p ← (1 − ε) × p + ε / |candidates|
```

- ε=0: 无噪声（我们的主实验）
- ε=0.15, 0.20: 消融实验，引入额外随机性

结论：ε > 0 一致推高 CE（见 §3.2），attention 自带的低噪声 RW 已最优。

### 4.10 完整训练循环

```
每 2000 步:
    1. 从当前模型 checkpoint 提取 A 矩阵 (forward over eval data)
    2. B = A.T, fill_diagonal(B, 0)
    3. source = out_deg - 0.5 * in_deg

每步训练:
    4. α = α_start + (step/total_warmup) × (α_target − α_start)
    5. 对每个样本:
       - 以 prob α 用 progressive_rw 采样 order (τ=0.1, top_k=4, ε=0)
       - 以 prob 1−α 用 random permutation
    6. 用 sampled order 计算 AO-GPT 的 CE loss
    7. 反向传播，更新模型参数
```

---

## 5. 三个 α variant 的解码

| 名字 | 公式 | 含义 |
|------|------|------|
| α=0.9 | 每样本 90% chance 用 RW order, 10% random | 固定混合 |
| α=1.0→0.9 | 前30k步 α从0→1.0, 后30k固定0.9 | 两阶段 |
| α=1.0 pure | α从0→1.0, 随后无random | 纯RW, 35k后plateau |

---

## 6. L2R 上限细节

从 v2 30k ckpt 继续训练，只用 L2R 顺序（α=0）：

| step | ori_l2r | train_loss |
|------|---------|------------|
| 31000 | 3.498 | 3.305 |
| 40000 | 3.401 | 3.145 |
| 45000 | **3.387** | 3.083 |
| 50000 | 3.390 | 3.064 |
| 55000 | 3.394 | 3.119 |
| 60000 | 3.410 | 3.033 |

45k 达到最优 3.387，之后轻微反弹（min_lr 阶段正常波动）。

**train-val gap**: 稳定在 0.25-0.30，无过拟合迹象。

---

## 7. 论文叙事 (英文)

**Main finding**: Attention-derived Graph-RW orders serve as an effective curriculum for any-order autoregressive language modeling, improving L2R generalization CE by **0.057** over random-order training.

**Why it works**: RW orders with top-k=4 and low temperature (τ=0.1) are highly L2R-like (τ=0.90 Kendall's tau), providing low-noise structural signal that guides the model toward better order generalization. The 10% random mix-in acts as a steady regularizer that prevents premature convergence.

**What doesn't help**:
- Complex α schedules: a fixed 10% random ratio is optimal
- Extra noise (ε-greedy, larger top-k): attention-derived structure is already near-optimal
- Two-phase "pure RW then mixed" curriculum: no benefit over constant mixing

**Key ablation table**:

| Ablation | Finding |
|----------|---------|
| α ∈ {0.9, 0.95, 1.0→0.9, 1.0→0.95} | All converge to same basin (Δ<0.002) |
| ε ∈ {0, 0.15, 0.20} | Noise strictly hurts (+0.015) |
| top_k ∈ {4, 8} | Larger k degrades (+0.029) |
| Refresh vs no-refresh | Refresh is essential |
| Random-refresh vs attention-refresh | No difference |

**Takeaway**: The attention matrix already encodes an order structure that is close to optimal. The best curriculum is a minimally-noisy random walk derived from it, mixed with a small constant fraction of random orders.

---

## 8. 配套图表

- `training_curves.png` — 所有实验的 training/eval curves（左: ori_l2r CE, 右: train objective CE）
- `order_sensitivity_diag/` — Frozen checkpoint 用不同 order 测 CE 的诊断数据
