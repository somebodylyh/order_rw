# Frozen Old Step-Policy Prior + AO-GPT Local-Utility Reranker

## Context

CrossAttentionOrderNetwork 已经是 step-wise policy：
old_on(A, visited_mask, last_node) → next logits  (B, N)
它不是 one-shot ON。问题不是 "one-shot → sequential"，而是旧 ON 的 prefix context 太弱（仅靠 cross-attention over revealed blocks 的 edge features），导致局部排序不够好。

真正目标：学到一个能给 AO-GPT 带来更好训练顺序的 order policy。局部 swap 只是诊断指标，不是目标本身。

前置实验要回答的问题：

> Frozen old ON 的 step-wise prior 能不能 + cheap A-edge features + 小 MLP，生成比 old ON / random 更好的 order（按 AO-GPT NLL 衡量）？

## 核心设计

### 两阶段

```
┌────────┬───────────────────────────────────────────────────────┬──────────────────────┬───────────────────────┐
│  阶段  │                         内容                          │         训练         │      AOGPT 参与       │
├────────┼───────────────────────────────────────────────────────┼──────────────────────┼───────────────────────┤
│ Stage  │ old ON + normalized A-edge linear combination         │ 无（grid search      │ 仅 eval              │
│ 1      │                                                       │ α,β,γ）              │                       │
├────────┼───────────────────────────────────────────────────────┼──────────────────────┼───────────────────────┤
│ Stage  │ old ON + MLP adapter，用 AOGPT per-step NLL ranking   │ 小 MLP (~8K params)  │ 离线生成训练标签 +    │
│ 2      │ 做训练标签                                            │                      │ eval                  │
└────────┴───────────────────────────────────────────────────────┴──────────────────────┴───────────────────────┘
```

### Stage 1: No-training Greedy Reranker

每步 scoring（logit space，z-score normalized within candidates）：

```
score_i = α * norm(z_on_i) + β * norm(A_last_i) + γ * norm(A_i_last)
```

其中 norm(x) = (x - mean(candidates)) / std(candidates)，在当前步未访问候选集合内做 z-score。所有项在 logit space 统一 scale。

Grid search α,β,γ 最小化 tune_val_set 平均 AOGPT NLL（不能用 test set）。

### Stage 2: MLP Adapter（训练）

特征 (7-dim，per-candidate)：

```
features_i = [
  on_logit_i,              # old ON raw logit (unvisited candidates only)
  on_rank_distance_i,      # -|rank_old(i) - t| / N
  A_last_i,                # edge last → candidate
  A_i_last,                # edge candidate → last
  row_mean_i,              # mean A[i,:] excl diag
  col_mean_i,              # mean A[:,i] excl diag
  step_norm                # t / N
]
```

on_rank_distance 计算方式：先跑 old ON autoregressive 得到 σ_old，rank_old(i) = i 在 σ_old 中的位置。（σ_old 是固定的，每步仅依赖它。）

#### 训练标签：per-step AOGPT candidate-local NLL（关键修正 #1）

对训练集每条序列、每一步，枚举所有剩余 candidate i：
- 构造 token order: `prefix_tokens + candidate_i_block_tokens + rest_tokens`
- Batched AOGPT forward → token-level NLL
- **只取 candidate_i block 的 first-k tokens 的 mean NLL 作为 label**
- rest_tokens 只为 AOGPT 提供完整上下文和 order 形状，不参与 loss 计算
- target_i = -NLL(candidate_i first-k tokens | prefix)

需要新写 `compute_candidate_prefix_nll()` —— 现有 `compute_per_seq_nll` 只能返回整条平均 NLL，会混入 rest 的损失。

#### Prefix distribution（修正 #2）

V0 默认用 old_ON greedy prefix 生成标签（因为 adapter 是 old ON 的修正器，应该学 old ON rollout 附近的状态）：

```
100% old_ON greedy prefix
```

后续若 distribution shift 严重，可扩展为：
```
70% old_ON greedy prefix
20% random valid prefix
10% A-DP teacher prefix
```

#### 训练细节

- 小训练集（~100-200 序列，离线生成标签，batch candidate 评估，约 200-400 次 AOGPT forward）
- MLP: 7 → 64 → 64 → 1，dropout=0.1
- **默认 Loss: listwise soft CE（修正 #6）**
  ```
  q_i = softmax(-NLL_i / τ_label)
  loss = CrossEntropy(q, log_softmax(score))
  ```
  比 pairwise margin 更稳定（per-step candidate NLL 差异可能很小）。
- 保留 `margin_ranking_loss` 作为可选 alternative
- 推理时不需要 AOGPT（只用 cheap features）

### Old ON Prior 提取（实现细节不变）

get_logits_safe() 正确处理 visited：
1. 调用 old_on(A, visited, last) → raw_logits (B,N)，visited 位置为 -inf
2. 对 unvisited candidates 做 z-score normalization
3. visited candidates 的 feature 置 0
4. 最终 adapter 输出层统一 mask visited → -inf
不把 -inf 替换成有限值喂 MLP——visited block 不该参与输入分布。

## 数据集划分（修正 #3）

```
train_label_set: MLP label generation (~100-200 seqs)
tune_val_set:   grid search / early stop / hyperparam selection (~200 seqs)
test_set:       final comparison only (~500 seqs)
```

Stage 1 的 α,β,γ grid search 只在 tune_val_set 上做，不能用 test set。

## 主要指标（按优先级）

### 一级：AO-GPT Utility

| 指标 | 含义 |
|------|------|
| NLL(order) | full-order AOGPT NLL，越低越好 |
| ΔNLL vs random_N16 | random_N16_MC_nll - policy_nll，正数 = 比 random 好 |
| ΔNLL vs old_on | old_on_nll - policy_nll，正数 = 比旧 ON 好 |

**Baselines:** L2R, random_N16 (MC avg, 动作空间对齐，修正 #4), old_ON autoregressive greedy, A-DP teacher

### 二级：Order Quality（diagnostic only）

| 指标 | 含义 |
|------|------|
| Kendall τ vs A-DP teacher | 整体排序相似度 |
| Pairwise relative order accuracy | a 是否在 b 前面（对 problem pairs） |
| Exact adjacent swap repair rate | (6,8) 这种局部错位是否被修回 |
| Invalid/duplicate rate | 必须为 0 |

### 三级：Order Diversity

| 指标 | 含义 |
|------|------|
| Policy entropy | 不同温度下的 order 多样性 |
| Unique orders / total | 生成 order 的唯一率 |

## Baselines 命名约定（修正 #4, #5）

- `random_N16` — N16 排列 → N16→N64 展开（主要对比基线，动作空间对齐）
- `random_N64` — diagnostic only，不作为主要基线
- `L2R` — 左到右
- `old_ON` — CrossAttentionOrderNetwork autoregressive greedy
- `A-DP teacher` — DP-on-A 顺序（不是 "optimal"，不是 AO-GPT NLL 上界）

## Sanity Checks

| # | Check |
|---|-------|
| 1 | old ON 每步 visited popcount == step（prefix 长度正确） |
| 2 | label candidate 在 visited 中为 False（target 未被 mask） |
| 3 | get_logits_safe: visited features = 0, unvisited 已 z-score |
| 4 | generate: 先 score 后 update（无 future leakage） |
| 5 | sequential_generate 输出是合法排列（无重复/遗漏） |
| 6 | on_rank_distance ∈ [-1, 0]，每步仅依赖 σ_old（固定） |
| 7 | MLP adapter 推理时不调用 AOGPT |
| 8 | per-step AOGPT label 只取 candidate block first-k token loss（修正 #1） |

## 成功标准

主要：
```
old_ON + MLP full-order AOGPT NLL < old_ON
old_ON + MLP full-order AOGPT NLL < random_N16_MC
held-out test 上成立
```

辅助：
```
invalid rate = 0
Kendall τ / swap 只报告，不作为通过标准
```

## 新增文件

1. **config.py** — 追加 RerankerConfig
2. **reranker.py** — 核心模块 (~500 lines，含 `compute_candidate_prefix_nll`)
3. **train_reranker.py** — 训练 + sanity overfit (~350 lines)
4. **eval_reranker.py** — grid search + 全对比 (~350 lines)

## 实现顺序

1. config.py — RerankerConfig
2. reranker.py — 核心组件（含 compute_candidate_prefix_nll）
3. train_reranker.py — 训练 + sanity overfit
4. eval_reranker.py — grid search + 全对比

## 不做的

- 不训练 AOGPT（frozen）
- 不修改 old ON（frozen）
- 不引入 GRPO/RL
- 不做 cotrain
- Swap pair accuracy 只报告，不作为 success criteria
- 不做完整 V1 sequential ON（留给后续）
