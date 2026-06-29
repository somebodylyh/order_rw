# TiT-AO-GPT Project Summary (2026-05-02)

## 1. 项目目标

研究 **reading order 对 autoregressive language model 的泛化影响**。

**已知事实 (已证伪):** HC (Hill Climbing) 暴力搜索已证明：对 frozen AO-GPT，存在比 L2R 更好的 reading order（12.8% NLL 改善）。这不是 open question。

**真正的问题:** 用这些更好的 order 做 continual pretraining，能否改善模型本身在 L2R decoding 下的泛化能力（val_l2r_loss）？即：order 是仅仅"适配当前模型参数"的捷径，还是包含了可被模型内化的、与 sequence structure 相关的信息？

最终目标：**TiT (Transformer-in-Transformer)**——outer Transformer 预测 reading order，inner AO-GPT 按该 order 做 token prediction，端到端联合优化。

## 2. 粒度错配的发现

这是整个项目最重要的元发现之一。

- **AO-GPT pretrained with `block_len=4` (64 blocks)**: 模型在 64-block 粒度下预训练的，每个 block 4 tokens
- **Route A 全部实验用 `block_len=16` (16 blocks)**: ON、DP、HC 最初都在 16-block 粒度运行
- **切换到原生 64-block 粒度**: HC NLL improvement 从 ~6% 跳到 **12.8%**

**教训**: Order 问题的有效粒度取决于模型训练时的 tiling strategy。在模型原生粒度下做 search/optimization 是前提条件——粗粒度遗漏了大量 search space 中的 structuring opportunities。

---

## 3. 方法论路线：EM-Style Alternating Optimization

放弃 REINFORCE（64! 离散空间方差过大）和 Gumbel-Softmax（straight-through 梯度太弱，loss 全程 flat at ~3.72）后，采用 EM-style 交替优化：

- **E-step**: Hill Climbing (HC) 暴力搜索——给定 frozen AO-GPT，对每序列搜索最优 block permutation（argmin NLL）
- **M-step**: 用 HC 搜出的 "golden orders" 做 AO-GPT continual pretraining

## 4. 先前相关结果 (Route A)

### Order Network (ON) 演进

Route A 的核心思路：用 DP (Dynamic Programming) 或 HC 生成 per-sequence optimal orders → 训 ON 学习从 A 矩阵到 optimal order 的映射。

| 模型 | 输入特征 | Val Acc | 参数 |
|------|---------|---------|------|
| GNN | topo features (degree) | 32.6% | — |
| MLP | compact A-matrix features | 67.7% | — |
| **Cross-Attention ON v2** | A-matrix + [None] signals | **76.9%** | 51K |

- **结论**: A 矩阵直接特征提取是关键（MLP 67.7% >> GNN 32.6%）；Cross-Attention 架构进一步捕获了 block 间非局部依赖
- **Joint ON-AO-GPT (Gumbel-Softmax)**: 失败。val NLL 全程 flat at ~3.72，early stop at epoch 145/200，最优在 epoch 20 (NLL 3.7181)。straight-through 梯度太弱，ON 在平坦 loss 面上漂移

### A 矩阵定义的三次 Pivot

这是项目中最重要的实验演进路径之一：

| 阶段 | A 矩阵定义 | 行为 | 结果 |
|------|-----------|------|------|
| 1. Block-to-block attention | 块间 attention 均值 | Sink pattern——几乎所有 attention 流向 token 0 | ❌ 不可用 |
| 2. [None]-delta cosine similarity | `cos(attn_to_None, attn_to_block_i)` | 信号强但 Toeplitz 结构——相邻块高度相关，DP 100% 单调 (0 inversions) | ❌ 无优化空间 |
| 3. **NLL pair score** | `NLL(permute(i,j)) - NLL(baseline)` | 非对称、instance-specific、DP 100% 非单调 (1.6/15 avg inversions) | ✅ 当前标准 |

**教训**: 前两次尝试的 A 矩阵本质上是结构性的（sink pattern、Toeplitz），不包含 instance-specific ordering information。NLL pair score 是第一个真正 capture "这条序列该怎么排" 的信号。

### Attention 信号证伪链

这条链记录了合作者最初声称的 "attention contains order information" 主张是如何被证伪的：

1. **`permute_data=False` 导致 PE leak**: 早期 probe 训练时没有随机 permute 数据（固定 block 顺序），即使 attention 真的纯随机、仅用 positional encoding 也能学到顺序 → 合作者声称的 80%+ accuracy 是因为 position 泄露，与 attention 无关
2. **Cosine similarity 只捕捉内容相似度**: [None]-delta cosine 的高信号被证明来自 token embeddings 的语义聚类（相似词 → 相似 attention → 高 cosine），而非 attention 本身携带 ordering 信息
3. **合作者的信号实际来自 NLL probe**: 最终分析确认——能预测 order 的特征来自直接对 AO-GPT 做的 NLL evaluation，不是 attention。attention 在这个问题上是弱信号源

**教训**: `permute_data=True` 的消融是 order 预测的金标准——任何声称 attention 能预测 order 的实验必须通过此项检检验。

## 5. 当前实验：M-step Continual Pretraining

### 核心脚本
`block_lo_arm_order_network/m_step_train.py`

### 实验设计
三组对比，每组从同一 pretrained AO-GPT checkpoint 开始：

| 组别 | 训练时使用的 order | 评估指标 |
|------|-------------------|---------|
| Baseline | 100% random block permutations | val_l2r_loss |
| Golden | 100% HC golden orders | val_l2r_loss |
| Mixed | 50% golden + 50% random | val_l2r_loss |

- **评估指标**: val_l2r_loss（L2R order 下的 NLL，非 training order 下的 NLL）
- **关键问题**: 用更好的 order 做 continual pretraining，能否改善模型本身的 L2R decoding 性能？

### Model Checkpoint
- AO-GPT-MDM: 47M params, trained on wikitext-103 with random block permutations
- Path: `~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt`
- 架构: AdaLN6, NoRep, conditional 128-dim, truncated QK norm
- 配置: seq_len=256, n_blocks=64, block_len=4

### 实验历史

#### v1 (❌ 作废)
- 配置: HC 500 seqs × 2000 steps, M-step 500 seqs × 5000 steps, lr=3e-4, 无 warmup, 无 optimizer state
- 结果: **三组全部发散**——val_l2r_loss 从 3.59 飙到 9.39~9.67，train loss 降到 0.09
- 根因: 500 seqs 数据量过小 (160 epochs overfitting), lr 过高, optimizer state 从零初始化
- 教训: 小数据集 + 高 lr + 无 warmup = 教科书级过拟合，golden ordering 效果被完全淹没

#### v2 (✅ 完成)
- 配置: HC 1000 seqs × 2000 steps, M-step 5000 seqs × 936 steps (max_epochs=3 cap), lr=3e-5, warmup=500, 加载 pretraining optimizer state
- HC 结果: best_nll=3.1593 (vs L2R baseline 3.5629, 11.3% improvement)
- M-step 结果:
  - Baseline: 3.5532 (−0.0161 vs original)
  - Golden: 3.5557 (−0.0136)
  - Mixed: 3.5543 (−0.0150)
- **结论**: Baseline 稳定不发散，训练管线正确。但 golden orders 仅覆盖 1000/5000 (20%) 训练数据，三组差异 0.0025，噪声级别，无法区分

#### v3 (🔄 运行中, 2026-05-02 18:30 启动)
- 配置: **HC 5000 seqs × 2000 steps (100% 覆盖)**, M-step 5000 seqs × 936 steps, 其他同 v2
- HC 进度: Step 600/2000, best_nll=3.2289 (vs L2R 3.5805)
- 预期: HC ~2 小时, M-step ~10 分钟, 总计 ~2h10m
- **关键决策点**: 100% 覆盖率下，golden vs baseline 的 val_l2r_loss 差异能否从噪声拉开到统计显著

### 训练管线关键参数
| 参数 | v1 (bug) | v2/v3 (fixed) | 说明 |
|------|---------|---------------|------|
| 训练数据量 | 500 seqs | 5000 seqs | 10× 增加 |
| lr | 3e-4 | 3e-5 | 1/10 |
| warmup | 无 | 500 steps linear | Adam 二阶矩重新积累 |
| optimizer state | 新鲜 Adam | 从 pretraining ckpt 加载 | 避免破坏性更新 |
| epoch 上限 | 无 (160 epochs) | max_epochs=3 | 硬上限防过拟合 |
| HC 覆盖 | 500/500 (100%) | v2: 1000/5000 (20%), v3: 5000/5000 (100%) | |

## 6. 项目文件结构

```
block_lo_arm_order_network/
├── m_step_train.py              # M-step 训练主脚本 (EM pipeline)
├── probe_results/
│   ├── m_step/
│   │   ├── m_step_train_v2.log  # v2 完整训练日志
│   │   ├── m_step_train_v3.log  # v3 进行中
│   │   ├── hc_golden_orders.npz # HC 搜索结果
│   │   ├── m_step_baseline.pt   # baseline 组 checkpoint
│   │   ├── m_step_golden.pt     # golden 组 checkpoint
│   │   ├── m_step_mixed.pt      # mixed 组 checkpoint
│   │   └── m_step_summary.json  # 最终对比结果
│   └── joint_train.log          # 早期 Gumbel-Softmax ON+AO-GPT 日志 (失败)
├── eval_results/
│   └── eval_report.png          # Route A 评估报告
└── *.py                         # Route A 相关脚本 (ON, DP 等)

AO-GPT-MDM/
├── model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py  # AO-GPT 模型定义
└── config.py                   # 训练配置

nanogpt-learned-order/          # 上游 pretrained model
docs/
└── project_summary_2026-05-02.md  # 本文档
```

## 7. 待回答的核心问题

1. **Go/No-go (v3 实验)**: HC golden orders 在 continual pretraining 中能否改善模型本身的 L2R val loss？
   - **成功标准**: golden 组 val_l2r_loss 比 baseline 低 **>0.01 nats**（~0.3%），且在多个 random seed 下一致
   - **说明**: 0.01 nats 的阈值排除了 training noise (~0.0025 in v2) 和随机波动
2. 如果 v3 通过 → 下一步训 ON 替代 HC 作为 E-step（省掉暴力搜索），做多轮 EM 或端到端 TiT
3. 如果 v3 未通过 → golden orders 本身对 continual pretraining 的泛化无效，需重新审视方向（order 可能仅是"适配当前参数"的捷径，不包含可被模型内化的信息）

## 8. 已知限制

- HC 仅搜索 block-level permutation (64! 空间)，非 token-level (256! 空间)
- 当前仅用 wikitext-103 train 集的 5000 条序列，非全量数据
- M-step 只做 1 轮 EM 迭代 (~3 epochs)，多轮迭代效果未知
- Order Network 尚未集成到 pipeline 中
- HC 的成本使其无法直接用于训练时 augmentation
- **Golden orders 是 checkpoint-specific 的**：HC 在 frozen model 上搜索，模型权重更新后最优 order 会变化。多轮 EM 需要周期性刷新 golden orders（重新运行 E-step），不能用同一批 order 一直训下去。
