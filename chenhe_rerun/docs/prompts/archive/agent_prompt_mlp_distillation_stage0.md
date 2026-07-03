> Archive note: this prompt targets the old L1H2 direct-asym-eig
> `distribution_mlp` stage-0 experiment and pre-cleanup `Report/...` paths.
> For current MLP work, start from
> `Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md`
> and the L0 layer-mean pairwise-max Fiedler try25/try28/try29 line.

# Agent Prompt：L1H2 Direct-Asym-Eig Teacher → Attn-MLP 蒸馏

Execution note: this archived prompt originally asked for `CUDA_VISIBLE_DEVICES=0`
and not using GPU1. The later user override for the executed run was
`CUDA_VISIBLE_DEVICES=1`.

你正在处理仓库：

```text
Yangxiaohehehe/nanogpt-learned-order
```

目标分支：

```text
distribution_mlp
```

请不要停留在方案设计。你需要实际检查代码、实现数据采集与训练脚本、生成并保存数据集、训练一个新的 MLP、运行对照实验、评估结果，并把所有命令、配置、指标和结论写入 `Report/MLP_distillation/try_1/`。

---

## 1. 研究目标

完成老师优先级中的 Stage 0，并为 Stage 1 做好可部署准备：

> 验证一个简单 MLP 能否从 current-frame attention 中学习并复现当前已经验证有效的 Distribution teacher 顺序信号。

本实验只验证 **teacher compression / imitation**，不要求：

- 自动发现最佳 attention head；
- 让 MLP 更新 AO-GPT backbone；
- 使用 validation PPL 选择 student；
- 证明 MLP 自主恢复了 OriginalL2R；
- 直接进入 teacher-free 端到端训练。

主张必须限定为：

> MLP 能否学习并复现固定 Distribution teacher 的 current-frame order mapping。

---

## 2. 开始前必须阅读

请先在 `distribution_mlp` 分支阅读并核对：

```text
Report/head_singal_Stable/current_distribution_methodology_summary.md
Report/head_singal_Stable/try_20/results.md
Report/MLP_distillation/current_distillation_methodology_summary.md
Report/MLP_distillation/try_1/experiment_design.md
Report/MLP_loss_training/current_mlp_methodology_summary.md
attn_mlp_order_policy.py
online_spectral_order_policy.py
train.py
```

同时定位并复用现有实现中的：

- current-frame attention 聚合与 remap；
- `direct_asym_eig` order recovery；
- raw/reverse current-model `linear_profile_loss` 比较；
- `FlatAttentionOrderMLP`；
- block order、rank、pairwise precedence 工具。

不要在已有可靠实现可复用时重新写一套近似版本。若文档与代码冲突，以当前分支代码为准，并在报告中明确记录。

---

## 3. 固定 Teacher

固定实验条件：

```text
task              WikiText103
sequence          seq256
data frame         permute
order unit         block64（64 个 block，每个 block 4 token）
backbone           clean Random-10k checkpoint，完全冻结
teacher head       L1H2
attention export   with_none
teacher method     direct asymmetric eigendecomposition
eig mode           raw_right_largest_real_real
orientation        current-model train linear_profile_loss 比较 raw/reverse
coordinate frame   current-frame block ids
```

首选 checkpoint：

```text
out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-clean-10k-for-attn-mlp-resume/ckpt.pt
```

运行前确认 checkpoint 存在，并核对：

- `iter_num == 10000`；
- backbone 是 Random-order base；
- 没有已训练的 `attn_mlp_policy_state` 污染；
- 数据 permutation metadata 完整。

### Teacher 生成过程

每个独立 probe item：

```text
current L1H2 with_none attention A
→ eig(A)
→ 选择实部最大的 eigenvalue 对应的右 eigenvector
→ 使用 eigenvector.real 排序得到 raw_order
→ 生成 reverse_order
→ 在 current model、train split probe batches 上计算两者的 linear_profile_loss
→ 选择 loss 更低的方向为 teacher_order
```

不得使用：

- OriginalL2R；
- original-frame tau/distance；
- OriginalL2R/R2L PPL；
- validation PPL；
- 人工顺序标签；
- “如果 diagnostic sign 为负就反转”等 oracle 规则。

---

## 4. 数据采集

### 4.1 一个数据样本

一个独立的 train-stream probe batch 生成一个样本。每个样本保存：

```python
{
    "attention":     float16[64, 64],  # current-frame L1H2 with_none attention
    "teacher_rank":  uint8[64],        # 每个 block 在 teacher_order 中的位置
    "teacher_order": uint8[64],        # 冗余但便于审计
    "teacher_gap":   float32,           # abs(raw_loss - reverse_loss)
    "raw_loss":      float32,
    "reverse_loss":  float32,
    "chosen_loss":   float32,
    "probe_seed":    int64,
    "sample_id":     int64,
    "metadata": {
        "checkpoint": str,
        "teacher_head": "L1H2",
        "export_type": "with_none",
        "teacher_method": "direct_asym_eig",
        "eig_mode": "raw_right_largest_real_real",
        "orientation_score": "linear_profile_loss",
        "frame": "current"
    }
}
```

`teacher_q[64,64]` 不必保存；训练时由 `teacher_rank` 动态生成：

```python
Q[i, j] = 1 if teacher_rank[i] < teacher_rank[j] else 0
```

### 4.2 Attention 采集要求

- backbone 始终 `eval()` 且所有参数冻结；
- 从 train stream 抽取不同 token offsets / batches；
- 使用独立 Random reveal orders；
- 将 token attention 聚合为 block attention；
- 将 reveal-frame attention 映射回 current-frame block coordinates；
- 对 probe batch 聚合得到一个 `[64,64]` attention；
- 保存进入 student 前的实际矩阵；
- 不只保存一张长期平均矩阵；
- v1 不做 z-score、robust-z、row-normalization 或额外 feature engineering。

### 4.3 Teacher confidence 过滤

定义：

```text
teacher_gap = abs(raw_loss - reverse_loss)
```

v1 使用固定硬过滤：

```text
teacher_gap < 0.01 → 丢弃该样本
```

不要静默修改阈值。必须记录：

- 总尝试样本数；
- 接受样本数；
- 拒绝样本数；
- 接受率；
- gap 的均值、中位数、分位数和直方图。

如果接受率过低，先报告并分析，不要使用 original diagnostics 调阈值。

### 4.4 数据划分

所有 teacher 构造和 student 训练都来自 WikiText103 `train` stream，但按 **probe seed group + token batch** 严格隔离：

```text
train       seed group A
val_probe   seed group B
test_probe  seed group C
```

不要生成完后随机打散再划分。相同 probe seed、相同 token offsets 或重复 batch 不能跨 split。

推荐正式规模，均指过滤后的有效样本数：

```text
train       5000
val_probe   1000
test_probe  1000
```

先运行不作为研究结果的小型 pipeline smoke：

```text
train       256
val_probe    64
test_probe   64
```

smoke 仅用于确认代码、shape、loss 和保存流程正确；正式报告只能使用正式数据。

### 4.5 数据存储

建议目录：

```text
Report/MLP_distillation/dataset_l1h2_direct_asym_eig_v1/
├── manifest.json
├── collection_summary.json
├── train/
│   ├── shard_000.pt
│   └── ...
├── val_probe/
│   └── ...
└── test_probe/
    └── ...
```

每个 shard 保存 256 或 512 个样本。`manifest.json` 必须包含：

- git branch 和 commit SHA；
- checkpoint 路径和 SHA/metadata；
- teacher 设置；
- attention 设置；
- loss-profile 设置；
- gap threshold；
- split seed groups；
- 每个 split 的有效样本数；
- dtype 和 tensor shape；
- 生成命令。

---

## 5. Student MLP

v1 使用最简单模型：

```text
input       raw attention [64,64]
flatten     4096
hidden      1024 → GELU → 1024 → GELU
output      64 logits
dropout     0
normalization none
```

直接复用 `FlatAttentionOrderMLP`。

输出：

```text
s_i：block i 的 priority logit
```

约定：

```text
s_i 越大，block i 越早
student_order = argsort(s, descending=True)
```

---

## 6. 唯一主 Loss

第一版只使用一个 pairwise logistic loss，不加入 directed-ribbon、rank regression、NLL reward、move preference、std-floor 或显式 logit L2。

对所有 unordered pairs：

```text
0 <= i < j < 64
```

共：

```text
C(64,2) = 2016
```

个 pair。

定义：

```python
target_ij = 1 if teacher_rank[i] < teacher_rank[j] else 0
pred_ij   = (s[i] - s[j]) / tau
```

使用：

```python
L_pair = BCEWithLogitsLoss(pred_ij, target_ij)
```

固定：

```text
tau = 1.0
```

等价公式：

\[
L_{\text{pair}}
=
\frac{1}{2016}
\sum_{i<j}
\log\left(1+\exp[-t_{ij}(s_i-s_j)]\right),
\]

其中 \(t_{ij}=+1\) 表示 teacher 认为 \(i\) 早于 \(j\)，否则 \(t_{ij}=-1\)。

总 loss：

```text
L = L_pair
```

优化器可以使用 AdamW 的普通 weight decay，但不要把 weight decay 描述成额外研究 loss。

只有在主实验明确出现 logits 恒定塌缩时，才额外建立一个独立 ablation；不要修改 v1 主实验定义。

---

## 7. 训练设置

默认：

```text
optimizer       AdamW
learning rate   1e-4
weight decay    0.01
betas           (0.9, 0.99)
batch size      64
epochs          up to 50
grad clip       1.0
student seeds   1337, 2026, 2027
```

模型选择只使用 held-out `val_probe` imitation 指标：

主选择指标：

```text
val_probe pairwise BCE loss
```

辅助指标：

```text
pairwise accuracy
Kendall distance / tau to teacher order
```

禁止使用以下指标选择 checkpoint 或 early stopping：

- validation corpus PPL；
- OriginalL2R tau；
- original-frame distance；
- oracle order quality。

保存：

```text
best_by_val_pair_loss.pt
last.pt
```

checkpoint 格式应兼容或容易转换为 `load_frozen_attn_mlp_policy`：

```python
{
    "model_state_dict": ...,
    "config": {
        "num_blocks": 64,
        "hidden_dims": [1024, 1024],
        "dropout": 0.0,
        "activation": "gelu",
        "input_normalization": "none"
    },
    "policy_type": "l1h2_direct_asym_eig_teacher_distillation",
    "target_direction": "larger_logit_reveals_earlier",
    "training_meta": ...
}
```

---

## 8. 必须运行的 Baselines

### 8.1 Constant-order baseline

从 train split 的 teacher ranks 计算平均 priority，得到一条固定 order；对所有 val/test 样本输出同一条 order。

### 8.2 Zero-input MLP

同一 MLP 架构，但输入恒为全零矩阵，正常训练 pairwise loss。

### 8.3 Shuffled-input MLP

对 attention 使用与 label 无关的随机同步行列 permutation：

```python
A_shuffled = A[p][:, p]
```

teacher labels 保持原 current-frame block ids，不同步改动，以破坏可利用对应关系。

### 8.4 Teacher upper bound

报告 teacher 自身对 teacher labels 的 pairwise accuracy（应为 1）以及 teacher 的 current-model linear-profile loss，用作参考，不参与 student checkpoint 选择。

---

## 9. 评估

在 `val_probe` 和 `test_probe` 上至少报告：

- pairwise BCE loss；
- pairwise precedence accuracy；
- student order 与 teacher order 的 Kendall tau；
- Kendall distance；
- exact-order match rate；
- student/teacher `linear_profile_loss` 差值；
- student 与 teacher raw/reverse orientation 一致率；
- 不同 student seed 的均值与标准差；
- 正常 MLP、constant、zero-input、shuffled-input 的同表比较。

在 student checkpoint 已经由 imitation 指标选定后，才允许补充：

- original-frame tau/distance；
- validation NLL/PPL；
- Random / teacher / student order 对照。

这些必须标为 `post-hoc diagnostic only`。

同时测量：

```text
teacher pipeline latency:
attention → eig → raw/reverse loss evaluation → order

student latency:
attention → MLP → argsort
```

报告：

- 每个 order 的平均 wall-clock time；
- speedup；
- GPU memory；
- 是否省掉 raw/reverse AO-GPT loss forwards。

---

## 10. 成功判据

强成功需要同时满足：

1. 正常 attention-conditioned MLP 在 held-out `test_probe` 上明显优于 constant-order baseline；
2. 明显优于 zero-input 和 shuffled-input MLP；
3. student-teacher pairwise accuracy 和 Kendall 指标在三个训练 seed 下稳定；
4. student order 的 current-model linear-profile loss接近 teacher；
5. 提供清晰的 teacher→student 推理加速结果。

如果正常 MLP 与 constant/zero/shuffle 接近，结论必须是：

> 当前数据主要支持一个近似常量 teacher order，尚未证明 MLP 学到了 attention-conditioned mapping。

不要用 original-frame tau 或 validation PPL 挽救该结论。

---

## 11. 需要实现或复用的脚本

优先复用仓库现有组件，建议新增：

```text
scripts/data/build_attn_mlp_distillation_dataset.py
scripts/train/train_attn_mlp_distillation.py
scripts/eval/eval_attn_mlp_distillation.py
```

脚本必须：

- 支持 `--device cuda` 与 `--dtype bfloat16`；
- 支持 smoke/formal 数据规模；
- 固定随机 seed；
- 可断点续采；
- 不重复收集已存在 sample；
- 校验 tensor shape、finite values、permutation 合法性；
- 将完整命令写入报告目录；
- 避免把大 dataset/checkpoint 误提交到 Git，除非仓库约定允许。

使用：

```text
CUDA_VISIBLE_DEVICES=0
```

不要占用 GPU1。

---

## 12. 报告与交付物

保留已有：

```text
Report/MLP_distillation/try_1/experiment_design.md
```

新增：

```text
Report/MLP_distillation/try_1/
├── dataset_manifest.json
├── collection_summary.json
├── train_command.sh
├── eval_command.sh
├── config.json
├── metrics.csv
├── eval_summary.json
├── experiment_result.md
└── failure_analysis.md
```

模型保存到：

```text
checkpoints/attn_mlp_distillation/l1h2_direct_asym_eig_v1/
```

最终 `experiment_result.md` 必须清楚回答：

1. Teacher order 是否具有足够样本多样性？
2. MLP 是否超过 constant-order baseline？
3. MLP 是否超过 zero/shuffle baselines？
4. MLP 是否在 held-out probe batches 上复现 teacher？
5. 三个 student seed 是否稳定？
6. Student 比 teacher pipeline 快多少？
7. 当前证据支持“条件映射学习”，还是仅支持“固定 order 记忆”？
8. 下一步是否具备进入 Stage 1 drop-in teacher replacement 的资格？

---

## 13. Stage 1 的最小后续动作

只有 Stage 0 强成功后，再做一个最小 drop-in 测试：

```text
attention A
→ frozen distilled MLP
→ logits
→ external priority EMA（沿用 Distribution 的 decay=0.95）
→ MAP order
```

保持原 Distribution 的 EMA、MAP 和训练 schedule 不变，只替换：

```text
eig + raw/reverse loss orientation
```

为：

```text
MLP forward
```

先在冻结 checkpoint 上做 policy-equivalence 与速度评估，不要立即扩展为复杂联合训练。

---

## 14. 执行原则

- 先检查代码与已有工具，再实现；
- smoke 不能作为正式结果；
- 不要把 OriginalL2R、original tau、validation PPL 用于数据过滤、head 选择、teacher orientation、student checkpoint 选择或 early stopping；
- 不要静默修改 teacher、gap 阈值、loss 或 split；
- 失败也必须保存完整报告和 failure analysis；
- 不要只输出代码，必须实际完成数据采集、正式训练、对照评估和报告。
