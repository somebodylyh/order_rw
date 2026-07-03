> Archive note: this head-information prompt targets the old `distribution_mlp`
> branch and pre-cleanup report paths. Use it for mechanism provenance only.
> Current distribution work should start from the L0 layer-mean pairwise-max
> Fiedler summaries under
> `Report/language/wikitext103/order_teacher_distribution/`.

# Agent Prompt：AO-GPT 多层多头 Attention 信息与机制研究

你是一名严谨的 ML research engineer / mechanistic interpretability researcher。请在仓库中实际执行一轮完整研究：

```text
repository: Yangxiaohehehe/nanogpt-learned-order
branch: distribution_mlp
```

所有新结果必须自动记录到以下目录，目录名保持原样，不要改拼写：

```text
Report/Head_imformation/
```

本任务不是只画热力图，也不是继续优化 Distribution 或 MLP order。核心目标是：

1. 严格验证已经观察到的层间、head 间和 seed 间现象是否真实、稳定；
2. 通过干预实验解释这些现象主要来自 position、content、reveal mask、QK routing、softmax、V/OV 写回还是跨 head/跨层交互；
3. 区分“显式三角 attention 弱”与“该层真正没有顺序信息”；
4. 建立一套可复现的跨 seed、跨层 head 功能分析框架。

不要把尚未运行的假设写成结论。Smoke test 仅用于排错，不能进入正式研究结论。

---

# 一、当前观察：只能作为待验证假设

现有 clean Random-10k、WikiText103 `seq256 / permute / block64` 结果初步显示：

1. `without_none / original_l2r` 中，L0/L1 的显式三角方向通常较强，L2 较弱，L3 最弱。
2. `with_none / original_l2r` 中，最强方向信号更集中在中间层，尤其 L2；L3 仍整体较弱。
3. `current_l2r` 中多数 heads 看起来接近上下平衡，但映射到 `original_l2r` 后，一部分 heads 出现强上三角或强下三角。
4. 同一 seed 中，不同 heads 方向、尖锐度、稳定性差异很大。
5. 不同 training seed 中，承担强方向功能的 head 编号不同；相同 head 编号也可能在不同 seed 中表现完全不同。
6. 纯 asymmetry selector 对 seed 敏感；current-model loss filter 有时有效，但低 raw/reverse gap 的 head 仍可能失败。
7. 最后一层三角 attention 较弱，但这不等价于最后一层“没有信息”；顺序信息可能已存在 residual stream 或 OV output 中。
8. 多个 heads 可能具有互补、冗余或协同作用，单 head heatmap 不足以解释最终模型行为。

以上全部是预注册假设，不是本轮可以直接引用的事实。必须在多个 seeds、多个 checkpoints 和统一 probe 上重新计算。

---

# 二、研究边界

## 2.1 主要任务

```text
dataset           WikiText103
main setting      seq256 / permute / block64
model             AO-GPT
primary policy    pure Random-order checkpoints
unit              block-level attention, 64 blocks
```

联合训练 MLP checkpoints 可作为第二 cohort，但不能与 pure Random cohort 混为一谈。

## 2.2 Original-frame 使用边界

允许把 current-frame 矩阵映射到 `original_l2r`，但只用于：

- 诊断上下三角现象；
- 描述 hidden original coordinate 是否可见；
- 计算 post-hoc direction/tau/distance；
- 机制解释。

不得使用 OriginalL2R、original tau、original distance 或 validation PPL：

- 选择训练 checkpoint；
- 选择最佳 seed；
- 调整模型；
- 选择 order policy；
- 早停；
- 训练 MLP；
- 宣称 no-prior order learning 成功。

本任务是 interpretability 研究，允许预注册少量历史 reference heads，但正式结论必须依赖 all-head census 和跨 seed 统计，而不是挑图。

## 2.3 不得混淆

必须明确区分：

```text
显式 attention direction
QK pre-softmax geometry
softmax 后 routing
V 中携带的信息
AV head output
AVW^O 写回 residual 的贡献
residual stream 中可解码的信息
head 对 NLL 的因果贡献
```

“热力图弱”不能直接写成“该 head 无信息”。

---

# 三、开始前必须阅读并核对

至少阅读：

```text
Report/MLP_loss_training/random_clean_10k_base/all_head_attention_maps_val_random/README.md
Report/MLP_loss_training/random_clean_10k_base/all_head_attention_maps_val_random/head_attention_metrics.csv
Report/MLP_loss_training/head_selection_random10k_stable_v1/results.md
Report/MLP_loss_training/head_selection_random10k_stable_v2/results.md
Report/MLP_loss_training/try_35_37_seed_sweep_result.md
Report/MLP_loss_training/try_39_40_autohead_eigloss_seed_sweep_result.md
Report/MLP_loss_training/current_mlp_methodology_summary.md
Report/head_singal_Stable/SUMMARY.md
Report/head_singal_Stable/current_distribution_methodology_summary.md
Report/head_singal_Stable/try_20/results.md

scripts/analysis/export_all_head_attention_maps.py
scripts/analysis/select_stable_attention_heads.py
scripts/analysis/mechanism_probe_head_directionality.py
scripts/analysis/head_directionality_continuous_runner.py
AOGPT_block.py
train.py
order_utils.py
```

必须检查并写入 `files_read.md`：

- `[None]` 插入与 predictor-target 对齐；
- `with_none = attn[:-1, :-1]`；
- `without_none = attn[1:, 1:]`；
- causal mask；
- Q/K/V 分头方式；
- token→block 聚合；
- reveal→current remap；
- current→original diagnostic remap；
- `c_proj` 中各 head 输出块的位置；
- AdaLN/target-position embedding 的作用。

若文档与代码冲突，以当前 branch 代码为准，并记录冲突。

---

# 四、仓库隔离与自动记录

不要污染用户工作树。

## 4.1 开始状态

在原仓库记录：

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git status --porcelain=v1 -uall
git diff
git diff --cached
```

保存到：

```text
Report/Head_imformation/git_state/
├── branch_before.txt
├── head_before.txt
├── status_before.txt
├── diff_before.patch
└── diff_cached_before.patch
```

## 4.2 使用隔离 worktree

优先：

```bash
SOURCE_REPO=/home/chenhe/nanogpt-learned-order
BASE_HEAD=$(git -C "$SOURCE_REPO" rev-parse distribution_mlp)
WORKTREE=/tmp/nanogpt-head-information-<RUN_ID>
git -C "$SOURCE_REPO" worktree add --detach "$WORKTREE" "$BASE_HEAD"
```

临时代码只在 worktree 修改。正式 artifacts 最后复制到：

```text
Report/Head_imformation/
```

不得覆盖现有 Report、out 或 checkpoint。

## 4.3 自动环境记录

保存：

```text
environment/
├── pip_freeze.txt
├── torch_info.json
├── cuda_info.txt
├── gpu_info.txt
├── hostname.txt
└── run_manifest.json
```

`run_manifest.json` 至少记录：

- branch；
- commit SHA；
- date；
- hostname；
- GPU；
- checkpoint 清单；
- config；
- dataset；
- seeds；
- sample windows；
- reveal-order seeds；
- export type；
- frame；
- dtype；
-所有命令。

---

# 五、统一 Attention 定义

## 5.1 Primary view

机制研究以：

```text
without_none
```

作为主视图，因为行列都对应真实 token state。

Teacher 兼容和 predictor 语义以：

```text
with_none
```

作为并行 secondary view。

所有核心结果都同时报告两种 export，不能只展示一种。

## 5.2 without_none 必须保存两种版本

完整 attention probability：

\[
P \in \mathbb{R}^{(T+1)\times(T+1)}
\]

定义：

\[
A^{raw}=P[1:,1:]
\]

并保存每行被删除的 None mass：

\[
m_i^{none}=P_{i+1,0}.
\]

再定义 conditional real-token attention：

\[
A^{cond}_{ij}
=
\frac{A^{raw}_{ij}}
{\sum_k A^{raw}_{ik}+\epsilon}.
\]

正式分析同时保存：

```text
without_none_raw
without_none_conditional
none_mass
```

不要用 conditional 版本覆盖 raw 版本。

## 5.3 token→block

继续复用仓库正确的 token→block 和 remap 逻辑。另行生成具有清晰概率语义的 block matrix：

\[
B_{ab}
=
\frac{1}{L}
\sum_{u\in a}
\sum_{v\in b}
A_{uv},
\quad L=4.
\]

若复用当前 `mean(dim=(3,5))`，则在 block 层按行归一化；明确记录两者仅差固定 block-size scale。

## 5.4 Exposure correction

必须记录每个 query-key block pair 在 causal reveal order 下的可见次数：

\[
E_{ij}
=
\#\{j\text{ 对 }i\text{ 可见}\}.
\]

聚合矩阵使用：

\[
\bar B_{ij}
=
\frac{\sum_n E_{ij}^{(n)}B_{ij}^{(n)}}
{\sum_n E_{ij}^{(n)}+\epsilon}.
\]

同时保存不做 correction 的原始平均，比较二者差异。若方向主要由 exposure imbalance 造成，必须明确报告。

## 5.5 必须保存的张量

对每个 layer/head 至少保存：

```text
pre_softmax_qk
post_softmax_attention
without_none_raw
without_none_conditional
with_none
none_mass
V
AV
AVW^O contribution
valid causal mask / exposure count
```

大张量可分 shard 或只保存正式选定 probe 的样本级数据，但必须保存聚合结果和可复现脚本。

---

# 六、Phase A：现象存在性验证

## 6.1 两个 cohort

### Cohort A：Pure Random

主因果研究必须使用 pure Random-order checkpoints。

优先收集同架构、同 data permutation seed 的至少 3 个 training seeds，推荐 5 个：

```text
1337
2026
2027
2028
2029
```

若现有完整 Random checkpoints 不足，则训练缺失 seeds，并保存：

```text
0k, 1k, 3k, 5k, 10k, 20k, 35k, 50k
```

检查点。

### Cohort B：Order/MLP-trained

使用现有 try35–try40 等 checkpoints，作为“order policy 改变后，head 结构如何变化”的第二 cohort。

不得用 Cohort B 代替 Pure Random 对照。

## 6.2 统一 probe

对所有 seeds/checkpoints 使用相同：

- token windows；
- train/val split；
- Random reveal orders；
- probe seeds；
- sample count；
- dtype；
- batch size。

正式规模至少：

```text
2048 samples per checkpoint
8 independent repeats
```

每 repeat 使用不同 token windows 与 reveal orders，保存 manifest。

## 6.3 每个 head 的指标

定义 signed triangle direction：

\[
D_h
=
\frac{
\sum_{i<j}B_{ij}-\sum_{i>j}B_{ij}
}{
\sum_{i<j}B_{ij}+\sum_{i>j}B_{ij}+\epsilon
}.
\]

定义 unsigned asymmetry：

\[
S_h
=
\frac{
\sum_{i<j}|B_{ij}-B_{ji}|
}{
\sum_{i<j}(B_{ij}+B_{ji})+\epsilon
}.
\]

同时记录：

```text
abs_direction = |D_h|
attention entropy
none mass mean/std
top diagonal offset
offset profile g_h(d)
active edge fraction
Q = B - B^T
Q repeat cosine stability
hard recovered-order stability
pre-softmax QK direction
post-softmax direction
AV norm
AVW^O contribution norm
zero-head delta NLL
```

不要把 `D≈0` 自动解释为无结构；检查 per-sample `D` 分布是否是正负双峰。

## 6.4 Layer-level 预注册对比

对每个 seed、step、export 计算：

```text
mean |D| per layer
median |D| per layer
max |D| per layer
fraction of heads with |D| > 0.1 / 0.3 / 0.5
mean S per layer
mean Q stability per layer
mean zero-head delta NLL per layer
mean AVW^O norm per layer
```

重点检验：

- H1：`without_none` 中 L0/L1 的显式三角 routing 强于 L3。
- H2：`with_none` 中 L2 出现中层峰值。
- H3：L3 显式三角 routing 较弱，但 residual/OV 或 NLL 贡献不一定弱。
- H4：current frame 方向较弱，而 original diagnostic frame 显著增强。
- H5：不同 seed 中具体 head ID 不稳定，但功能类别可能重复。
- H6：训练过程中强方向首先在早层出现，随后中层形成或强化。

## 6.5 统计方法

不能把所有 head 当成完全独立样本。

使用：

1. seed-level paired summaries；
2. hierarchical bootstrap：
   ```text
   seed → repeat → sample/head
   ```
3. 95% bootstrap CI；
4. 预注册 contrasts，例如：
   \[
   C_{L3} = |D|_{L3} - \operatorname{mean}(|D|_{L0:L2}).
   \]
5. 报告 effect size，不只报告 p-value。

“L3 较弱”只有在多个 seeds 上方向一致、CI 不跨 0 时才能确认。

## 6.6 输出

保存：

```text
phenomenon_validation/
├── head_metrics_long.csv
├── layer_metrics.csv
├── seed_layer_summary.csv
├── temporal_trajectories.csv
├── bootstrap_summary.json
├── hypothesis_tests.md
├── all_heads/
│   ├── grids/
│   └── individual/
└── plots/
    ├── layer_abs_direction_by_seed.png
    ├── layer_qk_vs_attention.png
    ├── temporal_layer_trajectory.png
    ├── per_sample_direction_distributions.png
    └── current_vs_original.png
```

---

# 七、Phase B：为什么不同 head 学到不同方向

## 7.1 不要按 head ID 做跨 seed 结论

为每个 head 构造 functional signature：

\[
f_h =
[
D,\,
S,\,
g(d),\,
H(A),\,
m^{none},\,
Q\text{-stability},\,
\text{order stability},\,
\Delta NLL,\,
\|AVW^O\|
].
\]

跨 seed：

1. 标准化 signature；
2. 计算 head-head similarity matrix；
3. 使用 Hungarian matching；
4. 可选使用 CKA 比较 head outputs/residual contributions；
5. 比较“功能类别”而不是相同编号。

报告：

- exact head-ID overlap；
- matched functional similarity；
- matched heads 是否位于同层；
- strong-left / strong-right / local-band / None-heavy / neutral 等 cluster 是否跨 seed 重现。

输出：

```text
cross_seed/
├── functional_signatures.csv
├── similarity_matrices/
├── hungarian_matches.csv
├── cluster_assignments.csv
├── cka_results.csv
└── cross_seed_interpretation.md
```

---

# 八、Phase C：原因分解实验

## C1. QK geometry vs softmax

对相同输入保存：

\[
Z_h = Q_hK_h^\top/\sqrt d
\]

和：

\[
A_h = \operatorname{softmax}(\operatorname{mask}(Z_h)).
\]

只在有效 causal exposure 上比较：

```text
D(Z)
D(A)
S(Z)
S(A)
QK repeat stability
attention repeat stability
```

解释：

- QK 已强、attention 也强：方向来自 learned QK geometry。
- QK 弱、attention 强：softmax 放大微小偏置。
- QK 稳定但 attention 不稳定：row competition 或 context mass 变化。
- 二者都弱：三角图可能来自聚合/exposure artifact。

输出：

```text
causal/qk_softmax/
```

## C2. Position / target-position

同一 checkpoint、同一 tokens、同一 reveal orders，运行：

```text
full
no_wpe
no_wtpe
no_wpe_no_wtpe
```

测量所有 heads 的：

```text
ΔD
ΔS
ΔQ stability
ΔAVW^O norm
ΔNLL
```

注意：这是 inference-time OOD ablation，必须这样标注。

为增强因果证据，至少训练一个小规模 from-scratch control：

```text
standard
no_wpe
no_wtpe
```

保持其他设置一致，比较 10k checkpoint。若算力不允许完整 50k，10k 结果可用于机制，不得冒充最终语言模型结果。

输出：

```text
causal/position/
```

## C3. Content vs coordinate

设计两个干预：

### Block content shuffle within sample

打乱 block token content，但保持：

- block coordinate；
- reveal-order seed；
- position IDs；
- target-position IDs。

输入与 target 同步变换，使 loss 可计算。该干预破坏自然语言邻接内容，但保留坐标结构。

### Same-position cross-sample shuffle

在 batch 内交换不同样本的同一 block content，保留位置分布但破坏 sequence-level语义。

比较：

```text
clean
content_shuffle_within
content_shuffle_across
```

若三角方向在内容破坏后仍保留，更支持 coordinate-driven；若显著消失，更支持 content-driven。

输出：

```text
causal/content/
```

## C4. Reveal mask / exposure

固定同一批文本，对每个样本生成至少 32 组 reveal orders。

比较：

```text
uncorrected attention
exposure-corrected attention
per-reveal direction
repeat direction variance
```

如果 exposure correction 后方向大幅衰减，必须将原现象部分解释为 mask/exposure effect。

输出：

```text
causal/reveal_exposure/
```

## C5. Fixed permutation lookup

运行：

1. 训练 permutation seed 下的正常 probe；
2. inference-only unseen permutation probe；
3. 若算力允许，训练 per-sample-changing-permutation control。

解释：

- 方向绑定旧 current coordinate：支持固定 permutation lookup；
- 换 permutation 后仍能映射回真实 original relation：更支持内容/关系驱动；
- inference-only unseen permutation 属于 OOD，必须与 retrained control 分开解释。

输出：

```text
causal/permutation/
```

---

# 九、Phase D：V、OV 与功能贡献

## 9.1 捕获定义

对每个 head：

\[
V_h = XW_h^V
\]

\[
O_h = A_hV_h
\]

\[
C_h = O_hW_h^O
\]

其中 \(C_h\) 是 head 实际写回 residual stream 的贡献。

保存并报告：

```text
||V_h||
||O_h||
||C_h||
cosine(C_h, residual_update)
zero-head delta NLL
```

## 9.2 单 head zero ablation

将：

\[
C_h \leftarrow 0
\]

分别在：

```text
Random order
head recovered order
reverse order
```

下测 NLL。

热力图强但 `ΔNLL≈0` 的 head 不得称为功能关键 head。

## 9.3 Routing 与 payload 分离

在 clean 与 corrupted 条件间做 activation patching：

1. 只 patch \(A_h\)；
2. 只 patch \(V_h\)；
3. patch 完整 \(O_h=A_hV_h\)；
4. patch \(C_h=O_hW_h^O\)。

观察：

```text
下游 head direction 是否恢复
teacher/order 是否恢复
NLL 是否恢复
```

由此区分：

- routing 重要；
- value payload 重要；
- 两者组合重要。

输出：

```text
causal/value_ov/
```

---

# 十、Phase E：多个 heads 如何相互作用

## 10.1 机制背景

同一层：

\[
Y_l
=
\operatorname{Concat}(O_1,\ldots,O_H)W^O
=
\sum_h O_hW_h^O.
\]

同层 heads 并行计算，不直接读取彼此当前输出。它们在：

```text
concat → c_proj → residual
```

处第一次混合；下一层的 Q/K/V 再读取该混合 residual。

## 10.2 Head cluster

按 functional signature 聚类：

```text
strong positive direction
strong negative direction
local-band
None-heavy
content-sensitive
position-sensitive
neutral/weak
```

不要用 original tau 训练 cluster；可使用 all-head diagnostic direction 描述 cluster。

## 10.3 单头与 cluster ablation

计算：

\[
\Delta L_h
=
L_{\text{zero }h}-L_{\text{clean}}
\]

\[
\Delta L_C
=
L_{\text{zero cluster }C}-L_{\text{clean}}.
\]

解释：

- 单 head 大：专门化；
- 单 head 小、cluster 大：冗余；
- 三角强但两者都小：表象性结构；
- 正向和负向 cluster 同时移除远大于单独移除：互补。

## 10.4 Pairwise synergy

对选定 head 对 \(a,b\)：

\[
\operatorname{synergy}_{a,b}
=
\Delta L_{a,b}
-
\Delta L_a
-
\Delta L_b.
\]

报告：

- synergy > 0：联合移除损害超加和，可能协同；
- synergy < 0：功能冗余或替代。

## 10.5 Head-to-head influence matrix

对 source head \(a\) 做 ablation，重新测所有下游 heads \(b\)：

\[
I^{D}_{a\rightarrow b}
=
D_b^{(-a)}-D_b^{clean}
\]

\[
I^{Q}_{a\rightarrow b}
=
1-\cos(Q_b^{(-a)},Q_b^{clean})
\]

\[
I^{C}_{a\rightarrow b}
=
1-\cos(C_b^{(-a)},C_b^{clean})
\]

以及整体：

\[
I^{L}_{a}=L^{(-a)}-L^{clean}.
\]

生成：

```text
source head × downstream head
```

影响矩阵。重点检查早层方向 heads 是否影响中层方向 axes。

输出：

```text
interactions/
├── single_head_ablation.csv
├── cluster_ablation.csv
├── pairwise_synergy.csv
├── downstream_influence_D.csv
├── downstream_influence_Q.csv
├── downstream_influence_C.csv
├── influence_heatmaps/
└── circuit_hypotheses.md
```

---

# 十一、Phase F：最后一层“显式 attention 弱”是否等于“信息少”

这是本任务必须单独回答的问题。

## 11.1 Residual linear probes

对每层 residual block representation 训练只用于诊断的简单线性 probes：

```text
probe current block ID
probe original diagnostic rank
probe direct-asym-eig teacher rank
probe left/right pairwise precedence
```

严格划分 train-probe/test-probe，不用 validation PPL 选 probe。

报告每层：

```text
rank R² / Spearman
pairwise accuracy
cross-seed stability
```

## 11.2 比较三类指标

对每层同时画：

```text
显式 attention |D|
OV contribution norm
zero-head/cluster ΔNLL
residual probe accuracy
```

可能的结论模式：

- L3 `|D|` 低，但 residual probe 高：顺序信息已被前层写入 residual，L3 不需显式三角 routing。
- L3 `|D|`、probe、OV、ΔNLL 都低：才支持最后一层与顺序机制关联较弱。
- L3 `|D|` 低但 ΔNLL 高：最后一层重要，但使用的不是简单左右三角模式。

输出：

```text
layer_information/
├── residual_probe_metrics.csv
├── attention_vs_residual_vs_ov.csv
├── layer_information_plots/
└── last_layer_interpretation.md
```

---

# 十二、Head 选择与 reference heads

正式 census 使用全部 heads。

深入机制 probe 使用两套选择：

## Fixed historical references

为与已有结果衔接，可固定检查：

```text
L1H2
L0H6
L0H7
L2H6
L2H7
```

必须标记为 historical references，不代表跨 seed 固定功能。

## Per-seed current-frame selection

每 seed 选择：

- top-2 strong/stable current-frame asymmetric heads；
- top-2 highest Q-stability heads；
- 2 neutral matched controls；
- 通过 functional matching 找到的跨 seed counterparts。

不得只挑 original-frame 最漂亮的图作为唯一证据。

---

# 十三、必须生成的总目录

```text
Report/Head_imformation/
├── README.md
├── files_read.md
├── hypotheses.md
├── methodology.md
├── run_manifest.json
├── commands.sh
├── environment/
├── git_state/
├── repro/
│   ├── scripts/
│   ├── configs/
│   └── patches/
├── phenomenon_validation/
├── cross_seed/
├── causal/
│   ├── qk_softmax/
│   ├── position/
│   ├── content/
│   ├── reveal_exposure/
│   ├── permutation/
│   └── value_ov/
├── interactions/
├── layer_information/
├── tables/
├── figures/
├── final_report.md
├── executive_summary.md
├── limitations.md
└── failure_analysis.md
```

---

# 十四、建议新增脚本

优先复用已有脚本，但可在隔离 worktree 新建：

```text
scripts/analysis/head_information_census.py
scripts/analysis/head_information_qk_ov_capture.py
scripts/analysis/head_information_causal_ablation.py
scripts/analysis/head_information_cross_seed_match.py
scripts/analysis/head_information_interaction_matrix.py
scripts/analysis/head_information_residual_probe.py
scripts/analysis/summarize_head_information_report.py
```

所有脚本必须：

- 支持 `--device`；
- 支持 `--dtype`；
- 支持固定 sample manifest；
- 支持断点续跑；
- 避免重复计算；
- 检查 tensor finite、shape、permutation 合法性；
- 自动写 config 与命令；
- 输出 CSV/JSON，而不只输出图片。

---

# 十五、正式结果的最低要求

至少完成：

1. 3 个 pure Random seeds；
2. 10k checkpoint 的 all-head census；
3. 至少一个 seed 的 temporal checkpoints；
4. with_none / without_none / without_none conditional；
5. current/original frame；
6. exposure correction；
7. pre-softmax QK vs post-softmax attention；
8. no_wpe / no_wtpe inference ablation；
9. content shuffle；
10. single-head zero ablation；
11. AVW^O norm；
12. cross-seed functional matching；
13. layer residual probes；
14. 至少一个 head-to-head downstream influence experiment。

若某一项因 checkpoint 或算力缺失无法完成，必须在 `limitations.md` 中明确列出，不得用猜测补全。

---

# 十六、最终报告必须回答

`final_report.md` 必须逐条回答：

1. “最后一层显式三角 attention 较弱”是否跨 seed 成立？
2. 该现象是 `with_none` 和 `without_none` 都成立，还是只对某一视图成立？
3. L3 弱的是 attention routing、OV contribution、residual information，还是仅 triangle metric？
4. 强方向首先在哪一层、哪个训练阶段形成？
5. 不同 seed 中具体 head ID 是否稳定？
6. 功能类别是否可通过 Hungarian/CKA 跨 seed 对齐？
7. 上/下三角主要来自 position、target-position、content、reveal exposure 还是 QK geometry？
8. softmax 是否放大了 pre-softmax QK 中的弱方向？
9. `with_none` 与 `without_none` 的差异有多少来自 None mass 与 predictor shift？
10. 强三角 head 是否对 NLL 有因果贡献？
11. V/OV 是否携带与 heatmap 不同的信息？
12. 多个 heads 是专门化、互补、冗余还是协同？
13. 早层 heads 是否通过 residual 影响中层/后层方向？
14. 当前证据支持什么机制叙事？哪些仍只是相关性？
15. 哪些结果能够指导 Distribution teacher、MLP head selection 与 distillation？

---

# 十七、报告措辞规范

允许写：

> Across the tested seeds, explicit original-frame triangular routing was concentrated in early/middle layers, while the final layer showed weaker triangle indices.

前提是统计支持。

不允许直接写：

> The final layer has no order information.

除非 attention、OV、residual probe 和 causal ablation 都支持。

允许写：

> Head identities were seed-dependent, while matched functional signatures were more stable.

前提是跨 seed matching 支持。

不允许写：

> L1H2 is universally the language-direction head.

只能写：

> L1H2 is a strong teacher in the tested checkpoint/configuration.

---

# 十八、执行结束

1. 保存临时代码 patch；
2. 复制可复现脚本/config 到 `Report/Head_imformation/repro/`；
3. 关闭所有进程；
4. 删除隔离 worktree；
5. 重新记录原始 repo git 状态；
6. 确认除了新建 `Report/Head_imformation/` 外，用户原始工作树未被修改；
7. 更新 `executive_summary.md`，用一页总结：
   - 已确认现象；
   - 被否定现象；
   - 最强因果证据；
   - 最重要限制；
   - 下一步。

不要只返回口头总结。必须实际运行、保存数据、图片、表格、脚本、命令和最终报告。
