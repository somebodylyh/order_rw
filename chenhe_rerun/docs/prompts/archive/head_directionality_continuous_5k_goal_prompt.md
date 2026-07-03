> Archive note: this prompt is a historical single-head continuous
> directionality study. It can inform mechanism analysis, but the current
> distribution mainline is L0 layer-mean pairwise-max Fiedler with EMA/no-EMA
> ablations.

# Prompt：WikiText103 / AO-GPT 单头 Attention 连续方向性研究

/goal

你是一位严谨的 ML research engineer / mechanistic interpretability researcher。请在
`/home/chenhe/nanogpt-learned-order` 中进入 **Goal + Plan 模式**，推进一项只针对
WikiText103 语言方向的真实实验。

请用中文和我沟通；代码、文件名、命令、日志字段保持英文。不要泛泛讨论 attention
或 learned order。你必须阅读项目记录、检查现有实现、完成必要的临时代码改动、运行
真实训练和真实 probe、保存所有 artifacts、写完整报告，并在实验结束后安全清理所有
临时代码和配置改动。

任何没有实际运行完成的实验、没有产生的指标、没有验证的机制都不得写成成功。
Smoke test 只能用于排错，不能作为研究结论。

---

## 0. 本轮任务一句话

本轮首先不研究 spectral order、distribution 聚合或 MLP order 本身，而是验证并解释：

> 在 AO-GPT 使用大量 random reveal orders 训练时，单个 attention head 的原始 attention
> 在 remap 回 current/original block coordinates 后，是否会在连续 5000 个 optimizer
> steps 中持续表现为上三角、下三角或方向波动；这一现象在 `with_none` 和
> `without_none` 两种 export 中是否都存在、是否一致、何时形成、何时翻转，以及它
> 最终由 reveal mask、position、content、QK routing、softmax、OV 写回还是 head
> interaction 导致。

这里研究的是 **单个 head 的最基本 attention matrix**，不是：

- 对 attention 无向化后的 graph；
- spectral eigenvector 的符号；
- spectral candidate 的 forward/reverse；
- cross-head aggregation 后的 order；
- distribution 的 `priority_ema`；
- MLP 的 logits/order；
- 用 original tau 选 head 或选实验。

---

## 1. 不可违反的研究边界

### 1.1 只研究语言方向

本轮只允许研究：

```text
WikiText103
AO-GPT
seq256 / block64 为主
单头 attention directionality
连续训练动态
```

除非用户明确要求，不要扩展到已删除的非语言分支。

### 1.2 No-prior / diagnostic 边界

训练、probe order 生成、head selection、change-point detection、ablation selection 和
checkpoint selection 不得使用：

```text
OriginalL2R order 作为训练目标
original-frame Kendall tau 作为选择信号
OriginalL2R PPL 作为选择信号
human scan
手写 original-order labels
validation PPL 作为 head 选择或方向选择信号
```

允许将 current-frame attention remap 到 original frame，以便**诊断**其是否呈现文本
上三角/下三角，但必须明确：

```text
original-frame fields are diagnostics only
```

代表 head 的选择必须基于直接 attention 现象：

```text
triangle direction index
bootstrap stability
with_none / without_none consistency
continuous temporal trajectory
```

不能按 tau 选择。

### 1.3 第一阶段禁止的方法

在“现象是否存在”验证完成前，不得使用以下方法定义 head 方向：

```text
symmetrization
无向 affinity graph
spectral decomposition
candidate order recovery
loss rerank
cross-head consensus
distribution priority
MLP order
```

第一阶段只能分析单个 head 的原始 QK/attention。

---

## 2. 先阅读的项目记录

开始任何实现或实验前，按下面顺序阅读。若某个路径在当前分支不存在，先使用
`find`、`git ls-files` 或 repo 内搜索定位最新等价文件，并把实际阅读路径记录在
`files_read.md` 中，不得凭空假设内容。

### 2.1 项目与语言方向入口

```text
README.md
base.md
base_cn.md
research-state.yaml
research-log.md
literature/survey.md
docs/prompts/Prompt_language.md
docs/prompts/prompt_online_language_current.md
docs/prompts/prompt_pro.md
docs/findings/findings_language.md
docs/structure/PROJECT_STRUCTURE_language.md
config/WikiText103/README.md
```

### 2.2 当前 order / distribution / MLP 背景

这些文件用于理解项目现状和 frame 约束，不得把它们的 spectral/MLP 输出当成本轮
head 方向现象：

```text
online_spectral_order_policy.py
attn_mlp_order_policy.py
docs/prompts/prompt_MLPTRAIN.md
docs/prompts/prompt_attn_mlp_training_no_prior_goal.md
docs/prompts/prompt_attn_mlp_loss_design_continuation.md
```

重点读当前最新相关 configs，包括但不限于：

```text
config/WikiText103/seq256/permute/block64/random.py
config/WikiText103/seq256/non_permute/block64/random.py
config/WikiText103/seq256/non_permute/block64/
  online_spectral_try19_bridge_oriented_distribution_map_update20_start8k_anneal18k_freeze35k.py
config/WikiText103/seq256/permute/block64/
  attn_mlp_try22_fromscratch_headprofile_paramtuned_midfreeze15k_randombase_50k.py
```

### 2.3 核心模型和 attention 坐标实现

```text
train.py
AOGPT.py
AOGPT_block.py
AOGPT_token.py
order_utils.py
```

必须定位并理解：

```text
attention tensor 的 row/query 与 column/key 语义
[None] token 的插入位置
causal mask
with_none / without_none 的切片方式
token attention → block attention 的聚合方式
reveal-frame → current-frame 的 inverse remap
current-frame → original-frame 的 diagnostic mapping
gradient accumulation 与 optimizer step 的关系
```

重点检查现有函数：

```text
_aggregate_layerhead_attention_to_current_blocks
_aggregate_layerhead_attention_to_current_blocks_differentiable
_forward_with_explicit_block_orders
_forward_with_active_training_policy
_attn_mlp_extract_attentions
```

### 2.4 现有 head 方向性记录和脚本

优先阅读并核对这些记录；若路径有拼写差异，搜索等价文件：

```text
Report/analysis/wikitext103_signed_head_alignment_20260607_4096attn_4096loss_3seeds/README.md
Report/analysis/wikitext103_signed_head_alignment_20260607_4096attn_4096loss_3seeds/distillation_spec.md
Report/analysis/wikitext103_signed_head_alignment_20260607_4096attn_4096loss_3seeds/method_summary.csv

Report/head_singal_Stable/
Report/analysis/attn/
```

重点读现有分析脚本：

```text
scripts/analysis/head_attention_gradient_orientation_diagnostic.py
scripts/analysis/head_candidate_loss_profile_diagnostic.py
scripts/analysis/head_orientation_loss_flow_diagnostic.py
scripts/analysis/summarize_head_signal_stability.py
scripts/analysis/head_angle_tau_diagnostic.py
scripts/analysis/seq80_random_ckpt_online_order_attn_batch.py
scripts/analysis/seq80_loss_rerank_existing_attention_candidates.py
```

阅读目的：

1. 复用已经正确实现的 attention extraction/remap；
2. 避免重新引入已修复的 frame bug；
3. 明确现有 4096 样本结果究竟统计了什么；
4. 找出 `with_none` 和 `without_none` 目前是否来自同一次 forward；
5. 找出当前结果是离散 checkpoint 还是连续训练观测。

---

## 3. 仓库隔离与实验结束后的强制回撤

本轮允许为了实验临时修改代码，但**不得污染用户当前工作树，也不得永久修改其他
config**。最终允许保留的只有新建的 Report 子目录及其中的实验 artifacts。

### 3.1 开始前记录用户工作树

在原始 repo 中记录：

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git status --porcelain=v1 -uall
git diff
git diff --cached
```

保存到本轮 Report 目录：

```text
git_state/
  branch_before.txt
  head_before.txt
  status_before.txt
  diff_before.patch
  diff_cached_before.patch
```

注意：

- 用户可能已有未提交修改；
- 不得覆盖、stash、reset 或 clean 用户已有修改；
- 严禁在原始工作树运行 `git reset --hard`；
- 严禁在原始工作树运行 `git clean -fd`；
- 严禁用 `git checkout -- .` 粗暴回滚。

### 3.2 使用隔离 worktree

默认使用独立 detached worktree：

```bash
SOURCE_REPO=/home/chenhe/nanogpt-learned-order
BASE_HEAD=$(git -C "$SOURCE_REPO" rev-parse HEAD)
WORKTREE=/tmp/nanogpt-head-direction-<RUN_ID>

git -C "$SOURCE_REPO" worktree add --detach "$WORKTREE" "$BASE_HEAD"
```

所有临时代码、临时 config 和运行脚本只在 `$WORKTREE` 中修改。

若 worktree 因环境原因不可用，使用本地临时 clone，但仍不得直接修改用户原始工作树。
只有在两种隔离方式都不可行时，才允许在原 repo 进行最小修改；此时必须先逐文件备份，
并在结束时逐文件精确恢复，不能使用全局 reset。

### 3.3 数据和 checkpoint

worktree 中缺失的非 tracked 大文件，只链接具体文件，不覆盖 tracked 目录，例如：

```text
data/wikitext103/train.bin
data/wikitext103/val.bin
必要的 ckpt.pt
```

训练和 probe 的新输出全部写入本轮 Report 目录或其 runtime 子目录。不得写入或覆盖
现有 `out/`、`checkpoints/`、已有 `Report/` 实验目录。

### 3.4 结束时回撤

所有训练/probe 进程结束后：

1. 保存临时代码 patch：

```bash
git -C "$WORKTREE" diff > "$REPORT_DIR/patches/experiment_code.patch"
git -C "$WORKTREE" status --porcelain=v1 -uall \
  > "$REPORT_DIR/patches/worktree_final_status.txt"
```

2. 将需要复现实验的临时脚本/config副本复制到 Report 目录：

```text
repro/
  scripts/
  configs/
  commands.sh
```

3. 删除隔离 worktree：

```bash
git -C "$SOURCE_REPO" worktree remove --force "$WORKTREE"
git -C "$SOURCE_REPO" worktree prune
```

4. 再次记录原始 repo 状态，并与开始前逐项比较。除本轮新建 Report 目录外，原始
工作树的 branch、HEAD、tracked diff 和原有 untracked files 必须完全不变。

5. 写：

```text
changes_manifest.json
rollback.log
```

其中必须明确：

```text
original_branch_unchanged
original_head_unchanged
preexisting_changes_preserved
source_changes_remaining = []
config_changes_remaining = []
only_report_artifacts_remaining = true
```

若无法完成安全回撤，不得声称实验完成；必须在最终回复中明确列出残留文件。

---

## 4. Report 目录规范

在原始 repo 下自动新建，不覆盖已有目录：

```text
Report/analysis/head_directionality_continuous_5k/try_n/
```

其中 `n` 从已有目录自动递增。

最低目录结构：

```text
try_n/
  README.md
  files_read.md
  experiment_design.md
  experiment_result.md
  next_direction.md
  summary.json
  environment.md
  commands.sh

  git_state/
  patches/
  repro/
    scripts/
    configs/

  logs/
    launcher.log
    train.log
    probe.log
    mechanism_ablation.log

  runtime/
    checkpoints/
    out/

  probe_manifests/
    core_probe_manifest.json
    full_probe_manifest.json

  sanity/
    synthetic_remap/
    with_without_index_audit/
    uniform_attention_null/
    label_permutation_null/

  metrics/
    passive_per_step.jsonl
    passive_per_step.csv
    fixed_core_128.jsonl
    fixed_core_128.csv
    full_probe_4096.csv
    head_summary_by_step.csv
    distance_profiles.csv
    sample_direction_distributions.csv
    with_without_comparison.csv
    change_points.csv
    flip_events.json

  matrices/
    all_heads_every100/
    selected_heads_every20/

  plots/
    head_time_series/
    with_without/
    distance_profiles/
    sample_fraction_curves/
    matrix_animations/

  mechanism/
    paired_order/
    qk_softmax/
    position_content/
    head_output_ablation/
```

所有文件和字段都必须写明 frame 和 export type。

---

## 5. 实验环境

工作目录：

```text
/home/chenhe/nanogpt-learned-order
```

Python 优先使用：

```text
/data/users/chenhe/conda_envs/X1/bin/python
```

GPU 优先使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

运行前记录：

```text
python version
torch version
CUDA version
GPU model
free GPU memory
git HEAD
selected base config
checkpoint path
all config overrides
```

不得打印或泄漏任何 W&B key。若启用 W&B，只使用 repo 已有安全机制，并在报告中保存
run URL；不得把 secret 写入报告或最终回复。

---

## 6. 主实验 setting

### 6.1 首选 baseline

主实验必须使用 clean Random baseline，避免 distribution/MLP policy改变 probe order
分布后混淆现象。

首选：

```text
WikiText103
seq256
permute_data = True
block_size = 256
block_order_block_len = 4
num_blocks = 64
n_layer = 4
n_head = 8
aogpt_train_mode = Random
```

基础超参应与：

```text
config/WikiText103/seq256/permute/block64/random.py
```

保持一致。除：

```text
out_dir
日志
临时诊断开关
必要的 checkpoint saving
```

外，不得改变 backbone optimizer、batch size、learning rate、模型规模或 random-order
训练分布。

第二对照为 `non_permute / block64 / Random`，但只有在主实验完成后且资源允许时执行。
不得因为第二对照未完成而伪造结论。

### 6.2 连续观测窗口

主实验必须连续观察完整 5000 个 **optimizer steps**：

```text
observation_start_iter = S
observation_end_iter = S + 5000
```

默认优先观察 random backbone 的：

```text
0 → 5000
```

以研究方向首次形成。

若已有研究记录表明另一个连续窗口更关键，可以选择：

```text
S → S + 5000
```

但必须在 `experiment_design.md` 中用现有 repo 证据说明原因，且不得使用 tau 或
validation PPL 选择窗口。

注意：

- step 指 optimizer step；
- `gradient_accumulation_steps=2` 时，一个 step 包含两个 microbatches；
- passive statistics 必须聚合该 optimizer step 的所有 microbatches，不能只取最后一个。

---

## 7. Phase A：先验证现象真的存在

Phase A 是本轮强制主任务。未完成 Phase A，不得进入或宣称机制解释。

### 7.1 禁止 spectral 处理

该阶段不调用：

```text
robust_z → sym → affinity → eigendecomposition
spectral candidate generation
candidate/reverse orientation
cross-head consensus
distribution priority
MLP
```

只使用单个 head 的原始 attention 和 QK。

### 7.2 三个坐标视图

同一次 forward 必须输出：

```text
reveal-frame
current-frame
original-frame diagnostic
```

其中：

- reveal-frame 天然受 causal mask 影响，只作实现检查；
- current-frame 是 policy实际坐标；
- original-frame 是语言方向诊断。

### 7.3 两个 export 必须同一次 forward 导出

从同一个原始 `layer_attn` 同时导出：

```text
with_none
without_none
```

禁止分别跑两次不同 batch。

必须记录现有实现中两种 slice 的精确定义，并通过单样本 index audit 验证：

```text
with_none
without_none
```

各自对应的 query/key token位置。

---

## 8. Phase A0：现象验证前的 sanity checks

### 8.1 Synthetic remap

人工构造：

```text
纯上三角矩阵
纯下三角矩阵
对称矩阵
随机无方向矩阵
固定 +1 offset
固定 -1 offset
```

经过：

```text
original/current → reveal permutation → inverse remap
```

后必须恢复原矩阵。

报告相对 Frobenius error。未通过时立即停止正式训练，先修复 remap。

### 8.2 `with_none` / `without_none` 单样本 index audit

对 4–8 个真实样本保存：

```text
原 token attention
两种 slice
token→block 聚合
remap 前
remap 后
```

打印明确 index mapping，确认：

```text
row = query
column = key
[None] token 的位置
预测位置与输入位置是否有 offset
首尾 token 是否被丢弃
```

### 8.3 Uniform-visible-attention null

构造一个所有 visible keys 均匀 attention 的人工 head，在与正式实验相同的 fixed random
orders 上执行 remap。Exposure correction 后的三角方向应接近 0。

### 8.4 Label-permutation null

对真实 remap matrix 随机打乱 original block labels，重复至少 200 次，形成每个 head 的
null `|triangle_index|` 分布。

正式方向 effect 必须超过相应 null 阈值。

---

## 9. 连续 5000-step 的三层观测流

### 9.1 Stream A：每个 optimizer step 的 passive stream

直接从真实训练 forward 中提取 attention，每个 optimizer step 记录一次。

要求：

- 聚合所有 gradient-accumulation microbatches；
- 同时统计所有 4×8=32 heads；
- 同时统计 `with_none` / `without_none`；
- 不保存每步完整 token attention；
- 在 forward/hook 内尽快聚合为 block-level metrics 并 detach，避免显存爆炸；
- 诊断代码不得改变 loss 或 gradient。

每步每 head 保存：

```text
lower_triangle_mean
upper_triangle_mean
triangle_index_raw
triangle_index_row_normalized
triangle_index_exposure_corrected
attention_entropy
attention_mass
None-related mass
mean signed distance
mean absolute distance
direction_by_distance_bin
sample_positive_fraction
sample_negative_fraction
```

Stream A 表示真实训练分布下的动态，但包含 batch/order noise。

### 9.2 Stream B：固定核心 probe，每 10 steps

构造固定：

```text
32 个固定 text windows
每个 4 条固定 random reveal orders
总计 128 instances
```

在：

```text
S, S+10, ..., S+4990, S+5000
```

运行。约 501 个高频时间点。

必须：

```text
model.eval()
torch.no_grad()
固定 text ids
固定 token windows
固定 block orders
固定 batch grouping
```

probe 前后保存并恢复：

```text
PyTorch CPU RNG
CUDA RNG
NumPy RNG
data sampler / fixed manifest state
```

probe 不得更新：

```text
backbone optimizer
policy state
attention EMA
priority EMA
MLP
任何训练计数器
```

Stream B 是判断模型参数变化的主时间序列。

### 9.3 Stream C：完整 4096 probe，每 500 steps

固定：

```text
512 个 text windows
每个 8 条固定 random reveal orders
总计 4096 instances
```

运行时间点：

```text
S
S+500
S+1000
...
S+5000
```

共 11 次。

必须使用与 core probe 相同的 full manifest，其中 core 128 是 full 4096 的预先固定子集，
不能在看到结果后选择。

Stream C 用于：

- 校准 core 128 的代表性；
- 确认真实 sign flip；
- 验证 `with_none` / `without_none`；
- 给出 cluster-bootstrap CI；
- 区分小样本噪声与真实参数变化。

---

## 10. 方向指标

设 remap 后：

```text
row = query block i
column = key block j
```

约定：

- 下三角 \(i>j\)：later query reads earlier key，记为 L2R-like routing；
- 上三角 \(i<j\)：earlier query reads later key，记为 R2L-like routing。

### 10.1 基础三角指数

\[
L_h(t)=\operatorname{mean}_{i>j}A^h_{ij}(t)
\]

\[
U_h(t)=\operatorname{mean}_{i<j}A^h_{ij}(t)
\]

\[
T_h(t)=
\frac{L_h(t)-U_h(t)}
{L_h(t)+U_h(t)+\epsilon}.
\]

解释：

```text
T > 0：下三角
T < 0：上三角
T ≈ 0：无整体三角方向
```

必须同时报告 `L`、`U` 和 `T`，避免分母很小时产生虚假大值。

### 10.2 Row-normalized 指标

对每个 query row 归一化后再计算：

```text
T_row
```

用于排除不同 query rows 总 attention mass 差异。

### 10.3 Exposure-corrected 指标

对每个 directed pair 累计可观察次数：

\[
C_{ij}=\sum_s\mathbf 1[r_s(j)<r_s(i)].
\]

仅在 key 已经 reveal、edge 实际可见时累计 attention：

\[
\widehat A^h_{ij}
=
\frac{
\sum_s \mathbf 1[r_s(j)<r_s(i)]A^{h,s}_{ij}
}{
C_{ij}+\epsilon
}.
\]

然后计算：

```text
T_exposure
```

现象确认的主指标使用 `T_exposure`。

### 10.4 Sample-level 分布

每个固定 probe sample 都计算：

```text
T_sample
```

保存：

```text
median
q25
q75
std
positive_sample_fraction
negative_sample_fraction
near_zero_fraction
```

不能只看平均 matrix，因为：

```text
所有样本都无方向
```

和：

```text
一半强正、一半强负
```

可能有相同平均值。

### 10.5 距离分层

至少分：

```text
distance 1
distance 2–3
distance 4–7
distance 8–15
distance 16–31
distance 32–63
```

输出：

\[
T_h(d,t).
\]

用于区分：

```text
局部方向
全局方向
近距离正、远距离负
distance-specific 波动
```

---

## 11. `with_none` / `without_none` 连续联合验证

每个时间点、每个 head 保存：

```text
T_with_raw
T_without_raw
T_with_row
T_without_row
T_with_exposure
T_without_exposure
```

同时保存：

```text
sign agreement
flattened matrix Pearson
flattened matrix Spearman
distance-profile correlation
normalized Frobenius difference
```

定义：

\[
\Delta T_h(t)
=
T_h^{with}(t)-T_h^{without}(t).
\]

分类：

### Export-robust

两种 export：

```text
同号
CI 均不跨 0
轨迹高度相关
full 4096 中复现
```

### None-sensitive

仅一个 export 稳定有方向。

### Export-sign-flip

两种 export 显著反号。

### Coupled temporal flip

两种 export 在相近 step 同时完成 sign flip。

任何只在一个 export 出现的现象都必须明确标注，不能写成普遍 head direction。

---

## 12. 连续时间序列和 sign flip 判定

### 12.1 保存原始曲线

必须保存所有未平滑：

```text
per-step passive
per-10-step core
per-500-step full
```

指标。

### 12.2 平滑只用于事件检测

使用：

```text
EWMA alpha = 0.1
rolling window = 100 optimizer steps
```

不得只展示平滑曲线而隐去 raw 数据。

### 12.3 三状态

每个 head/export 每个时段分类：

```text
Positive
Negative
Uncertain
```

Positive：

```text
rolling T_exposure > δ
cluster-bootstrap 95% CI > 0
positive sample fraction >= 0.70
```

Negative 同理。

\(\delta\) 必须根据 uniform/null distribution 在实验开始前确定，不能看完真实结果后调。

### 12.4 确认 sign flip

必须同时满足：

1. 从显著 Positive 进入显著 Negative，或相反；
2. 新状态持续至少 100 optimizer steps；
3. 至少连续 10 个 core-probe points 同方向；
4. 最近 full 4096 calibration 同方向；
5. 不是仅由 `with_none` 或 `without_none` 单一 export 的 index异常造成。

否则只记录：

```text
transient excursion
amplitude oscillation
uncertain interval
```

### 12.5 Change-point detection

离线对以下曲线运行至少一种 change-point 方法：

```text
PELT
CUSUM
rolling likelihood ratio
```

输入：

```text
T_with_exposure
T_without_exposure
positive fraction
negative fraction
attention entropy
```

输出 `change_points.csv`，并与：

```text
train loss
learning rate
gradient norm
attention entropy
policy probability（若有）
```

对齐。

---

## 13. Matrix 快照和动画

### 13.1 所有 heads

每 100 optimizer steps 保存一次固定 core probe 的 averaged block matrix：

```text
with_none
without_none
current-frame
original-frame diagnostic
```

5000 steps 共约 51 个时间点。

### 13.2 代表 heads

基于前 500 steps 的 discovery subset，预先选：

```text
3 个稳定下三角候选
3 个稳定上三角候选
3 个波动候选
3 个中性 control
```

选择依据只能是 direct triangle metrics，不得使用 tau。

这些 heads 每 20 steps 保存 averaged matrix。

### 13.3 可视化要求

动画和 heatmap 必须：

```text
固定色阶
固定坐标
固定 export 标签
标出 optimizer step
标出 T_raw / T_row / T_exposure
```

禁止每帧自动缩放色阶。

---

## 14. Formal run 前的 smoke test

先运行：

```text
20 optimizer steps
core probe 2 次
full probe 可缩小到 64 samples
```

只检查：

```text
shape
memory
RNG restore
with/without 同次导出
remap
日志
Report 写入
```

Smoke 通过后必须启动正式 5000-step run。不得把 smoke 写成研究结论。

还要做一个短对照：

```text
相同初始 checkpoint和seed
诊断关闭的10–20步
诊断开启的10–20步
```

验证诊断 hook 不改变 loss、梯度或参数更新，允许只有浮点级误差。

---

## 15. Phase A 的成功条件

只有满足以下条件，才允许写“单头原始 attention 的上/下三角和波动现象存在”：

1. 完整连续 5000 optimizer steps 已完成；
2. 32 heads 全部被观测；
3. `with_none` 和 `without_none` 都有结果；
4. reveal/current/original 三个 frame 明确区分；
5. synthetic remap 通过；
6. uniform-attention null 通过；
7. 至少 11 次 full 4096 calibration 完成；
8. core 128 和 full 4096 对 head 轨迹结论一致；
9. 独立 confirmation subset 或后半数据复现；
10. stable/flip/oscillation 的定义来自统计标准，而不是肉眼图；
11. 报告 `with_none` / `without_none` 不一致的 heads；
12. 所有 head 分类均附 effect size、CI 和 sample fraction。

---

## 16. Phase B：现象确认后的机制消融

Phase B 只对 Phase A 中确认的代表 heads执行：

```text
至少 2 个 export-robust 下三角 heads
至少 2 个 export-robust 上三角 heads
至少 2 个确认 sign-flip / 波动 heads
至少 2 个 neutral controls
```

如果某一类别不存在，必须如实报告，不得强行选择。

### 16.1 Paired-order controlled probe

对同一文本、同一 pair \((i,j)\)、同一 prefix set \(S\)：

```text
S, i, j, remainder
S, j, i, remainder
```

控制：

```text
query reveal rank
visible context size
prefix set
remainder
```

同时记录：

```text
pre-softmax QK
post-softmax attention
with_none
without_none
```

比较多个 prefix size：

```text
0
8
16
32
```

判断 random-stream方向是否来自 reveal mask/context-set。

### 16.2 QK vs softmax

增加临时 hooks，至少返回代表 heads 的：

```text
q
k
pre-softmax qk logits
attention probabilities
```

判断：

- QK 和 attention 都稳定：QK directional circuit；
- QK 接近 0、attention 有方向：softmax competition；
- QK稳定、attention波动：visible-context competition。

### 16.3 Position/content factorial

在固定 checkpoint、固定 paired probes 上做 inference-only消融：

```text
Full
No-WPE
No-WTPE
No-WPE + No-WTPE
Reverse-WPE
Reverse-WTPE
Block-content shuffle
Text block reverse
Within-block token shuffle
Random-token control
```

所有条件同时报告：

```text
with_none
without_none
QK direction
attention direction
clean loss shift
```

### 16.4 Head-output ablation

对代表 heads 做：

```text
zero ablation
dataset-mean ablation
```

分别评估：

```text
forward-like paired orders
reverse-like paired orders
Random orders
prefix-4/8/16 loss
full loss
```

用：

\[
\Delta L_h^F
\quad\text{和}\quad
\Delta L_h^R
\]

判断 attention routing 方向是否具有功能因果性。

### 16.5 事件 checkpoint

为机制消融保存：

```text
window start
稳定方向形成前
change point 前
change point 附近
change point 后
window end
```

模型 snapshot。

为控制存储，使用：

```text
每250 steps model-only snapshot
或 event-triggered ring buffer
```

不得覆盖现有 checkpoint。

---

## 17. 统计规范

### 17.1 Bootstrap 单位

以完整 text window及其全部 reveal orders为 cluster bootstrap 单位，不能把 attention edge
当成独立样本。

### 17.2 Discovery / confirmation

```text
core 128：高频 discovery
full 4096：formal confirmation
```

代表 head 在 core 中选择，结论必须由 full 4096确认。

### 17.3 多重检验

32 heads × 多时间点 × 两种 export × 多 distance bins，使用 Benjamini–Hochberg FDR，
并同时报告 effect size 和 CI。

### 17.4 波动来源分解

利用固定：

```text
512 texts × 8 orders
```

分解：

```text
text variance
order variance
time/model variance
residual variance
```

至少给出每个 head 的近似 variance decomposition。

---

## 18. 必须记录的训练与系统信息

每个 optimizer step或相应频率记录：

```text
iter
train loss
learning rate
gradient norm
active training policy
batch/order seed
diagnostic overhead
GPU memory
```

每个 head记录：

```text
layer
head
export_type
frame
T_raw
T_row
T_exposure
lower_mean
upper_mean
attention entropy
sample positive fraction
sample negative fraction
distance-bin metrics
```

每个 full probe记录：

```text
num_texts
orders_per_text
actual sequences
probe manifest hash
bootstrap CI
null percentile
```

---

## 19. 报告内容

### `experiment_design.md`

必须写：

```text
research question
phenomenon definition
coordinate convention
with_none/without_none definition
continuous 5k design
probe budgets
null controls
sign-flip criteria
mechanism ablations
no-prior boundary
full command
```

### `experiment_result.md`

必须写：

```text
formal 5000 steps 是否完成
实际观测时间点数量
actual 4096 probe 次数
每类 head数量
稳定上三角 heads
稳定下三角 heads
sign-flip heads
amplitude-oscillation heads
neutral heads
with/without agreement
export-sensitive heads
null-control结果
core/full一致性
change points
mechanism ablation结果
失败或缺失内容
```

### `summary.json`

至少包含：

```json
{
  "try_id": "",
  "base_head": "",
  "base_config": "",
  "observation_start_iter": 0,
  "observation_end_iter": 5000,
  "formal_steps_completed": 0,
  "core_probe_points": 0,
  "full_probe_points": 0,
  "full_probe_sequences_per_point": 4096,
  "num_heads": 32,
  "with_none_completed": false,
  "without_none_completed": false,
  "synthetic_remap_passed": false,
  "uniform_null_passed": false,
  "stable_lower_heads": [],
  "stable_upper_heads": [],
  "confirmed_flip_heads": [],
  "amplitude_oscillation_heads": [],
  "neutral_heads": [],
  "export_sensitive_heads": [],
  "mechanism_ablations_completed": [],
  "original_repo_restored": false,
  "only_report_artifacts_remaining": false,
  "success": false,
  "next_direction": ""
}
```

### `next_direction.md`

必须根据证据判断下一步属于：

```text
修 export/remap
扩大 fixed probe
进一步 paired-order
position/content来源
QK/softmax来源
OV功能
head interaction
跨seed复现
```

不能泛泛写“继续优化”。

---

## 20. 运行监督

正式运行期间必须检查：

```bash
nvidia-smi -i 1
ps -p <PID>
tail -f train.log
tail -f probe.log
```

若训练、probe、W&B 或保存进程异常退出，必须定位原因、修复并从安全 checkpoint恢复。

不得在关键训练或 full probe 仍运行时提交最终回复。

---

## 21. 退出条件

只有同时满足以下条件，才允许 mark goal complete：

1. 已阅读并记录关键 repo 文件；
2. 已建立隔离 worktree；
3. Phase A sanity checks 完成；
4. 连续 5000 optimizer steps 正式观测完成；
5. passive、core 128、full 4096 三条流均有 artifacts；
6. `with_none` 和 `without_none` 均完成；
7. 32 heads 全部分类；
8. sign flip 使用严格持续性和 full-probe确认；
9. 至少完成 Phase B 的 paired-order与QK/softmax消融；
10. 对代表 heads至少完成一项 position/content消融和head-output ablation；
11. 写完全部报告和 summary；
12. 所有临时代码/config改动已从用户原工作树回撤；
13. 原 branch、HEAD 和原有未提交修改保持不变；
14. 只有新建 Report 子目录保留。

若资源不足导致 Phase B 未全部完成，Phase A 完整结果仍可作为阶段成果，但不得把未完成的
机制解释写成结论；必须在 `summary.json` 和最终回复中明确。

---

## 22. 最终回复格式

最终用中文给出：

1. 新建的 `try_n` Report 路径；
2. 实际阅读的关键文件；
3. 使用的 base config、checkpoint和连续窗口；
4. formal 5000 steps 是否完成；
5. core/full probe 的实际样本数和时间点；
6. `with_none` / `without_none` 的主要现象；
7. 稳定上三角、下三角、flip、oscillation和neutral heads；
8. 现象是否通过 null controls；
9. paired-order、QK/softmax、position/content和head ablation的结果；
10. 所有训练/probe命令；
11. W&B URL，如实际使用；
12. 失败、未完成和不确定部分；
13. rollback结果；
14. 明确确认原始 repo 是否恢复到改动前状态。

不要只说“完成了”。必须给出路径、实际数量、指标、图表位置和 caveats。
