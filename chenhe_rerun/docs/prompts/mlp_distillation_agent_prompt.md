# Agent Prompt: WikiText103 MLP Distillation for Continuous Fiedler Scores

你正在处理仓库：

```text
/home/chenhe/nanogpt-learned-order
```

当前主线是 WikiText103 / seq256 / block64 的 learned-order 语言实验。最新 distribution 方法论不是旧的 `L1H2 direct_asym_eig`，而是：

```text
L0 all-head mean attention
-> without_none
-> W = max(A, A.T)
-> graph Laplacian L = D - W
-> Fiedler vector, i.e. second-smallest eigenvector
-> choose raw/reverse direction by current-model loss profile
-> convert to continuous priority / order
```

本 prompt 的目标是指导 agent 完成当前正在推进的 **Attn-MLP 蒸馏**：优先让 MLP 学会从 attention 直接拟合 Laplacian/Fiedler 分解产生的 **连续分数**，而不是一开始就学离散 order 或 pairwise labels。

---

## 0. Hard GPU Constraint

所有会使用 GPU 的命令必须且只能使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

这是硬约束。

必须遵守：

- 不允许使用 `CUDA_VISIBLE_DEVICES=0`。
- 不允许使用 `CUDA_VISIBLE_DEVICES=0,1` 或任何多卡设置。
- 不允许在 1 卡繁忙时自动 fallback 到 0 卡。
- 不允许在脚本内部覆盖成 0 卡。
- 所有 `python train.py`、MLP 训练、eval、downstream backbone 训练、tmux/nohup/background 脚本都必须显式带上 `CUDA_VISIBLE_DEVICES=1`。
- 如果已有脚本名里带 `cuda0`，文件名本身不构成许可；运行时仍必须用 `CUDA_VISIBLE_DEVICES=1`，必要时新建/复制一份 `cuda1` run script。
- 在 Python 内部优先使用 `--device cuda`。在 `CUDA_VISIBLE_DEVICES=1` 约束下，进程内的 `cuda:0` 对应物理 GPU1。

如果 GPU1 不可用或已有任务占用，请暂停并汇报，不要改用其他 GPU。

---

## 0.1 Goal Mode And Exit Criteria

本任务必须以 `/goal` 模式推进。Agent 开始执行时，应创建一个明确 goal：

```text
Distill an Attn-MLP for continuous Fiedler scores, plug the validated frozen MLP into WikiText103 block64 backbone training, run at least two downstream seeds on CUDA_VISIBLE_DEVICES=1, and stop only when the mean final validation loss is <= 3.45 or when the experiment is genuinely blocked.
```

不要把“完成配置”当作 goal 完成。真正可退出条件是：

```text
1. MLP score distillation validation 已完成并通过；
2. frozen MLP downstream 至少跑完两组 seed；
3. 两组或更多 seed 的 final validation loss 均值 <= 3.45；
4. report 中记录了每组 seed 的 final val loss、均值、命令、checkpoint、W&B run。
```

这里使用的是 **final validation loss** 的均值，不是 best validation loss。即使某一组 seed 的 best val loss 很好，只要 final val loss 均值没有达到 `<= 3.45`，就不能把 goal 标记为完成。

如果两组 seed 完成后：

```text
mean(final_val_loss) <= 3.45
```

则可以停止并总结。

如果没有达到，则必须写清楚失败原因，并开启下一版 try。不要在没有达到门槛时提前退出。可以运行超过两组 seed，但不能少于两组 seed。

---

## 1. Current Project State

当前已有可复用数据，不要默认重新收集。

### 1.1 Existing Distillation Dataset

现成数据路径：

```text
Report/language/wikitext103/mlp/distillation/try_25/combined_try20_try24_l0_layermean_fiedler_dataset
```

当前数据规模：

```text
train: 13000 samples
val:   3000 samples
```

每个样本核心字段：

```text
attention:      L0 layer-mean block attention, shape [64, 64]
teacher_order: loss-profile oriented Fiedler order
teacher_rank:  inverse rank of teacher_order
raw_order / reverse_order
raw_rank / reverse_rank
pair_i / pair_j / pair_target
selected_reverse
loss_score_gap
eigval_real / eigval_imag / eigval_abs
vector_std
pairwise_Q
```

注意：这批数据保存了 attention 和离散 teacher/order 信息，但没有保存完整的 raw Fiedler vector。因此当前 score distillation 应该从 `attention` 重新计算 continuous Fiedler score label，并缓存到新的 try 目录。

### 1.2 Current Score-Distillation Try

当前已配置的 score-distillation try：

```text
Report/language/wikitext103/mlp/distillation/try_33
```

训练脚本：

```text
scripts/train/train_mlp_fiedler_score_distillation.py
```

计划输出 checkpoint：

```text
checkpoints/attn_mlp_distillation/try33_joint_try20_try24_l0_layermean_fiedler_score_mlp_h2048_1024/best_by_val_score.pt
```

兼容现有 loader 的 checkpoint 名：

```text
checkpoints/attn_mlp_distillation/try33_joint_try20_try24_l0_layermean_fiedler_score_mlp_h2048_1024/best_by_val_tau.pt
```

本 try 的 target cache 应生成在：

```text
Report/language/wikitext103/mlp/distillation/try_33/target_cache
```

如果 cache 不存在，这是正常的。第一次训练时应从 try_25 attention 生成。

---

## 2. Main Objective

优先完成以下流程：

```text
reuse try_25 L0 layer-mean attention dataset
-> regenerate continuous Fiedler priority labels
-> train FlatAttentionOrderMLP with continuous score MSE
-> validate that MLP approximates the Fiedler-score operator
-> run external diagnostic validation on head_information artifacts
-> only if validation passes, plug frozen MLP into backbone training
-> run at least two downstream seeds
-> stop only if mean final validation loss <= 3.45
-> report results and failure modes cleanly
```

当前不要优先重新收集数据，除非明确证明 try_25 数据损坏、不匹配或数量不足。

---

## 3. Method Definition

### 3.1 Input

MLP 输入：

```text
A: L0 all-head mean block attention matrix, shape [64, 64]
```

数据来自 existing try_25 dataset 的 `attention` 字段。

### 3.2 Teacher Continuous Score

对每个 attention matrix 重新构造 teacher score：

```text
values = nan_to_num(A)
diag(values) = 0
W = max(values, values.T)
W = max(W, 0)
diag(W) = 0
D = diag(W.sum(axis=1))
L = D - W
eigvals, eigvecs = eigh(L)
v = eigvecs[:, 1]
```

这里：

- `eigvals[1]` 是第二小特征值，是一个标量；
- `eigvecs[:, 1]` 是第二小特征值对应的 Fiedler vector，是 64 维连续坐标；
- MLP 要拟合的是 64 维连续坐标转成的 priority score，不是标量 eigenvalue。

### 3.3 Direction / Sign

Fiedler vector 的整体正负号是任意的，所以必须做方向选择。

当前已有数据中保存了 loss-profile oriented teacher direction。可以用 `teacher_rank` 或 `selected_reverse` 只解决 sign ambiguity。

这一步不是用 `teacher_rank` 当连续 target，而是用它判断：

```text
raw priority
reverse priority
which direction matches the saved loss-selected teacher direction
```

最终 target：

```text
z = oriented min-max Fiedler priority in [0, 1]
larger z = earlier reveal
```

统一 order convention：

```python
order = torch.argsort(scores, descending=True)
```

### 3.4 Forbidden Teacher Signals

`OriginalL2R`、original-frame tau、oracle validation loss 只能作为诊断指标，不允许用于：

- teacher 构造；
- sign/orientation 选择；
- MLP target；
- MLP checkpoint selection；
- downstream policy selection；
- early stopping。

---

## 4. MLP Training

### 4.1 Model

优先使用现有：

```text
FlatAttentionOrderMLP
```

推荐当前 try_33 架构：

```text
input: 64 x 64 attention
flatten -> 4096
Linear 4096 -> 2048
GELU
Linear 2048 -> 1024
GELU
Linear 1024 -> 64
```

记录参数量、hidden dims、input normalization。

### 4.2 Loss

当前主版本使用连续 score regression：

```python
pred = torch.sigmoid(mlp(A))
target = oriented_fiedler_priority  # in [0, 1]
loss = F.mse_loss(pred, target)
```

不要在本版本默认混入：

- pairwise BCE；
- rank MSE；
- Plackett-Luce / Gumbel sampling；
- logits EMA；
- attention EMA；
- online MLP parameter update；
- original L2R supervision。

如果 score MSE 版本失败，下一版 try 再显式引入新 loss，不要在同一个 try 里偷偷改方法。

### 4.3 Validation Metrics

至少记录：

```text
train_score_mse
val_score_mse
val_score_mae
val_score_pearson
val_target_kendall_tau: MLP order vs target-score order
val_teacher_kendall_tau: MLP order vs saved teacher order
val_pred_order_examples
val_teacher_order_examples
val_order_tau_examples
zero_input metrics
gaussian_input metrics
shuffled_input metrics
per-head / per-layer metrics if applicable
```

关键判断：

- `val_score_mse` 是否稳定下降；
- `val_score_pearson` 是否高；
- `val_target_kendall_tau` 是否接近 1；
- zero/Gaussian/shuffled 输入是否明显变差；
- 输出 score 是否坍缩成固定 order；
- MLP 是否只学到平均 order，而不是依赖输入 attention。

必须额外输出若干具体顺序样本，方便人工观察：

```text
sample_id
source_iter / record_index
MLP scores summary
MLP order first/last blocks
teacher order first/last blocks
Kendall tau: MLP order vs target-score order
Kendall tau: MLP order vs saved teacher_order
pairwise accuracy
selected_reverse / orientation choice
```

这些具体顺序样本应保存为：

```text
val_order_examples.csv
val_order_examples.md
```

不要只给一个平均 tau。平均指标和具体顺序例子都要有。

### 4.4 Checkpoint Selection

score distillation 版本推荐：

```text
primary: best val_score_mse
secondary: val_target_kendall_tau
```

保存：

```text
best_by_val_score.pt
best_by_val_tau.pt   # compatibility name for existing frozen policy loader
training_summary.json
config.json
metrics.csv
loss_curve.png
score_corr_tau_curve.png
```

### 4.5 External Diagnostic Validation

除了 try_25 的 3000 条 held-out val，必须额外检查历史 head information artifact。路径：

```text
Report/history/language/wikitext103/head_information
```

优先检查这些文件：

```text
Report/history/language/wikitext103/head_information/phenomenon_validation/checkpoints/*/aggregate_matrices.npz
Report/history/language/wikitext103/head_information/phenomenon_validation/checkpoints/*/head_metrics_long.csv
Report/history/language/wikitext103/head_information/phenomenon_validation/checkpoints/*/layer_metrics.csv
Report/history/language/wikitext103/head_information/raw_artifacts/**/*.npz
```

这些数据只用于诊断，不用于：

```text
training
validation loss for checkpoint selection
early stopping
downstream policy selection
claiming original-order recovery
```

对 `aggregate_matrices.npz`，优先使用与 MLP 输入最接近的矩阵：

```text
post_softmax_attention_without_none_raw_current_raw_average
```

该 key 通常形如：

```text
[num_layers=4, num_heads=8, 64, 64]
```

外部诊断时至少计算：

```text
L0 all-head mean: matrix[0].mean(axis=0)
optional: L0 each head separately
optional: other layers/head means
```

对每个可用矩阵：

```text
1. 用同一套 pairwise_max_fiedler 方法重新生成 continuous target score；
2. 用 MLP 生成 pred scores 和 pred order；
3. 记录 pred order vs Fiedler target order 的 tau / pairwise acc；
4. 如果 artifact 中有可对齐的 teacher_order，则额外记录 pred order vs teacher_order tau；
5. 如果没有 loss-profile oriented teacher_order，则同时记录 raw/reverse tau 或 sign-invariant axis tau，并明确写成 diagnostic only。
```

外部诊断结果必须保存到当前 MLP try 下，例如：

```text
Report/language/wikitext103/mlp/distillation/try_XX/external_head_information_eval.csv
Report/language/wikitext103/mlp/distillation/try_XX/external_head_information_order_examples.md
Report/language/wikitext103/mlp/distillation/try_XX/external_head_information_summary.json
```

如果现有脚本不能直接读这些 artifact，agent 应补一个小型 eval 脚本或在现有 eval 脚本中加 adapter。这个 eval 仍然只允许使用：

```bash
CUDA_VISIBLE_DEVICES=1
```

该外部诊断不替代 try_25 的 3000 val。必须同时报告：

```text
primary_val_3k metrics
external_head_information metrics
```

---

## 5. Running try_33

如果继续当前已配置版本，使用 GPU1：

```bash
CUDA_VISIBLE_DEVICES=1 bash Report/language/wikitext103/mlp/distillation/try_33/run_train_fiedler_score_mlp_cuda1.sh
```

当前应优先使用 `cuda1` 脚本。若该脚本不存在，必须先创建：

```text
Report/language/wikitext103/mlp/distillation/try_33/run_train_fiedler_score_mlp_cuda1.sh
```

其中必须显式包含：

```bash
export CUDA_VISIBLE_DEVICES=1
```

并且运行命令必须仍然是：

```bash
CUDA_VISIBLE_DEVICES=1 bash Report/language/wikitext103/mlp/distillation/try_33/run_train_fiedler_score_mlp_cuda1.sh
```

不要使用 0 卡。

训练完 MLP 后，必须先查看：

```text
metrics.csv
training_summary.json
training_result.md
val_order_examples.*
external_head_information_eval.*
```

只有确认 MLP 在 primary val 和外部诊断上表现足够好，才允许配置并启动主干训练。

---

## 6. When To Plug Into Backbone

只有当 MLP validation 通过后，才把 frozen MLP 接入主干训练。

最低进入 downstream 的条件：

```text
1. val_score_mse 收敛；
2. val_score_pearson 明显为正且较高；
3. val_target_kendall_tau 接近 1；
4. zero / gaussian / shuffled 输入性能明显下降；
5. 输出 score std 不接近 0；
6. 没有明显 constant-order collapse。
7. val_order_examples 中 MLP order 与 teacher order 的具体样本合理；
8. external_head_information_eval 已完成，且没有显示明显分布外崩塌。
```

如果不满足，不要跑 downstream。先写清楚失败原因，再开下一版 MLP distillation try。

这是硬 gate：

```text
Do not start backbone training until MLP training/evaluation is reviewed.
```

主干训练命令必须在 MLP gate 通过后再配置/启动。不要把 MLP 训练和 backbone 训练放进同一个无人值守脚本里连续启动，除非脚本内部显式检查 MLP gate 并在不达标时退出。

---

## 7. Frozen MLP Backbone Integration

如果 score MLP 通过验证，则接入 backbone 进行测试。

### 7.1 Principle

MLP 必须 frozen：

```python
mlp.eval()
for p in mlp.parameters():
    p.requires_grad = False
```

本阶段测试的是：

```text
whether distilled Fiedler-score operator can replace online eigendecomposition
```

不是训练一个 end-to-end MLP。

### 7.2 Policy

输入当前模型的 L0 layer-mean attention：

```text
A_current -> frozen MLP -> scores -> argsort_desc -> order
```

约定：

```text
larger score = earlier reveal
```

不要维护 MLP 参数更新。

是否维护 attention/order EMA 要严格跟当前 try 设定一致。如果用户没有要求，默认不要新增 EMA 技巧。

必须区分两种“更新”：

```text
MLP parameter update: forbidden in downstream; MLP remains frozen.
MLP policy/order update: required every training step before fixed stage.
```

也就是说，downstream 中 MLP 的参数不训练，但在 annealing 阶段必须每一步用当前 attention 重新算一次：

```text
A_current -> frozen MLP -> scores -> order
```

配置语义：

```text
attn_mlp_policy_update_every = 1
```

### 7.3 Schedule

从现在开始，所有后续 frozen MLP backbone try 默认使用这一组固定 schedule，除非用户显式修改：

```text
0k-15k:   Random warmup
15k-35k:  Random -> MLP order annealing
35k-end:  fixed order
```

如果从已有 checkpoint resume，例如 10k random base，则仍然按全局 step 解释：

```text
10k-15k: continue Random
15k-35k: anneal to MLP order
35k-end: fixed order
```

Annealing 阶段必须：

```text
policy_update_every = 1
```

即每一个 training step 都刷新当前 attention 输入和 MLP order。不要使用 `update_every=20`。

35k fixed 的含义：

```text
在 global step 35000 固定当时的 MLP-derived order
之后不再刷新 MLP order
之后不再计算 annealing
之后仍然不训练 MLP 参数
```

这些参数必须写进 config 和 report，不要只写自然语言：

```text
warmup_until = 15000
anneal_start = 15000
anneal_end = 35000
fixed_after = 35000
attn_mlp_policy_update_every = 1
mlp_frozen = true
```

### 7.4 Seeds

Downstream 是硬性至少两组 seed。两组配置除 seed 外不要混入其他差异。

不要只跑一组 seed。不要用单 seed 结果做最终结论。

退出标准使用两组或更多 seed 的 final validation loss 均值：

```text
mean_final_val_loss = mean(final_val_loss over completed downstream seeds)
pass_to_exit = mean_final_val_loss <= 3.45
```

如果 `mean_final_val_loss <= 3.45`，可以结束 goal。

如果 `mean_final_val_loss > 3.45`，不能结束 goal，必须分析原因并开启下一版 try。

所有 downstream 运行也必须：

```bash
CUDA_VISIBLE_DEVICES=1
```

---

## 8. Report And Artifact Rules

每个 try 必须记录：

```text
goal
method
data path
target definition
orientation rule
model config
loss
command
GPU constraint
checkpoint path
metrics
figures
diagnosis
next step
primary_val_3k metrics
external_head_information metrics
val_order_examples path
external_order_examples path
downstream seed list
final_val_loss per seed
mean_final_val_loss
whether mean_final_val_loss <= 3.45
```

Report path should follow current project style:

```text
Report/language/wikitext103/mlp/distillation/try_XX
```

Checkpoint path should follow:

```text
checkpoints/attn_mlp_distillation/tryXX_...
```

Do not overwrite old tries. Every meaningful method change gets a new try.

Remember that `Report/**` is gitignored in this repository. If the user asks to commit report artifacts, use `git add -f` for report files.

---

## 9. W&B Rules

Use W&B if the surrounding experiment family uses it, but never hard-code an API key.

Allowed:

```bash
export WANDB_API_KEY=...
```

Forbidden:

```text
putting a W&B key in source code, config, report, or run script
```

For MLP distillation, record:

```text
train_score_mse
val_score_mse
val_score_mae
val_score_pearson
val_target_kendall_tau
val_teacher_kendall_tau
val_order_examples
external_head_information_tau
external_head_information_pair_acc
learning_rate
epoch
zero/gaussian/shuffled ablations
```

For downstream backbone runs, record:

```text
train_loss
val_loss
best_val_loss
final_val_loss
mean_final_val_loss across seeds, in summary report
policy_mode
global_step
anneal_start = 15000
anneal_end = 35000
fixed_after = 35000
attn_mlp_policy_update_every = 1
mlp_score_mean
mlp_score_std
order_change_rate if available
original_tau diagnostics only
```

---

## 10. Failure Handling

If score distillation fails, do not immediately run backbone.

Write a report that answers:

```text
Did score MSE decrease?
Did Pearson improve?
Did target tau approach 1?
Did zero/Gaussian input collapse to a fixed output?
Was the target sign/orientation correct?
Was the target cache generated correctly?
Was MLP too small or was data/label noisy?
```

Possible next tries, in order:

```text
1. score MSE + small pairwise auxiliary loss
2. zscore-score MSE instead of sigmoid-to-[0,1] MSE
3. confidence-weighted score loss by eigengap / loss_score_gap
4. larger MLP only if metrics suggest underfit
5. regenerate data with stronger/more diverse attention states
```

Do not blur conclusions. If only the score operator is learned but downstream fails, report that distinction clearly.

If downstream final-loss mean is above threshold:

```text
mean_final_val_loss > 3.45
```

then the goal is not complete. Start a new try with one clearly stated change, while preserving these fixed downstream defaults unless the user explicitly changes them:

```text
CUDA_VISIBLE_DEVICES=1 only
at least two downstream seeds
15k-35k anneal
35k fixed
policy_update_every = 1
MLP frozen in downstream
exit only when mean final val loss <= 3.45
```

---

## 11. Final Response Requirements

At the end of the task, report:

```text
1. whether existing data was reused or new data was collected
2. dataset path and sample counts
3. exact command used, with CUDA_VISIBLE_DEVICES=1
4. MLP checkpoint path
5. primary 3k val score MSE / Pearson / target tau / teacher tau
6. concrete val order examples: MLP order vs teacher order tau
7. external head_information diagnostic metrics and example orders
8. zero / Gaussian / shuffled input behavior
9. whether MLP passed the gate for backbone integration
10. downstream seed count; must be at least two
11. per-seed best/final val loss and W&B run names
12. mean final val loss across seeds
13. whether `mean_final_val_loss <= 3.45`
14. whether the `/goal` can be marked complete
15. clear diagnosis and next try recommendation if threshold is not met
```

Keep original-order metrics diagnostic-only. Do not claim original L2R recovery unless the experiment explicitly supports it.
