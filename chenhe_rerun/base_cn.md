# 语言主线项目 Base

这个仓库现在只保留 WikiText103 / 语言方向的 AO-GPT learned-order 工作区。非语言实验材料已经先备份到远程清理分支，然后从当前工作树删除。

## 当前研究问题

AO-GPT 在没有显式 left-to-right 生成策略、使用 random reveal order 训练的情况下，是否会在 current-frame attention、loss 和 early reveal behavior 中暴露出可恢复的文本顺序结构？

## 当前范围

- 数据：`data/wikitext103`
- 配置：`config/WikiText103/`
- 报告：`Report/language/wikitext103/`
- 核心代码：`train.py`、`AOGPT.py`、`AOGPT_block.py`、`AOGPT_token.py`、`order_utils.py`、`online_spectral_order_policy.py`、`attn_mlp_order_policy.py`
- 当前文档：
  - `docs/prompts/Prompt_language.md`
  - `docs/prompts/prompt_language_block_current_task.md`
  - `docs/findings/findings_language.md`
  - `docs/structure/PROJECT_STRUCTURE_language.md`

## 方法边界

当前主线不是在模型内部直接学习一个 global order 参数，而是外部 recovery-and-curriculum loop：

1. 训练 random reveal-order AO-GPT backbone。
2. 从 current-frame 模型信号中挖 candidate pair 或 candidate order。
3. 用 attention、loss、margin、gain、stability 等信号打分和过滤。
4. 把 recovered units 或 order 喂回训练/评估。
5. 评估 recovered structure 是否接近文本 L2R 顺序。

## Frame 规则

如果 `permute_data=False`，current frame 等于文本 frame。

如果 `permute_data=True`，训练发生在固定打乱后的 current frame。checkpoint 保存：

- `block_perm`：current-frame block index 到 text-frame block index
- `inverse_block_perm`：文本 L2R order 在 current-frame ids 中的表达

训练、candidate selection、rerank、early stop 和 checkpoint choice 必须使用 current-frame signal。`*_original`、original tau 和 `OriginalL2R` 只能作为 diagnostic。`permute_data=True` 下的 `OriginalL2R` 是 oracle upper bound，不是 no-prior evidence。

## 当前主线

当前主线比完整历史语言树更窄：

1. Distribution 方法论：
   `L0/layer-mean attention -> W=max(A,A.T) -> graph Laplacian Fiedler axis ->
   current-model linear_profile_loss 定向 -> training order`。
2. EMA 消融：
   到底维护 rank/priority EMA、维护 continuous Fiedler-priority EMA，还是去掉
   EMA 直接使用当前 teacher/order，目前仍是消融问题，不是定论。
3. MLP 蒸馏：
   把 Laplacian/Fiedler teacher 压缩进 `attn_mlp_order_policy.py`，再测试
   frozen MLP 接入 AO-GPT 训练是否可靠。

Direct-asym-eig 和 fixed-head top-1 policy 仍是重要 baseline 和历史证据，但除非任务明确要求回看旧线，否则不要把它们当成当前默认入口。

常用文件：

```text
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/eval/eval_lm_original_order_ppl.py
scripts/eval/eval_lm_ppl_sweep.py
scripts/train/train_pairwise_mlp_operator_distillation.py
scripts/data/merge_pairwise_operator_datasets.py
scripts/eval/eval_pairwise_operator_mlp_generalization.py
```

常用配置根目录：

```text
config/WikiText103/seq80/
config/WikiText103/seq256/
config/WikiText103/seq384/
```

当前报告根目录：

```text
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/
Report/language/wikitext103/mlp/distillation/
Report/language/wikitext103/mlp/joint_training/
```

## 安全结论

- 当前最安全的 claim：可恢复的局部 block-level 文本顺序结构。
- 还不能宣称：完整 global L2R induction。
- token/block1 和 token-micro 是 diagnostic，不是当前优化目标。
- no-prior claim 必须排除 original-frame diagnostic 对训练时决策的影响。

## 清理说明

清理后，新任务应只通过上面列出的语言文档入口继续。不要依赖已经删除的非语言配置、数据、报告、外部依赖或命令。
