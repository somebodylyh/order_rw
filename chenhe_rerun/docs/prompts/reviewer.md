请使用 Codex skill `nature-reviewer` 对这个仓库做一次投稿前 reviewer-style 严审。

如果当前 Codex 会话已经能发现 skill，必须先加载并遵循 `nature-reviewer` 的工作流；如果当前会话还没有重启、无法自动发现该 skill，则先读取并遵循：

`/home/chenhe/.codex/skills/nature-reviewer/SKILL.md`

本 prompt 的目标不是让你给作者鼓励，而是用 `nature-reviewer` 的“shared fact base -> 3 reviewer reports -> cross-review synthesis -> unsupported claims”框架，最大化暴露当前主方法距离 CCF-A / top-tier ML-NLP 发表还差什么。请把仓库里的代码、report、metrics、try 结果视为 manuscript fact base 的来源；不要把没有证据的推断写成事实。

项目目录：

`/home/chenhe/nanogpt-learned-order`

## Scope: 只审当前主方法区

这个仓库当前工作树已经清理为语言主线，但仍保留多条历史分支、失败/正控/清理报告。请不要把任务扩展成全仓库泛评。

本次锐评的主对象是 WikiText103 language / `seq256/permute/block64` 上的这条方法链：

```text
AO-GPT random / online checkpoint
-> collect current-frame L0 layer-mean attention A
-> build W = max(A, A.T), zero diagonal
-> graph Laplacian L = D - W
-> sort Fiedler vector to get an unsigned order axis
-> use current-model train linear_profile_loss to choose axis vs reverse
-> use the resulting oriented Fiedler order as teacher / distribution update
-> distill this teacher/operator into FlatAttentionOrderMLP
-> after distillation succeeds, replace the slow teacher with the MLP policy
```

The author believes the `max(A,A.T) + Laplacian Fiedler` teacher has already shown that it is usable, especially through distribution try23/try24. MLP distillation also has completed try25/try28 evidence; the currently uncertain stages are EMA/no-EMA distribution ablations and frozen-MLP insertion tests such as try29/try30. Your job is to judge whether this narrower teacher-to-MLP methodology is scientifically strong enough, what evidence is still missing, and how a top-tier reviewer would attack it.

Other branches should be used only as context or controls:

- Attn-MLP joint training try29/39/40/42/43: use as context for why teacher reliability matters, not as the paper spine unless you argue the Fiedler-teacher spine is weaker.
- frozen loss-design try20/21/22: use as evidence about current-frame loss/reward signals, not as the main claim.
- oracle / original-tau positive controls: only positive controls, never no-prior evidence.

## Skill 调用约束

- 必须显式使用 `nature-reviewer` 的 reviewer assessment 角色，而不是 author rebuttal 或项目助手角色。
- 使用该 skill 的默认审稿结构：3 份 reviewer reports + 1 份 cross-review synthesis。
- 本任务的 venue 目标按 CCF-A / top-tier ML-NLP 会议或期刊理解；`nature-reviewer` 只提供审稿工作流、groundedness 规则和输出骨架，不要求把论文定位成 Nature。
- 三位 reviewer 只允许在审查重点上不同，不要编造 reviewer 身份、机构、履历或编辑决定。
- 所有判断必须 grounded in repo evidence：文件路径、try 编号、metric、脚本、配置、日志或 report 原文。如果证据不足，明确写“我无法判断，因为缺少 X”。

## 必读材料

请先阅读以下文件，不要跳过。读完后可以继续用 `rg`、`find`、`python`、`jq` 等方式追查它们引用的代码、CSV、JSON、日志和 report。

### Project orientation

1. `docs/README.md`
2. `base_cn.md`
3. `docs/structure/PROJECT_STRUCTURE_language.md`
4. `Report/_index/active_threads.md`

### Focused Fiedler teacher line

5. `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/SUMMARY.md`
6. `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md`
7. `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23/results.md`
8. `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24/results.md`
9. `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28/design.md`
10. `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29/design.md`
11. `config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py`
12. `config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py`
13. `config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k.py`
14. `config/WikiText103/seq256/permute/block64/online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k.py`

### Focused MLP distillation line

15. `Report/language/wikitext103/mlp/distillation/README.md`
16. `Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md`
17. `Report/language/wikitext103/mlp/distillation/try_25/results.md`
18. `Report/language/wikitext103/mlp/distillation/try_28/results.md`
19. `Report/language/wikitext103/mlp/distillation/try_29/experiment_design.md`
20. `Report/language/wikitext103/mlp/distillation/try_30/experiment_design.md`
21. `config/WikiText103/seq256/permute/block64/attn_mlp_try29_seed2027_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_noema_update1_warmup10k_anneal35k_freeze35k.py`
22. `config/WikiText103/seq256/permute/block64/attn_mlp_try30_seed2028_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_noema_update1_warmup10k_anneal35k_freeze35k.py`

### Context only

23. `Report/language/wikitext103/mlp/joint_training/TRY_REVIEW_20260624.md`
24. `Report/language/wikitext103/mlp/joint_training/try_42_43_layeravg_comparison.md`
25. `Report/language/wikitext103/mlp/joint_training/try_39_40_autohead_eigloss_seed_sweep_result.md`
26. `Report/language/wikitext103/mlp/loss_design_exploration/SUMMARY.md`

If a listed file is missing or stale, say so explicitly and locate the current replacement with `find` / `rg`; do not silently ignore it.

## 项目背景

我在做基于 nanoGPT 改造的 AO-GPT / learned-order 研究。当前重点不是图像，也不是所有历史 MLP 分支，而是 WikiText103 语言侧的 `seq256/permute/block64` teacher-to-student 方法论。

核心问题是：

> 在没有显式 OriginalL2R/order prior 的情况下，能否从 AO-GPT 自身 current-frame layer-mean attention 中构造一个稳定、可用、可蒸馏的 order teacher；并进一步训练一个 `FlatAttentionOrderMLP` 近似这个 teacher，使最终方法不再依赖昂贵的在线 Fiedler/loss-orientation pipeline？

当前作者想把论文主线收敛成：

1. **Teacher construction**: `W=max(A,A.T) + Laplacian Fiedler + current train loss orientation` 是一个 no-prior teacher。
2. **Teacher usefulness**: try23/try24 显示该 teacher 用于 distribution / MAP order 后在多个 seed 上稳定、可用。
3. **Student distillation**: try25/try28 显示 `FlatAttentionOrderMLP` 可以从 L0 layer-mean `64x64` attention matrix 直接输出 64 block logits/ranks，逼近 teacher order。
4. **Full methodology**: 只有当 frozen-MLP substitution / no-EMA insertion 也通过时，才形成完整方法论：slow teacher proves recoverable structure; MLP compresses it into a deployable order policy.

你要判断这条故事是否足够清晰、足够严格、足够 novel。

## 方法边界

请把下面几条当作 no-prior 审查红线：

- `OriginalL2R`、original tau、validation PPL、oracle order 只能作为 post-hoc diagnostic。
- Teacher 的 Fiedler axis 只能来自 current-frame attention matrix `A`。
- raw Fiedler axis 和 reverse(axis) 的方向选择只能来自 current-model train loss/profile，例如 `linear_profile_loss`。
- MLP distillation target 是 teacher order / teacher rank / pairwise labels，不是 OriginalL2R，不是 original tau，不是 validation PPL selection。
- train/val split 必须按 attention record 切分，不能同一个 attention record 同时出现在 student train 和 val。
- 如果发现任何隐性 prior leakage、selection leakage、diagnostic 被包装成方法贡献，请直接指出。
- positive control 必须和 no-prior evidence 分开写，不能把正控结果包装成主方法证据。
- smoke test、partial run、排错脚本、incomplete data collection 不能当正式结论。

## 当前已知证据

请独立核查，不要盲信这段摘要。

### Teacher 已证明可用的证据

Distribution try23:

- Method: L0 layer-mean attention -> `W=max(A,A.T)` -> Laplacian Fiedler -> current `linear_profile_loss` orientation -> priority/MAP.
- Report: `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23/results.md`
- Key metrics reported:
  - best val/loss `3.3958`
  - final val/loss `3.4106`
  - final MAP original tau `+0.916667`
  - selected probe tau mean `+0.904533`
  - selected probe sign flips `0`
  - raw probe sign flips `437`
  - consecutive MAP tau mean `0.998350`

Distribution try24:

- Same data permutation, different training seed `2027`.
- Report: `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24/results.md`
- Key metrics reported:
  - best val/loss `3.3594`
  - final val/loss `3.3643`
  - final MAP original tau `+0.959325`
  - max MAP original tau `+0.994048`
  - selected probe tau mean `+0.977012`
  - selected probe sign flips `0`
  - raw probe sign flips `490`
  - consecutive MAP tau mean `0.999004`

Distribution try28/try29:

- Continuous Fiedler-priority EMA designs.
- Report/design paths:
  - `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28/design.md`
  - `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29/design.md`
- Treat them as pending/in-progress unless completed logs/results are verified.

### Distillation 当前阶段

Distillation try25/try28:

- Reports:
  - `Report/language/wikitext103/mlp/distillation/try_25/results.md`
  - `Report/language/wikitext103/mlp/distillation/try_28/results.md`
- Goal: distill the graph-Laplacian/Fiedler teacher into `FlatAttentionOrderMLP`.
- Student input: one L0 layer-mean `64x64` attention matrix.
- Student output: 64 block-priority logits; order recovered by sorting logits.
- Target: full loss-oriented teacher order/rank/pairwise labels.
- Key evidence:
  - try25 reached held-out MLP-teacher tau `0.979898`, just below the strict
    `0.980` gate.
  - try28 reached held-out MLP-teacher tau `0.980612` with a smaller
    two-hidden-layer MLP, passing the strict gate.
- Baselines:
  - constant train-mean order
  - zero input
  - Gaussian input
  - shuffled input

Frozen insertion try29/try30:

- Reports:
  - `Report/language/wikitext103/mlp/distillation/try_29/experiment_design.md`
  - `Report/language/wikitext103/mlp/distillation/try_30/experiment_design.md`
- Goal: insert frozen try28 MLP into AO-GPT training with attention EMA and
  logits EMA disabled, update every step.
- You must verify logs/results before claiming insertion success or failure.

## 审查任务

请像 CCF-A reviewer 一样锐评这条 Fiedler-teacher-to-MLP methodology，重点找不足。不要泛泛而谈，要基于实际代码、report、metrics、try 结果给出证据。

重点审查：

- `max(A,A.T) + Laplacian Fiedler` teacher 的 claim 是否清楚。
- 该 teacher 的 no-prior 性是否经得起审查。
- try23/try24 是否足以证明 teacher 可用，还是有 seed / permutation / metric cherry-pick 风险。
- `linear_profile_loss` 用于 raw/reverse orientation 是否会被 reviewer 认为是隐藏的 direction prior。
- teacher 的 MAP original tau 很高，但 original tau 是 diagnostic-only；论文叙事如何避免“看起来像用 tau 证明方法”的风险。
- teacher 的 validation loss/PPL 改善是否和 order-quality claim 一致；若不一致，如何解释。
- distillation 数据收集设计是否足够证明 MLP 学的是 operator，而不是 common-order shortcut。
- MLP distillation 成功后，还缺什么 main-run substitution / frozen student evaluation 才能称为完整方法论。
- 与 previous joint-training / direct-asym-eig / single-head / all-head-mean 方法相比，Fiedler teacher 的 novelty 和 necessity 是否充分。
- 这个方法现在更像“一个可发表的收敛故事”，还是“强 teacher + pending student 的半成品”。

## 输出格式

请按下面结构输出，且不要省略任何部分。

### Review setup

- Input scope:
- What is deliberately out of scope:
- Assessment boundary:
- Shared manuscript/project claim summary:
- Visible evidence base:
- Missing materials affecting confidence:
- 一句话总评：如果论文主线限定为 Fiedler teacher -> MLP distillation，它目前更接近 reject / weak reject / borderline / weak accept / accept？

### Reviewer 1: methodology / no-prior emphasis

- Overall assessment:
- Who would be interested in the results, and why:
- Major strengths:
- Major concerns:
- Technical failings that need to be addressed before the case is established:
- Assessment against top-tier criteria:
- Recommendation posture:
- Hardest rebuttal problem:

### Reviewer 2: experiments / benchmark emphasis

使用同样小节结构。

### Reviewer 3: ML/NLP systems emphasis

使用同样小节结构。

### Cross-review synthesis

- Consensus strengths:
- Consensus technical risks:
- Where emphasis differs across reviewers:
- Broad-interest / significance readout:
- Is the Fiedler-teacher-to-MLP contribution story clear?
- Does the current evidence justify making this the paper spine?
- What exactly must be shown after distillation completes?
- Most important issues to resolve before a strong CCF-A case is established:

### Major Weaknesses

至少列 8 条。每条必须包含：

- 问题是什么
- 为什么这是 CCF-A 级别风险
- repo/report 中的证据
- 怎么补实验或改叙事

### Methodology / No-prior Audit

- current-frame vs original-frame boundary:
- teacher construction leakage:
- orientation leakage:
- distillation target leakage:
- selection leakage:
- seed/permutation cherry-picking:
- post-hoc explanation:
- positive control packaging:
- diagnostics that risk being mistaken for method evidence:

### Teacher Evidence Audit

- try23 evidence:
- try24 evidence:
- try25 status:
- robustness across seeds:
- robustness across data permutations:
- robustness across checkpoints / warmup stages:
- comparison to direct-asym-eig / single-head / layer-average alternatives:
- evidence that `W=max(A,A.T)` and Fiedler specifically matter:
- evidence that current-loss orientation matters:

### Distillation Evidence Audit

- data collection completeness:
- train/val record split:
- teacher-label construction:
- student architecture:
- student loss:
- held-out teacher tau gate:
- input ablations:
- common-order shortcut risk:
- main-run substitution evidence:
- missing reproducibility artifacts:

### Experimental Rigor

- Seed coverage:
- Permutation coverage:
- Baseline coverage:
- Ablation coverage:
- Metric relationships:
- Failed tries and negative evidence:
- Runtime/cost evidence:
- Missing reproducibility artifacts:

### Required Experiments

给出按优先级排序的实验清单：

- P0: 没有就很难投稿
- P1: 显著增强可信度
- P2: 加分但不是核心

每个实验都要写清楚：

- 要验证的 reviewer concern
- 最小可执行设计
- 需要对比的 baseline / seed / metric
- 通过或失败分别意味着什么

At minimum, consider whether the following are P0/P1/P2:

- complete try25 or equivalent third seed for teacher;
- repeat teacher with a different `permute_seed`;
- teacher ablations: `max(A,A.T)` vs `A`, `A+A.T`, direct-asym-eig, random symmetric graph, layer L1 mean;
- orientation ablation: current `linear_profile_loss` vs prefix/full loss variants vs no orientation;
- student distillation held-out real-input tau vs constant/zero/Gaussian/shuffled baselines;
- student generalization across checkpoints or seeds, not only same distribution as collection;
- frozen student substituted into main training/eval, compared to slow teacher and Random;
- cost comparison: slow teacher online pipeline vs MLP inference.

### Sharp Reject Simulation

模拟 3 条最尖锐的拒稿意见，不要编造 reviewer 身份：

- Methodology-focused rejection:
- Experiment-focused rejection:
- Systems / NLP-focused rejection:

每条都要给出 reject 理由和 rebuttal 难点。

### Risk / Unsupported Claims

- 列出所有证据不足、过早、容易被 reviewer 打的 claim。
- 对每条 claim 标注：supported / weakly supported / unsupported / not assessable。
- 对 unsupported 或 not assessable 的 claim，指出缺失证据是什么。

## 语气和证据要求

- 语气要严厉、具体、直接。
- 不要为了礼貌淡化问题。
- 不要只说“需要更多实验”，要说清楚缺什么实验、为什么缺、该怎么设计。
- 所有关键判断必须尽量引用具体 report 路径、try 编号、metric。
- 如果信息不足，请明确说“我无法判断，因为缺少 X”，不要脑补。
- 不要把 image 分支、旧 joint-training 分支或 frozen loss-design 分支当成主方法证据，除非你明确说明它们只是 context / baseline / risk evidence。
