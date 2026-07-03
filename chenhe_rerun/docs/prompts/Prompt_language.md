# 给新 Agent 的语言方向 Prompt

请先读取并综合项目根目录下的共享记忆和语言方向文件，再开始任何分析、实验设计或代码修改：

```text
research-state.yaml
docs/findings/findings_language.md
research-log.md
literature/survey.md
base.md
base_cn.md
docs/structure/PROJECT_STRUCTURE_language.md
README.md
```

当前工作树只保留语言主线；不要把任务扩展到已删除的非语言分支。

## 2026-06 当前主线

当前最重要的主线不是旧的 broad online-order tree，而是两条互相连接的方法论：

1. **Distribution / Laplacian graph decomposition**：
   从 current-frame layer-mean attention 构造 `W=max(A,A.T)`，用 graph
   Laplacian Fiedler vector 恢复一个无向 order axis，再用 current-model
   `linear_profile_loss` 在 raw/reverse 之间定向。这个方向目前是
   WikiText103 `seq256/permute/block64` 的主线。
2. **MLP distillation**：
   先把上面的 Fiedler teacher 蒸馏成 `attn_mlp_order_policy.py` 里的
   attention-conditioned MLP，再测试 frozen MLP 是否能稳定接入 AO-GPT
   training loop。

EMA 仍在消融：rank/priority EMA、continuous Fiedler-priority EMA、以及 no-EMA
直接用当前 teacher/order，都必须分开讨论。不要把“维护 EMA”写成默认结论。

## 语言方向目标

本方向研究 WikiText103 上的 AO-GPT learned-order signal。核心问题是：

> AO-GPT 在没有显式 left-to-right 生成策略、使用 random reveal order 训练的情况下，是否会在 loss / attention / early reveal behavior 中暴露出可恢复的文本局部生成结构？

当前主线里的 "learned order" 不是模型内部学习一个可微 global order 参数，而是外部 recovery / teacher / distillation pipeline：

1. 训练 random reveal-order AO-GPT backbone。
2. 从 current-frame attention 构造 block affinity graph。
3. 用 Laplacian/Fiedler 恢复无向 order axis。
4. 用 current-model train loss/profile 选择方向。
5. 比较 EMA / no-EMA policy 如何影响训练 loss 和 order stability。
6. 将稳定 teacher 蒸馏进 MLP，再评估 frozen MLP 接入训练。

## Frame 约定

如果 `permute_data=False`，current frame 等于 original text frame。

如果 `permute_data=True`，训练数据会先被固定 block permutation 打乱。checkpoint 中保存：

- `block_perm`: current-frame block index -> original-frame block index
- `inverse_block_perm`: original L2R order 在 current frame 中的顺序

训练和 curriculum 一律使用 current-frame units。`*_original` 字段只用于分析和报告，不能用于训练决策、候选筛选、early stop 或 policy 选择。

PPL 评估也必须遵守这个约定：`permute_data=True` 下的 `OriginalL2R` 是用
checkpoint permutation metadata 注入真实原始文本顺序的 oracle，只能作为 upper bound。
无先验主比较应看 current-frame AR、Random、`CheckpointOnlineSpectralOrder` 或
recovery/curriculum 自己产生的 order。

## 历史语言 Pipeline

下面这条 hierarchical pair/unit recovery 是重要历史证据，但不是当前默认主线：

1. attention-pruned candidate pairs 或 full candidate pairs。
2. directed pair scoring。
3. `margin_vs_reverse` filtering。
4. 直接把高分 non-conflicting pair 聚合成 `[i, j]`。
5. 下一 stage 把这些 units 当作新的单位继续挖 pair。
6. curriculum runner 用 `segment_source_json=stage_x/results.json` resume training。

当前 `hierarchical_structured_benchmark.py` 支持两种 pair score：

- `abs_tv`: `score(i -> j) = -mean(li, lj) - tv_weight * abs(lj - li)`
- `signed_drop`: `score(i -> j) = -mean(li, lj) + drop_weight * (li - lj)`

其中 `li` 是 first reveal loss，`lj` 是 second reveal loss。block-level 文本配置里 `signed_drop` 一直较有效。

另一个历史实验分支是 `OnlineSpectralFixedHeadOrder`。它在训练中维护一个 cached order，
每次从固定 layer/head attention matrix 枚举 spectral candidates，并缓存最高
attention-spectral score 的候选。这个方法会持续刷新 order，因此如果用于 no-prior 结论，
必须报告 order stability；更严格的协议应先 recover/freeze 一条 order，再围绕 frozen order
训练或评估。

当前更具体的目标已经转向：先围绕 layer-mean pairwise-max Fiedler teacher 验证
distribution/EMA/no-EMA，再把这个 teacher 蒸馏成 MLP。不要让新任务默认回到
fixed-head top-1 或 direct-asym-eig，除非用户明确要求比较旧线。

## 当前 Distribution / MLP 方法

优先阅读这些文件：

```text
train.py
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/eval/eval_lm_original_order_ppl.py
scripts/eval/eval_lm_ppl_sweep.py
scripts/train/train_pairwise_mlp_operator_distillation.py
scripts/eval/eval_pairwise_operator_mlp_generalization.py
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
Report/language/wikitext103/mlp/distillation/README.md
Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md
```

核心 configs 和报告：

```text
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k.py
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py
config/WikiText103/seq256/permute/block64/online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k.py
config/WikiText103/seq256/permute/block64/attn_mlp_try29_seed2027_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_noema_update1_warmup10k_anneal35k_freeze35k.py
config/WikiText103/seq256/permute/block64/attn_mlp_try30_seed2028_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_noema_update1_warmup10k_anneal35k_freeze35k.py
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23/results.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24/results.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28/design.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29/design.md
Report/language/wikitext103/mlp/distillation/try_25/results.md
Report/language/wikitext103/mlp/distillation/try_28/results.md
Report/language/wikitext103/mlp/distillation/try_29/experiment_design.md
```

### 历史 Seq80 / Fixed-Head 诊断背景

以下 seq80/block1 non-permute 内容是 2026-06-07 附近的 online/fixed-head
诊断背景。它可以解释为什么后来转向 layer-mean Fiedler teacher，但不是当前默认主线。

截至 2026-06-07，seq80/block1 non-permute 是当时最活跃的 online 诊断方向。
`random_save_attn_ckpts...` 已经跑完 50k，最终 `val loss 3.9655`，并保留了
`1k,2k,5k,8k,10k,15k,20k,25k,50k` checkpoints 用于 attention heatmap 演化分析。
这些 checkpoint 的 heatmap/order 诊断已经完成，报告在
`Report/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_online_order_warmup_diagnostic`。
修正后的 Random-probe 局部性报告在
`Report/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_random_probe_remap_locality_diagnostic`。
对当前 policy 真正相关的是固定 L0H7 `with_none`，这个视图在 Random probe 下确实
很乱：主要是列/全局结构，不是局部对角带，近邻/远邻 ratio 基本 `1.00`，谱排序
top1 tau 也接近 0。Random 多样本平均并 remap 回 current L2R 后，all-layer/all-head
`without_none` 有清楚局部对角带，近邻/远邻 ratio 从 1k 的 `1.06` 上升到 50k 的
`2.04`；这只能说明模型内部某处有 local signal，不能证明当前单头 `with_none`
policy path 能用好它。可比的 AR-probe 结果在
`Report/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_ar_probe_attention_diagnostic`；
AR-probe tau 在 `1k..50k` 是高正/高负波动（例如 `5k=-0.643`,
`10k=0.597`, `15k=-0.651`, `50k=0.431`），说明 random checkpoint 有强 order axis，
问题是方向选择和 probe policy。
后续 fixed-head no-freeze run 已自动接上并运行中：`0..5k` random warmup，
`5k..15k` anneal，`15k..50k` 每 20 step 持续更新 cached order，不在最后 freeze。
如果任务专门问 seq80/fixed-head 历史，再读
`docs/prompts/archive/prompt_online_language_current.md` 和
`docs/findings/findings_language.md` 的 seq80 online 小节。

### Fixed-Head Top-1

`OnlineSpectralFixedHeadOrder` 的训练 loop 是：

```text
current model attention
-> fixed layer/head matrix, usually layer 0 head 7
-> spectral candidate enumeration
-> current-frame attention-side scoring
-> optional current-frame loss reranking
-> cache top-1 order
-> later train samples use cached_order after anneal
```

attention-side score 来自 current-frame attention graph，包括 adjacency path、
directed path 和 band quality。不要用 original L2R、original tau、human scan 或
validation PPL 来选择 order。

loss-rerank 版本允许使用 current-frame train/probe teacher-forced loss：

```text
score = 1.0  * z(attention_spectral_score)
      - 0.25 * z(prefix_loss)
      - 0.15 * z(full_loss)
```

这不是 original-frame oracle，因为 loss 只来自 current model 和 current-frame candidate
order。默认 fixed-head loss-rerank block64 使用 `16 x 64 = 1024` train probe sequences
per candidate，`loss_rerank_top_k=64`。

### Distribution

`OnlineSpectralOrderDistribution` 不永久维护 32 条 order。它维护的是一个长度
`num_blocks` 的 priority EMA：

```text
attention matrix
-> top candidates
-> r_update priority vector
-> priority_ema
-> Gumbel-top-k sampled hard orders
```

对候选 order `m`：

```text
v_m(i) = 1 - pos_m(i) / (num_blocks - 1)
w_m = softmax(z(score_m) / teacher_temperature)
r_update(i) = sum_m w_m * v_m(i)
priority_ema = 0.95 * priority_ema + 0.05 * r_update
```

训练时：

```text
if priority_ema is missing or policy anneal misses:
    fallback random
else per sample:
    20% random mix by default
    otherwise sample Gumbel-top-k(priority_ema / 0.7)
```

distribution loss-rerank 版本先 recover/rerank top-64，然后用 reranked top-32 更新
`priority_ema`。默认 loss rerank 数据量是 `32 x 64 = 2048` train probe sequences
per candidate。

### Late-L2R Curriculum Ablation

late-L2R configs 是显式 AR prior ablation，不能写成 no-prior learned-order 结果。
它们测试：

```text
online spectral/distribution early curriculum
-> gradually anneal to L2R
```

block64 schedule：

```text
0..8000: random fallback -> online spectral policy
8000..50000: online spectral policy -> L2R
```

如果评价 late-L2R，请同时报告 AR/OriginalL2R PPL、checkpoint learned-order PPL、
Random PPL、late_l2r_prob/count，以及是否只是 AR finetune 改善。

## 语言结论边界

安全结论是：

> Random-order AO-GPT checkpoint 中存在可挖掘的局部结构信号。block-level setting 中，这些信号能较稳定恢复 original L2R 方向；token/block1 setting 中，信号强烈体现局部邻域关系但方向性弱。

不要写成：

> AO-GPT 从 any order 中学出了完整 global L2R 顺序。

不要把 token-level final unit collapse 当成 clean order recovery 证据。`token_micro_segments` 已退出主线，只能作为 diagnostic evidence。

## 当前最强语言结果

- config: `config/WikiText103/seq256/permute/block64/segment_curriculum_early_stop_open_level.py`
- report: `Report/curriculum/permute/seq256/block64/hierarchical_block64_early_stop_open_level_gain_only-6-stage`
- final original-frame diagnostic:

```text
[4..12, 0..3, 13..27]
[28..63]
```

final within-unit adjacency 为 `+1=60/62`，concat Kendall tau 为 `0.964`。这是强 block-level evidence，但 `[0..3]` 仍被放错位置，所以只能说恢复局部 order structure，不能说完整 global L2R。

## 语言对照结果

- `Report/curriculum/permute/seq256/block32/hierarchical_block32_early_stop_gain_only-6-stage`
  - good local recovery，final `+1=28/31`，concat tau `0.871`。
  - 仍有 prefix chunk error：`[3..12], [1..2], [0], [13..31]`。
- `Report/curriculum/permute/seq256/block128/hierarchical_block128_early_stop_gain_only-6-stage`
  - over-aggregation diagnostic。
  - 有很多 local runs，但 global chunk order 较差，final tau `0.469`。
- `Report/curriculum/permute/seq80/block1/hierarchical_block1_early_stop_open_level_gain_only-6-stage`
  - negative token-level diagnostic。
  - local adjacency 存在，但 direction mixed / reverse-biased，final concat tau `-0.594`。
- `Report/curriculum/permute/seq256/block64/hierarchical_block64_early_stop_open_level_gain_only_rope-6-stage`
  - RoPE ablation。
  - stage 1 top-50 original-frame pairs 仍然强：`+1=46`、`-1=4`。
  - final `+1=50/63`，concat tau `0.692`，说明 local signal 没消失，但当前 recipe 更容易 over-aggregate / chunk-order error。

## 重点文件

语言方向优先读：

- `train.py`
- `AOGPT.py`
- `AOGPT_block.py`
- `AOGPT_token.py`
- `order_utils.py`
- `scripts/runner/hierarchical_segment_curriculum_runner.py`
- `scripts/analysis/`
- `scripts/eval/`
- `scripts/train/`
- `scripts/analysis/pair_margin_stability_probe.py`
- `scripts/analysis/tv_weight_pair_sweep.py`
- `scripts/analysis/visualize_pair_score_heatmaps.py`
- `scripts/analysis/original_span_conditional_gain_probe.py`
- `scripts/analysis/original_span_permutation_loss_probe.py`
- `scripts/analysis/token_group_consistency_probe.py`
- `config/WikiText103/**/segment_curriculum*.py`

## 重要 Caveats

- `non_permute` 有 absolute-position prior。
- `block_order_block_len > 1` 有 block-internal fixed-order prior。
- `permute_data=True` 打破 current position 和 original position 的直接单调对应，但模型仍知道 current-frame absolute positions。
- benchmark 默认可能在 `train` 或 `val` 上 mining；如果用同一 split 做最终 eval，会有 model-selection leakage。
- curriculum 会放大前一阶段挖出的结构，所以 curriculum 成功不等于 base backbone 已经独立学出完整 global order。
- final unit collapse 不等于 order 正确；文本要看 original-frame Kendall、adjacent/gap 分布、loss baseline、forward/reverse symmetry。
- `token_micro_segments` 已退出主线，不要把它当作当前 active method。

## 修改或回答前请先复述

如果你要改代码或回答实验结论，在修改前请先复述：

1. 项目目标。
2. current-frame / original-frame 约定。
3. 当前 Distribution / Laplacian-Fiedler / EMA 消融 pipeline。
4. MLP 蒸馏和 frozen-MLP 接入训练分别在验证什么。
5. 文本主线为何只做 block-level，以及 token-micro 为什么退出主线。
6. 你准备修改哪些文件，为什么。

除非用户明确要求，否则不要改训练语义，不要把 `*_original` 字段用于训练，不要把 token-level final unit 数量当成 clean order recovery 证据。
