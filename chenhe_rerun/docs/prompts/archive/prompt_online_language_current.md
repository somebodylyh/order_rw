> Archive note: this prompt captures the older online attention/fixed-head
> diagnostic thread. It is useful background, but the current default task is
> the `seq256/permute/block64` Laplacian/Fiedler distribution method plus MLP
> distillation. Start from `prompt_language_block_current_task.md` for new work.

# Prompt: Current Online Language Order Research

你是一个没有本项目历史记忆的新 agent。请把下面内容当成本轮工作的项目背景。当前任务只关注语言 / WikiText103，不要主动扩展到已删除的非语言分支。

## 当前目标

本仓库是 `nanogpt-learned-order`，一个基于 nanoGPT 改造的 AO-GPT / learned-order 实验项目。

现在的目标不是马上训练 MLP，而是先验证：

> 在线使用当前模型的 attention 和 current-frame loss 信号，能否在 WikiText103 上持续产生有用的训练 order，并改善或稳定 PPL。

如果 online teacher 证明有效，下一步才是把 “attention 找顺序” 的正流程蒸馏成 `attn_mlp_order_policy.py` 中的小 MLP，让 order policy 更稳定、更可解释、更容易部署。

## 必须遵守的 frame 约定

训练和 order selection 只能使用 current-frame model-side signals。

禁止把这些用于训练决策、candidate selection、direction selection、temperature selection 或 early stop：

```text
original L2R order
original-frame Kendall tau
original-frame oracle diagnostics
human scan / raster / Morton / Hilbert / column-major
validation PPL for policy selection
```

允许使用：

```text
current-frame attention matrix
current-frame spectral graph recovery
current-frame candidate attention path score
current-frame directed attention score
band quality
current-frame train/probe teacher-forced loss
```

如果 `permute_data=True`：

```text
cached_order / priority_ema / sampled orders 都必须是 current-frame ids
*_original 只用于 logging / diagnostics
OriginalL2R 是 oracle upper bound，不是 no-prior learned-order 结果
```

如果 `permute_data=False`，current frame 等于自然文本 frame，但仍要把实验解释成使用 current-model attention/loss signal，而不是用 tau 或 oracle 选 order。

## 先读这些文件

```text
train.py
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/eval/eval_lm_original_order_ppl.py
scripts/eval/eval_lm_ppl_sweep.py
docs/findings/findings_language.md
docs/structure/PROJECT_STRUCTURE_language.md
base.md
base_cn.md
```

## 当前核心 configs

Fixed-head top-1:

```text
config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order.py
config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank.py
config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank_late_l2r_8000_50000.py
config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py
config/WikiText103/seq80/non_permute/block1/online_spectral_cached_order.py
config/WikiText103/seq80/non_permute/block1/online_spectral_fixed_head_order_loss_rerank_warmup5k_anneal15k_update20_top96_prefix16_no_freeze.py
```

Distribution:

```text
config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution.py
config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank.py
config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_late_l2r_8000_50000.py
config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_fast_b128.py
config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_fast_b256.py
config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution_loss_rerank_top4_start8k_update20_top96_prefix16.py
config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution_loss_rerank_top4_warmup15k_dist15k40k_freeze10k_update20_top96_prefix16.py
```

Seq80/block1 diagnostic baseline:

```text
config/WikiText103/seq80/non_permute/block1/random.py
config/WikiText103/seq80/non_permute/block1/random_save_attn_ckpts_1k2k5k8k10k15k20k25k50k.py
```

## 2026-06-07 Seq80 实验现状

当前用户正在聚焦 seq80/block1 non-permute，因为之前 online order 的 attention
热力图看起来过于凌乱，需要用 clean random baseline 的中间 checkpoint 判断
“attention 乱”是 random AO-GPT 本身的训练现象，还是 online policy 造成的。

已完成：

```text
config: config/WikiText103/seq80/non_permute/block1/random_save_attn_ckpts_1k2k5k8k10k15k20k25k50k.py
out:    out/base/nonpermute/seq80/block1/out-wikitext103-seq80-random-b1-nonpermute-save-attn-ckpts-50000-iters
final:  step 50000, train loss 3.8768, val loss 3.9655
saved:  ckpt_iter0001000.pt, ckpt_iter0002000.pt, ckpt_iter0005000.pt,
        ckpt_iter0008000.pt, ckpt_iter0010000.pt, ckpt_iter0015000.pt,
        ckpt_iter0020000.pt, ckpt_iter0025000.pt, ckpt_iter0050000.pt
```

这个 run 的训练参数和原始 `random.py` 一致；它只是额外保存 attn 诊断 checkpoint。
这些 checkpoint 的 attention/order 诊断已经完成：

```text
random-probe report: Report/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_online_order_warmup_diagnostic
locality report:     Report/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_random_probe_remap_locality_diagnostic
AR-probe report:     Report/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_ar_probe_attention_diagnostic
attention script: scripts/analysis/seq80_random_ckpt_online_order_attn_batch.py
loss-rerank script: scripts/analysis/seq80_loss_rerank_existing_attention_candidates.py
```

结论修正：`random-probe report` 是 Random reveal 后 remap 到 current L2R 的视图，
对当前 policy 真正相关的是固定 L0H7 `with_none`。这个视图在 Random probe 下确实
很乱：主要是列/全局结构，不是局部对角带，近邻/远邻 ratio 基本 `1.00`，谱排序
top1 tau 也接近 0。`locality report` 另有一个 sanity check：Random 多样本平均并
remap 回 current L2R 后，all-layer/all-head `without_none` 有清楚局部对角带，
近邻/远邻 ratio 从 1k 的 `1.06` 上升到 50k 的 `2.04`。这只说明模型内部某处有
local signal，不说明当前单头 `with_none` policy path 能用好它。`AR-probe report` 才是和
AR-style heatmap 可比的视图，固定 L0H7 在 50k 有清楚 causal/对角结构，并且
recovered tau 是高正或高负：
`2k=0.461`、`5k=-0.643`、`10k=0.597`、`15k=-0.651`、`50k=0.431`。

warmup 解释：random checkpoint 中并不是没有 order signal；甚至 2k/5k 就已经有强
局部 order axis。真正的问题是方向会翻转，而且 probe policy 很关键。online 训练中的
attention collection 使用 `_forward_with_active_training_policy`，5k 后会逐渐混入
cached order，15k 后基本按 cached order probe，不是强制 Random probe。因此后续
seq80 调参要重点控制 probe mode、aggregation 和 current-frame direction-breaking。

当前 active follow-up：

```text
config: config/WikiText103/seq80/non_permute/block1/online_spectral_fixed_head_order_loss_rerank_warmup5k_anneal15k_update20_top96_prefix16_no_freeze.py
out:    out/base/nonpermute/seq80/block1/out-wikitext103-seq80-online-spectral-fixed-head-loss-rerank-warmup5k-anneal15k-update20-top96-prefix16-kall-no-freeze-b1-nonpermute-50000-iters
mode:   OnlineSpectralFixedHeadOrder
```

Schedule:

```text
0..5000:     pure random, no online probe/update
5000..15000: random -> fixed-head cached order anneal
15000..50000: use latest fixed-head order and keep updating every 20 steps
no final fixed-order freeze
```

At documentation time this run had already passed about 33k training steps.
The latest checkpoint snapshot was around `iter_num=33250`,
`best_val_loss=4.0702`, with more than 1400 policy updates. Its order history
is `online_spectral_policy_order_history.jsonl`. Treat this as an in-progress
run, not a final result.

## Online fixed-head top-1 方法

`aogpt_train_mode='OnlineSpectralFixedHeadOrder'`。

每次 online update：

```text
current train/probe batch
-> current model forward
-> collect attention from fixed layer/head, usually layer 0 head 7
-> aggregate to current-frame block attention matrix
-> enumerate spectral candidate orders
-> score candidates with current-frame attention-side score
-> optional loss rerank
-> cache top-1 order
-> later train samples use cached_order when policy anneal selects policy
```

典型 block64 设置：

```text
block_size = 256
block_order_block_len = 4
num_blocks = 64
batch_size = 64
gradient_accumulation_steps = 2
```

训练数据量：

```text
per optimizer step = 64 x 2 = 128 sequences
tokens per optimizer step = 128 x 256 = 32768
```

普通 fixed-head loss-rerank config：

```text
online_spectral_policy_update_every = 10
attention probe = 64 x 64 = 4096 sequences per update
loss rerank    = 16 x 64 = 1024 sequences per candidate
loss_rerank_top_k = 64
prefix_k = 8 blocks
```

## Candidate scoring

基础 attention-spectral candidate score 使用 current-frame attention graph：

```text
attention adjacency path score
directed attention path score
band quality
```

loss-rerank 版本先 recover top-64 attention-spectral candidates，然后对每个 candidate 用 current-frame train/probe teacher-forced loss 重新打分：

```text
final_score =
  1.0  * z(attention_spectral_score)
- 0.25 * z(prefix_loss)
- 0.15 * z(full_loss)
```

`prefix_loss` 默认看前 `8` 个 block 的 reveal loss，`full_loss` 看完整 order 的平均 block loss。loss 只用于 current-frame candidate rerank，不使用 original-frame 信息。

## OnlineSpectralOrderDistribution 方法

`aogpt_train_mode='OnlineSpectralOrderDistribution'`。

它不是永久维护 32 条 order，而是维护一个 soft priority vector：

```text
attention matrix A_t
-> spectral candidates
-> optional loss rerank
-> top-32 candidates
-> r_update priority vector
-> priority_ema
-> sampled hard orders
```

对候选 order `m`：

```text
pos_m(i) = block i 在 order_m 中的位置
v_m(i) = 1 - pos_m(i) / (num_blocks - 1)
```

越早 reveal 的 block，`v_m(i)` 越大。

候选权重：

```text
scores = candidate scores
scores_norm = zscore(scores)
w_m = softmax(scores_norm / teacher_temperature)
```

priority update：

```text
r_update(i) = sum_m w_m * v_m(i)
priority_ema = decay * priority_ema + (1 - decay) * r_update
map_order = argsort(priority_ema, descending=True)
```

默认参数：

```text
top_m = 32
teacher_temperature = 1.0
score_normalization = zscore
priority_ema_decay = 0.95
sample_temperature = 0.7
random_mix_prob = 0.2
distribution_per_sample = True
```

训练采样：

```text
if priority_ema is None or policy anneal misses:
    fallback random
else per sample:
    with probability random_mix_prob:
        uniform random order
    else:
        Gumbel-top-k(priority_ema / sample_temperature)
```

distribution loss-rerank config 的数据量：

```text
online_spectral_policy_update_every = 10
attention probe = 64 x 64 = 4096 sequences per update
loss rerank    = 32 x 64 = 2048 sequences per candidate
loss_rerank_top_k = 64
final priority update uses reranked top-32
```

fast variants 保持样本预算相近但加大 probe batch size：

```text
fast_b128:
  attention probe = 32 x 128 = 4096
  loss rerank     = 16 x 128 = 2048

fast_b256:
  attention probe = 16 x 256 = 4096
  loss rerank     = 8 x 256 = 2048
```

fast variants 还设置：

```text
random_mix_prob = 0.0
update_stop_iter = 40000
freeze_to_map_order_after_stop = True
```

## Late-L2R curriculum ablation

Late-L2R 是显式 AR/L2R prior ablation，不能当作 no-prior learned-order 证据。

它测试：

```text
online spectral/distribution early curriculum
-> gradually anneal to L2R
```

block64 schedule：

```text
0..8000:
  random fallback -> online spectral policy

8000..50000:
  online spectral policy -> L2R
```

大致：

```text
iter 8000:  L2R prob = 0
iter 29000: L2R prob ~= 0.5
iter 50000: L2R prob = 1
```

如果评估 late-L2R，请明确它是 AR-prior curriculum，对比 pure AR baseline、no-late online baseline、Random eval、checkpoint-policy eval，并检查是否只是 AR finetune 在起作用。

## Eval modes

固定头 top-1：

```text
Random
AR
OriginalL2R
CheckpointOnlineSpectralOrder
```

Distribution：

```text
Random
AR
OriginalL2R
CheckpointOnlineSpectralDistributionMAP
CheckpointOnlineSpectralDistributionSampled
CheckpointOnlineSpectralDistributionMixSampled
```

关键指标：

```text
mean_nll_original_frame
ppl_original_frame
mean_nll_reveal_frame
ppl_reveal_frame
origin Kendall tau diagnostics
priority entropy
candidate weight entropy
sampled order diversity
```

对 `permute_data=True`，`OriginalL2R` 是 oracle diagnostic。对 `non_permute`，AR/OriginalL2R 通常等价，但仍要说明这是 AR-direction eval，不是无先验选择。

## 当前结论边界

安全说法：

```text
AO-GPT 的当前模型 attention/loss 中存在可用于 online order recovery 的信号。
Fixed-head top-1 和 distribution 是验证 online teacher 是否有用的实验路径。
Loss rerank 是 current-frame train/probe model-side rerank，不是 original oracle。
Late-L2R 是显式 AR-prior curriculum ablation。
```

不要说：

```text
模型已经无监督学出了完整 global L2R order。
OriginalL2R PPL 证明了 no-prior order recovery。
top-32 distribution 永久维护了 32 条 order。
late-L2R 结果是无先验 learned order。
```

## 下一步 MLP 化

当前不要优先训练 MLP。先用 online fixed-head / distribution / loss-rerank 证明 teacher 有用。

当 teacher 有稳定收益后，再做：

```text
attention matrix / attention features
-> MLP order policy logits
-> Plackett-Luce / Gumbel-top-k sampled order
-> train AO-GPT with learned policy
```

MLP 化的目的：

```text
把复杂 spectral enumeration 正流程蒸馏成可解释、可保存、可部署的 order policy
减少每步 expensive spectral/loss rerank
避免 two-stage mismatch
让 policy 可以和 backbone 更稳定地共同训练或冻结评估
```
