# Attention Story Summary

## 1. 最初的问题

这两天 attention 讨论的起点是：我们一开始以为 attention graph 可以直接恢复 L2R block order。这个判断本身方向是对的，但旧证据里混了几个 protocol 问题：

- B0 / B1 extraction 不一致；
- B1 predictor frame 的现有 result files 实际用了 `none_mode=predictor`；
- AO-GPT 是 token-level AR next-token training，有 predictor-target shift；
- data block、query/predictor block、source/context block 不能混成同一个 block label；
- `[None] -> phys0` 会提供 canonical start anchor，不能作为 label-free start discovery；
- content-only graph 能看到 order axis，但 anchor 可能不稳定。

所以这两天的核心工作不是证明“attention 有信号”这一点，而是把这个信号放进一个不会偷用起点、不会混淆坐标、能解释给老板和 reviewer 的 strict protocol。

## 2. AR next-token shift 的澄清

AO-GPT 的训练 loss 仍然是 token-level next-token AR。block-level graph 是从 attention 聚合出来的，不是训练时直接优化的 block-level loss。

关键对齐关系：

```text
data/physical block0              = x0,x1,x2,x3
query positions predicting block0 = [None],x0,x1,x2
source/content block0             = x0,x1,x2,x3
```

因此不能用同一套 block label 同时标 query 和 key。旧式把 `[None],x0,x1,x2` 和 `x0,x1,x2,x3` 当成同一个 block 的做法，会混淆 predictor block、target block 和 content block。

结论：如果要讲 block-level discovery，None 不能并入 block0，必须作为独立 BOS node。

## 3. 协议收紧

最终采用的主协议是：

> strict 65-node None-separated block graph

定义：

- node None = independent BOS/start node；
- node `1+i` = physical content block `i = x_{4i},...,x_{4i+3}`；
- rollout starts from None；
- first content block is selected by graph edge/readout；
- None is not manually attached to physical0。

这解决了 `[None]->phys0` folding 的 start-anchor 问题。它也把 “model frame / original L2R / posthoc scoring” 的边界拆开：图构造可以按坐标 remap，rollout 不能用 oracle order，posthoc scoring 可以翻译回物理 L2R 来评估。

## 4. 最新机制结论

Strict 65-node label-free block-level L2R discovery is supported in selected early heads.

当前最强证据是 collaborator @50k strict LF full sweep：L0H1-L0H4 在 `L` 和 `C-D+L` readout 下 strong pass，`tau=1.000`、`first=0`、`phys0_rank=0`、`prefix@4=4`、`prefix@8=8`；L0H7 失败，destroyed controls 接近随机。

边界必须同时保留：

- head-specific：不是所有 heads 都 discover L2R；
- extraction-frame-dependent：B1 predictor content-only 可以恢复 cyclic axis，但对 L0H1/L0H2/L0H4 会丢 anchor；
- not all heads：L0H7 fail 是重要负例；
- existing frozen hook acceleration still legacy controller path unless strict 65-node hook has been run。

## 5. Token-level attention map 的坐标解释

这两天还讨论了 token-level attention map 的坐标系。model/reveal frame 下，causal future region 是 0；original L2R label 下，因为 random reveal order remap，attention 会出现在原始对角线两侧。这个现象不是双向 attention 的证据，而是坐标系变换的结果。

Relevant figures:

- reveal/model frame: `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.png`
- original L2R labels: `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.png`

