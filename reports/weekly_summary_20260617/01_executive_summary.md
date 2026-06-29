# Weekly Summary - AO-GPT Block-level Order Discovery

## 一句话结论

本周我们把 AO-GPT 的 attention-order 机制从旧的 B0/B1 extraction 混乱中重新收紧到 strict 65-node label-free block graph：None 单独作为 BOS node，64 个 content nodes 是真实 physical blocks，rollout 从 None 出发。结果显示，selected early heads 可以在不手动指定 block0 的情况下恢复完整 L2R block order；该现象 head-specific、extraction-frame-dependent，但在 5k-60k clean-base sweep 中稳定存在，destroyed controls 接近随机。

同时，现有 frozen `g_beta` / hook acceleration 结果仍是 legacy controller path；strict 65-node discovery 目前主要支撑 mechanism claim。下一步需要把 strict 65-node teacher 蒸馏到 `g_beta` / hook 中，闭合 training story。

## 本周最重要的进展

1. 重新核对旧 evidence package，修正 seed42 baseline：seed42 random baseline @50k 是 `3.466`，不是 `3.354`；`3.354` 属于 frozen_beta run，不是 clean random baseline。
2. 明确 B1 result-bearing files 实际使用 `none_mode=predictor`，B1/predictor diagnostic signal 仍强，但不能自动替代 legacy B0 hook acceleration。
3. 澄清 token-level AR next-token shift：data block、query/predictor block、source/content block 不能共用同一套 block label。
4. 发现 `[None]->phys0` folding 会提供 canonical start，因此不能作为 label-free start discovery；strict protocol 必须把 None 单独作为 node。
5. 在 strict 65-node label-free 协议下，collaborator @50k 的 L0H1-L0H4 strong pass，clean-base 5k-60k 也稳定出现 10-16 个 strong rows；destroyed controls 接近随机。

## 当前最强结果

Strict 65-node LF graph 下，collaborator @50k 的 L0H1-L0H4 在 `L` 和 `C-D+L` readout 中都能从 None 出发恢复完整 L2R：`tau=1.000`、`first=0`、`phys0_rank=0`、`prefix@4=4`、`prefix@8=8`。同一 full sweep 中 L0H7 失败，说明这不是所有 head 的 trivial 行为。

Clean-base 9-step ladder 显示：0/1k 没有 strong heads；从 5k 到 60k，每个 checkpoint 有 10-16 个 strong rows，best tau 均达到 1.000。强头身份会漂移，但现象本身稳定。

## 当前不能 claim 的边界

- 不能说所有 heads 都 discover L2R；结果是 selected-head、head-specific。
- 不能说 head identity 固定；clean-base 和 collaborator 的强头不同，同一 ladder 中 best head 也漂移。
- 不能说 B1 predictor content-only alone sufficient；对 L0H1/L0H2/L0H4，它恢复 order axis 但 anchor 错。
- 不能说当前 frozen hook acceleration 已经使用 strict 65-node teacher；现有 acceleration 是 legacy B0 controller path。
- 不能说训练 loss 是 true block-level；模型训练仍是 token-level AR，block graph 是 attention aggregation。

## 下周最重要的计划

P0 是把 mechanism-to-controller loop 闭合：用 strict 65-node teacher 训练/蒸馏 `g_beta`，做短 hook smoke，对比 legacy B0 hook，并在 strict teacher 下重做 destroyed-B sanity。P1 是定义干净的 label-free head audition protocol，避免用 oracle tau 选头。P1/P2 继续补 robust checks、317M strict diagnostic、paper figures 和 method/limitations 文本。

