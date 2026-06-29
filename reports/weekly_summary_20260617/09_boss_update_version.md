# Boss Update Version

本周我们先清理旧结果，再把 attention-order discovery 的 protocol 收紧。

第一步是清理旧 evidence package 里的 training-result confusion。最重要的修正是 seed42 random baseline：它不是 3.354，而是 3.466；3.354 实际来自 frozen_beta run。这个修正之后，seed42 的 random-vs-L2R gap 是 0.125，和 seed-123 的 0.140 同级，不再是之前误以为的 narrow-gap case。基于这个修正，legacy frozen `g_beta` acceleration 仍然成立：seed-123 和 seed42 两个 matched groups 都有明显 recovery，保守 step saving 在 from10k/from20k 设置下是 33-42%。但这些训练加速结果要明确标注为 legacy B0 controller path。

第二步是重新检查 B1 attention extraction。我们发现当前 result-bearing B1 files 实际使用的是 `none_mode=predictor`，也就是 collaborator-aligned predictor frame diagnostic。B1 下 signal 仍然很强：clean-base B1 ladder 在 10k/50k/60k 都有 L0H0 tau=1.000；最新 continuous B1 tracking 的 final best head 是 L0H4，tau 约 0.958。这说明 order-bearing signal 不是旧 B0 extraction artifact。但是，B1 diagnostic 还不能自动替代 frozen hook acceleration，因为现有 hook/g_beta 数据仍然来自 legacy B0 selected-head path。

第三步是发现旧 extraction 中有几个概念混淆。AO-GPT 训练仍是 token-level AR next-token loss；block graph 是从 attention 聚合出来的。这里必须区分 data block、query/predictor block 和 source/content block。例如 data block0 是 `x0,x1,x2,x3`，但预测这些 token 的 query positions 是 `[None],x0,x1,x2`，source/content block0 又是 `x0,x1,x2,x3`。所以不能用同一套 block label 同时标 query 和 key。

这周最大的变化是：我们不再把 None 手动并到 block0，而是把 None 作为独立 BOS node；在这个更严格协议下，L0H1-L0H4 仍能从 None 出发自动选出 physical block0 并走出完整 L2R。

这个 strict 65-node protocol 的定义是：node0 是 None / BOS，nodes 1..64 是真实 physical content blocks，rollout 从 None 出发，first physical block 由 graph structure 决定。这样避免了 `[None]->phys0` folding 带来的 canonical-start anchor。换句话说，旧 `[None]->phys0` 可以作为 anchored controller evidence，但不能作为 label-free discovery 证据。

在这个更严格协议下，collaborator @50k 的 L0H1-L0H4 strong pass。它们在 `L` 和 `C-D+L` readout 下都能得到 tau=1.000，first=0，phys0_rank=0，prefix@4=4，prefix@8=8。destroyed controls 接近随机，强头的 destroyed mean |tau| 大约在 0.05-0.067。反过来，L0H7 失败：C-D+L 下 tau 只有约 0.292，first=45，phys0_rank=20，prefix@4=0。这说明结果不是所有 head 的 trivial bias，而是 selected-head mechanism。

clean-base ladder 进一步说明这个现象不是单个 checkpoint 偶然出现。0/1k 没有 strong heads；从 5k 到 60k，每个 checkpoint 都有 10-16 个 strong rows，best tau 都是 1.000。需要注意的是 head identity 会漂移：clean-base 的稳定强头主要在 L1H0-L1H4，collaborator strong heads 是 L0H1-L0H4。这说明现象稳定，但具体 head index 不固定。

当前最稳妥的 framing 是：strict 65-node discovery 支撑 mechanism claim；legacy frozen `g_beta` / hook 支撑 historical controller acceleration claim。两者都是真的，但现在还不是同一条完全闭合的 method chain。下一步要做的是把 strict 65-node teacher 接到 `g_beta` / hook 上，跑短 smoke 和 destroyed-B sanity，然后和 legacy B0 hook 做对照。如果这个闭环跑通，paper story 就能从 mechanism discovery 连到 training acceleration。

