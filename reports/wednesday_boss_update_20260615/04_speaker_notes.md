# Speaker Notes

## Slide 1 - One-Line Result

这页先给老板一个判断点：text-side story 已经不是单个现象，而是一条 evidence chain。random-order AO-GPT 里出现 sparse order-bearing heads，我们把 selected-head graph 蒸馏成 frozen `g_beta`，在两个 matched seed groups 上看到 33-42% step saving。B1 更新的作用是确认 attention signal 不是旧 B0 extraction 的 artifact。记住：B1 当前增强的是 mechanism robustness，acceleration 已验证部分仍是 B0 legacy hook。

老板应记住：核心结果是 text-side attention-derived order controller can accelerate canonical-order training.

## Slide 2 - Problem and Hypothesis

AO-GPT 理论上允许任意 reveal order，但 random order 对 text 来说可能浪费训练效率。如果模型内部 attention 已经出现 physical sequence structure，我们可以把它读出来变成训练 order controller。这里的 hypothesis 很窄：不是所有 heads 都有 order，也不是发现 beyond-L2R order，而是 random-order training 会产生少量 L2R-like order-bearing heads。

老板应记住：我们利用模型自己产生的 attention order signal，而不是手工指定一个固定 prior。

## Slide 3 - Clean-Permutation Protocol

这页要防止误解。我们用 fixed clean permutation，不是 per-sample permutation。模型看到的是 model-coordinate order，physical order 不直接给模型。诊断时再 inverse-remap 到 physical coordinate，看 attention-derived order 是否接近 physical L2R。这个设置让问题可控，也说明我们现在不是在做 document unscrambling 或 per-sample adaptive ordering。

老板应记住：protocol 是固定 permutation 下的 text-side order recovery and training acceleration。

## Slide 4 - Method Pipeline

完整 pipeline 是从 checkpoint 抽 selected-head `B^{l,h}`，用 CDL 得到 teacher order，训练 `g_beta` 做 pairwise ranking readout，然后把 `g_beta` freeze 后 hook 回训练。CDL 是 offline teacher，不是部署时在线算 CDL；部署时用的是 learned readout。order 是 batch-global canonical order，所以 evaluation 也主要看 canonical-order validation loss。

老板应记住：`g_beta` 是 attention graph 到 order 的 learned controller，不是直接把 L2R 写进去。

## Slide 5 - Attention Signal Evidence

B0 clean-base 10k 已经有 9/32 strong heads，best `tau=1.0`。B1/predictor 后 clean-base ladder 到 50k/60k 仍然有 `L0H0 tau=1.0`，latest continuous seed124 final best `L0H4 tau=0.957589`。317M B0 也有 6/256 strong heads。这里要强调 sparse：不是所有 heads collapse 成 L2R，而是少数 heads  carrying signal。

老板应记住：B1 后 signal 还在，因此不是 B0 extraction artifact。

## Slide 6 - `g_beta` Sanity

这里回答最关键质疑：`g_beta` 是不是只是输出 L2R？不是。real B 输出 `tau=0.9675`，Gaussian / shuffled / row-col shuffled 都接近 0，Gaussian family pairwise tau 也接近 0。zero B 输出 L2R 是 margin=0 tie-breaking artifact，不代表 prior。边界是：这个 sanity 是 legacy B0 controller path；B1 `g_beta` sanity 还没在这包里闭合。

老板应记住：当前证据支持 `g_beta` reads structured B, not a constant prior。

## Slide 7 - Training Acceleration

主训练结果来自 two matched seed groups。seed123 L0H2 from10k/from20k/from40k recovery 是 86.0/83.0/58.7%；seed42 L0H4 是 106.1/101.5/66.9%。从 step saving 看，from10k/from20k 的 conservative range 是 33-42%。要主动说明 ori-L2R 是 reference，不是 upper bound；seed42 recovery 超过 100% 说明 controller order 在该模型上低于 L2R reference loss。

老板应记住：已验证 acceleration 是 legacy B0 hook path，但它是当前最强 training evidence。

## Slide 8 - B1 Update and Impact

B1 不是简单替换一句名字。当前 result-bearing B1 用的是 `none_mode=predictor`，也就是 predictor-frame extraction；代码 enum `none_mode=b1` 是另一个 physical-remap variant，结果里还没找到。B1 的已完成价值是：collaborator-aligned diagnostic 下 signal 仍强。因为我们正在做 B1 替代 B0 的程序，最终目标可以是 B1 unified protocol；但今天汇报时不能把 B0 hook acceleration 说成 B1 hook acceleration。

老板应记住：B1 is target protocol, but current acceleration claim remains B0 until B1 hook finishes.

## Slide 9 - Limitations / Reviewer Attack Surface

主动列边界更稳。第一，val_unstructured 会退化，这是 canonical-order specialization 的代价，不是 universal likelihood improvement。第二，B1 hook acceleration 和 B1 `g_beta` sanity 尚未完成。第三，label-free audition 没有 end-to-end 闭合，因为 `g_beta` 训练还用 CDL teacher。第四，317M 只有 diagnostic，没有 hook。第五，CDL teacher matched run 如果没有最终 verified 数字，就只能标 pending 或 supplementary。

老板应记住：最稳 paper claim 是 canonical-order training acceleration, not universal order robustness。

## Slide 10 - Decisions Needed

最后把问题变成决策。我们可以先写 text-side paper，把 image/multimodal 放 future work；也可以继续做 B1 hook、third seed、317M hook 或 label-free audition。我的建议是先把 text-side claim lock 和 B1 diagnostic 写进 paper draft，同时用小成本 B1 hook smoke 或 label-free audition closure 作为加分项，而不是立刻把所有方向都展开。

老板应记住：下一步要选择 paper scope 和 compute priority，而不是继续无限补诊断。

