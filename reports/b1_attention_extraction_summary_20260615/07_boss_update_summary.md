# 07 - Boss Update Summary

## 1. 我们更新了什么

这次把 attention extraction 从旧的 B0/canonical 路径，整理到新的 B1/predictor-aligned 诊断路径。最终定位是：B1 是 collaborator-aligned attention diagnostic protocol，不是当前 frozen-hook acceleration protocol。最新已有结果文件里，B1 实际对应的是 `none_mode=predictor`：用 AO-GPT 的 predictor frame `attn[:-1, :-1]` 做 token-to-block 聚合，然后做 batch mean，再用 `B=A.T` 给 CDL/readout 评估。

需要注意一个命名细节：代码里还存在 `none_mode=b1`，它是 predictor frame + physical remap；但我这次没有找到任何已有结果文件实际使用 `none_mode=b1`。当前 B1 结果目录和脚本使用的是 `predictor`。这个命名差异需要在论文/代码注释里澄清。

## 2. B1 和合作者如何对齐

本地脚本把它称为 B1 predictor-aligned extraction，核心对齐点是使用模型原始预测帧 `attn[:-1, :-1]`，而不是旧版 B0 的 `[None] -> physical block 0` 折叠方式。连续数据加载也在代码里标注为 collaborator-style continuous loading。

但我没有在当前仓库里找到外部合作者的 extraction 脚本，因此不能写成“已和外部实现 bit-for-bit 对齐”。最稳的说法是：我们采用了本地 B1 predictor-aligned convention；它与合作者约定的 predictor-frame 方向一致，但外部代码级一致性仍未在本仓库中直接验证。

## 3. B1 下信号是否还在

还在，而且很强。

clean-base B1 ladder 里，10k 时三个 scan seed 都是 `L0H0 tau=1.000000`，`|tau|>0.9` 的 head 数是 7/32。50k 和 60k 时，仍然是 `L0H0 tau=1.000000`，`|tau|>0.9` 的 head 数是 6/32。这个结果明显降低了“order signal 只是早期 transient”的风险，但要谨慎说明它是 clean-base ladder，不是 seed-123 continuous L0H2 的 matched B1 scan。来源是 `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json`。

最新连续训练 B1 all-head tracking 里，最终 60000 步最强 head 是 `L0H4`，`tau=0.957589`，`mean_pairwise_tau=0.930060`；最终 `|tau|>0.9` 的 head 是 1 个，`|tau|>0.7` 是 9 个。全程看 `L0H4` 在 5768/6001 次 tracking 中是 top head，平均 `tau=0.966056`。来源是 `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`。

## 4. 对之前结论有什么影响

最直接增强的是 Claim 1：random-order AO-GPT 会形成 sparse order-bearing attention heads，而且这个信号 survives B1/predictor-aligned extraction。B1 说明这个发现不是 B0 提取方式的偶然产物。

对 g_beta 和 frozen hook 的结论要分开讲。现有 g_beta dataset、g_beta sanity、frozen hook acceleration 都还是 B0 路径。代码里 `selected_head_dataset.py` 默认 `none_mode="b0"`，`hook_order_provider.py` 直接用 `_attn_to_A_block_b0_vec`。所以 B1 现在证明的是“信号存在且 survives protocol alignment”，不是“B1 hook 已经加速训练”。总 story 应该是：B1 strengthens the mechanism story, but does not replace the B0 acceleration evidence.

## 5. 需要不需要重跑 frozen g_beta

不需要为了保留现有 B0 acceleration claim 而立刻重跑。现有训练加速结论仍然是 B0 legacy 结果。

如果论文要把主方法完全改成 B1 controller，或者要写“B1 frozen hook 也加速训练”，那就需要 B1 hook rerun。当前最稳的表述是：

> B1 confirms that the order-bearing signal is not an artifact of the legacy B0 extraction. Existing acceleration numbers are legacy-B0 controller results; B1-controller acceleration remains a future rerun if needed.

## 6. 下一步建议

第一，先统一命名：当前结果目录叫 B1，但结果字段是 `none_mode=predictor`；代码 enum `b1` 又表示 predictor frame + physical remap。建议在代码和 paper method 里明确“result-bearing B1 = predictor-aligned convention”，或者补一个真正 `none_mode=b1` 的小规模 scan。

第二，如果 paper 主结果要全部 post-alignment，建议补三个便宜诊断：B1 shuffled-L2R control、seed-42 continuous B1 scan、seed-123 continuous B1 scan。第三，如果 boss/paper 需要 scale claim 也 post-alignment，再补 317M B1 scan；否则 317M 继续作为 B0 legacy appendix。

一句话给老板/合作者：我们把 attention extraction 更新到和合作者对齐的 B1/predictor convention 后，order-bearing signal 依然很强：clean-base ladder 在 10k/50k/60k 都有 head 达到 `tau=1.0`，最新 continuous B1 run 60k 时 best head `L0H4` 也有 `tau=0.958`。这说明 attention-order signal 不是旧 B0 提取方式的 artifact。不过现有 g_beta pretrain 和 frozen hook 加速结果仍是 B0 legacy path，所以 B1 目前增强的是 diagnostic robustness；如果要 claim B1-controller acceleration，需要单独 rerun B1 hook。
