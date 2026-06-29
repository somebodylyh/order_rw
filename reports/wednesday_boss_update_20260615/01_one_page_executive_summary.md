# 周三汇报一页摘要

## 一句话结论

Text-side core story 已基本闭合：random-order AO-GPT 产生稀疏 order-bearing heads，我们用 selected-head `B` 蒸馏 frozen `g_beta` controller，在两个 matched seed groups 上带来 33-42% step saving；B1/predictor-aligned extraction 进一步确认 signal 不是旧 B0 提取方式的 artifact。当前程序方向是 B1 替代 B0，但已完成的 frozen-hook acceleration 证据仍是 legacy B0 hook path。

## 当前主结果

1. Attention signal exists: B0 clean-base 10k 有 9/32 heads 满足 `|tau|>0.9`，best head `L0H0 tau=1.000`。Source: `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json`。
2. Signal survives B1/predictor alignment: clean-base B1 ladder 在 10k/50k/60k 都有 `L0H0 tau=1.000000`；latest continuous seed124 final best `L0H4 tau=0.957589`。Sources: `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json`; `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`。
3. Legacy B0 frozen `g_beta` accelerates: seed123 L0H2 recovery = 86.0/83.0/58.7%; seed42 L0H4 recovery = 106.1/101.5/66.9%; conservative step saving = 33-42%。Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`。

## 这次更新

1. B1/predictor-aligned diagnostic 已纳入主 mechanism story：signal survives collaborator-aligned extraction。
2. Claim boundary 更清楚：B1 目前证明 signal robustness，不自动证明 B1 hook acceleration。
3. 总 story 改为迁移期框架：B1 是 target unified protocol；B0 是当前 verified acceleration evidence 和 future appendix/legacy baseline。

## 现在可以 claim

- Random-order AO-GPT develops sparse order-bearing attention heads.
- The order-bearing signal survives B1/predictor-aligned extraction.
- Under the legacy B0 controller path, `g_beta` reads structured selected-head `B`, not a constant L2R prior.
- Under the legacy B0 hook path, frozen `g_beta` accelerates canonical-order training across two matched seed groups.
- Granularity and 317M diagnostics show the signal is not limited to the 47M/64-block setting, but those are diagnostic rather than hook-acceleration results.

## 现在不能 claim

- B1 frozen-hook acceleration is established.
- Existing acceleration runs already use B1.
- `g_beta` discovers orders beyond L2R.
- ori-L2R is an upper bound.
- val_unstructured also improves.
- Fully label-free head selection and training are end-to-end verified.
- Image/multimodal extension is solved.

## 需要老板决定

1. Paper scope: 先投 text-side order-controller paper，还是继续扩到 scale/image/multimodal 后再投。
2. B1 migration depth: 是否为了主文统一协议补 B1 hook smoke/full rerun，还是把 B1 作为 diagnostic robustness、B0 acceleration 作为 legacy controller result。
3. Compute priority: third seed、B1 hook、317M hook、label-free audition closure、CDL matched run placement，哪个优先。

