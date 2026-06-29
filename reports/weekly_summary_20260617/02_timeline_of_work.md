# Timeline of Work

## Phase 1 - Evidence Package Verification

本周首先清理旧 evidence package 中的 training-result confusion。重点是把 seed group、baseline、L2R reference 和 method run 对齐。

- Matched seed groups were reconstructed from primary sources: `config.json` and `eval_curve.tsv`.
- seed42 baseline 被修正：`random_baseline_continuous_jun05` 的 `val_ori_l2r_block @50k = 3.466`；旧的 `3.354` 实际来自 frozen_beta run。
- seed42 gap 从错误的 narrow gap 修正为正常 gap：`3.466 - 3.341 = 0.125`。
- Recovery formula 和 step-saving formula 被重新验证并写入 verified tables。
- `g_beta` sanity raw JSON 被保存：`reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`。
- CDL teacher seed mismatch 被标注：seed-2 CDL teacher 不能和 seed-123 frozen_beta 直接混成 matched comparison。

Primary sources:

- `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`
- `reports/evidence_package_20260613_verified/03_DEPRECATED_NUMBERS.md`
- `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`

## Phase 2 - B1 Attention Extraction Update

B1 audit 的核心结论是：当前 result-bearing B1 files 使用的是 collaborator-aligned / predictor-frame diagnostic convention，但文件中实际记录的是 `none_mode=predictor`，不是代码 enum `none_mode=b1`。

- B1/predictor convention 使用 `attn[:-1, :-1]`，保留 AO-GPT predictor frame。
- Clean-base B1 ladder 显示 late checkpoints 仍有强 signal：10k L0H0 tau=1.000，50k/60k L0H0 tau=1.000。
- 最新 continuous B1 tracking seed=124 在 final 60k 的 best head 是 L0H4，`tau=0.957589`，`mean_pairwise_tau=0.930060`。
- 但 B1 diagnostic 不自动等于 B1 hook acceleration。现有 frozen hook acceleration 仍是 legacy B0 selected-head/controller path。

Primary sources:

- `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md`
- `reports/b1_attention_extraction_summary_20260615/03_b1_signal_summary.md`
- `reports/b1_attention_extraction_summary_20260615/05_impact_on_existing_claims.md`

## Phase 3 - Predictor-Target Shift / Block Definition Clarification

本周把 block-level discovery 和 token-level AR loss 的对齐问题重新拆清楚。AO-GPT 训练是 token-level next-token AR；attention graph 是从 token attention 聚合成 block graph，不是训练时直接优化 block-level loss。

关键不对称：

- data block0 = `x0,x1,x2,x3`
- query positions predicting block0 = `[None],x0,x1,x2`
- source/content block0 = `x0,x1,x2,x3`

因此不能用同一套 block label 同时标 query 和 key。旧式 `[None],x0,x1,x2,x3 -> x0,x1,x2,x3` 的聚合会混淆 predictor block、target block 和 content block。

结论：如果要讲 block-level discovery，None 不能被偷偷折到 physical block0；None 必须作为独立 BOS node。

Primary sources:

- `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`
- `scripts/search_none_separated_65_heads.py`

## Phase 4 - None Anchor / Leak Audit

旧的 `[None]->phys0` folding 提供了 canonical start，因此它可以作为 anchored controller evidence，但不能证明 label-free start discovery。

- `[None]->phys0` 不是无锚点发现；它把起点信息注入了 block0。
- content-only graph 可以恢复 order axis，但起点不稳定。例如 collaborator @50k 的 L0H1/L0H2/L0H4 在 B1 content-only 中 best cyclic tau=1.000，但 anchored tau 只有 0.655，first physical 从 6 开始。
- 因此本周决定将主 protocol 收紧为 strict 65-node None-separated graph。

Primary sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`
- `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json`

## Phase 5 - Strict 65-node Label-free Discovery

Strict graph definition:

- Node 0 = None / BOS start node.
- Nodes 1..64 = physical content blocks `x_{4i}..x_{4i+3}`.
- Rollout starts from None.
- First physical block is determined by graph structure, not manually specified.

Results:

- Collaborator @50k full sweep: strict LF and oracle-remapped protocols have identical gate distribution: 20 strong, 11 weak, 225 fail out of 256 rows.
- L0H1-L0H4 strong pass under `L` and `C-D+L`: `tau=1.000`, `first=0`, `phys0_rank=0`, `p4=4`, `p8=8`.
- L0H7 fails: `C-D+L tau=0.292`, `first=45`, `phys0_rank=20`, `p4=0`, `p8=0`.
- Destroyed controls are near random for the strong heads: roughly 0.05-0.067 destroyed mean |tau| in collaborator strong-head rows.
- Clean-base 9-step ladder: 0/1k no strong heads; 5k-60k has 10-16 strong rows per checkpoint.

Primary sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/04_ckpt_sweep_results.md`
- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`

## Phase 6 - Current Framing

Current framing:

- Mechanism discovery is supported under strict 65-node LF protocol.
- The effect is head-specific and extraction-frame-dependent.
- strict 65-node discovery and legacy acceleration are separate evidence streams.
- Existing frozen `g_beta` acceleration remains valid, but as legacy B0 controller evidence.
- Next step is strict 65-node teacher distillation / hook rerun to make the paper story one continuous chain.

