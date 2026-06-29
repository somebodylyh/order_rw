# AO-GPT Attention 机制总结

## Strict 65-node Block-level Order Discovery

这份文档是中文总版，整合这两天关于 attention extraction / block-level order discovery 的讨论，并把目前找到的相关图片全部嵌入。重点是 attention 协议如何从 B0/B1 混乱，收敛到 strict 65-node label-free block-level discovery。

---

## 一句话结论

我们现在应把 AO-GPT 的 attention-order discovery 表述为 **strict 65-node None-separated block graph**：

- None 是独立 BOS node；
- 64 个 content nodes 是真实 physical blocks；
- rollout 从 None 出发；
- first content block 由 graph edge / readout 选择；
- selected early heads 可以恢复 L2R；
- 该现象是 head-specific、extraction-frame-dependent；
- 现有 frozen hook acceleration 仍是 legacy controller path，不能说已经由 strict 65-node teacher 导致。

---

## 1. 为什么要重新整理 attention extraction

最初我们以为：

> attention graph 能直接恢复 L2R block order。

这个方向是对的，但旧结果里混了几个 protocol 问题：

- B0 / B1 extraction 不一致；
- B1 predictor frame 的现有 result files 实际使用 `none_mode=predictor`；
- AO-GPT 是 token-level AR next-token training，有 predictor-target shift；
- data block、query/predictor block、source/context block 不能混成一个；
- `[None] -> phys0` 可能提供 canonical start anchor；
- content-only graph 能看到 order axis，但 anchor 可能不稳定。

所以这两天的核心不是重新证明“attention 有信号”，而是把信号放进一个不会偷用起点、不会混淆坐标、能解释给老板和 reviewer 的 strict protocol。

---

## 2. AR next-token shift 的关键澄清

AO-GPT 的训练 loss 仍然是 token-level next-token AR。Block graph 是从 token attention 聚合出来的，不是训练时直接优化的 block-level loss。

关键对齐关系：

```text
data/physical block0              = x0,x1,x2,x3
query positions predicting block0 = [None],x0,x1,x2
source/content block0             = x0,x1,x2,x3
```

因此不能用同一套 block label 同时标 query 和 key。旧式把 `[None],x0,x1,x2` 和 `x0,x1,x2,x3` 当成同一个 block 的做法，会混淆 predictor block、target block 和 content block。

结论：

> 如果要讲 block-level discovery，None 不能并入 block0，必须作为独立 BOS node。

---

## 3. 最终协议：strict 65-node None-separated graph

定义：

- node 0 = None / independent BOS start node；
- node `1+i` = physical content block `i = x_{4i},...,x_{4i+3}`；
- rollout starts from None；
- first content block is selected by graph edge/readout；
- None is not manually attached to physical0。

这个协议解决了 `[None]->phys0` folding 的 start-anchor 问题。它也把 model frame、original L2R、posthoc scoring 的边界拆开：

- 图构造可以按坐标 remap；
- rollout 不能用 oracle order；
- posthoc scoring 可以翻译回 physical L2R 来评估。

### Protocol evolution

| Protocol | None handling | 用途 | 当前状态 |
|---|---|---|---|
| B0 legacy controller extraction | `[None]` folded into physical block0 | Existing frozen `g_beta` / hook controller path | legacy acceleration evidence，不是 strict discovery |
| B1 predictor frame | 现有 B1 result files 使用 `none_mode=predictor` | collaborator-aligned diagnostic | signal strong，但不是当前 hook path |
| B1 predictor + content-only | None removed | 测 order axis 是否存在 | supporting diagnostic，会丢 anchor |
| `[None]->phys0` anchored protocol | None 手动并到 phys0 | anchored controller / older diagnostics | 不是 label-free discovery |
| loss-aligned AR extraction | None 作为 separate source column | strict 65-node graph 基础 | 当前 strict mechanism protocol |
| strict 65-node label-free graph | None independent BOS/start node | main mechanism evidence | 当前 preferred discovery protocol |
| oracle-remapped graph | labels consistent remap | equivariance check | 与 strict LF 一致，不是 leak |
| destroyed controls | edges shuffled / labels permuted | 排除 tie-break / graph-statistic confound | near random |

---

## 4. `inv_perm` 边界

| Stage | `inv_perm` allowed? | Reason |
|---|---:|---|
| graph construction | yes | graph labels may be remapped; edge weights unchanged |
| CDL rollout | no | rollout must rely only on edge weights |
| posthoc scoring | yes | only translate predicted sigma for evaluation |

Strict label-free 和 oracle-remapped 一致，不是 leak，而是 permutation equivariance。真正的 leak 是在 rollout decision 里用 `inv_perm`，当前 strict LF framing 排除了这一点。

---

## 5. 关键结果

### 5.1 Strict 65-node collaborator @50k

Source:

`reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`

| Head | Method | tau | first | phys0_rank | prefix@4 | prefix@8 | destroyed \|tau\| | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L0H1 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0498 | strong_pass |
| L0H2 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0531 | strong_pass |
| L0H3 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0501 | strong_pass |
| L0H4 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0554 | strong_pass |
| L0H7 | C-D+L | 0.292 | 45 | 20 | 0 | 0 | 0.0825 | fail |

解释：

- L0H1-L0H4 可以从 independent None/BOS 出发恢复完整 L2R；
- L0H7 fail 说明这不是 universal head property，也不是 trivial tie-break。

### 5.2 Gate distribution

Source:

`reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`

| Protocol | strong_pass | weak_pass | fail | total |
|---|---:|---:|---:|---:|
| Strict label-free | 20 | 11 | 225 | 256 |
| Oracle-remapped | 20 | 11 | 225 | 256 |

解释：

- 两者完全一致支持 permutation equivariance；
- 这不代表 rollout 用了 `inv_perm`；
- 它表示等价 label graph 在只使用 edge weights 的 rollout 下恢复等价 order。

### 5.3 Clean-base 9-step ladder

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv`
- `reports/strict_65node_discovery_ckpt_verification_20260617/04_ckpt_sweep_results.md`

| Step | strong_total | Best head/method | best tau | destroyed mean \|tau\| |
|---:|---:|---|---:|---:|
| 0 | 0 | L0H2 C-D+L | 0.191 | 0.063 |
| 1000 | 0 | L2H4 none_edge | 0.142 | 0.069 |
| 5000 | 10 | L1H2 L | 1.000 | 0.050 |
| 10000 | 10 | L1H2 L | 1.000 | 0.050 |
| 20000 | 16 | L1H4 L | 1.000 | 0.052 |
| 30000 | 15 | L0H1 L | 1.000 | 0.053 |
| 40000 | 15 | L1H2 L | 1.000 | 0.055 |
| 50000 | 14 | L0H0 L | 1.000 | 0.057 |
| 60000 | 16 | L0H0 L | 1.000 | 0.054 |

总结：

- 0/1k：0 strong heads；
- 5k-60k：每个 checkpoint 有 10-16 strong rows；
- stable heads 包括 L1H0-L1H4，L0H0 later emerges；
- phenomenon stable，但 exact head identity drifts。

### 5.4 Extraction frame comparison

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`
- `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json`

| Head | B1 predictor + content-only | loss-aligned AR + None-separated |
|---|---:|---:|
| L0H1 | 0.655 | 1.000 |
| L0H2 | 0.655 | 1.000 |
| L0H3 | 1.000 | 1.000 |
| L0H4 | 0.655 | 1.000 |

解释：

B1 predictor content-only 可以恢复 order axis，甚至经常是 perfect cyclic order，但 L0H1/L0H2/L0H4 会丢 anchor。Loss-aligned AR + None-separated 保留 independent None/BOS，因此能恢复正确起点。

### 5.5 Destroyed controls

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv`

总结：

- collaborator strong heads 的 destroyed mean |tau| 大约 0.05-0.067；
- real tau 是 1.000；
- real - destroyed gap 约 0.95；
- clean-base ladder destroyed means 大约 0.05-0.069；
- 安全说法是 “destroyed controls near random”，不要依赖脆弱的 hard threshold。

---

## 6. 所有相关图片

> 下面全部用 Markdown 图片语法嵌入。如果 Notion 无法直接读取本地相对路径，可以用路径作为本地 source trace。

### 6.1 Token attention — model / reveal frame

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.png`

用途：

- 展示 model/reveal frame 下的 causal attention；
- future region 为 0；
- 说明我们的模型在真实模型序列里不是 bidirectional。

Caveat:

- 这是 token-level figure，不是 block-level B65 graph。

![Token attention model frame](../collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.png)

### 6.2 Token attention — original L2R labels

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.png`

用途：

- 展示 original L2R 坐标下为什么会看起来两侧都有 signal；
- 用于回应 collaborator 关于 bidirectional-looking map 的问题。

Caveat:

- original label 下两侧有信号，不代表模型双向；
- causality 要看 reveal/model frame。

![Token attention original L2R](../collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.png)

### 6.3 L0H7 B65 None-separated heatmap

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated_heatmap.png`

用途：

- fail example；
- 说明不是所有 head 都成功；
- 可以放在 “not trivial” slide。

Caveat:

- 这是负例，不是主要正例。

![L0H7 B65 None-separated heatmap](../collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated_heatmap.png)

### 6.4 L0H7 loss-aligned A maps

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/loss_aligned_L0H7_A_maps.png`

用途：

- 展示 L0H7 在 loss-aligned extraction 下的 A maps；
- 可作为 appendix。

Caveat:

- L0H7 是 fail case。

![L0H7 loss-aligned A maps](../collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/loss_aligned_L0H7_A_maps.png)

### 6.5 L0H7 B content

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/loss_aligned_L0H7_B_content.png`

用途：

- 展示 L0H7 content-only B view；
- 可作为 appendix。

Caveat:

- fail/caveat figure。

![L0H7 B content](../collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/loss_aligned_L0H7_B_content.png)

### 6.6 Clean-base strict A/B65 L0H0

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_none65_step60000_L0H0_strict_A_B65.png`

用途：

- 当前已有的 positive strict-style heatmap；
- 可用于展示 strict A/B65 形式。

Caveat:

- 这是 clean-base L0H0，不是 collaborator L0H1-L0H4。

![Clean-base strict A/B65 L0H0](../collaborator_ckpt_b1_scan_20260616/figures/cleanbase_none65_step60000_L0H0_strict_A_B65.png)

### 6.7 B1 model-frame vs physical-frame heatmap

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_b1_image1_style_step60000_L0H0_model_vs_phys.png`

用途：

- 展示 model frame vs physical/original frame remap；
- 用来解释 coordinate-system effects。

Caveat:

- B1 diagnostic，不是 strict 65-node teacher。

![B1 model-frame vs physical-frame heatmap](../collaborator_ckpt_b1_scan_20260616/figures/cleanbase_b1_image1_style_step60000_L0H0_model_vs_phys.png)

### 6.8 Collaborator B1 L0H3 avg model-to-physical

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_avg_model_to_physical_nomask.png`

用途：

- B1 remap illustration for collaborator L0H3。

Caveat:

- B1 diagnostic，不是 strict 65-node。

![Collaborator B1 L0H3 avg model-to-physical](../collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_avg_model_to_physical_nomask.png)

### 6.9 B1 L0H2 step30000 model-vs-phys

Path:

`analyses/figures/attn_map_b1/b1_L0_H2_step30000_model_vs_phys.png`

用途：

- earlier B1 model/physical heatmap；
- 可辅助说明 B1 坐标 remap。

Caveat:

- older figure，作为 main evidence 前需要确认 exact run。

![B1 L0H2 step30000 model-vs-phys](../../analyses/figures/attn_map_b1/b1_L0_H2_step30000_model_vs_phys.png)

### 6.10 Content-anchor then None

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_L1H3_content_anchor_then_none.png`

用途：

- 解释 content anchor / None diagnostic；
- 用于说明 anchor 为什么重要。

Caveat:

- 不是 final strict protocol。

![Content-anchor then None](../collaborator_ckpt_b1_scan_20260616/figures/cleanbase_L1H3_content_anchor_then_none.png)

### 6.11 Content-only L1H3 fast model-vs-phys

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_content_only_L1H3_step5000_fast_model_vs_phys.png`

用途：

- content-only model vs physical remap；
- 展示 order-axis / anchor issue。

Caveat:

- content-only diagnostic，不是 final proof。

![Content-only L1H3 fast model-vs-phys](../collaborator_ckpt_b1_scan_20260616/figures/cleanbase_content_only_L1H3_step5000_fast_model_vs_phys.png)

### 6.12 Content-only L1H3 model-vs-phys

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_content_only_L1H3_step5000_model_vs_phys.png`

用途：

- content-only graph heatmap；
- 展示 anchor-loss 相关现象。

Caveat:

- content-only diagnostic，不是 final proof。

![Content-only L1H3 model-vs-phys](../collaborator_ckpt_b1_scan_20260616/figures/cleanbase_content_only_L1H3_step5000_model_vs_phys.png)

### 6.13 Collaborator attention maps summary

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_ckpt_attention_maps_M20.png`

用途：

- panel of collaborator attention-derived maps；
- 可作为 overview appendix。

Caveat:

- mixed protocols，使用时必须标清 panel protocol。

![Collaborator attention maps summary](../collaborator_ckpt_b1_scan_20260616/figures/collaborator_ckpt_attention_maps_M20.png)

### 6.14 B0/B1 before-after remap

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_before_after_remap.png`

用途：

- before/after remap illustration。

Caveat:

- B1/predictor remap，不是 final strict 65-node。

![B0/B1 before-after remap](../collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_before_after_remap.png)

### 6.15 B1 fixed-order axis relabel

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_fixed_order_axis_relabel.png`

用途：

- axis relabel effect。

Caveat:

- coordinate demonstration only。

![B1 fixed-order axis relabel](../collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_fixed_order_axis_relabel.png)

### 6.16 B1 pure axis relabel

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_pure_axis_relabel.png`

用途：

- pure relabel of B1 map。

Caveat:

- coordinate demonstration only。

![B1 pure axis relabel](../collaborator_ckpt_b1_scan_20260616/figures/collaborator_b1_L0H3_pure_axis_relabel.png)

### 6.17 Legacy method overview

Path:

`analyses/figures/fig1_method_overview.png`

用途：

- method overview。

Caveat:

- 主要用于 controller/training，不是 strict attention protocol 主图。

![Legacy method overview](../../analyses/figures/fig1_method_overview.png)

### 6.18 Legacy g_beta sanity bar

Path:

`analyses/figures/fig2_gbeta_sanity_bar.png`

用途：

- legacy g_beta sanity。

Caveat:

- legacy B0 controller，不是 strict 65-node teacher。

![Legacy g_beta sanity bar](../../analyses/figures/fig2_gbeta_sanity_bar.png)

### 6.19 Legacy catch-up curve

Path:

`analyses/figures/fig3_catchup_curve.png`

用途：

- training catch-up。

Caveat:

- training acceleration，不是 attention protocol 主证据。

![Legacy catch-up curve](../../analyses/figures/fig3_catchup_curve.png)

### 6.20 Legacy multistart comparison

Path:

`analyses/figures/fig4_multistart_comparison.png`

用途：

- multi-start training comparison。

Caveat:

- training acceleration，不是 attention protocol 主证据。

![Legacy multistart comparison](../../analyses/figures/fig4_multistart_comparison.png)

### 6.21 CDL tau heatmap 3-panel

Path:

`analyses/attention_diagnostic_20260609/cdl_tau_heatmap_3panel.png`

用途：

- older attention diagnostic。

Caveat:

- older protocol，不是 final strict 65-node。

![CDL tau heatmap 3-panel](../../analyses/attention_diagnostic_20260609/cdl_tau_heatmap_3panel.png)

---

## 7. 缺失但建议生成的图

### 7.1 strict 65-node graph schematic

Status: NEEDS_FIGURE

建议画法：

- 左侧一个 node0=None/BOS；
- 右侧 64 个 content nodes；
- rollout arrow 从 None 指向 first selected content block；
- 标注 “no manual None->phys0 folding”。

### 7.2 unified L0H1-L0H4 strong-pass heatmap panel

Status: NEEDS_FIGURE

已有数据：

`reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/A_with_none_lh_mean_MODEL_FRAME.npy`

建议画法：

- 4-panel B65 heatmap；
- panels: L0H1, L0H2, L0H3, L0H4；
- 每个 panel 标注 tau=1.000, first=0, phys0_rank=0。

### 7.3 gate distribution bar chart

Status: OPTIONAL

已有表格：

Strict label-free = 20 strong / 11 weak / 225 fail  
Oracle-remapped = 20 strong / 11 weak / 225 fail

---

## 8. Claim boundaries

### Can claim

1. Strict 65-node label-free block-order discovery exists in selected early heads.
2. None is an independent BOS node; first content block is selected by graph rollout.
3. L0H1-L0H4 recover full L2R under loss-aligned AR + None-separated extraction.
4. L0H7 fail shows the result is head-specific, not trivial.
5. Destroyed controls are near random relative to real strong-head tau=1.000.
6. strict label-free and oracle-remapped consistency follows permutation equivariance, not leak.
7. B1 predictor content-only provides supporting evidence that an order axis exists, but it is not the primary anchored-discovery proof.

### Cannot claim

1. All heads discover L2R.
2. B1 predictor content-only alone is sufficient for full anchored discovery.
3. Current frozen hook already uses strict 65-node teacher.
4. The training loss is true block-level; it remains token-level AR.
5. There is no head drift across runs.
6. 317M strict 65-node is verified unless evidence exists.
7. Image or multimodal extension is solved.
8. `[None]->phys0` anchored results are strict label-free results.
9. `inv_perm` is a leak; the correct statement is stage-dependent usage plus permutation equivariance.

Preferred wording:

> The token-level AR model's attention can be aggregated into a strict 65-node block graph where selected early heads recover L2R from an independent None/BOS node.

Avoid:

> The model is trained with block-level loss.

Preferred wording:

> Existing frozen hook acceleration remains legacy controller-path evidence until rerun with strict 65-node teacher.

Avoid:

> Strict 65-node teacher already caused the acceleration.

---

## 9. Boss update slide plan

### Slide 1 — Why we revisited attention extraction

- Old story: attention graph seems to recover L2R.
- Problem: B0/B1 extraction, predictor shift, and None anchor were mixed.
- Goal: separate diagnostic signal from anchored controller protocol.

### Slide 2 — Correct strict 65-node graph protocol

- node0=None/BOS；
- nodes1..64=physical blocks；
- rollout starts from None；
- first content block is selected by graph edges/readout；
- None is not manually folded into phys0。

Visual: NEEDS_FIGURE strict 65-node schematic.

### Slide 3 — Strict label-free discovery result

- Collaborator @50k L0H1-L0H4 recover full L2R；
- tau=1.000；
- first=0；
- phys0_rank=0；
- prefix@4=4；
- prefix@8=8。

Visual: result table or generated L0H1-L0H4 heatmap panel.

### Slide 4 — Not trivial: L0H7 fail + destroyed controls

- L0H7 fails: tau=0.292, first=45, phys0_rank=20；
- destroyed controls near random；
- not all heads；
- not tie-break；
- not graph-statistic artifact。

Visual:

`reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated_heatmap.png`

### Slide 5 — Extraction frame matters

- B1 predictor content-only can recover order axis but can lose anchor；
- loss-aligned AR + None-separated retains independent None/BOS and recovers start。

Visual: extraction comparison table。

### Slide 6 — Claim update

Can claim:

- selected-head strict discovery；
- None independent BOS；
- destroyed controls near random；
- strict LF vs oracle-remapped = permutation equivariance。

Cannot claim:

- all heads；
- B1 content-only sufficiency；
- strict-teacher hook acceleration；
- block-level training loss。

### Slide 7 — Next step

Close mechanism-to-controller loop:

strict 65-node teacher -> `g_beta` distillation -> hook smoke -> compare to legacy B0 hook.

---

## 10. Open questions / next actions

| Question | Why it matters | Current evidence | Suggested action |
|---|---|---|---|
| Has strict 65-node teacher been distilled to `g_beta`? | Needed to connect mechanism discovery to controller training. | No strict-teacher `g_beta` result found. | Train/distill strict-teacher `g_beta`. |
| Has strict 65-node hook been run? | Needed before claiming strict teacher acceleration. | Existing acceleration is legacy B0 controller path. | Run short hook smoke with strict teacher. |
| Can strong heads be selected by label-free audition? | Avoids oracle tau head selection. | Strong heads exist, but end-to-end label-free audition not closed. | Define audition score and test whether it selects strong-pass heads. |
| Is 317M strict protocol verified? | Needed for scale claim under final protocol. | 317M evidence found only as older B0 diagnostic. | Run strict 65-node diagnostic on 317M if scale claim is needed. |
| Is multi-seed strict discovery verified? | Needed for robustness. | Clean-base and collaborator both show selected strong heads, but head identity drifts. | Add more seeds/checkpoints under strict LF. |
| Do we need a strict 65-node schematic figure? | Boss update needs a clean visual of final protocol. | No existing schematic found. | Generate schematic. |
| Do we need unified L0H1-L0H4 heatmaps? | Strong-pass figure would make mechanism evidence easier to communicate. | A_with_none data exists, but no unified PNG found. | Generate 4-panel B65 heatmap. |

---

## 11. Source trace

Primary summary package:

- `reports/attention_summary_and_figures_20260617/`

Strict 65-node results:

- `reports/strict_65node_discovery_ckpt_verification_20260617/`
- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`
- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.json`
- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/A_with_none_lh_mean_MODEL_FRAME.npy`

B1 protocol:

- `reports/b1_attention_extraction_summary_20260615/`

Figures and heatmaps:

- `reports/collaborator_ckpt_b1_scan_20260616/figures/`
- `analyses/figures/`
- `analyses/attention_diagnostic_20260609/`

Implementation trace:

- `block_lo_arm_order_network/per_head_order_scan.py`
- `block_lo_arm_order_network/none_separated_block_graph.py`
- `scripts/search_strict_label_free_65.py`
- `scripts/search_none_separated_65_heads.py`

Relevant tests:

- `block_lo_arm_order_network/tests/test_none_separated_block_graph.py`
- `block_lo_arm_order_network/tests/test_per_head_scan_b0.py`
- `block_lo_arm_order_network/tests/test_search_raw_none_separated_65_heads.py`

