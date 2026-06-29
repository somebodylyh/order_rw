# AO-GPT Attention Mechanism Summary

## Strict 65-node Block-level Order Discovery

This is the consolidated Notion-ready version of the recent attention extraction / block-level order-discovery discussion. It focuses on the mechanism story, protocol cleanup, attention maps, claim boundaries, and next steps. It does not treat training acceleration as the main result.

## One-line Conclusion

我们现在应把 AO-GPT 的 attention-order discovery 表述为 strict 65-node None-separated block graph：None 是独立 BOS node，64 个 content nodes 是真实 physical blocks，rollout 从 None 出发。Selected early heads 可以恢复 L2R，但该现象 head-specific、extraction-frame-dependent；现有 frozen hook acceleration 仍是 legacy controller path，不能说已经由 strict 65-node teacher 导致。

## Why We Revisited Attention Extraction

最初我们以为 attention graph 可以直接恢复 L2R block order。这个方向没错，但旧证据里混了几个 protocol 问题：

- B0 / B1 extraction 不一致。
- B1 predictor frame 的现有 result files 实际使用 `none_mode=predictor`。
- AO-GPT 是 token-level AR next-token training，有 predictor-target shift。
- data block、query/predictor block、source/context block 不能混成同一个 block label。
- `[None] -> phys0` 可能提供 canonical start anchor，不能作为 label-free start discovery。
- content-only graph 能看到 order axis，但 anchor 可能不稳定。

所以这两天的核心不是重新证明“attention 有信号”，而是把这个信号放进一个不会偷用起点、不会混淆坐标、能解释给老板和 reviewer 的 strict protocol。

## AR Next-token Shift Clarification

AO-GPT 的训练 loss 仍然是 token-level next-token AR。Block graph 是从 token attention 聚合出来的，不是训练时直接优化的 block-level loss。

关键对齐关系：

```text
data/physical block0              = x0,x1,x2,x3
query positions predicting block0 = [None],x0,x1,x2
source/content block0             = x0,x1,x2,x3
```

因此不能用同一套 block label 同时标 query 和 key。旧式把 `[None],x0,x1,x2` 和 `x0,x1,x2,x3` 当成同一个 block 的做法，会混淆 predictor block、target block 和 content block。

结论：如果要讲 block-level discovery，None 不能并入 block0，必须作为独立 BOS node。

## Final Protocol

最终采用的主协议是：

> strict 65-node None-separated block graph

定义：

- node 0 = None / independent BOS start node
- node `1+i` = physical content block `i = x_{4i},...,x_{4i+3}`
- rollout starts from None
- first content block is selected by graph edge/readout
- None is not manually attached to physical0

这解决了 `[None]->phys0` folding 的 start-anchor 问题。它也把 model frame、original L2R、posthoc scoring 的边界拆开：图构造可以按坐标 remap，rollout 不能用 oracle order，posthoc scoring 可以翻译回 physical L2R 来评估。

## Protocol Evolution

| Protocol | None handling | Used for | Current status |
|---|---|---|---|
| B0 legacy controller extraction | `[None]` folded into physical block0 | Existing frozen `g_beta` / hook controller path | Legacy acceleration evidence; not strict discovery. |
| B1 predictor frame | Existing result files use `none_mode=predictor`; `[None]` stays in predictor frame position 0 and is averaged by reshape | Collaborator-aligned diagnostic | Strong diagnostic signal; not current hook path. |
| B1 predictor + content-only | None removed | Tests whether order axis survives without None anchor | Supporting diagnostic; can lose anchor. |
| `[None]->phys0` anchored protocol | None manually folded into phys0 | Anchored controller / older diagnostics | Not label-free start discovery. |
| loss-aligned AR extraction | None retained as separate source column in with-none variant | Basis for strict 65-node graph | Current strict mechanism protocol. |
| strict 65-node label-free graph | None independent BOS/start node | Main mechanism evidence | Current preferred discovery protocol. |
| oracle-remapped graph | None fixed, content labels remapped consistently | Equivariance check / comparison | Matches strict LF; not a leak. |
| destroyed controls | None fixed; edges shuffled or labels permuted | Tests tie-break / graph-statistic confounds | Near random; supports real structure. |

## `inv_perm` Boundary

| Stage | `inv_perm` allowed? | Reason |
|---|---:|---|
| graph construction | yes | graph labels may be remapped; edge weights unchanged |
| CDL rollout | no | rollout must rely only on edge weights |
| posthoc scoring | yes | only translate predicted sigma for evaluation |

Strict label-free and oracle-remapped consistency is not a leak. It is permutation equivariance: if graph labels are consistently permuted and rollout uses only graph edges, the recovered order translates back the same way. A leak would be using `inv_perm` inside rollout decisions.

## Key Result A: Strict 65-node Collaborator @50k

Source: `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`

| Head | Method | tau | first | phys0_rank | prefix@4 | prefix@8 | destroyed \|tau\| | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L0H1 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0498 | strong_pass |
| L0H2 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0531 | strong_pass |
| L0H3 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0501 | strong_pass |
| L0H4 | L-transition-only (`L`) | 1.000 | 0 | 0 | 4 | 8 | 0.0554 | strong_pass |
| L0H7 | C-D+L | 0.292 | 45 | 20 | 0 | 0 | 0.0825 | fail |

Interpretation:

- L0H1-L0H4 recover full L2R from independent None/BOS under strict 65-node LF.
- L0H7 fail shows this is head-specific, not a universal head property or trivial tie-break.

## Key Result B: Gate Distribution

Source: `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`

| Protocol | strong_pass | weak_pass | fail | total |
|---|---:|---:|---:|---:|
| Strict label-free | 20 | 11 | 225 | 256 |
| Oracle-remapped | 20 | 11 | 225 | 256 |

Interpretation:

- Exact match supports permutation equivariance.
- This does not mean `inv_perm` is used inside rollout.
- It means equivalent graph labelings recover equivalent orders.

## Key Result C: Clean-base 9-step Ladder

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

Summary:

- 0/1k: 0 strong heads.
- 5k-60k: 10-16 strong rows per checkpoint.
- Stable heads include L1H0-L1H4; L0H0 emerges later.
- Phenomenon is stable; exact head identity drifts.

## Key Result D: Extraction Frame Comparison

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`
- `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json`

| Head | B1 predictor + content-only | loss-aligned AR + None-separated |
|---|---:|---:|
| L0H1 | 0.655 | 1.000 |
| L0H2 | 0.655 | 1.000 |
| L0H3 | 1.000 | 1.000 |
| L0H4 | 0.655 | 1.000 |

Interpretation:

B1 predictor content-only can recover the order axis, often as a perfect cyclic order, but loses anchor for L0H1/L0H2/L0H4. Loss-aligned AR + None-separated keeps independent None/BOS and recovers the correct start.

## Key Result E: Destroyed Controls

Sources:

- `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv`

Summary:

- Collaborator strong heads have destroyed mean |tau| around 0.05-0.067 while real tau is 1.000.
- Real - destroyed gap is approximately 0.95 for strong heads.
- Clean-base ladder destroyed means are about 0.05-0.069 across steps.
- Safe claim: destroyed controls are near random. Avoid relying on a fragile exact threshold.

## Attention Maps / Figures

### 1. Token Attention — Model / Reveal Frame

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.png`

Use:

- Show model-frame causal attention.
- Future region is zero.
- Useful for explaining that our model is not bidirectional in the actual reveal/model frame.

Caveat:

- Token-level figure, not block-level B65 graph.

### 2. Token Attention — Original L2R Labels

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.png`

Use:

- Show why original L2R coordinates can look two-sided after random reveal-order remap.
- Useful for explaining collaborator's "bidirectional-looking" concern.

Caveat:

- Two-sided signal in original labels does not imply bidirectional attention. Causality is judged in reveal/model frame.

### 3. L0H7 B65 None-separated Heatmap

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated_heatmap.png`

Use:

- Fail example.
- Supports "not all heads discover L2R".

Caveat:

- This is a negative/control example, not the main positive heatmap.

### 4. Clean-base Strict A/B65 L0H0

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_none65_step60000_L0H0_strict_A_B65.png`

Use:

- Positive strict-style heatmap already available.

Caveat:

- Clean-base L0H0, not collaborator L0H1-L0H4.

### 5. B1 Model-frame vs Physical-frame Heatmap

Path:

`reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_b1_image1_style_step60000_L0H0_model_vs_phys.png`

Use:

- Shows model frame vs physical/original frame remap.
- Useful to explain coordinate-system effects.

Caveat:

- B1 diagnostic, not strict 65-node teacher.

## Figures Missing / Need Generation

1. strict 65-node graph schematic  
   Suggested content: node0=None/BOS, nodes1..64=physical blocks, rollout arrow from None, label "no manual None->phys0".

2. unified L0H1-L0H4 strong-pass heatmap panel  
   Data exists at:
   `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/A_with_none_lh_mean_MODEL_FRAME.npy`

3. gate distribution bar chart  
   Optional; table is already sufficient.

## Claim Boundaries

### Can Claim

1. Strict 65-node label-free block-order discovery exists in selected early heads.
2. None is an independent BOS node; first content block is selected by graph rollout.
3. L0H1-L0H4 recover full L2R under loss-aligned AR + None-separated extraction.
4. L0H7 fail shows the result is head-specific, not trivial.
5. Destroyed controls are near random relative to real strong-head tau=1.000.
6. strict label-free and oracle-remapped consistency follows permutation equivariance, not leak.
7. B1 predictor content-only provides supporting evidence that an order axis exists, but it is not the primary anchored-discovery proof.

### Cannot Claim

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

## Boss Update Slide Plan

### Slide 1 — Why we revisited attention extraction

Message:

- Old story: attention graph seems to recover L2R.
- Problem: B0/B1 extraction, predictor shift, and None anchor were mixed.
- Goal: separate diagnostic signal from anchored controller protocol.

### Slide 2 — Correct strict 65-node graph protocol

Message:

- node0=None/BOS, nodes1..64=physical blocks.
- Rollout starts from None.
- First content block is selected by graph edges/readout.
- None is not manually folded into phys0.

Visual:

- NEEDS_FIGURE: strict 65-node schematic.

### Slide 3 — Strict label-free discovery result

Message:

- Collaborator @50k L0H1-L0H4 recover full L2R.
- tau=1.000, first=0, phys0_rank=0, prefix@4=4, prefix@8=8.

Visual:

- Use table above.
- Optional generated L0H1-L0H4 heatmap panel.

### Slide 4 — Not trivial: L0H7 fail + destroyed controls

Message:

- L0H7 fails: tau=0.292, first=45, phys0_rank=20.
- Destroyed controls near random.
- Not all heads, not tie-break, not graph-statistic artifact.

Visual:

- `B65_none_separated_heatmap.png`
- destroyed control summary.

### Slide 5 — Extraction frame matters

Message:

- B1 predictor content-only can recover order axis but can lose anchor.
- Loss-aligned AR + None-separated retains independent None/BOS and recovers start.

Visual:

- Extraction comparison table.
- Optional B1 model-vs-physical heatmap.

### Slide 6 — Claim update

Message:

- Can claim selected-head strict discovery.
- Cannot claim all heads, B1 content-only sufficiency, or strict-teacher hook acceleration.

### Slide 7 — Next step

Message:

- Close mechanism-to-controller loop:
  strict 65-node teacher -> `g_beta` distillation -> hook smoke -> compare to legacy B0 hook.

## Open Questions / Next Actions

| Question | Why it matters | Current evidence | Suggested action |
|---|---|---|---|
| Has strict 65-node teacher been distilled to `g_beta`? | Needed to connect mechanism discovery to controller training. | No strict-teacher `g_beta` result found. | Train/distill strict-teacher `g_beta`. |
| Has strict 65-node hook been run? | Needed before claiming strict teacher acceleration. | Existing acceleration is legacy B0 controller path. | Run short hook smoke with strict teacher. |
| Can strong heads be selected by label-free audition? | Avoids oracle tau head selection. | Strong heads exist, but end-to-end label-free audition not closed. | Define audition score and test whether it selects strong-pass heads. |
| Is 317M strict protocol verified? | Needed for scale claim under final protocol. | 317M evidence found only as older B0 diagnostic. | Run strict 65-node diagnostic on 317M if scale claim is needed. |
| Is multi-seed strict discovery verified? | Needed for robustness. | Clean-base and collaborator both show selected strong heads, but head identity drifts. | Add more seeds/checkpoints under strict LF. |
| Do we need a strict 65-node schematic figure? | Boss update needs a clean visual of final protocol. | No existing schematic found. | Generate schematic. |
| Do we need unified L0H1-L0H4 heatmaps? | Strong-pass figure would make mechanism evidence easier to communicate. | A_with_none data exists, but no unified PNG found. | Generate 4-panel B65 heatmap. |

## Source Trace

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

Implementation trace:

- `block_lo_arm_order_network/per_head_order_scan.py`
- `block_lo_arm_order_network/none_separated_block_graph.py`
- `scripts/search_strict_label_free_65.py`
- `scripts/search_none_separated_65_heads.py`

Relevant tests:

- `block_lo_arm_order_network/tests/test_none_separated_block_graph.py`
- `block_lo_arm_order_network/tests/test_per_head_scan_b0.py`
- `block_lo_arm_order_network/tests/test_search_raw_none_separated_65_heads.py`

