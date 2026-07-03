# Language Findings

本文件只记录 WikiText103 / language direction 的结果、解释边界和 open questions。当前工作树已经清理为语言主线；被删除的非语言/图像方向只从 Git 历史或备份分支追溯。

## Research Question

Random-order AO-GPT checkpoint 是否在 loss / attention / early reveal behavior 中暴露出可恢复的文本局部顺序信号？在 `permute_data=True` 下，这些信号是否更接近 original-frame L2R block order，而不是 current-frame L2R？

## Safe Claim

Random-order AO-GPT checkpoint 中存在可挖掘的局部结构信号。block-level setting 中，这些信号能较稳定恢复 original L2R 方向；token/block1 setting 中，信号强烈体现局部邻域关系但方向性弱。

不要写成 AO-GPT 从 arbitrary order 中学出了完整 global L2R order。

对于 `permute_data=True` 的 PPL 或 loss 评估，不要把 `OriginalL2R` 当作无先验结果。
`OriginalL2R` 是用 checkpoint 的 permutation metadata 把真实原始文本顺序映射回 current
frame 的 oracle upper bound。无先验主比较应使用 current-frame AR、sampled Random、
checkpoint learned order、或 recovery/curriculum 自身产生的 order。

## Current Mainline Snapshot

当前主线已经从 broad recovery/fixed-head online 诊断收敛到两条方法论：

1. **Distribution / Laplacian graph decomposition**：
   L0 layer-mean attention -> `W=max(A,A.T)` -> graph Laplacian Fiedler axis ->
   current-model `linear_profile_loss` raw/reverse orientation -> training
   order policy。
2. **MLP distillation**：
   用当前 Fiedler teacher 构造 pairwise/rank supervision，训练
   `FlatAttentionOrderMLP`，再测试 frozen MLP 是否能替代慢 teacher。

EMA 仍是 open ablation。rank/priority EMA、continuous Fiedler-priority EMA、以及
no-EMA/direct-current-order 需要分开报告。不能把“保留 EMA”写成已经证明的结论。

Current distribution evidence:

- Try 23: L0 layer-mean pairwise-max Fiedler rank EMA, final MAP original tau
  `+0.916667`, best val/loss `3.3958`, selected sign flips `0`.
- Try 24: same method with seed `2027`, final MAP original tau `+0.959325`,
  max MAP tau `+0.994048`, best val/loss `3.3594`, selected sign flips `0`.
- Try 28/29: continuous-priority EMA variants are configured/pending in this
  documentation snapshot.

Current MLP evidence:

- Try 25: MLP fits the merged Fiedler teacher dataset with held-out teacher tau
  `0.979898`, just below the strict `0.980` gate.
- Try 28: smaller two-hidden-layer MLP reaches held-out teacher tau `0.980612`
  and passes the strict gate.
- Try 56-60: latest online-shadow score-MLP schedule with no attention/logits
  EMA. The best retained baseline is try60, final val `3.3245`, best val
  `3.3128`.
- Try 61-64: matched follow-ups to try60/57/58/59. They keep the MLP
  parameters open through 32k and continue teacher-score MSE during the 18k-32k
  policy-refresh phase; they are not yet completed results.

## Historical Main Result

当前最强语言证据：

- config: `config/WikiText103/seq256/permute/block64/segment_curriculum_early_stop_open_level.py`
- report: `Report/curriculum/permute/seq256/block64/hierarchical_block64_early_stop_open_level_gain_only-6-stage`
- budget: `warmup_iters=8000`, `stage_iters=7000`, `num_curriculum_stages=6`
- `benchmark_num_levels=64`
- `pair_mining_batches=96`
- `attn_num_batches=64`, `attn_batch_size=64`
- margin schedule: `0.06,0.055,0.05,0.045,0.045,0.045`
- aggregation top-k: `128,96,48,16,6,2`
- drop weights: `0.20,0.18,0.15,0.10,0.05,0.00`
- gain gate: `early_stop_gain_threshold=0.01`, `early_stop_gain_positive_rate=0.70`

Final original-frame diagnostic:

```text
[4..12, 0..3, 13..27]
[28..63]
```

Final within-unit adjacency 为 `+1=60/62`，concat Kendall tau 为 `0.964`。这是强 block-level evidence，但 `[0..3]` displacement 仍然存在，因此只能主张 local order-structure recovery。

## Latest PPL Frame Sanity Check

最新 full-val PPL sweep：

- report: `Report/eval/wikitext103_random_b64_permute_block64_order_ppl_fullval`
- split: `val`
- sample mode: sequential full validation chunks
- main question: order 是否来自模型/训练过程本身，还是来自 hidden original L2R oracle

核心结果：

| checkpoint | eval mode | PPL | no-prior interpretation |
| --- | ---: | ---: | --- |
| `seq256-random-b64-permute-block` | `AR` current permuted | `37.96` | valid no-prior baseline in current frame |
| `seq256-random-b64-permute-block` | `Random` | `38.04` | valid random-order baseline |
| `seq256-random-b64-permute-block` | `OriginalL2R` | `31.61` | oracle, not no-prior |
| `seq256-online-spectral-fixed-head-b64-permute-block` | `AR` current permuted | `38.57` | valid current-frame baseline |
| `seq256-online-spectral-fixed-head-b64-permute-block` | `Random` | `38.57` | valid random-order baseline |
| `seq256-online-spectral-fixed-head-b64-permute-block` | `CheckpointOnlineSpectralOrder` | `37.24` | valid learned-order evaluation |
| `seq256-online-spectral-fixed-head-b64-permute-block` | `OriginalL2R` | `29.71` | oracle, not no-prior |

Order diagnostics:

- final permuted online cached order vs current-frame L2R: Kendall tau `0.092`
- final permuted online cached order vs `OriginalL2R` mapped order: Kendall tau `-0.667`
- permuted online cached order from 21k checkpoint to 50k checkpoint: Kendall tau `0.689`
- non-permute online cached order from 25k checkpoint to 50k checkpoint: Kendall tau `-0.296`

Interpretation:

- `OriginalL2R` PPL is low because evaluation injects true original text order.
- `CheckpointOnlineSpectralOrder` improves over Random/current-frame AR for the
  permuted online run (`37.24` vs about `38.57`), but the gain is modest and far
  from the oracle.
- The learned online order is not aligned with hidden original L2R and can drift
  during training. This supports separating order discovery from order training:
  explore/recover an order, freeze it, then train/evaluate that fixed order.

## Historical Online Attention-Spectral Method

This section is retained as historical context for why the project moved toward
the layer-mean Fiedler distribution teacher. It is not the current default
entry point. Related report products now live under
`Report/history/language/wikitext103/`.

At the time of these notes, the language thread was online order recovery
during training. The current cleaned report mainline is the
Distribution/Fiedler teacher plus MLP distillation path described in
`Report/language/wikitext103/MAINLINE_AND_HISTORY.md`.

Core configs:

- `config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank_late_l2r_8000_50000.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_late_l2r_8000_50000.py`
- `config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_cached_order.py`
- `config/WikiText103/seq80/non_permute/block1/random_save_attn_ckpts_1k2k5k8k10k15k20k25k50k.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_fixed_head_order_loss_rerank_warmup5k_anneal15k_update20_top96_prefix16_no_freeze.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution_loss_rerank_top4_start8k_update20_top96_prefix16.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution_loss_rerank_top4_warmup15k_dist15k40k_freeze10k_update20_top96_prefix16.py`

All policy decisions use current-frame ids. `*_original` and original-frame
Kendall diagnostics are report-only. In non-permute text, current frame equals
natural text frame, but the method should still be described as using model-side
attention/loss signals rather than tau or original-frame oracle selection.

### Seq80/Block1 Current Diagnostics

The user is currently diagnosing why seq80 online attention heatmaps look noisy.
The immediate control is a clean random non-permute run with milestone
checkpoints:

```text
config: config/WikiText103/seq80/non_permute/block1/random_save_attn_ckpts_1k2k5k8k10k15k20k25k50k.py
out:    out/base/nonpermute/seq80/block1/out-wikitext103-seq80-random-b1-nonpermute-save-attn-ckpts-50000-iters
final:  step 50000, train loss 3.8768, val loss 3.9655
ckpts:  1k, 2k, 5k, 8k, 10k, 15k, 20k, 25k, 50k
```

This run is intentionally not a new hyperparameter setting. It matches the
original seq80/block1 random baseline (`batch_size=256`,
`gradient_accumulation_steps=2`, `block_size=80`, `n_layer=4`, `n_head=8`,
`n_embd=384`, `aogpt_train_mode='Random'`) and only adds
`save_iter_checkpoints`. Its purpose is to answer whether attention maps become
cleaner during ordinary random-order AO-GPT training.

Completed diagnostic reports:

```text
random-probe report: Report/history/language/wikitext103/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_online_order_warmup_diagnostic
locality report:     Report/history/language/wikitext103/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_random_probe_remap_locality_diagnostic
official check:      Report/history/language/wikitext103/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_random_probe_official_export/ckpt_050000_random_2048
AR-probe report:     Report/history/language/wikitext103/analysis/attn/nonpermute/seq80/block1/random_baseline_ckpt_ar_probe_attention_diagnostic
scripts:
  scripts/analysis/seq80_random_ckpt_online_order_attn_batch.py
  scripts/analysis/seq80_loss_rerank_existing_attention_candidates.py
```

Important correction: the `random-probe` report uses Random reveal orders and
then remaps the averaged matrix back to current L2R coordinates. For the actual
current policy, the relevant view is fixed L0H7 `with_none`. Under this view the
attention map is visually messy, dominated by column/global structure rather
than a local diagonal band, and its near/far ratio stays around `1.00`.

The corrected locality report also shows a separate sanity check:
all-layer/all-head `without_none` attention does become local after Random probes
are remapped to current L2R. Its near/far block-attention ratio rises from
`1.06` at 1k to `2.04` at 50k. The official export script reproduces this at 50k
with 2048 Random-probe samples (`near/far=2.05`). This sanity check shows that
the model contains local structure somewhere, but it does not validate the
current single-head `with_none` policy path.

The part that remains near zero is the fixed-head order recovery used in the
online policy: under this Random-probe view, fixed L0H7 spectral top-1 tau is
near zero (`[-0.094, 0.057]`), and fast top96/prefix16 loss-rerank tau is also
near zero (`[-0.057, 0.094]`).

The AR-probe report is the comparable view for earlier AR-style heatmaps. It
shows a strong causal/diagonal attention pattern and high positive or negative
recovered tau:

| step | AR-probe tau | adjacent rate | note |
| ---: | ---: | ---: | --- |
| 1k | `0.254` | `0.127` | weak but nonzero axis |
| 2k | `0.461` | `0.354` | local AR-like chunks appear |
| 5k | `-0.643` | `0.291` | strong reverse-oriented axis |
| 8k | `-0.280` | `0.456` | very local, reversed/fragmented |
| 10k | `0.597` | `0.278` | strong positive axis |
| 15k | `-0.651` | `0.329` | strong reverse axis |
| 20k | `-0.420` | `0.241` | reverse-oriented chunks |
| 25k | `-0.481` | `0.215` | reverse-oriented chunks |
| 50k | `0.431` | `0.392` | strong local AR-like structure |

Interpretation for warmup: the issue is not absence of order signal in random
checkpoints. Even by 2k/5k, AR-probe attention can expose a strong local order
axis. The hard part is that the direction flips and depends on probe policy:
Random-probe all-head remap is local but policy-relevant fixed-head `with_none`
is messy and its top-1 order is near zero. AR-probe can be strongly positive or
negative, and the online run probes with the active training policy rather than
forced Random. So simply lengthening pure-random warmup is not the clean lever;
future seq80 tuning should explicitly control probe mode, aggregation, and
current-frame direction-breaking.

The active follow-up is a seq80 fixed-head loss-rerank run with no final freeze:

```text
config: config/WikiText103/seq80/non_permute/block1/online_spectral_fixed_head_order_loss_rerank_warmup5k_anneal15k_update20_top96_prefix16_no_freeze.py
out:    out/base/nonpermute/seq80/block1/out-wikitext103-seq80-online-spectral-fixed-head-loss-rerank-warmup5k-anneal15k-update20-top96-prefix16-kall-no-freeze-b1-nonpermute-50000-iters
schedule: 0..5k random, 5k..15k anneal, 15k..50k keep updating every 20 steps
```

At the 2026-06-07 documentation snapshot, this fixed-head run was in progress
past 33k steps. The latest checkpoint snapshot was around `iter_num=33250`,
`best_val_loss=4.0702`, with more than 1400 online policy updates. The
`online_spectral_policy_order_history.jsonl` file showed that the first order
at iter 5000 was highly jumpy, while later orders had more local chunks but
were still not stable global L2R. This is evidence for "attention/order signal
is changing and partly local," not yet evidence that online fixed-head has
solved seq80.

Updated live order-history read at iter `37240`: selected cached-order tau over
the 1-card run spans approximately `[-0.716, 0.740]`, with latest selected tau
around `-0.409`. This confirms the live online json is high-positive/high-negative
rather than near-zero. The discrepancy with the random-probe report comes from
probe policy: online attention collection uses `_forward_with_active_training_policy`,
so after annealing it probes with the current cached order, not forced Random.

### Fixed-Head Top-1 Online Order

`aogpt_train_mode='OnlineSpectralFixedHeadOrder'` maintains one cached hard
order. Each policy refresh:

1. collects no-grad train-probe attention from fixed layer `0`, head `7`;
2. aligns attention back to the current block frame;
3. averages the probe matrices;
4. runs spectral candidate enumeration over that fixed-head matrix;
5. scores candidates with attention-side adjacency path, directed path, and
   band-quality terms;
6. caches the highest-scoring order for subsequent training samples.

For seq256/block64, `block_order_block_len=4`, so `num_blocks=64`. Typical
probe budget is `64 x 64 = 4096` sequences per refresh. The basic online
configs anneal from random fallback to the cached order over the warmup window.

### Current-Frame Loss Reranking

Loss reranking is an optional candidate reranker layered on top of the same
attention-spectral enumeration. It is allowed because it uses only
current-frame train/probe teacher-forced losses from the current model. It does
not use original L2R, tau, grid priors, validation PPL, or human-designed scan
orders.

For fixed-head loss-rerank block64:

```text
recover top-64 attention-spectral candidates
evaluate each candidate on current-frame train probe batches
score = 1.0 * z(attention_spectral_score)
      - 0.25 * z(prefix_loss)
      - 0.15 * z(full_loss)
select top-1 as cached_order
```

The default non-permute block64 loss-rerank config uses:

```text
online_spectral_policy_update_every = 10
attention probe = 64 x 64 = 4096 sequences
loss rerank    = 16 x 64 = 1024 sequences per candidate
loss_rerank_top_k = 64
prefix_k = 8 blocks
```

For distribution loss-rerank, the candidate data budget is larger:

```text
attention probe = 64 x 64 = 4096 sequences
loss rerank    = 32 x 64 = 2048 sequences per candidate
loss_rerank_top_k = 64
final priority update uses reranked top-32
```

### Top-32 Priority Distribution

`aogpt_train_mode='OnlineSpectralOrderDistribution'` replaces a single cached
top-1 order with a soft priority vector. It does not permanently maintain 32
orders. The top candidates are used only to update a length-`num_blocks`
priority vector:

```text
attention matrix A_t
-> spectral candidates
-> top-M candidates, M = 32 after optional loss rerank
-> r_update
-> priority_ema
-> sampled hard orders by Gumbel-top-k
```

For candidate order `m`, block `i` receives:

```text
v_m(i) = 1 - pos_m(i) / (num_blocks - 1)
```

Early reveal positions therefore have larger priority. Candidate scores are
z-scored and converted to weights:

```text
w_m = softmax(z(score_m) / teacher_temperature)
r_update(i) = sum_m w_m * v_m(i)
priority_ema = 0.95 * priority_ema + 0.05 * r_update
```

Training then samples hard orders:

```text
with probability online_spectral_policy_random_mix_prob:
    uniform random order
else:
    Gumbel-top-k(priority_ema / sample_temperature)
```

Default distribution parameters:

```text
top_m = 32
teacher_temperature = 1.0
score_normalization = zscore
priority_ema_decay = 0.95
sample_temperature = 0.7
random_mix_prob = 0.2
distribution_per_sample = True
```

### Late-L2R Curriculum Ablation

The late-L2R configs explicitly add an AR/L2R prior as an ablation, so they
must not be reported as no-prior learned-order evidence. They test whether
online spectral order can be useful as an early curriculum before gradually
finetuning toward L2R.

The block64 late-L2R schedule is:

```text
0..8000:
  random fallback -> online spectral policy

8000..50000:
  online spectral policy -> L2R
```

At each sample, the late-L2R probability increases linearly from `0` to `1`.
For distribution late-L2R, the non-L2R branch still samples from
`priority_ema` with its configured random mix.

Current caveat: continually refreshing either a cached order or a priority
distribution can make order policy and backbone co-adapt. Report update counts,
order stability, priority entropy, sampled-order diversity, and PPL under
Random/AR/OriginalL2R/checkpoint-policy eval modes. The planned next step is to
turn the attention-to-order recovery into an MLP policy after the online teacher
shows useful PPL/stability behavior.

## Comparison Runs

- `Report/curriculum/permute/seq256/block32/hierarchical_block32_early_stop_gain_only-6-stage`
  - strong local recovery。
  - final `+1=28/31`，concat tau `0.871`。
  - prefix chunk error：`[3..12], [1..2], [0], [13..31]`。
- `Report/curriculum/permute/seq256/block128/hierarchical_block128_early_stop_gain_only-6-stage`
  - negative / diagnostic over-aggregation result。
  - many local runs, poor chunk order。
  - final tau `0.469`。
- `Report/curriculum/permute/seq80/block1/hierarchical_block1_early_stop_open_level_gain_only-6-stage`
  - negative token-level result。
  - local adjacency exists, but direction is mixed / reverse-biased。
  - final concat tau `-0.594`。
- `Report/curriculum/permute/seq256/block64/hierarchical_block64_early_stop_open_level_gain_only_rope-6-stage`
  - RoPE ablation。
  - stage 1 still shows strong forward local pair evidence: top-50 original-frame pairs are `+1=46`, `-1=4`, `other=0`。
  - final `+1=50/63`，concat tau `0.692`。
  - interpretation: local directed-pair signal remains, but current open-level recipe over-aggregates / misorders chunks more easily。

## Historical Noise-Amplification Baseline

- config: `config/WikiText103/seq256/permute/block64/segment_curriculum.py`
- report: `Report/curriculum/permute/seq256/block64/hierarchical_block64_margin_aligned_eval-6-stage`

This older run is important because it shows that noise appears early and is amplified later. Stage 2 already has small noisy seeds such as `[7,4,8]`, `[32,30,33]`, and `[5,2,3,0]` in original-frame diagnostics. Stage 4 amplifies a large noisy unit:

```text
[31,34,35,9,10,11,12,5,2,3,0,6,1]
```

This supports the current gain-only / early-stop direction: the method needs conservative model-side gates, not just more aggregation.

## Pair-Score Findings

The main pair-score scripts are:

- `scripts/analysis/pair_margin_stability_probe.py`
- `scripts/analysis/tv_weight_pair_sweep.py`
- `scripts/analysis/visualize_pair_score_heatmaps.py`

Relevant outputs:

- `Report/analysis/pair_margin_stability/permute_seq256_block64_base`
- `Report/analysis/pair_margin_stability/permute_seq256_block128_base`
- `Report/analysis/pair_margin_stability/permute_seq256_block1_base_cap1024`
- `Report/analysis/tv_weight_sweep_seq256_fuller`
- `Report/analysis/signed_drop_sweep_seq256_fuller`
- `Report/analysis/signed_drop_sweep_seq80`

At `drop_weight=0.2`, seq256 permuted block64 is the clean regime:

- `i -> i+1` mean score: `-4.4923`
- `i -> i-1` mean score: `-4.6615`
- adjacent L2R margin mean: `0.1692`
- top-50 and top-100 score cells have adjacent share `1.000`

Token/block1 is different:

- seq256 token/block1 adjacent L2R margin mean: `-0.0379`
- seq80 token/block1 adjacent L2R margin mean: `-0.0054`
- top score cells are still local adjacent, but direction is weak or slightly reverse-biased。

Interpretation: block-level pair scores expose both local membership and stable direction; token-level pair scores mostly expose local membership without stable direction.

## Why Token-Micro Is Retired

`token_micro_segments` was introduced because token pair scores were local but weakly directional. It built undirected affinity graphs, mutual top-k neighborhoods, component filters, and internal permutation search. This is now retired because:

- the main claim can be supported more cleanly at block level;
- token-micro adds method complexity;
- it shifts the story from directed pair recovery to undirected locality proposal plus verification;
- token-level final collapse is not clean evidence of L2R recovery.

Historical token-micro reports can be cited only as diagnostic evidence that token-level locality is present but direction is weak.

## Caveats

- `non_permute` has absolute-position prior。
- `block_order_block_len > 1` has block-internal fixed-order prior。
- `permute_data=True` breaks monotonic current/original alignment, but the model still sees current-frame absolute positions。
- `*_original` fields must remain analysis-only。
- Mining and final evaluation on the same split can create model-selection leakage。
- Curriculum success does not prove the base random checkpoint independently learned a complete order。
- final unit collapse is not enough; report original-frame Kendall, adjacent/gap distribution, loss baselines, and forward/reverse symmetry。

## Open Questions

- Can a stricter model-side gate reduce the remaining `[0..3]` displacement in block64 without hurting local adjacency?
- Which margin/gain/stability statistic best predicts late-stage over-aggregation?
- Is there a clean held-out validation protocol for pair mining that avoids model-selection leakage?
- Can token-level results be summarized as locality evidence without distracting from the block-level main claim?
