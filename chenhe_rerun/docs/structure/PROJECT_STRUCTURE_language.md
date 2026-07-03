# Language Project Structure

This file maps the language / WikiText103 side of the project. The repository
has been cleaned to keep this language mainline as the active working tree.

## Entry Documents

- `docs/prompts/Prompt_language.md`
  Language-specific new-agent prompt.
- `docs/findings/findings_language.md`
  Language-specific results and interpretation.
- `base.md`, `base_cn.md`
  Shared conceptual handoff. These remain mixed because they explain the whole project.
- `docs/structure/PROJECT_STRUCTURE.md`
  Lightweight router to the direction-specific structure files.
- `README.md`
  Human-facing overview.
- `RUN_COMMANDS.md`
  Shared command list. Use only the WikiText103 sections for language tasks.

## Core Shared Code Used By Language Experiments

- `train.py`
  Main training loop: config overrides, data permutation, checkpointing, evaluation, random/AR/explicit-order AO-GPT training, segment-guided orders.
- `AOGPT.py`
  Dispatcher that selects block-level or token-level AO-GPT.
- `AOGPT_block.py`
  Block-order AO-GPT. This is the active language path for block32/64/128.
- `AOGPT_token.py`
  Token-order AO-GPT. Mostly diagnostic now.
- `order_utils.py`
  Shared order helpers: block expansion, permutation metadata, Kendall metrics, local diagnostics.
- `path_layout.py`
  Output path helpers.

## Language Data

- `data/wikitext103/prepare.py`
  Dataset preparation script.
- `data/wikitext103/train.bin`
  Tokenized training split.
- `data/wikitext103/val.bin`
  Tokenized validation split.

The code expects `np.uint16` memmap files named `train.bin` and `val.bin`.

## Language Configs

Main tree:

```text
config/WikiText103/
  seq80/
  seq128/
  seq256/
  seq512/
```

Important axes:

- frame: `non_permute`, `permute`
- granularity: `block1`, `block16`, `block32`, `block64`, `block128`
- presets: `random.py`, `ar.py`, `segment_curriculum*.py`

Active evidence configs:

- `config/WikiText103/seq256/permute/block64/segment_curriculum_early_stop_open_level.py`
- `config/WikiText103/seq256/permute/block32/segment_curriculum_early_stop_general.py`
- `config/WikiText103/seq256/permute/block128/segment_curriculum_early_stop_general.py`
- `config/WikiText103/seq80/permute/block1/segment_curriculum_early_stop_open_level.py`

Historical online fixed-head / attention-spectral configs:

- `config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank_late_l2r_8000_50000.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_late_l2r_8000_50000.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_fast_b128.py`
- `config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_fast_b256.py`
- `config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_cached_order.py`
- `config/WikiText103/seq80/non_permute/block1/random_save_attn_ckpts_1k2k5k8k10k15k20k25k50k.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_fixed_head_order_loss_rerank_warmup5k_anneal15k_update20_top96_prefix16_no_freeze.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution_loss_rerank_top4_start8k_update20_top96_prefix16.py`
- `config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution_loss_rerank_top4_warmup15k_dist15k40k_freeze10k_update20_top96_prefix16.py`
- `config/WikiText103/seq80/non_permute/block1/segment_curriculum_attention_spectral_crossaxis_svd.py`

The list above is historical online/fixed-head coverage. The current
`seq256/permute/block64` distribution mainline uses:

- `config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py`
- `config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py`
- `config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k.py`
- `config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py`
- `config/WikiText103/seq256/permute/block64/online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k.py`

Current MLP distillation / frozen insertion configs:

- `config/WikiText103/seq256/permute/block64/attn_mlp_try29_seed2027_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_noema_update1_warmup10k_anneal35k_freeze35k.py`
- `config/WikiText103/seq256/permute/block64/attn_mlp_try30_seed2028_fromscratch_frozen_try28_l0_layermean_fiedler_mlp_noema_update1_warmup10k_anneal35k_freeze35k.py`

RoPE ablations:

- `config/WikiText103/seq256/permute/block64/random_rope.py`
- `config/WikiText103/seq256/permute/block64/ar_rope.py`
- `config/WikiText103/seq256/permute/block64/segment_curriculum_early_stop_open_level_rope.py`
- `config/WikiText103/seq256/non_permute/block64/ar_rope.py`

Historical comparison:

- `config/WikiText103/seq256/permute/block64/segment_curriculum.py`
- `config/WikiText103/seq256/permute/block64/segment_curriculum_strict_front_margin.py`

Token-micro configs may exist, but the token-micro path is retired from the active workflow.

## Language Runner And Helper Scripts

- `scripts/runner/hierarchical_segment_curriculum_runner.py`
  Main curriculum runner. Performs warmup training, frozen-checkpoint mining, segment-guided resume training, and stage repetition.
- `scripts/analysis/`
  Language analysis and diagnostic helpers.
- `scripts/eval/`
  Language evaluation helpers.
- `scripts/train/`
  Language distillation and policy-training helpers.
- `online_spectral_order_policy.py`
  Online order policy utilities. Current mainline usage is layer-mean
  pairwise-max Fiedler distribution and EMA/no-EMA ablation; older fixed-head
  top-1 and direct-asym-eig modes remain as baselines.
- `attn_mlp_order_policy.py`
  Frozen/learnable MLP order-policy utilities. Current usage is to distill the
  L0 layer-mean Fiedler teacher and test frozen-MLP insertion.

## Language Analysis Scripts

- `scripts/analysis/pair_margin_stability_probe.py`
- `scripts/analysis/tv_weight_pair_sweep.py`
- `scripts/analysis/visualize_pair_score_heatmaps.py`
- `scripts/analysis/original_span_conditional_gain_probe.py`
- `scripts/analysis/original_span_permutation_loss_probe.py`
- `scripts/analysis/token_group_consistency_probe.py`
- `scripts/eval/eval_ckpt_modes.py`
- `scripts/eval/block_permutation_sanity_check.py`
- `scripts/eval/permuted_ckpt_generate_step_eval.py`
- `scripts/eval/eval_prefix_pairs.py`
- `scripts/eval/eval_lm_original_order_ppl.py`
  Language PPL eval for explicit checkpoint orders, including online spectral
  fixed-head and distribution checkpoint modes.
- `scripts/eval/eval_lm_ppl_sweep.py`
  Multi-checkpoint/multi-mode PPL sweep. Use this for Random/AR/OriginalL2R,
  `CheckpointOnlineSpectralOrder`, and distribution MAP/sampled/mix-sampled
  evaluation.

## Language Report Directories

Current-vs-history router:

- `Report/language/wikitext103/MAINLINE_AND_HISTORY.md`
  Read this before browsing old report products.

Current distribution / MLP mainline:

- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md`
  Compact current method summary for the Laplacian/Fiedler teacher and
  EMA/no-EMA status.
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23`
  First completed L0 layer-mean pairwise-max Fiedler rank-EMA run.
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24`
  Seed ablation that strengthens the rank-EMA Fiedler result.
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28`
  Continuous Fiedler-priority EMA design.
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29`
  Seed ablation for continuous Fiedler-priority EMA.
- `Report/language/wikitext103/mlp/distillation/try_25`
  Merged Fiedler-teacher distillation dataset and three-hidden-layer MLP.
- `Report/language/wikitext103/mlp/distillation/try_28`
  Smaller two-hidden-layer MLP that passes the strict held-out teacher-tau gate.
- `Report/language/wikitext103/mlp/distillation/try_29`
  Frozen try28 MLP insertion, no attention/logits EMA, update every step.
- `Report/language/wikitext103/mlp/distillation/try_30`
  Seed variant of the frozen try28 MLP no-EMA insertion.

Historical products:

- `Report/history/language/wikitext103/`
  Reference-only tree for old head-information artifacts, analysis,
  curriculum/eval/recovery outputs, direct-asym-eig and fixed-head baselines,
  joint-training/loss-design MLP attempts, and older distillation tries.

Do not route new work into history unless the user explicitly asks to revisit
one of those branches.

## Runtime Outputs

- `out/base/nonpermute/seq*/...`
- `out/base/permute/seq*/...`
- `out/curriculum/nonpermute/seq*/...`
- `out/curriculum/permute/seq*/...`

These are experiment artifacts, not source. Do not move them while doing documentation cleanup.

## Language Caveats

- `permute_data=True` uses current-frame units for training; `*_original` is report-only.
- `OriginalL2R` under `permute_data=True` is an oracle mapped through
  checkpoint permutation metadata. It is not a no-prior learned-order result.
- `non_permute` has absolute-position prior.
- `block_order_block_len > 1` has block-internal fixed-order prior.
- Token/block1 results show locality but weak direction.
- Online fixed-head cached-order training keeps changing the target order unless
  explicitly frozen; report order stability when using it as evidence.
- Token-micro is diagnostic only.
- Strongest claim: recoverable local block-level L2R structure, not complete global L2R induction.
