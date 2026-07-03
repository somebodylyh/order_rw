# Language Block64 Current Task For Agents

This is the current handoff for WikiText103 `seq256/permute/block64`. The
repository is language-only after cleanup. Do not route new work into removed
image data, old top-level report paths, or pre-cleanup prompt instructions.

## Main Objective

The current mainline has two linked method families:

1. **Distribution methodology work**:
   use current-frame attention to build a block affinity graph, recover an
   order axis with graph Laplacian/Fiedler decomposition, orient that axis with
   current-model train loss/profile, and test how the teacher should drive
   AO-GPT training.
2. **MLP distillation methodology**:
   compress the Laplacian/Fiedler distribution teacher into a small
   attention-conditioned MLP, then test whether the frozen MLP can safely
   replace the slower teacher inside training.

EMA is still an ablation. Treat rank/priority EMA, continuous Fiedler-priority
EMA, and no-EMA/direct-current-order variants as separate hypotheses.

## No-Prior Rule

Do not use any of the following as training targets, orientation oracle,
runtime selection criterion, early stopping, or model-selection rule:

- `OriginalL2R`
- original-frame tau or Kendall distance
- OriginalL2R/R2L PPL
- hand-written original order labels
- validation PPL as a policy-selection oracle
- historical sign alignment such as "if sign flips, reverse it"

These may be logged only as diagnostics after the current-frame order already
exists.

Allowed signals include current-model/current-sample information:

- current-frame attention matrices
- graph-Laplacian/Fiedler axis from current-frame attention affinity
- current reveal-step/full/prefix loss profiles
- current frozen or active model NLL differences
- current same-probe cross-head or layer-level consensus
- current block loss features

## Read These First

Repository orientation:

- `README.md`
- `base.md`
- `base_cn.md`
- `docs/README.md`
- `docs/structure/PROJECT_STRUCTURE_language.md`
- `docs/findings/findings_language.md`
- `Report/language/wikitext103/README.md`
- `Report/language/wikitext103/MAINLINE_AND_HISTORY.md`

Current distribution line:

- `Report/language/wikitext103/order_teacher_distribution/README.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/README.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/_index/classification.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23/results.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24/results.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28/design.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29/design.md`

Current MLP distillation line:

- `Report/language/wikitext103/mlp/README.md`
- `Report/language/wikitext103/mlp/distillation/README.md`
- `Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md`
- `Report/language/wikitext103/mlp/distillation/try_25/results.md`
- `Report/language/wikitext103/mlp/distillation/try_28/results.md`
- `Report/language/wikitext103/mlp/distillation/try_29/experiment_design.md`
- `Report/language/wikitext103/mlp/distillation/try_30/experiment_design.md`
- `Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/README.md`
- `Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/README.md`

Historical MLP joint-training, directed-ribbon, loss-design, and older
distillation attempts are under `Report/history/language/wikitext103/mlp/`.
Use them as reference only unless the user explicitly asks to revive a past
branch.

Core code:

- `train.py`
- `online_spectral_order_policy.py`
- `attn_mlp_order_policy.py`
- `scripts/train/train_pairwise_mlp_operator_distillation.py`
- `scripts/eval/eval_pairwise_operator_mlp_generalization.py`

## Current Distribution Method

The active distribution teacher is the layer-level pairwise-max Fiedler family:

```text
current-frame L0 attention from all heads
-> layer-mean matrix A
-> undirected affinity W = max(A, A.T), diagonal zeroed
-> graph Laplacian Fiedler vector v
-> raw order from sorting v
-> reverse order from sorting -v
-> choose direction by current-model linear_profile_loss
-> feed the oriented teacher into the training policy
```

Key completed evidence:

- Try 23: rank/priority EMA, seed default, final MAP original tau `+0.916667`,
  best val/loss `3.3958`, selected sign flips `0`.
- Try 24: rank/priority EMA, seed `2027` with same `permute_seed=42`, final MAP
  original tau `+0.959325`, max MAP tau `+0.994048`, best val/loss `3.3594`,
  selected sign flips `0`.

Important interpretation:

- The raw Fiedler axis may flip sign frequently.
- Current-model `linear_profile_loss` is the allowed direction chooser.
- Original-frame tau is a post-hoc diagnostic only.
- Direct-asym-eig and fixed-head top-1 are baselines, not the current default.

## EMA Ablation Status

Do not assume EMA should remain. The open comparison is:

- **Rank EMA**:
  `Fiedler vector -> hard order -> rank priority -> priority_ema`.
- **Continuous Fiedler-priority EMA**:
  `Fiedler vector -> oriented continuous [0,1] priority -> priority_ema`.
- **No EMA / direct update**:
  use the current recovered teacher/order without smoothing, or cache only the
  latest accepted order.

Relevant configs include:

```text
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k.py
config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py
config/WikiText103/seq256/permute/block64/online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k.py
```

## Current MLP Distillation Method

The clean distillation target is the same L0 layer-mean pairwise-max Fiedler
teacher:

```text
input:  current-frame L0 layer-mean attention matrix, without_none
teacher: pairwise_max_fiedler order, raw/reverse oriented by train linear_profile_loss
student: FlatAttentionOrderMLP
target: pairwise precedence / teacher-rank fit
```

Key completed evidence:

- Try 25: merged try20+try24 dataset, held-out teacher tau `0.979898`, strong
  real-input use, but missed strict `0.980` gate by about `1e-4`.
- Try 28: two-hidden-layer MLP, held-out teacher tau `0.980612`, passes the
  strict gate with fewer parameters.

Current insertion tests:

- Try 56-60: latest no-EMA online-shadow score-MLP schedule. The best retained
  baseline is try60: final val `3.3245`, best val `3.3128`.
- Try 61-64: configured trainable online-MSE refinement counterparts to
  try60/57/58/59. They keep MLP updates open through 32k, while try57-60 stop
  MLP parameter updates after the 10k-18k shadow window.

Do not conflate:

- teacher imitation success;
- frozen-MLP insertion into AO-GPT training;
- online-coupled MLP MSE refinement;
- fully differentiable end-to-end MLP policy learning through `argsort`.

## Required Reporting Style

For any formal run:

- Use the path under `Report/language/wikitext103/...`.
- Do not present smoke tests as research results.
- Record commands, configs, metrics, checkpoints, and failure analysis.
- Explicitly state which signals were used for training/selection and which
  were diagnostic-only.
- Report validation NLL/PPL, MAP/order diagnostics, sign/order stability, and
  whether EMA was rank, continuous, or disabled.
- Keep original-order diagnostics post-hoc.

If GPU placement matters, inspect the live machine state before launching. Do
not occupy a GPU that the user did not authorize.
