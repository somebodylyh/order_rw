# nanoGPT Learned Order

This repository is now maintained as a language-focused AO-GPT learned-order
research fork on top of nanoGPT.

The non-language experiment tree, datasets, checkpoints, external dependencies,
and obsolete documentation were removed after being backed up on the remote
branch `language-graph-laplacian-decomposition-20260624`.

## Active Scope

The active research scope is WikiText103 language order recovery:

- train AO-GPT style models under `AR`, `Random`, and explicit order policies
- mine current-frame attention and loss signals from random-order backbones
- recover local block-level left-to-right structure under permutation settings
- develop the distribution methodology based on Laplacian graph decomposition
  of attention-derived block affinity
- distill the validated distribution teacher into small Attn-MLP order policies

The mainline is language-only. Do not route new work into deleted image
datasets, image configs, image reports, or image benchmark scripts.

## Start Here

- `docs/README.md`
  Documentation router.
- `docs/prompts/Prompt.md`
  Top-level agent prompt router.
- `docs/prompts/Prompt_language.md`
  WikiText103 language prompt.
- `docs/prompts/prompt_language_block_current_task.md`
  Current seq256/block64 Fiedler distribution + MLP distillation handoff.
- `docs/structure/PROJECT_STRUCTURE_language.md`
  Language-side layout and active paths.
- `docs/findings/findings_language.md`
  Current language conclusions and interpretation boundaries.
- `Report/language/wikitext103/`
  Language reports and generated artifacts.

## Important Boundaries

- `permute_data=True` means training happens in a fixed current frame.
- Checkpoints save `block_perm` and `inverse_block_perm`.
- Training and curriculum use current-frame units.
- `*_original` fields are diagnostics only.
- In permuted WikiText103 evaluations, `OriginalL2R` is an oracle upper bound,
  not no-prior learned-order evidence.
- Token/block1 and token-micro paths are diagnostics, not the active target.
- The safe current claim is recoverable local block-level language structure,
  not complete global L2R recovery.

## Current Language Mainline

The current mainline has two linked method tracks.

1. Distribution methodology: layer-level attention is converted into a
   symmetric affinity graph, usually `W = max(A, A.T)`, and the graph Laplacian
   Fiedler vector recovers an undirected block-order axis. Current-model
   `linear_profile_loss` chooses the axis direction. The active question is
   whether to keep EMA over this teacher signal, and if so whether EMA should
   be over ranks, continuous Fiedler priorities, or not used at all.

2. MLP distillation methodology: train a compact attention-conditioned MLP to
   imitate the Laplacian/Fiedler distribution teacher, then test whether the
   frozen MLP can safely replace the slower teacher inside AO-GPT training.

Direct asymmetric eigendecomposition, fixed-head top-1 order, and earlier
priority-distribution variants remain important baselines and historical
evidence, but they are not the preferred current entry point unless a task
explicitly asks to revisit them.

Active entry reports:

- `Report/language/wikitext103/order_teacher_distribution/block/seq256/block64/head_signal_stability/current_distribution_methodology_summary.md`
- `Report/language/wikitext103/mlp/block/seq256/block64/distillation/README.md`
- `Report/language/wikitext103/mlp/block/seq256/block64/distillation/HISTORY_ARCHIVE_20260624.md`
- `Report/language/wikitext103/mlp/block/seq256/block64/distillation/try_25/results.md`
- `Report/language/wikitext103/mlp/block/seq256/block64/distillation/try_28/results.md`
- `Report/language/wikitext103/mlp/block/seq256/block64/distillation/try_29/experiment_design.md`
