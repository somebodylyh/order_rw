# Project Prompt For New Agents

This repository is now language-focused. The active work is WikiText103
AO-GPT learned-order research.

Start with:

1. `docs/prompts/Prompt_language.md`
2. `docs/prompts/prompt_language_block_current_task.md`
3. `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md`
4. `Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md`
5. `docs/structure/PROJECT_STRUCTURE_language.md`
6. `docs/findings/findings_language.md`

## Project Goal

The central question is whether an AO-GPT trained without an explicit
left-to-right generation strategy exposes recoverable language order signals in
attention, loss, and early reveal behavior.

The current mainline is not the older broad recovery tree. It has two active
method tracks:

1. Distribution methodology: recover an order axis from a current-frame
   attention graph with Laplacian/Fiedler decomposition, orient it with
   current-model loss, and test how it should drive training.
2. MLP distillation methodology: compress that teacher into an
   attention-conditioned MLP, then test frozen-MLP insertion into AO-GPT
   training.

EMA is an active ablation. Do not assume it should be maintained; compare rank
EMA, continuous Fiedler-priority EMA, and no-EMA variants as separate
experiments.

## Boundaries

- Work in the current frame for training and curriculum.
- Treat `*_original` and `OriginalL2R` as diagnostics or oracle baselines.
- Do not use original-frame values as training signals.
- Keep token/block1 and token-micro results as diagnostics.
- The safe claim is local block-level language order recovery, not complete
  global L2R induction.
- Treat direct-asym-eig and fixed-head top-1 policies as baselines unless the
  task explicitly asks to revisit them.

The old non-language track has been removed from the working tree and should
not be used as an active prompt target.
