# Prompt Index

This directory contains prompt-style handoffs for the language-focused
WikiText103 repository.

## Current Entry Points

- `Prompt.md`
  Top-level prompt router.
- `Prompt_language.md`
  Main WikiText103 language prompt.
- `prompt_language_block_current_task.md`
  Current seq256/block64 distribution + MLP-distillation handoff.
- `prompt_attn_mlp_training_no_prior_goal.md`
  Goal-mode prompt for real WikiText103 block64 Attn-MLP no-prior attempts.
- `prompt_for_MLP.md`
  MLP policy prompt. Use after reading the current Fiedler-teacher reports.
- `prompt_pro.md`
  Research-review / professionalization prompt.
- `reviewer.md`
  Strict reviewer-style prompt for the current Fiedler teacher and MLP
  distillation methodology.

## Routing Rule

Read the prompt matching the current language task first:

- WikiText103 seq256/block64 distribution, EMA, or MLP distillation work:
  `prompt_language_block_current_task.md`
- General language-side project orientation:
  `Prompt.md`, then `Prompt_language.md`

Do not route new work to deleted non-language prompts or datasets.

## Archived Prompt Files

The following files are kept for provenance, but should not be the first
instruction source for new work because they mention pre-cleanup paths, older
branches, or superseded teacher choices:

- `archive/prompt_for_distribution.md`
- `archive/agent_prompt_mlp_distillation_stage0.md`
- `archive/prompt_attn_mlp_loss_design_continuation.md`
- `archive/prompt_MLPTRAIN.md`
- `archive/prompt_attn_mlp_implicit_axis_frozen_random.md`
- `archive/prompt_attn_mlp_logits_v1.md`
- `archive/head_directionality_continuous_5k_goal_prompt.md`
- `archive/agent_prompt_head_information_research.md`
- `archive/prompt_online_language_current.md`
- `archive/README.md`

If one of these is useful, first translate its paths into
`Report/language/wikitext103/...` and check it against the current method
summaries.
