# Markdown Cleanup Candidates

This inventory tracks Markdown files that are safe to keep, archive, merge, or
delete after the language-only cleanup. It is intentionally conservative:
language experiment provenance should be archived or indexed before deletion.

Status: applied on 2026-06-26. Active prompt files were kept in
`docs/prompts/`, superseded prompt files were moved to `docs/prompts/archive/`,
and pure non-language/image entry points remain deleted.

## Keep As Current Entry Points

- `README.md`
- `base.md`
- `base_cn.md`
- `RUN_COMMANDS.md`
- `research-log.md`
- `docs/README.md`
- `docs/prompts/Prompt.md`
- `docs/prompts/Prompt_language.md`
- `docs/prompts/prompt_language_block_current_task.md`
- `docs/findings/findings.md`
- `docs/findings/findings_language.md`
- `docs/structure/PROJECT_STRUCTURE.md`
- `docs/structure/PROJECT_STRUCTURE_language.md`
- `Report/README.md`
- `Report/language/wikitext103/README.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md`
- `Report/language/wikitext103/mlp/distillation/README.md`
- `Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md`
- `Report/language/wikitext103/mlp/README.md`

## Keep As Current Evidence

- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23/*.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24/*.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28/*.md`
- `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29/*.md`
- `Report/language/wikitext103/mlp/distillation/try_25/*.md`
- `Report/language/wikitext103/mlp/distillation/try_28/*.md`
- `Report/language/wikitext103/mlp/distillation/try_29/*.md`
- `Report/language/wikitext103/mlp/distillation/try_30/*.md`

## Archive, Do Not Use As First Prompt

These files are useful provenance but mention old paths, older branches, or
superseded method choices:

- `docs/prompts/archive/prompt_for_distribution.md`
- `docs/prompts/archive/README.md`
- `docs/prompts/archive/agent_prompt_mlp_distillation_stage0.md`
- `docs/prompts/archive/prompt_attn_mlp_loss_design_continuation.md`
- `docs/prompts/archive/prompt_MLPTRAIN.md`
- `docs/prompts/archive/prompt_attn_mlp_implicit_axis_frozen_random.md`
- `docs/prompts/archive/prompt_attn_mlp_logits_v1.md`
- `docs/prompts/archive/head_directionality_continuous_5k_goal_prompt.md`
- `docs/prompts/archive/agent_prompt_head_information_research.md`
- `docs/prompts/archive/prompt_online_language_current.md`

These have already been moved to `docs/prompts/archive/` and should stay out
of the active prompt directory.

Current active prompt directory should contain only:

- `Prompt.md`
- `Prompt_language.md`
- `README.md`
- `prompt_attn_mlp_training_no_prior_goal.md`
- `prompt_for_MLP.md`
- `prompt_language_block_current_task.md`
- `prompt_pro.md`
- `reviewer.md`

## Merge Or Compress Later

These are intentionally not deleted in this cleanup pass. They are language
provenance, and the deletion rule below requires a summary/manifest before
removing them.

- `Report/language/wikitext103/head_information/summaries/*.md`
  Keep the executive/final reports, but merge duplicated file-read,
  methodology, limitation, and hypothesis notes when a paper-facing summary is
  ready.
- `Report/language/wikitext103/mlp/loss_semantics/*.md`
  Keep `SUMMARY.md` and `try_summary.csv`; archive individual old try designs
  after their lessons are merged into one failure-analysis note.
- `Report/language/wikitext103/mlp/loss_design_exploration/*.md`
  Keep as historical loss-design evidence; compress after the current Fiedler
  distillation/insertion path settles.

## Already Deleted Or Safe To Leave Deleted

These were pure non-language/image-side entry points and should remain deleted
in the cleaned tree:

- `docs/prompts/Prompt_image.md`
- `docs/findings/findings_image.md`
- `docs/structure/PROJECT_STRUCTURE_image.md`
- image-side literature notes and benchmark prompts
- pure image report indexes under the removed `Report/image/` tree

## Deletion Rule

Do not delete a language Markdown report just because it is old. Delete only
when one of these is true:

- it is a pure non-language/image file already covered by the remote backup;
- it is a duplicate prompt whose content is fully represented in a current
  entry point or archive note;
- it is a generated raw note with no unique metrics, commands, paths, or
  interpretation;
- its parent directory has a summary file that explicitly records the deleted
  content.
