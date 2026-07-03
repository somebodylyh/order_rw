# Project Docs

This directory is the project-facing documentation entry point for the cleaned
language-focused repository.

The repository now keeps only the WikiText103 / language mainline. Obsolete
non-language docs, prompts, structure maps, configs, data, checkpoints,
benchmarks, and external dependencies were removed after backup.

## Start Here

- `docs/prompts/README.md`
  Prompt index and routing rules.
- `docs/prompts/Prompt.md`
  Top-level language prompt router.
- `docs/prompts/Prompt_language.md`
  WikiText103 language prompt.
- `docs/prompts/prompt_language_block_current_task.md`
  Current seq256/block64 distribution + MLP-distillation handoff.
- `Report/language/wikitext103/MAINLINE_AND_HISTORY.md`
  Current-vs-history report routing after the Report cleanup.
- `Report/language/wikitext103/order_teacher_distribution/block/seq256/block64/head_signal_stability/current_distribution_methodology_summary.md`
  Current Laplacian/Fiedler distribution methodology and EMA ablation status.
- `Report/language/wikitext103/mlp/block/seq256/block64/distillation/HISTORY_ARCHIVE_20260624.md`
  Current MLP distillation line and kept/deleted history.
- `Report/language/wikitext103/mlp/block/seq256/block64/distillation/README.md`
  Compact MLP distillation reading path.
- `docs/structure/README.md`
  Structure-document index.
- `docs/structure/PROJECT_STRUCTURE.md`
  Top-level language-only structure router.
- `docs/structure/PROJECT_STRUCTURE_language.md`
  Language structure index.
- `docs/findings/README.md`
  Findings index.
- `docs/findings/findings.md`
  Top-level findings router.
- `docs/findings/findings_language.md`
  Language findings.
- `docs/inventory/markdown_cleanup_candidates.md`
  Markdown keep/archive/delete inventory after cleanup.

## Report Indexes

`Report/` contains generated artifacts and historical experiment reports. Use
these indexes before browsing it:

- `Report/README.md`
- `Report/_index/README.md`
- `Report/_index/top_level_buckets.md`
- `Report/language/wikitext103/README.md`
- `Report/language/wikitext103/MAINLINE_AND_HISTORY.md`

Root-level `base.md`, `base_cn.md`, `README.md`, `RUN_COMMANDS.md`,
`research-state.yaml`, and `research-log.md` remain at the repository root
because they are broad project entry points or active state files.

## Current Mainline

Use the documentation as a language-mainline router:

1. Distribution work means the Laplacian/Fiedler graph-decomposition line:
   layer-mean attention, `W=max(A,A.T)`, Fiedler axis recovery, and
   current-model loss orientation.
2. EMA is unsettled. Treat rank EMA, continuous-priority EMA, and no-EMA
   variants as active ablations.
3. MLP work means teacher compression first, then frozen-MLP integration. Do
   not treat old teacher-free or direct-asym-eig prompts as the default path.
