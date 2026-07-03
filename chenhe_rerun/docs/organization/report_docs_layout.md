# Report And Docs Layout Policy

This file defines the intended organization for `docs/` and `Report/` after
the language-only cleanup.

## Scope

This cleanup layer may touch:

- `docs/**/*.md`
- `Report/**/*.md`
- lightweight `Report/_index/*` inventory files

It should not casually touch:

- source code
- active configs
- `data/wikitext103`
- `out/` runtime checkpoints
- `wandb/`
- active training outputs

## Docs Policy

`docs/` is the human-readable source of truth for project navigation. It should
stay small, stable, and language-focused.

Preferred layout:

```text
docs/
  README.md
  organization/
  inventory/
  prompts/
  findings/
  structure/
```

Use `docs/structure/` for where language things live, `docs/findings/` for what
is known, `docs/prompts/` for agent handoffs, `docs/inventory/` for current
filesystem maps, and `docs/organization/` for cleanup rules.

## Report Policy

`Report/` is an artifact tree. It contains generated files, historical
experiments, smoke tests, logs, and analysis reports. It should be navigated
through indexes rather than treated as source documentation.

Current active report root:

```text
Report/language/wikitext103/
```

Current active subroots:

```text
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/
Report/language/wikitext103/mlp/distillation/
Report/language/wikitext103/mlp/joint_training/
```

Use the first two as the default current-method entry points. Treat
`mlp/joint_training/` as important history unless a task explicitly asks to
extend joint training.

Future language report directories should include dataset, task, and short run
name:

```text
Report/language/wikitext103/<task>/<run_name>/
Report/logs/<yyyymmdd>_<short_task>/
```

When a report is temporary, put `tmp` or `smoke` in the path so it is easy to
archive later.

## Migration Rule

Use a three-step migration for any future physical reorganization:

1. Add or update an index file describing the current location.
2. Move only directories that are not active and not referenced by current
   docs/configs.
3. Leave a manifest recording old path, new path, reason, and date.
