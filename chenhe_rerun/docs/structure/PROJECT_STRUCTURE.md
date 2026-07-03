# Project Structure Index

This repository is a language-focused learned-order / AO-GPT research fork
built from nanoGPT.

The active structure map is:

- `docs/structure/PROJECT_STRUCTURE_language.md`

The removed non-language maps, datasets, configs, reports, auxiliary scripts,
and external dependencies are no longer part of the active working tree.

## Language Paths

Primary source and configuration:

```text
train.py
AOGPT.py
AOGPT_block.py
AOGPT_token.py
order_utils.py
online_spectral_order_policy.py
attn_mlp_order_policy.py
config/WikiText103/
data/wikitext103/
scripts/
tests/
```

Primary documentation and report paths:

```text
docs/prompts/Prompt_language.md
docs/prompts/prompt_language_block_current_task.md
docs/findings/findings_language.md
docs/structure/PROJECT_STRUCTURE_language.md
Report/language/wikitext103/
Summarize/
```

Runtime artifact paths:

```text
out/
checkpoints/
wandb/
Report/language/
```

## Shared Rules

- `permute_data=True` means training happens in a fixed current frame.
- Checkpoints save `block_perm` and `inverse_block_perm`.
- Training and curriculum use current-frame units.
- `*_original` fields are analysis/reporting fields only.
- Language conclusions should emphasize recoverable local block-level L2R
  structure.
- Do not claim complete global order recovery without direct evidence.
- Current active method work is the `seq256/permute/block64`
  Laplacian/Fiedler distribution teacher plus MLP distillation line.
- EMA is unresolved: rank EMA, continuous Fiedler-priority EMA, and no-EMA
  variants must be tracked separately.
