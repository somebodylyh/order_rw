# Current Layout Inventory

This inventory reflects the cleaned language-focused repository.

## Source And Configuration

| Path | Role | Cleanup stance |
| --- | --- | --- |
| `train.py`, `AOGPT*.py`, `order_utils.py`, `path_layout.py` | Core training/model/order source | Keep |
| `online_spectral_order_policy.py`, `attn_mlp_order_policy.py` | Current online order and MLP policy utilities | Keep |
| `config/WikiText103/` | Language experiment configs | Keep |
| `scripts/` | Language runners, analysis, eval, and data helpers | Keep and audit before deleting |
| `tests/` | Tests | Keep |

## Documentation

| Path | Role | Cleanup stance |
| --- | --- | --- |
| `docs/` | Canonical documentation tree | Keep language-only |
| `base.md`, `base_cn.md` | Broad language handoffs | Keep |
| `README.md`, `RUN_COMMANDS.md` | Root entry points | Keep |
| `research-state.yaml`, `research-log.md` | Active research state/log | Keep |
| `literature/`, `paper/` | Research references | Keep only language-relevant notes |

## Artifacts And Runtime Outputs

| Path | Role | Cleanup stance |
| --- | --- | --- |
| `Report/language/wikitext103/` | Language reports and analysis artifacts | Keep, later split summaries from raw artifacts |
| `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/` | Current distribution/Fiedler reports and EMA ablations | Keep current summaries and try23/24/28/29; archive older probes only after index coverage |
| `Report/language/wikitext103/mlp/distillation/` | Current Fiedler-teacher MLP distillation and frozen insertion reports | Keep README, try20/21/25/28/29/30 plus archive note |
| `out/` | Training runtime outputs and checkpoints | Audit language runs before deleting |
| `wandb/` | Local W&B logs | Keep until a separate log cleanup |
| `checkpoints/` | Standalone language checkpoints/assets | Audit before deleting |
| `Summarize/` | Meeting and progress summaries | Keep |

## Data

| Path | Role | Cleanup stance |
| --- | --- | --- |
| `data/wikitext103/` | Prepared language dataset | Keep |

## Current Cleanup Priority

1. Keep docs and prompts language-only.
2. Preserve active WikiText103 distribution/Fiedler and MLP-distillation
   summaries/configs.
3. Mark old prompt files as archival instead of using them as active entry
   points.
4. Separate readable language summaries from raw artifacts.
5. Audit `out/`, `wandb/`, and `checkpoints/` before any further deletion.
