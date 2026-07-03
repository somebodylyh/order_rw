# Prompt: WikiText103 Attn-MLP Order Policy

Use this prompt when the task is to continue the language-side Attn-MLP order
policy line.

Current default: start from the L0 layer-mean pairwise-max Fiedler teacher and
the try25/try28 distillation evidence. Do not start from direct-asym-eig,
teacher-free policy learning, or old top-1 fixed-head prompts unless the user
explicitly asks for those baselines.

## Goal

Train or evaluate a small MLP that maps current-frame attention/features to an
order policy for WikiText103 AO-GPT runs.

The MLP must learn from model-side language signals, not from original-frame
oracle labels.

## Required Reading

```text
README.md
base.md
base_cn.md
docs/findings/findings_language.md
docs/prompts/prompt_language_block_current_task.md
docs/structure/PROJECT_STRUCTURE_language.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
Report/language/wikitext103/mlp/distillation/README.md
train.py
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/train/train_pairwise_mlp_operator_distillation.py
scripts/eval/eval_pairwise_operator_mlp_generalization.py
```

## Hard Rules

- Do not train on `OriginalL2R`, original tau, or mapped original-frame labels.
- Do not use original-frame diagnostics for checkpoint selection.
- Keep `with_none` and `without_none` semantics explicit.
- Report MAP order, sampled order, heldout tau diagnostics, and current-frame
  loss separately.
- Treat token/block1 results as diagnostics unless the user explicitly changes
  the target.

## Useful Setting

Start with WikiText103 `seq256/permute/block64`, the current layer-mean /
Fiedler teacher family, and the frozen try28 MLP insertion tests before
broadening to other sequence lengths.
