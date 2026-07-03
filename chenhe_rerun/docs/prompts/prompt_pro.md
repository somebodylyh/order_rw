# Prompt Pro: WikiText103 Fiedler Distribution + Attn-MLP Distillation

Use this prompt for rigorous research review, claim polishing, or next-step
planning on the current WikiText103 language mainline.

## Current One-Line Goal

Develop and audit a no-prior order-learning path for WikiText103
`seq256/permute/block64`:

```text
current-frame attention
-> Laplacian/Fiedler distribution teacher
-> EMA/no-EMA training-policy ablation
-> Attn-MLP teacher compression
-> frozen-MLP insertion into AO-GPT training
```

## Required Reading

```text
README.md
base.md
base_cn.md
docs/prompts/prompt_language_block_current_task.md
docs/findings/findings_language.md
docs/structure/PROJECT_STRUCTURE_language.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23/results.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24/results.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28/design.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29/design.md
Report/language/wikitext103/mlp/distillation/README.md
Report/language/wikitext103/mlp/distillation/try_25/results.md
Report/language/wikitext103/mlp/distillation/try_28/results.md
Report/language/wikitext103/mlp/distillation/try_29/experiment_design.md
```

Core code:

```text
train.py
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/train/train_pairwise_mlp_operator_distillation.py
scripts/eval/eval_pairwise_operator_mlp_generalization.py
```

## Method Summary

Current distribution teacher:

```text
L0 layer-mean current-frame attention
-> W=max(A,A.T), diagonal zeroed
-> graph Laplacian Fiedler vector
-> raw/reverse order
-> current-model train linear_profile_loss chooses direction
```

Current EMA question:

```text
rank EMA vs continuous Fiedler-priority EMA vs no EMA
```

Current MLP question:

```text
Can FlatAttentionOrderMLP imitate the Fiedler teacher, and can the frozen
student be inserted into training without attention/logits EMA smoothing?
```

## Evidence Boundaries

- Try23/try24 support the Fiedler distribution teacher with rank EMA.
- Try28/try29 distribution runs are designs/configured follow-ups unless their
  results are audited from logs.
- Try25/try28 support teacher imitation by MLP.
- Try29/try30 MLP insertion results must not be claimed without reading the
  run logs/results.
- Original-frame tau, distance, and `OriginalL2R` are diagnostics only.
- Do not claim complete global L2R induction.
- Do not turn smoke tests into conclusions.

## Review Questions

When reviewing a claim or plan, ask:

1. Which signal was used for selection/training: attention graph, loss profile,
   EMA memory, MLP logits, or original-frame diagnostic?
2. Is the Fiedler axis recovery separated from direction selection?
3. Is EMA treated as an ablation rather than a default assumption?
4. Is teacher imitation separated from successful insertion into AO-GPT
   training?
5. Are direct-asym-eig, fixed-head top-1, and seq80/block1 described as
   baselines/history unless explicitly under review?

## Answer Style

Answer in Chinese unless asked otherwise. Cite concrete files, configs,
reports, metrics, and uncertainty. Keep claims bounded to what the evidence
actually supports.
