# Prompt: No-Prior Attn-MLP Order Policy Training Goal

Use this prompt for a fresh agent that must run real WikiText103 block64
Attn-MLP experiments until it obtains an effective no-prior MLP design, or gives
a rigorous failure report with concrete next steps.

Current default: continue from the L0 layer-mean pairwise-max Fiedler teacher,
try25/try28 distillation, and try29/try30 frozen-MLP no-EMA insertion tests.
Treat older direct-asym-eig or teacher-free designs as baselines/history.

## Objective

Work in `/home/chenhe/nanogpt-learned-order`.

Given the current WikiText103 `seq256/permute/block64` Fiedler teacher and
distilled MLP checkpoints, evaluate or improve an attention-conditioned MLP
order policy without using original-frame oracle labels.

Input:

```text
current-frame block attention/features
```

Output:

```text
logits in R^64
order = argsort(logits, descending=True)
```

Target:

- good heldout current-frame NLL / full loss
- strong post-hoc absolute Kendall tau diagnostic
- no use of oracle/original-frame signals during training or selection

## Hard Constraints

The following are diagnostics only and must not be used as training targets,
candidate sources, rerank signals, early stopping rules, checkpoint selection
rules, or hyperparameter selection rules:

- `OriginalL2R`
- original-frame tau
- original-frame PPL
- hand-written order labels
- validation oracle comparisons

## Required Reading

```text
README.md
base.md
base_cn.md
docs/findings/findings_language.md
docs/structure/PROJECT_STRUCTURE_language.md
docs/prompts/prompt_language_block_current_task.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
Report/language/wikitext103/mlp/distillation/README.md
train.py
AOGPT.py
AOGPT_block.py
AOGPT_token.py
order_utils.py
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/train/train_pairwise_mlp_operator_distillation.py
scripts/eval/eval_pairwise_operator_mlp_generalization.py
```

## Reporting Requirements

Save artifacts under `Report/language/wikitext103/mlp/` or an explicitly named
language report subdirectory.

Every result must include:

- command used
- config path
- checkpoint path
- train/eval split
- current-frame loss metrics
- MAP order metrics
- sampled-order metrics if sampling is used
- post-hoc diagnostics clearly labeled as diagnostics
- failure analysis if the run does not improve over baseline
