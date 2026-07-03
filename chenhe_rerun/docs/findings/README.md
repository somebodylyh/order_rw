# Findings Index

Use this directory to answer "what do we currently believe?" questions for the
language-focused repository.

## Files

- `findings.md`
  Top-level findings router.
- `findings_language.md`
  WikiText103 / language findings, interpretation boundaries, and active online
  order status.

## Interpretation Rule

Avoid claiming complete global order recovery unless a specific result directly
supports that stronger claim. In the current mainline, `OriginalL2R` is an
oracle diagnostic under permutation, not no-prior evidence.

Current method claims should focus on the L0 layer-mean pairwise-max Fiedler
distribution teacher, EMA/no-EMA ablation, and MLP distillation/insertion
rather than older fixed-head top-1 online policies.
