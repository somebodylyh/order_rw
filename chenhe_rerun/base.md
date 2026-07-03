# Language-Only Project Base

This repository is now a WikiText103 / language-only AO-GPT learned-order
workspace. Non-language experiment material was backed up on the remote cleanup
branch and removed from the working tree.

## Active Question

Can a random reveal-order AO-GPT language backbone expose recoverable
left-to-right structure through current-frame attention and loss, and can that
signal be turned into a stable training order without using original-frame
oracles?

## Active Scope

- Dataset: `data/wikitext103`
- Configs: `config/WikiText103/`
- Reports: `Report/language/wikitext103/`
- Core code: `train.py`, `AOGPT.py`, `AOGPT_block.py`, `AOGPT_token.py`,
  `order_utils.py`, `online_spectral_order_policy.py`,
  `attn_mlp_order_policy.py`
- Current docs:
  - `docs/prompts/Prompt_language.md`
  - `docs/prompts/prompt_language_block_current_task.md`
  - `docs/findings/findings_language.md`
  - `docs/structure/PROJECT_STRUCTURE_language.md`

## Method Boundary

The current mainline does not learn one global order parameter directly inside
the model. It uses a recovery-and-curriculum loop:

1. train a random reveal-order AO-GPT backbone
2. mine candidate pair relations or candidate orders from current-frame model
   signals
3. score and filter with attention, loss, margin, gain, and stability signals
4. feed recovered units or orders back into training/evaluation
5. evaluate whether the recovered structure approaches language L2R order

## Frame Rule

If `permute_data=False`, current frame equals text frame.

If `permute_data=True`, training happens in a fixed permuted current frame.
Checkpoints save:

- `block_perm`: current-frame block index to text-frame block index
- `inverse_block_perm`: text L2R order expressed in current-frame ids

Training, candidate selection, rerank, early stop, and checkpoint choice must use
current-frame signals. `*_original`, original tau, and `OriginalL2R` are
diagnostics only. Under `permute_data=True`, `OriginalL2R` is an oracle upper
bound, not no-prior evidence.

## Current Mainline

The current mainline is narrower than the full historical language tree:

1. Distribution methodology:
   `L0/layer-mean attention -> W=max(A,A.T) -> graph Laplacian Fiedler axis ->
   current-model linear_profile_loss orientation -> training order`.
2. EMA ablation:
   maintain rank/priority EMA, maintain continuous Fiedler-priority EMA, or
   remove EMA and use the current teacher/order directly. This is still an
   open ablation, not a settled default.
3. MLP distillation:
   compress the Laplacian/Fiedler teacher into `attn_mlp_order_policy.py` and
   test frozen-MLP insertion into AO-GPT training.

Direct-asym-eig and fixed-head top-1 policies remain useful baselines. Do not
use them as the current default unless the task explicitly asks for that older
line.

Useful files:

```text
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/eval/eval_lm_original_order_ppl.py
scripts/eval/eval_lm_ppl_sweep.py
scripts/train/train_pairwise_mlp_operator_distillation.py
scripts/data/merge_pairwise_operator_datasets.py
scripts/eval/eval_pairwise_operator_mlp_generalization.py
```

Useful config roots:

```text
config/WikiText103/seq80/
config/WikiText103/seq256/
config/WikiText103/seq384/
```

Current report roots:

```text
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/
Report/language/wikitext103/mlp/distillation/
Report/language/wikitext103/mlp/joint_training/
```

## Safe Claims

- Strongest claim: recoverable local block-level language order structure.
- Not yet safe: complete global L2R induction.
- Token/block1 and token-micro work are diagnostics, not the active target.
- No-prior claims must exclude original-frame diagnostics from training-time
  decisions.

## Cleanup Note

After the cleanup, documentation should route new work only through the
language files listed above. Do not rely on removed non-language configs,
datasets, reports, external dependencies, or commands.
