# Run Commands

This file keeps language-only commands that remain valid after the cleanup.

## Environment

```bash
cd /home/chenhe/nanogpt-learned-order
```

Preferred Python on the workstation:

```bash
/data/users/chenhe/conda_envs/X1/bin/python
```

## Tiny Smoke Test

```bash
python train.py config/WikiText103/seq256/permute/block64/random.py \
  --max_iters=1 \
  --eval_interval=1 \
  --eval_iters=1 \
  --wandb_log=False \
  --compile=False
```

## WikiText103 Evaluation Helpers

```bash
python scripts/eval/eval_lm_original_order_ppl.py --help
python scripts/eval/eval_lm_ppl_sweep.py --help
```

## Pairwise MLP Distillation Helpers

```bash
python scripts/data/merge_pairwise_operator_datasets.py --help
python scripts/train/train_pairwise_mlp_operator_distillation.py --help
python scripts/eval/eval_pairwise_operator_mlp_generalization.py --help
```

## Current Mainline References

Distribution / Fiedler reports:

```text
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_23/results.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_24/results.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28/design.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29/design.md
```

MLP distillation reports:

```text
Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md
Report/language/wikitext103/mlp/distillation/README.md
Report/language/wikitext103/mlp/distillation/try_25/results.md
Report/language/wikitext103/mlp/distillation/try_28/results.md
Report/language/wikitext103/mlp/distillation/try_29/experiment_design.md
Report/language/wikitext103/mlp/distillation/try_30/experiment_design.md
Report/language/wikitext103/mlp/distillation/tries_by_stage/05_latest_shadow_schedule_seed_sweep/README.md
Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/README.md
```

Configured trainable online-MSE try61-64:

```bash
Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/try_61/run_main_try61_seed2053_cuda1.sh
Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/try_62/run_main_try62_seed2050_cuda1.sh
Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/try_63/run_main_try63_seed2051_cuda1.sh
Report/language/wikitext103/mlp/distillation/tries_by_stage/06_trainable_online_mse_refinement/try_64/run_main_try64_seed2052_cuda1.sh
```

Serial GPU1 queue for the remaining trainable online-MSE counterparts:

```bash
scripts/runner/run_wikitext103_attn_mlp_try62_try63_try64_gpu1_after_try61_20260630.sh
```

## Current Reading Order

```text
README.md
base.md
base_cn.md
docs/README.md
docs/prompts/Prompt_language.md
docs/prompts/prompt_language_block_current_task.md
Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md
Report/language/wikitext103/mlp/distillation/README.md
Report/language/wikitext103/mlp/distillation/HISTORY_ARCHIVE_20260624.md
docs/findings/findings_language.md
docs/structure/PROJECT_STRUCTURE_language.md
Report/language/wikitext103/README.md
```

## Safety Notes

- Do not use `OriginalL2R`, original tau, or `*_original` fields for training
  decisions.
- For `permute_data=True`, compare no-prior orders in the current frame.
- Token/block1 and token-micro runs are diagnostic.
- The active claim is local block-level language order recovery, not complete
  global L2R recovery.
- EMA is unsettled; rank EMA, continuous Fiedler-priority EMA, and no-EMA runs
  must be reported separately.
