# Rerun of the seq256/block64 permuted random baseline for comparison.
#
# Training hyperparameters are inherited from random.py. Only out_dir and W&B
# naming are changed so this run does not overwrite the existing baseline.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-rerun-compare-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-permute-rerun-compare-50000-iters'
