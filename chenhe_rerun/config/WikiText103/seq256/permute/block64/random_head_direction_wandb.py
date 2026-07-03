# Permuted random baseline with a read-only 32-head direction logger.
#
# Training hyperparameters are inherited from random.py. This config only changes
# the output/W&B identity and enables diagnostic logging of per-head attention
# maps and direction metrics. The logger uses no loss/order signal and does not
# feed anything back into training.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-head-direction-wandb-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-random-b64-permute-head-direction-wandb-50000-iters'

head_direction_logger_enabled = True

# Default: log at eval checkpoints. For literal per-iteration logging, override
# with --head_direction_logger_interval=1, but that will create very large W&B
# image volume over 50k steps.
head_direction_logger_interval = 250
head_direction_logger_start_iter = 0
head_direction_logger_stop_iter = -1

# Independent no-grad probe batch. This does not change the training batch size
# or the random training policy.
head_direction_logger_batch_size = 16
head_direction_logger_batches = 1

# Match the clean single-head view used in the current analysis thread.
head_direction_logger_export_type = 'without_none'
head_direction_logger_frame = 'true_original'

head_direction_logger_wandb_prefix = 'head_direction'
head_direction_logger_log_grid = True
head_direction_logger_log_individual_maps = True
head_direction_logger_individual_interval = 250
head_direction_logger_cmap = 'viridis'
head_direction_logger_vmax_percentile = 99.5

# Keep only latest local diagnostic files under out_dir; the full history is in W&B.
head_direction_logger_save_latest = True
head_direction_logger_out_dir = ''

# Preserve the training random stream around the diagnostic probe.
head_direction_logger_restore_rng = True
head_direction_logger_deterministic = False
head_direction_logger_seed = 24681357
