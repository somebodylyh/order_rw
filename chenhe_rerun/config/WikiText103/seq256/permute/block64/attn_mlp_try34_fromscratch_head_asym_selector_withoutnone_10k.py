# WikiText103 seq256 permuted block64 try_34:
# Pure Random warmup to 10k, then select exactly one head by current-frame
# without_none attention asymmetry.
#
# Selection rule:
#   asym_score_h = sum_{i!=j} |A_ij - A_ji| / sum_{i!=j} |A_ij|
#   choose argmax_h asym_score_h.
#
# No-prior rule:
# Original-frame upper/lower triangles, OriginalL2R, original tau, validation
# loss, oracle permutations, and distribution-produced orders are not used for
# selection. The selector only sees current-frame attention under Random order.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try34-head-asym-selector-withoutnone-10k-b64-permute-block'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 10000
lr_decay_iters = 50000
eval_interval = 10000
always_save_checkpoint = True

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try34-head-asym-selector-withoutnone-10k-b64-permute'

aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = False
online_spectral_policy_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False

head_asym_selector_enabled = True
head_asym_selector_target_iter = 10000
head_asym_selector_split = 'train'
head_asym_selector_batch_size = 32
head_asym_selector_batches = 4
head_asym_selector_export_type = 'without_none'
head_asym_selector_out_dir = 'Report/MLP_loss_training/try_34/head_asym_selector'
head_asym_selector_wandb_prefix = 'head_asym_selector'
head_asym_selector_log_wandb = True
head_asym_selector_save_matrices = True
head_asym_selector_restore_rng = True
head_asym_selector_deterministic = True
head_asym_selector_seed = 24681357
