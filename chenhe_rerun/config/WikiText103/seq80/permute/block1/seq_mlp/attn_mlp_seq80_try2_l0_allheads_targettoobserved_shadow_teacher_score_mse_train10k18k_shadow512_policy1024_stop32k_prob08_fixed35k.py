# WikiText103 seq80 permuted block1 Attn-MLP try_2.
#
# Token-level counterpart to the block64 try60 schedule, using the newer seq
# readout found after try_1:
# - inherit the seq80 random base, including 4 layers, 8 heads, 384 emb
# - use layer-0 all-head mean attention as MLP input
# - use target_to_observed attention export with exposure correction
# - shadow-train a random-init score MLP on teacher_score_mse from 10k to 18k
# - stop MLP optimizer updates at 18k, keep refreshing cached MLP orders to 32k
# - anneal cached MLP order probability to 0.8 by 32k and 1.0 by 35k

_base_config = 'config/WikiText103/seq80/permute/block1/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2053
permute_seed = 42

out_dir = (
    'out/seq/permute/seq80/block1/'
    'out-wikitext103-seq80-attnmlp-try2-l0-allheads-targettoobserved-'
    'shadow-teacher-score-mse-train10k18k-shadow512-policy1024-'
    'stop32k-prob08-fixed35k-b1-permute'
)

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False
eval_batch_size = 64

wandb_log = True
wandb_project = 'AOGPT-order-seq80-final'
wandb_run_name = (
    'seq80-attnmlp-try2-l0-allheads-targettoobserved-shadow-teacher-score-mse-'
    'train10k18k-shadow512-policy1024-stop32k-prob08-fixed35k'
)

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False
attn_mlp_policy_train_loss = 'teacher_score_mse'

# New seq readout: L0 all-head mean target-to-observed attention. Keeping
# use_global=False prevents later layers from washing out the stronger L0 signal.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = -1
attn_mlp_policy_export_type = 'target_to_observed'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_hidden_dims = '2048,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_feature_clip = 0.0
attn_mlp_policy_input_normalization = 'zscore'

attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'

# Seq80 uses 256 samples per micro-batch and grad_acc=2, so each optimizer
# iteration sees 512 raw samples. The batch counts below are sample-equivalent
# to block64 try60: 512 samples per shadow item and 1024 samples per policy
# attention/update.
shadow_train_start = 10000
warmup_until = 18000
update_stop = 32000
anneal_start = 18000
anneal_end = 35000
fixed_after = 35000

attn_mlp_policy_start_iter = warmup_until
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = anneal_start
attn_mlp_policy_anneal_end_iter = anneal_end
attn_mlp_policy_prob_schedule = 'piecewise'
attn_mlp_policy_prob_points = '0:0.0,9999:0.0,10000:0.0,17999:0.0,18000:0.0,32000:0.8,35000:1.0,50000:1.0'
attn_mlp_policy_collect_warmup_attention = True
attn_mlp_policy_update_every = 1
attn_mlp_policy_attention_batches_per_update = 2
attn_mlp_policy_update_stop_iter = update_stop
attn_mlp_policy_log_interval = 50
attn_mlp_policy_save_orders = True

mlp_frozen = False
attn_mlp_policy_lr = 2e-4
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_lr_anneal_enabled = False

attn_mlp_policy_shadow_mse_enabled = True
attn_mlp_policy_shadow_mse_start_iter = shadow_train_start
attn_mlp_policy_shadow_mse_stop_iter = warmup_until
attn_mlp_policy_shadow_mse_train_frac = 0.75
attn_mlp_policy_shadow_mse_batches_per_item = 1
attn_mlp_policy_shadow_mse_train_items_per_update = 2
attn_mlp_policy_shadow_mse_val_items_per_update = 0
attn_mlp_policy_shadow_mse_train_samples_per_step = 0
attn_mlp_policy_shadow_mse_val_samples_per_step = 0
attn_mlp_policy_shadow_mse_label_orientation = 'linear_profile_loss'
attn_mlp_policy_shadow_mse_orientation_x_mode = 'last_step'
attn_mlp_policy_shadow_mse_prefix_k = 16
attn_mlp_policy_shadow_mse_prefix_weight = 0.7
attn_mlp_policy_shadow_mse_full_weight = 0.3

_report_root = 'Report/language/wikitext103/mlp/token/seq80/try_2'
attn_mlp_policy_shadow_mse_log_path = _report_root + '/input_attn/attn_mlp_shadow_mse_history.jsonl'
attn_mlp_policy_shadow_mse_eval_after_stop_enabled = True
attn_mlp_policy_shadow_mse_eval_after_stop_interval = 1

attn_mlp_policy_attention_ema_enabled = False
attn_mlp_policy_ema_decay = 0.0
attn_mlp_policy_logits_ema_enabled = False

attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = (
    'seq80_attnmlp_try2_l0_allheads_targettoobserved_shadow_teacher_score_mse_'
    'train10k18k_shadow512_policy1024_stop32k_prob08_fixed35k_input_attn'
)
attn_mlp_policy_log_input_attn_out_dir = _report_root + '/input_attn'
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0

attn_mlp_policy_order_history_enabled = True
attn_mlp_policy_order_history_interval = 1
attn_mlp_policy_order_history_include_scores = True
attn_mlp_policy_order_history_include_input_stats = True
attn_mlp_policy_teacher_diag_enabled = True
attn_mlp_policy_teacher_diag_interval = 1
attn_mlp_policy_teacher_diag_orientation = 'match_mlp'
attn_mlp_policy_teacher_diag_include_scores = True

# Keep this MLP try clean: MSE-to-teacher-score only.
attn_mlp_policy_loss_stop_enabled = False
attn_mlp_policy_axis_profile_weight = 0.0
attn_mlp_policy_axis_profile_dir_weight = 0.0
attn_mlp_policy_sampled_orders_per_state = 0
attn_mlp_policy_random_baseline_orders = 0
attn_mlp_policy_pg_weight = 0.0
attn_mlp_policy_prefix_reward_weight = 0.0
attn_mlp_policy_move_pref_weight = 0.0
attn_mlp_policy_move_pref_pairs_per_state = 0
attn_mlp_policy_attn_pair_weight = 0.0
attn_mlp_policy_attn_close_weight = 0.0
attn_mlp_policy_head_profile_weight = 0.0
attn_mlp_policy_head_profile_every = 0

online_spectral_policy_enabled = False
online_spectral_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
head_asym_selector_enabled = False

assert bool(permute_data)
assert str(permute_mode) == 'block'
assert int(block_size) == 80
assert int(block_order_block_len) == 1
assert str(order_impl) == 'token'
assert int(n_layer) == 4
assert int(n_head) == 8
assert int(n_embd) == 384
assert int(batch_size) == 256
assert int(gradient_accumulation_steps) == 2
assert str(aogpt_train_mode) == 'AttnMLPFrozenOrder'
assert bool(attn_mlp_policy_enabled)
assert bool(attn_mlp_policy_random_init)
assert not bool(attn_mlp_policy_freeze)
assert str(attn_mlp_policy_train_loss) == 'teacher_score_mse'
assert not bool(attn_mlp_policy_use_global)
assert int(attn_mlp_policy_layer) == 0
assert int(attn_mlp_policy_head) == -1
assert str(attn_mlp_policy_export_type) == 'target_to_observed'
assert str(attn_mlp_policy_shadow_mse_label_orientation) == 'linear_profile_loss'
assert int(attn_mlp_policy_attention_batches_per_update) * int(batch_size) * int(gradient_accumulation_steps) == 1024
assert int(attn_mlp_policy_shadow_mse_batches_per_item) * int(batch_size) * int(gradient_accumulation_steps) == 512
assert (
    int(attn_mlp_policy_shadow_mse_batches_per_item)
    * int(attn_mlp_policy_shadow_mse_train_items_per_update)
    * int(batch_size)
    * int(gradient_accumulation_steps)
) == 1024
assert int(attn_mlp_policy_shadow_mse_start_iter) == 10000
assert int(attn_mlp_policy_shadow_mse_stop_iter) == 18000
assert int(attn_mlp_policy_update_stop_iter) == 32000
assert int(attn_mlp_policy_anneal_end_iter) == 35000
assert not bool(attn_mlp_policy_attention_ema_enabled)
assert not bool(attn_mlp_policy_logits_ema_enabled)
assert not bool(online_spectral_policy_enabled)
assert not bool(head_signal_probe_enabled)
