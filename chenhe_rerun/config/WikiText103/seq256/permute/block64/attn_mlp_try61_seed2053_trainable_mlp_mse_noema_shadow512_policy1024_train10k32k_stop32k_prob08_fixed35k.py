# WikiText103 seq256 permuted block64 try_61:
# Follow-up to try60. Keep the latest no-EMA score-MLP distillation protocol,
# but keep the MLP optimizer active through the 18k-32k policy-refresh phase.
# The run tests whether online teacher-score MSE refinement improves over the
# frozen-after-shadow try56-60 family.

import os

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2053
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try61-seed2053-trainable-mlp-mse-online-noema-shadow512-policy1024-train10k32k-stop32k-prob08-fixed35k-b64-permute-block'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try61-seed2053-trainable-mlp-mse-online-noema-shadow512-policy1024-train10k32k-stop32k-prob08-fixed35k'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False
attn_mlp_policy_train_loss = 'teacher_score_mse'

# L0 all-head mean, without_none, matching the latest score-distillation input.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = -1
attn_mlp_policy_export_type = 'without_none'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_hidden_dims = '2048,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_feature_clip = 0.0
attn_mlp_policy_input_normalization = 'zscore'

attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'

# 0..10k: pure Random without attention collection.
# 10k..18k: Random backbone, shadow-train MLP with teacher-score MSE.
#            Each shadow item averages 512 raw samples:
#            4 optimizer iters * 2 micro-batches * 64 samples = 512.
#            Two train items are consumed per 1024-sample update.
# 18k..32k: anneal MLP policy probability 0.0 -> 0.8 while continuing
#            teacher-score MSE updates from current attention. The policy input
#            attention and MSE update each aggregate 1024 raw samples.
# 32k..35k: stop MLP updates and cached-order refresh; anneal the final cached
#            MLP order probability 0.8 -> 1.0.
# 35k..50k: deterministic fixed cached MLP order.
shadow_train_start = 10000
warmup_until = 18000
mse_train_stop = 32000
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
attn_mlp_policy_attention_batches_per_update = 8
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
attn_mlp_policy_shadow_mse_stop_iter = mse_train_stop
attn_mlp_policy_shadow_mse_train_frac = 0.75
attn_mlp_policy_shadow_mse_batches_per_item = 4
attn_mlp_policy_shadow_mse_train_items_per_update = 2
attn_mlp_policy_shadow_mse_val_items_per_update = 0
attn_mlp_policy_shadow_mse_train_samples_per_step = 0
attn_mlp_policy_shadow_mse_val_samples_per_step = 0
attn_mlp_policy_shadow_mse_label_orientation = 'linear_profile_loss'
attn_mlp_policy_shadow_mse_orientation_x_mode = 'last_step'
attn_mlp_policy_shadow_mse_prefix_k = 16
# Prefix/full weights are retained for legacy loss_profile mode; this try
# orients labels with distribution-style per-block linear_profile_loss.
attn_mlp_policy_shadow_mse_prefix_weight = 0.7
attn_mlp_policy_shadow_mse_full_weight = 0.3
attn_mlp_policy_shadow_mse_log_path = 'Report/language/wikitext103/mlp/distillation/try_61/input_attn/attn_mlp_shadow_mse_history.jsonl'
attn_mlp_policy_shadow_mse_eval_after_stop_enabled = True
attn_mlp_policy_shadow_mse_eval_after_stop_interval = 1

# No EMA in this controlled follow-up.
attn_mlp_policy_attention_ema_enabled = False
attn_mlp_policy_ema_decay = 0.0
attn_mlp_policy_logits_ema_enabled = False

attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = 'try61_seed2053_trainable_mlp_mse_online_noema_shadow512_policy1024_train10k32k_stop32k_prob08_fixed35k_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/language/wikitext103/mlp/distillation/try_61/input_attn'
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

# Keep the MLP ablation clean: MSE-to-teacher-score only.
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
