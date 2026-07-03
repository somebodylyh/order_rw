# WikiText103 seq256 permuted block64 try_53:
# Same as try52, but with a new seed and more grouped shadow MSE data per MLP
# update: each attention matrix is aggregated from 1024 raw samples, and each
# update uses fifteen train items plus three held-out val items.

import os

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2036
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try53-seed2036-shadow-teacher-score-mse-train8k12k-attn1024-train15val3-fixed12k-anneal12k35k-b64-permute-block'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try53-seed2036-shadow-teacher-score-mse-train8k12k-attn1024-train15val3-fixed12k-anneal12k35k-b64-permute'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False
attn_mlp_policy_train_loss = 'teacher_score_mse'

# L0 all-head mean, without_none, matching the score-distillation input family.
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

# 0..8k: pure Random without attention collection.
# 8k..12k: Random backbone, shadow-train MLP from current attention.
#           Eight optimizer-step batches form one 1024-sample matrix item;
#           each MLP update uses fifteen train items and three held-out val items.
# 12k..35k: stop MLP optimizer updates, anneal Random -> fixed-parameter MLP.
#           Order refresh also uses eight optimizer-step batches per attn.
# 35k..50k: stop refreshing and train on the last cached MLP order.
shadow_train_start = 8000
warmup_until = 12000
anneal_start = 12000
anneal_end = 35000
fixed_after = 35000

attn_mlp_policy_start_iter = warmup_until
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = anneal_start
attn_mlp_policy_anneal_end_iter = anneal_end
attn_mlp_policy_collect_warmup_attention = True
attn_mlp_policy_update_every = 1
attn_mlp_policy_attention_batches_per_update = 8
attn_mlp_policy_update_stop_iter = fixed_after
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
attn_mlp_policy_shadow_mse_batches_per_item = 8
attn_mlp_policy_shadow_mse_train_items_per_update = 15
attn_mlp_policy_shadow_mse_val_items_per_update = 3
attn_mlp_policy_shadow_mse_label_orientation = 'loss_profile'
attn_mlp_policy_shadow_mse_prefix_k = 16
attn_mlp_policy_shadow_mse_prefix_weight = 0.7
attn_mlp_policy_shadow_mse_full_weight = 0.3
attn_mlp_policy_shadow_mse_log_path = 'Report/language/wikitext103/mlp/distillation/try_53/input_attn/attn_mlp_shadow_mse_history.jsonl'
attn_mlp_policy_shadow_mse_eval_after_stop_enabled = True
attn_mlp_policy_shadow_mse_eval_after_stop_interval = 1

attn_mlp_policy_attention_ema_enabled = False
attn_mlp_policy_ema_decay = 0.0
attn_mlp_policy_logits_ema_enabled = False

attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = 'try53_seed2036_shadow_teacher_score_mse_train8k12k_attn1024_train15val3_fixed12k_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/language/wikitext103/mlp/distillation/try_53/input_attn'
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
