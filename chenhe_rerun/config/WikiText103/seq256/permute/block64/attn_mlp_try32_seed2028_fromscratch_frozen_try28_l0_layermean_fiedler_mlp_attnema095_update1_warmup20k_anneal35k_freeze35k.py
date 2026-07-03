# WikiText103 seq256 permuted block64 try_32:
# Seed ablation of try_31 with seed=2028. Same frozen try_28 MLP, attention EMA
# input, every-step updates, warmup20k, anneal20k-35k, then fixed.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2028
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try32-seed2028-fromscratch-frozen-try28-l0-layermean-fiedler-mlp-attnema095-update1-warmup20k-anneal35k-freeze35k-b64-permute-block'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try32-seed2028-fromscratch-frozen-try28-l0-layermean-fiedler-mlp-attnema095-update1-warmup20k-anneal35k-freeze35k-b64-permute'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = False
attn_mlp_policy_path = 'checkpoints/attn_mlp_distillation/try28_joint_try20_try24_l0_layermean_fiedler_mlp_h2048_1024/best_by_val_tau.pt'
attn_mlp_policy_freeze = True
attn_mlp_policy_train_loss = 'none'

# L0 all-head mean, without_none, matching the try_28 distilled student.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = -1
attn_mlp_policy_export_type = 'without_none'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_feature_clip = 0.0
attn_mlp_policy_input_normalization = 'zscore'

attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'

attn_mlp_policy_start_iter = 20000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 20000
attn_mlp_policy_anneal_end_iter = 35000
attn_mlp_policy_collect_warmup_attention = False
attn_mlp_policy_update_every = 1
attn_mlp_policy_update_stop_iter = 35000
attn_mlp_policy_log_interval = 50
attn_mlp_policy_save_orders = True

# Restore the older MLP-input attention EMA for this controlled ablation.
attn_mlp_policy_attention_ema_enabled = True
attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_logits_ema_enabled = False

attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = 'try32_seed2028_frozen_try28_l0_layermean_fiedler_mlp_attnema095_update1_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/language/wikitext103/mlp/distillation/try_32/input_attn'
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0

# Keep this run a pure frozen-distillation backbone test.
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
online_attn_probe_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
head_asym_selector_enabled = False
head_signal_consistency_enabled = False
head_signal_position_regularizer_enabled = False
