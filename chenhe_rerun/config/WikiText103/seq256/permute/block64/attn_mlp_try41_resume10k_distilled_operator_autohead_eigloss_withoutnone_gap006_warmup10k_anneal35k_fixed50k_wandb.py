# WikiText103 seq256 permuted block64 try_41:
# Resume from the clean Random 10k checkpoint, select one head at iter 10k by
# current-frame asymmetry top-k + direct-asym-eig raw/reverse train-loss prior,
# then drive the order with a frozen operator-distilled Attn-MLP.
#
# This is a controlled distillation test, not an end-to-end MLP training run.
# The MLP checkpoint is try10, trained on try8 all-head data with
# loss_score_gap >= 0.06. The policy input is the selected head's without_none
# attention matrix; larger MLP logits mean earlier blocks.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try41-resume10k-distilled-operator-autohead-eigloss-withoutnone-gap006-warmup10k-anneal35k-fixed50k-b64-permute-block'

init_from = 'resume'
resume_optimizer_state = True
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try41-resume10k-distilled-operator-autohead-eigloss-withoutnone-gap006-warmup10k-anneal35k-fixed50k-b64-permute'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = False
attn_mlp_policy_path = 'checkpoints/attn_mlp_distillation/try10_high_conf_operator_mlp_gap006/best_by_val_tau.pt'
attn_mlp_policy_freeze = True
attn_mlp_policy_train_loss = 'none'

# Placeholder until head_asym_selector assigns the selected head at iter 10000.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = 0
attn_mlp_policy_export_type = 'without_none'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_feature_clip = 0.0
attn_mlp_policy_input_normalization = 'zscore'

attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'

# 10k..35k Random/MLP anneal, then fixed cached MLP order.
attn_mlp_policy_start_iter = 10000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 10000
attn_mlp_policy_anneal_end_iter = 35000
attn_mlp_policy_collect_warmup_attention = False
attn_mlp_policy_update_every = 20
attn_mlp_policy_update_stop_iter = 35000
attn_mlp_policy_log_interval = 50
attn_mlp_policy_save_orders = True

attn_mlp_policy_ema_decay = 0.995
attn_mlp_policy_logits_ema_enabled = False

attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = 'try41_distilled_operator_autohead_gap006_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_distillation/try_12/main_block64/input_attn'
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0

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

head_asym_selector_enabled = True
head_asym_selector_target_iter = 10000
head_asym_selector_split = 'train'
head_asym_selector_batch_size = 32
head_asym_selector_batches = 4
head_asym_selector_export_type = 'without_none'
head_asym_selector_out_dir = 'Report/MLP_distillation/try_12/main_block64/head_asym_selector'
head_asym_selector_wandb_prefix = 'head_asym_selector'
head_asym_selector_log_wandb = True
head_asym_selector_save_matrices = True
head_asym_selector_save_all_head_maps = True
head_asym_selector_assign_to_attn_mlp_policy = True
head_asym_selector_init_attn_mlp_policy = False
head_asym_selector_restore_rng = True
head_asym_selector_deterministic = True
head_asym_selector_seed = 24681357

head_asym_selector_selection_rule = 'asym_topk_direct_asym_eig_loss'
head_asym_selector_asym_top_k = 8
head_asym_selector_direct_asym_eig_mode = 'raw_right_largest_real_real'
head_asym_selector_loss_batches = 4
head_asym_selector_loss_batch_size = 16
head_asym_selector_loss_candidate_batch_size = 8
head_asym_selector_loss_prefix_k = 16
head_asym_selector_loss_score = 'linear_profile'
head_asym_selector_loss_exp_tau = 16.0

online_spectral_policy_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
