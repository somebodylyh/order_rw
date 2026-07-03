# WikiText103 seq256 permuted block64 try_48:
# Paired seed with try_46, but maintain an EMA over the MLP continuous scores.

import os

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2029
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try48-seed2029-fromscratch-frozen-try33-score-mlp-scoreema095-update1-warmup15k-anneal35k-freeze35k-b64-permute-block'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True
compile = False

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try48-seed2029-fromscratch-frozen-try33-score-mlp-scoreema095-update1-warmup15k-anneal35k-freeze35k-b64-permute'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = False
attn_mlp_policy_path = 'checkpoints/attn_mlp_distillation/try33_joint_try20_try24_l0_layermean_fiedler_score_mlp_h2048_1024/best_by_val_tau.pt'
attn_mlp_policy_freeze = True
attn_mlp_policy_train_loss = 'none'

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

warmup_until = 15000
anneal_start = 15000
anneal_end = 35000
fixed_after = 35000

attn_mlp_policy_start_iter = warmup_until
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = anneal_start
attn_mlp_policy_anneal_end_iter = anneal_end
attn_mlp_policy_collect_warmup_attention = False
attn_mlp_policy_update_every = 1
attn_mlp_policy_update_stop_iter = fixed_after
attn_mlp_policy_log_interval = 50
attn_mlp_policy_save_orders = True

mlp_frozen = True
attn_mlp_policy_attention_ema_enabled = False
attn_mlp_policy_ema_decay = 0.0

# Continuous-score EMA: sigmoid(MLP logits) -> EMA -> argsort_desc.
# This mirrors the teacher priority_ema path and does not smooth attention
# matrices or discrete orders.
attn_mlp_policy_logits_ema_enabled = True
attn_mlp_policy_logits_ema_decay = 0.95
attn_mlp_policy_logits_ema_normalize = 'sigmoid'
attn_mlp_policy_logits_ema_start_iter = warmup_until

attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = 'try48_seed2029_fromscratch_frozen_try33_score_mlp_scoreema095_update1_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/language/wikitext103/mlp/distillation/try_48/input_attn'
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

online_spectral_policy_enabled = False
online_spectral_enabled = False
online_attn_probe_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
head_asym_selector_enabled = False
head_signal_consistency_enabled = False
head_signal_position_regularizer_enabled = False
