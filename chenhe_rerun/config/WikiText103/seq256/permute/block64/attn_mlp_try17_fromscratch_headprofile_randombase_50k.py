# WikiText103 seq256 permuted block64 try_17:
# try15 head-profile no-prior MLP-order method, with the non-policy training
# configuration inherited directly from the pure Random block64 base.
#
# Alignment rule:
# This file first executes random.py and then overrides only the fields needed
# to attach the trainable Attn-MLP/head-profile policy and to use a distinct
# output directory. In particular, random.py's batch/model/LR/max_iters,
# eval_interval, W&B default, and compile default are not redefined here.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, or model selection. Tau is logged only as a diagnostic.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try17-fromscratch-headprofile-randombase-50k-b64-permute-block'

# Attach the same trainable Attn-MLP order path used by try15.
aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False

attn_mlp_policy_layer = 2
attn_mlp_policy_head = -1
attn_mlp_policy_use_global = False
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_feature_mode = 'try20'
attn_mlp_policy_feature_clip = 8.0
attn_mlp_policy_input_normalization = 'none'
attn_mlp_policy_input_channels = 3
attn_mlp_policy_hidden_dims = '1024,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_activation = 'gelu'

# Same schedule as try15:
# 0-5k random/shadow MLP, 5k-20k anneal, 20k-30k 100% MLP while still
# trainable, then 30k-50k fixed MLP argsort order for backbone adaptation.
attn_mlp_policy_start_iter = 5000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 5000
attn_mlp_policy_anneal_end_iter = 20000
attn_mlp_policy_collect_warmup_attention = True

attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_update_every = 4
attn_mlp_policy_update_stop_iter = 30000
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True

attn_mlp_policy_lr = 4e-5
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_train_loss = 'sampled_nll_pg_move_pref'

attn_mlp_policy_sampled_orders_per_state = 4
attn_mlp_policy_random_baseline_orders = 2
attn_mlp_policy_nll_states_per_update = 1
attn_mlp_policy_pg_weight = 0.05
attn_mlp_policy_prefix_reward_weight = 0.05
attn_mlp_policy_prefix_k = 8
attn_mlp_policy_reward_scale_floor = 0.001
attn_mlp_policy_advantage_clip = 5.0

attn_mlp_policy_move_pref_weight = 1.0
attn_mlp_policy_move_pref_pairs_per_state = 4
attn_mlp_policy_move_pref_window = 4
attn_mlp_policy_move_pref_tau = 2.0
attn_mlp_policy_move_pref_margin = 0.02
attn_mlp_policy_move_pref_max_weight = 4.0
attn_mlp_policy_move_pref_prefix_weight = 0.05

attn_mlp_policy_attn_pair_weight = 0.25
attn_mlp_policy_attn_pair_top_frac = 0.10
attn_mlp_policy_attn_pair_min_z = 0.0
attn_mlp_policy_attn_pair_max_weight = 4.0
attn_mlp_policy_attn_pair_tau = 2.0
attn_mlp_policy_attn_close_weight = 0.0
attn_mlp_policy_attn_close_tau = 4.0
attn_mlp_policy_attn_close_margin = 0.0

attn_mlp_policy_head_profile_weight = 2.0
attn_mlp_policy_head_profile_start_iter = 5000
attn_mlp_policy_head_profile_every = 1000
attn_mlp_policy_head_profile_tau = 2.0
attn_mlp_policy_head_profile_min_abs_q = 0.05
attn_mlp_policy_head_profile_max_weight = 4.0
attn_mlp_policy_head_profile_use_cached = True

attn_mlp_policy_logit_l2 = 5e-4
attn_mlp_policy_min_logit_std = 1.0
attn_mlp_policy_std_floor_weight = 0.05
attn_mlp_policy_min_entropy = 3.0
attn_mlp_policy_max_entropy = 3.8
attn_mlp_policy_entropy_floor_weight = 0.1
attn_mlp_policy_entropy_ceiling_weight = 0.02

# Head-profile target settings. They are reused by the MLP auxiliary loss and
# intentionally do not use original order, tau, validation loss, or position
# anchors as selection signals.
head_signal_probe_heads = 'all'
head_signal_probe_batches = 8
head_signal_probe_batch_size = 16
head_signal_probe_loss_batches = 2
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'with_none'
head_signal_probe_fixed_angle_idx = 4
head_signal_probe_top_m = 12
head_signal_probe_num_components = 4
head_signal_probe_component_pairs = '1-2'
head_signal_probe_num_angles = 16
head_signal_probe_k_values = '8,10'
head_signal_probe_group_methods = 'gap'
head_signal_probe_threshold_percentile = 60.0
head_signal_probe_transform = 'relu'
head_signal_probe_temperature = 1.0
head_signal_probe_direction_lambdas = '0,0.1,0.25'
head_signal_probe_directed_score_weight = 0.25
head_signal_probe_band_quality_weight = 0.05
head_signal_probe_score_adjacency_sym = 'max'
head_signal_probe_consensus_enabled = True
head_signal_probe_consensus_leave_one_out = True
head_signal_probe_deterministic = False
head_signal_probe_orientation_rule = 'linear_profile_consensus_candidate'
head_signal_probe_candidate_loss_profile_enabled = True
head_signal_probe_candidate_loss_profile_score = 'linear_profile'
head_signal_probe_candidate_loss_profile_max_rank = 4
head_signal_probe_candidate_loss_profile_include_reverse = True
head_signal_probe_candidate_loss_profile_exp_tau = 16.0
head_signal_probe_candidate_loss_profile_min_gap = 0.005
head_signal_probe_candidate_loss_profile_min_alignment = 0.0
head_signal_probe_candidate_loss_profile_low_confidence_fallback = 'global_q'
head_signal_probe_position_anchor_enabled = False
head_signal_probe_position_consensus_enabled = False
