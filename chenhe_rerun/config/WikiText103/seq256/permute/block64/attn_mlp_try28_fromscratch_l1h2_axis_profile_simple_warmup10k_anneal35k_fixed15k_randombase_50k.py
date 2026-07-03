# WikiText103 seq256 permuted block64 try_28:
# Minimal no-prior trainable Attn-MLP that keeps only the two principles that
# worked in the distribution/direct_asym_eig line:
# 1. Use the strongly directional L1H2 with_none attention head as the input
#    signal.
# 2. Use current-model reveal/profile loss only to choose the sign of the
#    learned axis.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, distribution
# orders, and hand-written original-order labels are not used for training,
# reranking, early stopping, runtime target acceptance, or selection. Tau is a
# post-hoc diagnostic only.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try28-fromscratch-l1h2-axis-profile-simple-warmup10k-anneal35k-fixed15k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try28-fromscratch-l1h2-axis-profile-simple-warmup10k-anneal35k-fixed15k-randombase-50k-b64-permute'

# Attach a fresh trainable Attn-MLP policy to the random-base training recipe.
aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False

# Direct input: only the distribution-effective L1H2 head.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 1
attn_mlp_policy_head = 2
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_feature_mode = 'attention_direct'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_feature_clip = 8.0
attn_mlp_policy_input_normalization = 'none'

# Same moderate policy capacity used in the recent Attn-MLP line.
attn_mlp_policy_hidden_dims = '1024,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_activation = 'gelu'
attn_mlp_policy_lr = 4e-5
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'

# Schedule: 0-10k random warmup/shadow collection, 10k..35k annealed MLP use,
# 35k..50k fixed cached MLP order for backbone adaptation.
attn_mlp_policy_start_iter = 10000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 10000
attn_mlp_policy_anneal_end_iter = 35000
attn_mlp_policy_collect_warmup_attention = True
attn_mlp_policy_update_every = 20
attn_mlp_policy_update_stop_iter = 35000
attn_mlp_policy_log_interval = 50
attn_mlp_policy_save_orders = True

# No temporal smoothing in try28: the MLP sees the latest collected L1H2 matrix,
# and the fixed order is direct argsort over current logits.
attn_mlp_policy_ema_decay = 0.0
attn_mlp_policy_logits_ema_enabled = False
attn_mlp_policy_log_input_attn = True
attn_mlp_policy_log_input_attn_interval = 1000
attn_mlp_policy_log_input_attn_prefix = 'try28_l1h2_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/try_28/input_attn'
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0

# Minimal loss:
# - axis term: align logits_i - logits_j with the antisymmetric L1H2 signal up
#   to sign, using only high-signal pairs.
# - direction term: compare the MLP order with its reverse under current-model
#   linear loss profile, then orient logits toward the lower-profile direction.
attn_mlp_policy_train_loss = 'axis_profile_simple'
attn_mlp_policy_axis_profile_weight = 1.0
attn_mlp_policy_axis_profile_dir_weight = 0.25
attn_mlp_policy_axis_profile_dir_margin = 0.003
attn_mlp_policy_axis_profile_min_abs_q = 0.05
attn_mlp_policy_axis_profile_score = 'linear_profile'
attn_mlp_policy_nll_states_per_update = 1
attn_mlp_policy_prefix_k = 16

# Disable the heavier try15/17/27 auxiliary machinery.
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

# Regularization: keep logits ordered enough to be useful, but do not impose an
# external order.
attn_mlp_policy_logit_l2 = 5e-4
attn_mlp_policy_min_logit_std = 1.0
attn_mlp_policy_std_floor_weight = 0.05
attn_mlp_policy_min_entropy = 3.0
attn_mlp_policy_max_entropy = 3.8
attn_mlp_policy_entropy_floor_weight = 0.10
attn_mlp_policy_entropy_ceiling_weight = 0.02

# Keep unrelated head-signal probes off. L1H2 is used as an input signal, not as
# a teacher order.
head_signal_probe_enabled = False
head_signal_probe_heads = '1:2'
head_signal_probe_export_type = 'with_none'
