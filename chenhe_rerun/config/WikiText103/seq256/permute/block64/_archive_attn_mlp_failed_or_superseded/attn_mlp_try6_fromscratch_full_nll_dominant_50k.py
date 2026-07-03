# WikiText103 seq256 permuted block64 try_6:
# from-scratch backbone + from-scratch trainable Attn-MLP.
#
# Hard target:
# - CheckpointAttnMLPOrder PPL must beat the random checkpoint OriginalL2R
#   200-batch baseline, about 31.6167.
# - Learned cached order post-hoc original-frame tau must be > 0.4.
#
# Training uses no OriginalL2R/tau/permutation-oracle signal. Original-frame
# metrics are diagnostics only.

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try6-fromscratch-full-nll-dominant-warmup10k-anneal35k-update40k-fixed10k-b64-permute-block'
eval_interval = 1000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try6-fromscratch-full-nll-dominant-warmup10k-anneal35k-update40k-fixed10k-b64-permute'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0
compile = False

block_order_block_len = 4

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False

# Keep broad all-layer/all-head features from try_5, but remove the direct
# attention-pair objective that destabilized late training.
attn_mlp_policy_layer = 2
attn_mlp_policy_head = -1
attn_mlp_policy_use_global = True
attn_mlp_policy_export_type = 'with_none'
attn_mlp_policy_feature_mode = 'try20'
attn_mlp_policy_feature_clip = 8.0
attn_mlp_policy_input_normalization = 'none'
attn_mlp_policy_input_channels = 3
attn_mlp_policy_hidden_dims = '1024,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_activation = 'gelu'

attn_mlp_policy_start_iter = 10000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 10000
attn_mlp_policy_anneal_end_iter = 35000
attn_mlp_policy_collect_warmup_attention = True

attn_mlp_policy_ema_decay = 0.99
attn_mlp_policy_update_every = 16
attn_mlp_policy_update_stop_iter = 40000
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True

attn_mlp_policy_lr = 5e-5
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_train_loss = 'sampled_nll_pg'

attn_mlp_policy_pair_weight = 0.0
attn_mlp_policy_pair_tau = 1.0
attn_mlp_policy_pair_margin = 0.25
attn_mlp_policy_pair_max_weight = 4.0
attn_mlp_policy_axis_weight = 0.0
attn_mlp_policy_axis_sign = 'easy_first'
attn_mlp_policy_axis_target_scale = 1.0

# Full-NLL-dominant policy gradient. More samples and random baselines reduce
# the reward noise versus try_5; prefix and move-preference signals are disabled
# for this cleaner 50k retry.
attn_mlp_policy_sampled_orders_per_state = 4
attn_mlp_policy_random_baseline_orders = 4
attn_mlp_policy_nll_states_per_update = 2
attn_mlp_policy_pg_weight = 1.0
attn_mlp_policy_prefix_reward_weight = 0.0
attn_mlp_policy_prefix_k = 8
attn_mlp_policy_reward_scale_floor = 0.001
attn_mlp_policy_advantage_clip = 5.0

attn_mlp_policy_move_pref_weight = 0.0
attn_mlp_policy_move_pref_pairs_per_state = 0
attn_mlp_policy_move_pref_window = 1
attn_mlp_policy_move_pref_tau = 3.0
attn_mlp_policy_move_pref_margin = 0.02
attn_mlp_policy_move_pref_max_weight = 6.0
attn_mlp_policy_move_pref_prefix_weight = 0.0

attn_mlp_policy_attn_pair_weight = 0.0
attn_mlp_policy_attn_pair_top_frac = 0.10
attn_mlp_policy_attn_pair_min_z = 0.0
attn_mlp_policy_attn_pair_max_weight = 8.0
attn_mlp_policy_attn_pair_tau = 2.0
attn_mlp_policy_attn_close_weight = 0.0
attn_mlp_policy_attn_close_tau = 4.0
attn_mlp_policy_attn_close_margin = 0.0

attn_mlp_policy_logit_l2 = 1e-4
attn_mlp_policy_min_logit_std = 0.8
attn_mlp_policy_std_floor_weight = 0.03
attn_mlp_policy_min_entropy = 2.7
attn_mlp_policy_max_entropy = 3.9
attn_mlp_policy_entropy_floor_weight = 0.04
attn_mlp_policy_entropy_ceiling_weight = 0.01
