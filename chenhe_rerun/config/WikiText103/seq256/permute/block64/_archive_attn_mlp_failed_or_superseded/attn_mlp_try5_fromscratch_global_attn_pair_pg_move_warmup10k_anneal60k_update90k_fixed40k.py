# WikiText103 seq256 permuted block64 try_5:
# from-scratch backbone + from-scratch trainable Attn-MLP.
#
# New hard target:
# - CheckpointAttnMLPOrder PPL must beat the random checkpoint OriginalL2R
#   full sweep baseline, about 31.6167.
# - Learned cached order post-hoc original-frame tau must be > 0.4.
#
# Training still uses no OriginalL2R/tau/permutation oracle signal. The new
# attention-pair terms consume only current-frame all-layer/all-head attention.

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try5-fromscratch-global-attn-pair-pg-move-warmup10k-anneal60k-update90k-fixed40k-b64-permute-block'
eval_interval = 1000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try5-fromscratch-global-attn-pair-pg-move-warmup10k-anneal60k-update90k-fixed40k-b64-permute'

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
max_iters = 130000
lr_decay_iters = 130000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False

# Key change from try_4: use all layers/all heads instead of layer-2 mean heads.
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
attn_mlp_policy_anneal_end_iter = 60000
attn_mlp_policy_collect_warmup_attention = True

attn_mlp_policy_ema_decay = 0.97
attn_mlp_policy_update_every = 8
attn_mlp_policy_update_stop_iter = 90000
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True

attn_mlp_policy_lr = 2e-5
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_train_loss = 'sampled_nll_pg_move_pref'

attn_mlp_policy_pair_weight = 0.0
attn_mlp_policy_pair_tau = 1.0
attn_mlp_policy_pair_margin = 0.25
attn_mlp_policy_pair_max_weight = 4.0
attn_mlp_policy_axis_weight = 0.0
attn_mlp_policy_axis_sign = 'easy_first'
attn_mlp_policy_axis_target_scale = 1.0

attn_mlp_policy_sampled_orders_per_state = 2
attn_mlp_policy_random_baseline_orders = 1
attn_mlp_policy_nll_states_per_update = 1
attn_mlp_policy_pg_weight = 0.3
attn_mlp_policy_prefix_reward_weight = 0.02
attn_mlp_policy_prefix_k = 8
attn_mlp_policy_reward_scale_floor = 0.001
attn_mlp_policy_advantage_clip = 5.0

# Broader current-loss preferences than try_4: sample global swaps, still from
# current-model loss comparisons only.
attn_mlp_policy_move_pref_weight = 4.0
attn_mlp_policy_move_pref_pairs_per_state = 16
attn_mlp_policy_move_pref_window = 63
attn_mlp_policy_move_pref_tau = 3.0
attn_mlp_policy_move_pref_margin = 0.02
attn_mlp_policy_move_pref_max_weight = 6.0
attn_mlp_policy_move_pref_prefix_weight = 0.02

# New no-prior attention-pair regularization. Directed pair loss: if current
# all-head attention A[i,j] is large, prefer block j earlier than block i.
# Close loss: high-attention pairs should be close in soft rank.
attn_mlp_policy_attn_pair_weight = 5.0
attn_mlp_policy_attn_pair_top_frac = 0.10
attn_mlp_policy_attn_pair_min_z = 0.0
attn_mlp_policy_attn_pair_max_weight = 8.0
attn_mlp_policy_attn_pair_tau = 2.0
attn_mlp_policy_attn_close_weight = 1.0
attn_mlp_policy_attn_close_tau = 4.0
attn_mlp_policy_attn_close_margin = 0.05

attn_mlp_policy_logit_l2 = 1e-4
attn_mlp_policy_min_logit_std = 1.0
attn_mlp_policy_std_floor_weight = 0.05
attn_mlp_policy_min_entropy = 2.2
attn_mlp_policy_max_entropy = 3.8
attn_mlp_policy_entropy_floor_weight = 0.08
attn_mlp_policy_entropy_ceiling_weight = 0.02
