# WikiText103 seq256 permuted block64 try_13:
# from-scratch backbone + from-scratch trainable Attn-MLP.
#
# try_13 keeps the no-prior rule. OriginalL2R, original tau, oracle
# permutations, validation PPL, and hand-written original-order labels are not
# used for training, reranking, early stopping, or model selection. Tau is
# reported only after/in validation diagnostics.
#
# Method change vs try_9/try_12:
# each policy update evaluates the current MLP MAP/sample/random candidates
# with current-model train-split full/prefix loss, chooses the best candidate
# inside that on-policy pool, and adds a dense pairwise BCE objective that
# increases the MLP logits' probability of that loss-selected candidate. This
# is a current-frame internal loss signal, not attention-order imitation and
# not original-frame supervision.

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try13-fromscratch-bestpref-earlyfreeze-50k-b64-permute-block'
eval_interval = 1000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try13-fromscratch-bestpref-earlyfreeze-50k-b64-permute'

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

# 0-5k random/shadow MLP, 5k-15k anneal, 15k-20k 100% MLP while still
# trainable, then 20k-50k fixed cached order for long backbone adaptation.
attn_mlp_policy_start_iter = 5000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 5000
attn_mlp_policy_anneal_end_iter = 15000
attn_mlp_policy_collect_warmup_attention = True

attn_mlp_policy_ema_decay = 0.95
attn_mlp_policy_update_every = 4
attn_mlp_policy_update_stop_iter = 20000
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True

attn_mlp_policy_lr = 3e-5
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

attn_mlp_policy_sampled_orders_per_state = 4
attn_mlp_policy_random_baseline_orders = 2
attn_mlp_policy_nll_states_per_update = 1
attn_mlp_policy_pg_weight = 0.2
attn_mlp_policy_prefix_reward_weight = 0.05
attn_mlp_policy_prefix_k = 8
attn_mlp_policy_reward_scale_floor = 0.001
attn_mlp_policy_advantage_clip = 5.0

attn_mlp_policy_move_pref_weight = 4.0
attn_mlp_policy_move_pref_pairs_per_state = 8
attn_mlp_policy_move_pref_window = 4
attn_mlp_policy_move_pref_tau = 2.0
attn_mlp_policy_move_pref_margin = 0.02
attn_mlp_policy_move_pref_max_weight = 8.0
attn_mlp_policy_move_pref_prefix_weight = 0.05

attn_mlp_policy_global_pref_weight = 0.0
attn_mlp_policy_global_pref_proposals = 0
attn_mlp_policy_global_pref_start_iter = 0
attn_mlp_policy_global_pref_tau = 1.0
attn_mlp_policy_global_pref_margin = 0.02
attn_mlp_policy_global_pref_max_weight = 8.0
attn_mlp_policy_global_pref_prefix_weight = 0.0

# New on-policy best-of-candidates objective. Candidate selection uses only
# train-split current-model loss. The selected candidate is converted into
# dense pairwise preferences for the MLP logits.
attn_mlp_policy_best_pref_weight = 1.0
attn_mlp_policy_best_pref_pair_weight = 1.0
attn_mlp_policy_best_pref_pl_weight = 0.0
attn_mlp_policy_best_pref_random = 0
attn_mlp_policy_best_pref_local_swaps = 0
attn_mlp_policy_best_pref_tau = 2.0
attn_mlp_policy_best_pref_margin = 0.02
attn_mlp_policy_best_pref_max_weight = 4.0
attn_mlp_policy_best_pref_prefix_weight = 0.10

attn_mlp_policy_attn_pair_weight = 0.75
attn_mlp_policy_attn_pair_top_frac = 0.10
attn_mlp_policy_attn_pair_min_z = 0.0
attn_mlp_policy_attn_pair_max_weight = 4.0
attn_mlp_policy_attn_pair_tau = 2.0
attn_mlp_policy_attn_close_weight = 0.0
attn_mlp_policy_attn_close_tau = 4.0
attn_mlp_policy_attn_close_margin = 0.0

attn_mlp_policy_logit_l2 = 2e-4
attn_mlp_policy_min_logit_std = 1.0
attn_mlp_policy_std_floor_weight = 0.05
attn_mlp_policy_min_entropy = 3.0
attn_mlp_policy_max_entropy = 3.8
attn_mlp_policy_entropy_floor_weight = 0.1
attn_mlp_policy_entropy_ceiling_weight = 0.02

attn_mlp_policy_cache_rerank_enabled = False
