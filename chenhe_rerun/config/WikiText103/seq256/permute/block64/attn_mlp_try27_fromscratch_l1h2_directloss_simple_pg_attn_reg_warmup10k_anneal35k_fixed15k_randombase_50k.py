# WikiText103 seq256 permuted block64 try_27:
# Simplified no-prior trainable Attn-MLP using the strongest head from the
# distribution/direct_asym_eig evidence.
#
# Controlled simplification:
# - Direct MLP input head: L1H2, with_none, matching the successful
#   direct_asym_eig distribution run.
# - Feature tensor keeps only channel 0 and channel 2 from the try20 feature
#   stack: directed attention robust-z and loss_diff. The symmetric attention
#   channel is removed.
# - Loss keeps only sampled full-loss policy gradient, attention-pair score,
#   and logit regularization. Head-profile, move preference, global preference,
#   best preference, cache rerank, and logits EMA are disabled.
# - No temporal EMA smoothing is used for MLP inputs or output logits:
#   attn_mlp_policy_ema_decay = 0.0 and logits EMA is off.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, runtime target acceptance, or selection. Tau is diagnostic
# only after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try17_fromscratch_headprofile_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try27-fromscratch-l1h2-directloss-simple-pg-attn-reg-warmup10k-anneal35k-fixed15k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try27-fromscratch-l1h2-directloss-simple-pg-attn-reg-warmup10k-anneal35k-fixed15k-randombase-50k-b64-permute'

# Direct input: distribution-effective single head.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 1
attn_mlp_policy_head = 2
attn_mlp_policy_export_type = 'with_none'

# Two-channel input: [directed attention, loss_diff].
attn_mlp_policy_feature_mode = 'attention_loss_direct'
attn_mlp_policy_input_channels = 2
attn_mlp_policy_feature_clip = 8.0
attn_mlp_policy_input_normalization = 'none'

# Fresh trainable MLP. Keep capacity moderate because the input/loss are simpler.
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False
attn_mlp_policy_hidden_dims = '1024,1024'
attn_mlp_policy_lr = 4e-5
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_order_mode = 'argsort_desc'

# Schedule: 10k random warmup, 10k..35k anneal, 35k..50k fixed cached order.
attn_mlp_policy_start_iter = 10000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 10000
attn_mlp_policy_anneal_end_iter = 35000
attn_mlp_policy_collect_warmup_attention = True
attn_mlp_policy_update_every = 4
attn_mlp_policy_update_stop_iter = 35000
attn_mlp_policy_fallback = 'random'
attn_mlp_policy_log_interval = 100
attn_mlp_policy_save_orders = True

# No temporal EMA smoothing. The variable name remains A_ema in train.py, but
# decay 0.0 makes it equal to the newest collected feature tensor.
attn_mlp_policy_ema_decay = 0.0
attn_mlp_policy_logits_ema_enabled = False

# Simple loss: policy gradient + attention pair score + regularization.
attn_mlp_policy_train_loss = 'sampled_nll_pg'
attn_mlp_policy_sampled_orders_per_state = 8
attn_mlp_policy_random_baseline_orders = 4
attn_mlp_policy_nll_states_per_update = 1
attn_mlp_policy_pg_weight = 0.10
attn_mlp_policy_prefix_reward_weight = 0.0
attn_mlp_policy_prefix_k = 0
attn_mlp_policy_reward_scale_floor = 0.001
attn_mlp_policy_advantage_clip = 5.0

attn_mlp_policy_move_pref_weight = 0.0
attn_mlp_policy_move_pref_pairs_per_state = 0
attn_mlp_policy_head_profile_weight = 0.0
attn_mlp_policy_head_profile_every = 0

attn_mlp_policy_attn_pair_weight = 0.50
attn_mlp_policy_attn_pair_top_frac = 0.10
attn_mlp_policy_attn_pair_min_z = 0.0
attn_mlp_policy_attn_pair_max_weight = 4.0
attn_mlp_policy_attn_pair_tau = 2.0
attn_mlp_policy_attn_close_weight = 0.0

# Regularization to keep logits useful without forcing an external order.
attn_mlp_policy_logit_l2 = 5e-4
attn_mlp_policy_min_logit_std = 1.0
attn_mlp_policy_std_floor_weight = 0.05
attn_mlp_policy_min_entropy = 3.0
attn_mlp_policy_max_entropy = 3.8
attn_mlp_policy_entropy_floor_weight = 0.10
attn_mlp_policy_entropy_ceiling_weight = 0.02

# Head-profile machinery is intentionally off in this try. Keep these fields
# harmless in case inherited logging inspects them.
head_signal_probe_enabled = False
head_signal_probe_heads = '1:2'
head_signal_probe_export_type = 'with_none'
