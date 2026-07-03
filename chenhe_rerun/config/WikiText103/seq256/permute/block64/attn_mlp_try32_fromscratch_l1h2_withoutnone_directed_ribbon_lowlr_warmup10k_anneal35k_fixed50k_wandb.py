# WikiText103 seq256 permuted block64 try_32:
# From-scratch version of the raw L1H2 without_none directed-ribbon Attn-MLP
# method. Train Random-only for 10k iters, anneal Random/MLP usage from 10k to
# 35k, then use the fixed cached MLP order until 50k. No checkpoint resume.
#
# No-prior rule:
# OriginalL2R, original tau, validation PPL, oracle permutations, distribution
# orders, and hand-written original-order labels are not used for training,
# reranking, early stopping, runtime target acceptance, or selection. Tau is a
# post-hoc diagnostic only.

_base_config = 'config/WikiText103/seq256/permute/block64/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try32-fromscratch-l1h2-withoutnone-directed-ribbon-lowlr-warmup10k-anneal35k-fixed50k-b64-permute-block'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try32-fromscratch-l1h2-withoutnone-directed-ribbon-lowlr-warmup10k-anneal35k-fixed50k-b64-permute'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False

# Same current-frame, non-symmetric input family as try31.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 1
attn_mlp_policy_head = 2
attn_mlp_policy_export_type = 'without_none'
attn_mlp_policy_feature_mode = 'attention'
attn_mlp_policy_input_channels = 1
attn_mlp_policy_feature_clip = 0.0
attn_mlp_policy_input_normalization = 'none'

attn_mlp_policy_hidden_dims = '1024,1024'
attn_mlp_policy_dropout = 0.0
attn_mlp_policy_activation = 'gelu'
attn_mlp_policy_lr = 1e-4
attn_mlp_policy_weight_decay = 0.01
attn_mlp_policy_beta1 = 0.9
attn_mlp_policy_beta2 = 0.99
attn_mlp_policy_grad_clip = 1.0
attn_mlp_policy_order_mode = 'argsort_desc'
attn_mlp_policy_per_sample = False
attn_mlp_policy_fallback = 'random'

# 0..10k: Random warmup only. 10k..35k: Random/MLP usage anneal.
# 35k..50k: frozen cached MLP order.
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
attn_mlp_policy_log_input_attn_prefix = 'try32_l1h2_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/try_32/input_attn'
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0

attn_mlp_policy_train_loss = 'directed_ribbon'
attn_mlp_policy_directed_ribbon_flow_sign = -1.0
attn_mlp_policy_directed_ribbon_rank_tau = 1.0
attn_mlp_policy_directed_ribbon_margin = 1.0
attn_mlp_policy_directed_ribbon_band_width = 8.0
attn_mlp_policy_directed_ribbon_band_weight = 0.05

attn_mlp_policy_logit_l2 = 1e-4
attn_mlp_policy_min_logit_std = 1.0
attn_mlp_policy_std_floor_weight = 0.01
attn_mlp_policy_min_entropy = 0.0
attn_mlp_policy_max_entropy = 10.0
attn_mlp_policy_entropy_floor_weight = 0.0
attn_mlp_policy_entropy_ceiling_weight = 0.0

# Lower MLP LR than try31. LR annealing is training-internal and does not use
# validation, tau, or original-frame diagnostics.
attn_mlp_policy_lr_anneal_enabled = True
attn_mlp_policy_lr_anneal_start_iter = 10000
attn_mlp_policy_lr_anneal_end_iter = 35000
attn_mlp_policy_lr_anneal_min_lr = 1e-5
attn_mlp_policy_lr_anneal_style = 'cosine'

# MLP-only loss-stop. If triggered, only MLP optimizer steps stop; attention EMA
# and cached-order refresh continue until update_stop_iter=35000.
attn_mlp_policy_loss_stop_enabled = True
attn_mlp_policy_loss_stop_metric = 'attn_mlp_train_loss'
attn_mlp_policy_loss_stop_start_iter = 25000
attn_mlp_policy_loss_stop_patience = 5
attn_mlp_policy_loss_stop_min_delta = 0.001
attn_mlp_policy_loss_stop_check_every = 1

# Keep the method simple: disable old auxiliary objectives.
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
head_signal_probe_enabled = False
head_direction_logger_enabled = False
