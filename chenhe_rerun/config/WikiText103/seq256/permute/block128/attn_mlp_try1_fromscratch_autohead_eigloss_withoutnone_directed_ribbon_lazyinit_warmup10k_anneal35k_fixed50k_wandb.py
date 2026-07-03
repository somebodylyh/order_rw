# WikiText103 seq256 permuted block128 try_1:
# Block128 port of the latest no-prior Attn-MLP method.
#
# Base training shape is inherited from seq256/permute/block128/random.py:
# - block_size=256, block_order_block_len=2, num_blocks=128
# - batch_size=64, gradient_accumulation_steps=2
# - permute_data=True, permute_seed=42, permute_mode='block'
# - lr/max_iters/eval cadence follow the block128 random base.
#
# Schedule:
# - 0..10k: pure Random warmup. The Attn-MLP policy is lazy-initialized, so
#   MLP parameters and optimizer are not created before the selector fires.
# - iter 10k: select one head from current-frame without_none attention:
#     1) rank all heads by asymmetry
#     2) keep the top-k asymmetric heads
#     3) recover a direct_asym_eig order for each candidate head
#     4) orient raw/reverse by current-model training loss
#     5) choose the lowest-loss candidate
# - 10k..35k: train the MLP/order with directed-ribbon loss.
# - 35k..50k: freeze cached MLP order and continue backbone training.
#
# No-prior rule:
# Original-frame upper/lower triangles, OriginalL2R, original tau, validation
# loss, oracle permutations, and distribution-produced orders are not used for
# head selection, MLP training, reranking, or early stopping. Tau remains a
# post-hoc diagnostic only.

_base_config = 'config/WikiText103/seq256/permute/block128/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block128/out-wikitext103-seq256-try1-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup10k-anneal35k-fixed50k-b128-permute-block'

init_from = 'scratch'
resume_optimizer_state = False
max_iters = 50000
lr_decay_iters = 50000
always_save_checkpoint = True

wandb_log = True
wandb_project = 'AOGPT-order-block-128-final'
wandb_run_name = 'seq256-try1-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup10k-anneal35k-fixed50k-b128-permute'

aogpt_train_mode = 'AttnMLPFrozenOrder'
main_eval_mode = 'Random'
generalization_eval_mode = ''

attn_mlp_policy_enabled = True
attn_mlp_policy_random_init = True
attn_mlp_policy_path = ''
attn_mlp_policy_freeze = False
attn_mlp_policy_lazy_init_enabled = True
attn_mlp_policy_lazy_init_iter = 10000

# Placeholder until head_asym_selector assigns the selected head at iter 10000.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = 0
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
attn_mlp_policy_log_input_attn_prefix = 'seq256_block128_try1_autohead_eigloss_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/seq256_block128_try_1/input_attn'
attn_mlp_policy_log_input_attn_save_latest = True
attn_mlp_policy_log_input_attn_cmap = 'coolwarm'
attn_mlp_policy_log_input_attn_vmax_percentile = 99.0

attn_mlp_policy_train_loss = 'directed_ribbon'
attn_mlp_policy_directed_ribbon_flow_sign = -1.0
attn_mlp_policy_directed_ribbon_rank_tau = 1.0
attn_mlp_policy_directed_ribbon_margin = 1.0
attn_mlp_policy_directed_ribbon_band_width = 16.0
attn_mlp_policy_directed_ribbon_band_weight = 0.05

attn_mlp_policy_logit_l2 = 1e-4
attn_mlp_policy_min_logit_std = 1.0
attn_mlp_policy_std_floor_weight = 0.01
attn_mlp_policy_min_entropy = 0.0
attn_mlp_policy_max_entropy = 10.0
attn_mlp_policy_entropy_floor_weight = 0.0
attn_mlp_policy_entropy_ceiling_weight = 0.0

attn_mlp_policy_lr_anneal_enabled = True
attn_mlp_policy_lr_anneal_start_iter = 10000
attn_mlp_policy_lr_anneal_end_iter = 35000
attn_mlp_policy_lr_anneal_min_lr = 1e-5
attn_mlp_policy_lr_anneal_style = 'cosine'

# MLP-only loss stop. If triggered, only MLP optimizer steps stop; attention EMA
# and cached-order refresh continue until update_stop_iter=35000.
attn_mlp_policy_loss_stop_enabled = True
attn_mlp_policy_loss_stop_metric = 'attn_mlp_train_loss'
attn_mlp_policy_loss_stop_start_iter = 25000
attn_mlp_policy_loss_stop_patience = 5
attn_mlp_policy_loss_stop_min_delta = 0.001
attn_mlp_policy_loss_stop_check_every = 1

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
head_asym_selector_out_dir = 'Report/MLP_loss_training/seq256_block128_try_1/head_asym_selector'
head_asym_selector_wandb_prefix = 'head_asym_selector'
head_asym_selector_log_wandb = True
head_asym_selector_save_matrices = True
head_asym_selector_save_all_head_maps = True
head_asym_selector_assign_to_attn_mlp_policy = True
head_asym_selector_init_attn_mlp_policy = True
head_asym_selector_restore_rng = True
head_asym_selector_deterministic = True
head_asym_selector_seed = 24681357

head_asym_selector_selection_rule = 'asym_topk_direct_asym_eig_loss'
head_asym_selector_asym_top_k = 8
head_asym_selector_direct_asym_eig_mode = 'raw_right_largest_real_real'
head_asym_selector_loss_batches = 4
head_asym_selector_loss_batch_size = 16
head_asym_selector_loss_candidate_batch_size = 8
head_asym_selector_loss_prefix_k = 32
head_asym_selector_loss_score = 'linear_profile'
head_asym_selector_loss_exp_tau = 32.0

online_spectral_policy_enabled = False
head_signal_probe_enabled = False
head_direction_logger_enabled = False
