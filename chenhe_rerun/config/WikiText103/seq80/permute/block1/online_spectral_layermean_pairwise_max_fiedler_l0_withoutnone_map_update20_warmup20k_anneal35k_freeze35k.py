# WikiText103 seq80 / permute / block1 token-level distribution config.
#
# Method:
# - train from scratch with the seq80 permuted random-base training shape;
# - use all heads in layer 0 by averaging their token-level attention matrices;
# - use the real-token attention view (`without_none`);
# - build W=max(A,A.T), diag(W)=0;
# - recover one axis with the graph Laplacian Fiedler vector;
# - choose axis vs reverse(axis) with current-model linear_profile_loss;
# - update priority_ema with the loss-oriented order and train with MAP order.
#
# Schedule:
# - 0..20000: pure random base, no EMA/probe updates.
# - 20000..35000: every 20 steps, update priority_ema from L0 layer mean.
# - 20000..35000: train anneals from random to EMA MAP.
# - 35000..50000: stop EMA updates and train fixed with current EMA MAP.
#
# No original L2R/tau/PPL/history is used for candidate recovery or direction.
# Original-frame metrics remain diagnostic-only.

_base_config = 'config/WikiText103/seq80/permute/block1/random.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

init_from = 'scratch'

out_dir = 'out/base/permute/seq80/block1/out-wikitext103-seq80-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-map-update20-warmup20k-anneal35k-freeze35k-b1-permute-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-seq80-final'
wandb_run_name = 'seq80-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-map-update20-warmup20k-anneal35k-freeze35k-b1-permute-50000-iters'

compile = False
eval_batch_size = 64

aogpt_train_mode = 'OnlineSpectralOrderDistribution'
main_eval_mode = 'Random'
generalization_eval_mode = ''

# Disable the old spectral-distribution updater; all distribution updates come
# from the oriented head-signal bridge below.
online_spectral_policy_enabled = True
online_spectral_policy_probe_batches = 0
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_loss_rerank_enabled = False

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 20000
online_spectral_policy_update_stop_iter = 35000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_anneal_start_iter = 20000
online_spectral_policy_anneal_end_iter = 35000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

online_spectral_policy_distribution_sample_mode = 'map'
online_spectral_policy_distribution_per_sample = False
online_spectral_policy_random_mix_prob = 0.0
online_spectral_policy_priority_ema_decay = 0.95
online_spectral_policy_sample_temperature = 0.7
online_spectral_policy_teacher_temperature = 1.0
online_spectral_policy_score_normalization = 'zscore'

online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'oriented_distribution'
online_spectral_policy_try19_bridge_order_field = 'loss_profile_selected_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 1e-4
online_spectral_policy_try19_bridge_max_candidates = 1
online_spectral_policy_try19_bridge_start_iter = 20000
online_spectral_policy_try19_bridge_stop_iter = 35000

online_spectral_policy_order_history_enabled = True
online_spectral_policy_order_history_top_candidates = 1
online_spectral_policy_order_history_include_priority = True

# L0 all-head mean. On seq80/block1 this is a token-level 80x80 matrix.
head_signal_probe_enabled = True
head_signal_probe_start_iter = 20000
head_signal_probe_stop_iter = 35000
head_signal_probe_interval = 20
head_signal_probe_heads = '0:mean'
head_signal_probe_out_dir = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/seq80_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup20k_anneal35k_freeze35k_permute/online_training_probe'

# 16 x 256 = 4096 attention samples.
head_signal_probe_batches = 16
head_signal_probe_batch_size = 256
# 8 x 128 = 1024 loss-profile samples for steadier order-vs-reverse direction.
head_signal_probe_loss_batches = 8
head_signal_probe_loss_batch_size = 128
head_signal_probe_candidate_batch_size = 2
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'without_none'

head_signal_probe_candidate_source = 'pairwise_max_fiedler'
head_signal_probe_direct_asym_eig_mode = 'raw_right_largest_real_real'
head_signal_probe_top_m = 1
head_signal_probe_candidate_loss_profile_max_rank = 1
head_signal_probe_candidate_loss_profile_include_reverse = True

head_signal_probe_consensus_enabled = False
head_signal_probe_consensus_leave_one_out = False
head_signal_probe_deterministic = True
head_signal_probe_seed = 24681357
head_signal_probe_orientation_rule = 'linear_profile_candidate'
head_signal_probe_candidate_loss_profile_enabled = True
head_signal_probe_candidate_loss_profile_score = 'linear_profile'
head_signal_probe_candidate_loss_profile_exp_tau = 16.0
head_signal_probe_candidate_loss_profile_min_gap = 0.0
head_signal_probe_candidate_loss_profile_min_alignment = 0.0
head_signal_probe_candidate_loss_profile_low_confidence_fallback = 'none'
head_signal_probe_position_anchor_enabled = False
head_signal_probe_position_consensus_enabled = False

# Distribution-only diagnostic: log the exact L0 layer-mean matrix consumed by
# pairwise_max_fiedler in current-L2R and original-L2R frames.
online_spectral_policy_log_input_attn = True
online_spectral_policy_log_input_attn_interval = 1000
online_spectral_policy_log_input_attn_prefix = 'seq80_permute_layermean_pairwise_max_fiedler_l0_distribution_input_attn'
online_spectral_policy_log_input_attn_out_dir = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/seq80_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup20k_anneal35k_freeze35k_permute/input_attn'
online_spectral_policy_log_input_attn_save_latest = True
online_spectral_policy_log_input_attn_cmap = 'coolwarm'
online_spectral_policy_log_input_attn_vmax_percentile = 99.0

# Keep milestone checkpoints for this try, especially the 20k warmup boundary.
save_iter_checkpoints = True
save_iter_checkpoint_steps = '20000,35000,50000'
save_iter_checkpoint_keep = 0
