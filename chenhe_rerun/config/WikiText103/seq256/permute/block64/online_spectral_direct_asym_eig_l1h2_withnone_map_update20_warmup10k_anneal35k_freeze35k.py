# Clean single-head asymmetric order discovery.
#
# Method:
# - candidate source: direct_asym_eig
# - attention input: with_none block attention from one head, default L1H2
# - raw order: A -> eig(A) -> right eigenvector of largest-real eigenvalue
#   -> eigenvector.real sorted as one complete order
# - direction: compare raw order vs reverse(raw order) with current-model
#   linear_profile_loss; choose the lower-loss direction
# - no sym, no threshold, no affinity graph, no spectral coordinates, no
#   multi-angle candidates, no cross-head voting, no leave-one-out consensus
# - original tau/L2R/PPL/history remain diagnostic-only
#
# Schedule:
# - 0..10000: pure random base, no EMA/probe updates.
# - 10000..35000: every 20 steps, L1H2 produces one directed candidate; update
#   priority_ema with the loss-oriented order. Training anneals from random to
#   EMA MAP over the same 10k..35k interval.
# - 35000..50000: stop EMA updates and train fixed with the current EMA MAP.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_order_distribution.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-direct-asym-eig-l1h2-withnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-direct-asym-eig-l1h2-withnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

compile = False

# Disable the old single-head spectral distribution update; all updates come
# from the direct_asym_eig head-signal bridge below.
online_spectral_policy_probe_batches = 0
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_loss_rerank_enabled = False

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 10000
online_spectral_policy_update_stop_iter = 35000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_anneal_start_iter = 10000
online_spectral_policy_anneal_end_iter = 35000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

online_spectral_policy_distribution_sample_mode = 'map'
online_spectral_policy_distribution_per_sample = False
online_spectral_policy_random_mix_prob = 0.0
online_spectral_policy_priority_ema_decay = 0.95

online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'oriented_distribution'
online_spectral_policy_try19_bridge_order_field = 'loss_profile_selected_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 1e-4
online_spectral_policy_try19_bridge_max_candidates = 1
online_spectral_policy_try19_bridge_start_iter = 10000
online_spectral_policy_try19_bridge_stop_iter = 35000

online_spectral_policy_order_history_enabled = True
online_spectral_policy_order_history_top_candidates = 1
online_spectral_policy_order_history_include_priority = True

head_signal_probe_enabled = True
head_signal_probe_start_iter = 10000
head_signal_probe_stop_iter = 35000
head_signal_probe_interval = 20
head_signal_probe_out_dir = 'Report/head_singal_Stable/direct_asym_eig_l1h2_withnone_map_update20_warmup10k_anneal35k_freeze35k_permute/online_training_probe'

# Change this to '0:7' for the L0H7 version.
head_signal_probe_heads = '1:2'
head_signal_probe_batches = 32
head_signal_probe_batch_size = 16
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'with_none'

head_signal_probe_candidate_source = 'direct_asym_eig'
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
