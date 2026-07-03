# Try19 ablation: with-none attention export, oriented EMA MAP distribution.
#
# Schedule:
# - 0..10000: pure random training orders; no try19 bridge/probe cost.
# - 10000..20000: update try19-oriented priority_ema every 20 steps, but keep
#   training orders random so the EMA has time to settle before use.
# - 20000..40000: anneal policy usage 0 -> 1. Training batches mix random
#   orders and the current EMA MAP order through the policy probability.
# - 40000..50000: stop updates and train fixed with the final EMA MAP order.
#
# This is no-prior in the try19 sense: attention proposes candidates; current
# reveal/profile loss plus same-probe leave-one-out cross-head consensus orients
# candidate/reverse before the candidates enter priority_ema. Original L2R,
# original tau, validation PPL, target-position anchors, and history are
# diagnostic-only.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_order_distribution.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-try19-bridge-oriented-distribution-map-update20-withnone-warmup10k-probe10k-anneal20k40k-freeze40k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-try19-bridge-oriented-distribution-map-update20-withnone-warmup10k-probe10k-anneal20k40k-freeze40k-b64-permute-50000-iters'

compile = False

# Replace the old single-head distribution update with a three-head try19
# bridge. Consensus is pairwise-Q within L1H2, L2H7, and L2H6 only.
online_spectral_policy_probe_batches = 0
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_loss_rerank_enabled = False

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 10000
online_spectral_policy_update_stop_iter = 40000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_anneal_start_iter = 20000
online_spectral_policy_anneal_end_iter = 40000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

# Deterministic EMA MAP use; no Gumbel/per-sample priority sampling.
online_spectral_policy_distribution_sample_mode = 'map'
online_spectral_policy_distribution_per_sample = False
online_spectral_policy_random_mix_prob = 0.0
online_spectral_policy_priority_ema_decay = 0.95

online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'oriented_distribution'
online_spectral_policy_try19_bridge_order_field = 'loss_profile_consensus_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 1e-4
online_spectral_policy_try19_bridge_max_candidates = 3
online_spectral_policy_try19_bridge_start_iter = 10000
online_spectral_policy_try19_bridge_stop_iter = 40000

online_spectral_policy_order_history_enabled = True
online_spectral_policy_order_history_top_candidates = 3
online_spectral_policy_order_history_include_priority = True

# Try19 head-signal stability probe used by the bridge.
head_signal_probe_enabled = True
head_signal_probe_start_iter = 10000
head_signal_probe_stop_iter = 40000
head_signal_probe_interval = 20
head_signal_probe_out_dir = 'Report/head_singal_Stable/try19_bridge_oriented_distribution_map_update20_withnone_warmup10k_probe10k_anneal20k40k_freeze40k_permute/online_training_probe'
head_signal_probe_heads = '1:2,2:7,2:6'
head_signal_probe_batches = 32
head_signal_probe_batch_size = 16
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'with_none'
head_signal_probe_fixed_angle_idx = 4
head_signal_probe_top_m = 32
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
head_signal_probe_deterministic = True
head_signal_probe_seed = 24681357
head_signal_probe_orientation_rule = 'linear_profile_consensus_candidate'
head_signal_probe_candidate_loss_profile_enabled = True
head_signal_probe_candidate_loss_profile_score = 'linear_profile'
head_signal_probe_candidate_loss_profile_max_rank = 6
head_signal_probe_candidate_loss_profile_include_reverse = True
head_signal_probe_candidate_loss_profile_exp_tau = 16.0
head_signal_probe_candidate_loss_profile_min_gap = 0.005
head_signal_probe_candidate_loss_profile_min_alignment = 0.0
head_signal_probe_candidate_loss_profile_low_confidence_fallback = 'global_q'
head_signal_probe_position_anchor_enabled = False
head_signal_probe_position_consensus_enabled = False
