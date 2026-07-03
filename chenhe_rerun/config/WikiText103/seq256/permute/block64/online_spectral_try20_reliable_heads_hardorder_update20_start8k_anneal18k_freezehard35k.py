# Try20: reliable-head try19 bridge, wired into training as a hard complete order.
#
# Goal:
# - keep the try19 no-prior orientation rule:
#     current attention -> spectral candidate orders -> current-model
#     reveal/profile loss -> same-probe cross-head consensus
# - reduce the old distribution failure mode where oriented candidates are
#   flattened into a per-block priority_ema and then partly canceled by
#   reverse-like candidates or sampling noise.
#
# Schedule:
# - 0..8000: pure random training orders, no head-signal bridge/probe cost.
# - 8000..18000: policy usage anneals 0 -> 1; hard complete-order usage grows
#   from 0.50 -> 0.85 inside the policy.
# - 18000..35000: update every 20 steps; train mostly on the oriented hard
#   complete order, while keeping a small EMA/MAP component as a stabilizer.
# - 35000..50000: freeze to the hard complete order and train it fixed.
#
# No original-frame L2R/R2L, original tau, PPL oracle, target-position anchor,
# or historical sign is used to orient an order. Original-order metrics are
# diagnostics only.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_order_distribution.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-try20-reliable-heads-hardorder-update20-start8k-anneal18k-freezehard35k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-try20-reliable-heads-hardorder-update20-start8k-anneal18k-freezehard35k-b64-permute-50000-iters'

# Keep the same base training/eval setup as the distribution baseline. The
# changes below only affect the online order policy and its diagnostics.
compile = False

# Replace the old single-head online spectral update with the try19 bridge.
online_spectral_policy_probe_batches = 0
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_loss_rerank_enabled = False

# Try20 uses a small reliable head set instead of all 32 heads. This avoids
# letting weak/ambiguous heads dilute the consensus order before it reaches the
# training sampler.
head_signal_probe_enabled = True
head_signal_probe_start_iter = 8000
head_signal_probe_interval = 20
head_signal_probe_heads = '1:2,0:6,0:7'
head_signal_probe_out_dir = 'Report/head_singal_Stable/try20_reliable_heads_hardorder_update20_start8k_anneal18k_freezehard35k_permute/online_training_probe'

# Use a slightly larger current-sample probe than try19 so the per-update
# orientation rule has less batch noise.
head_signal_probe_batches = 64
head_signal_probe_batch_size = 16
head_signal_probe_loss_batches = 8
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
head_signal_probe_candidate_loss_profile_max_rank = 8
head_signal_probe_candidate_loss_profile_min_gap = 0.010
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
head_signal_probe_deterministic = True
head_signal_probe_seed = 24681357

# Keep the try19 recovery/orientation rule. Attention proposes candidates;
# current reveal-step loss and same-probe consensus choose their orientation.
head_signal_probe_orientation_rule = 'linear_profile_consensus_candidate'
head_signal_probe_candidate_loss_profile_enabled = True
head_signal_probe_candidate_loss_profile_score = 'linear_profile'
head_signal_probe_candidate_loss_profile_include_reverse = True
head_signal_probe_candidate_loss_profile_exp_tau = 16.0
head_signal_probe_candidate_loss_profile_min_alignment = 0.0
head_signal_probe_candidate_loss_profile_low_confidence_fallback = 'global_q'
head_signal_probe_consensus_enabled = True
head_signal_probe_consensus_leave_one_out = True
head_signal_probe_position_anchor_enabled = False
head_signal_probe_position_consensus_enabled = False

# Online policy cadence.
online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 8000
online_spectral_policy_update_stop_iter = 35000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_anneal_start_iter = 8000
online_spectral_policy_anneal_end_iter = 18000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_random_mix_prob = 0.0

# Bridge rows are already oriented complete orders. Only three heads are used,
# so keep the bridge candidate list small and interpretable.
online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'hybrid_direct_ema'
online_spectral_policy_try19_bridge_order_field = 'loss_profile_consensus_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 0.005
online_spectral_policy_try19_bridge_max_candidates = 3
online_spectral_policy_try19_bridge_start_iter = 8000
online_spectral_policy_try19_bridge_stop_iter = 35000

# The EMA/MAP side is still recorded and available, but training is deliberately
# dominated by the hard complete order selected by the current no-prior bridge.
online_spectral_policy_distribution_sample_mode = 'map'
online_spectral_policy_distribution_per_sample = False
online_spectral_policy_hybrid_per_sample = True
online_spectral_policy_hybrid_direct_prob_schedule = 'piecewise'
online_spectral_policy_hybrid_direct_prob_points = '0:0.0,7999:0.0,8000:0.50,18000:0.85,35000:1.0,50000:1.0'
online_spectral_policy_hybrid_direct_prob_start = 0.50
online_spectral_policy_hybrid_direct_prob_end = 1.0
online_spectral_policy_hybrid_direct_prob_anneal_start_iter = 8000
online_spectral_policy_hybrid_direct_prob_anneal_end_iter = 35000
online_spectral_policy_hybrid_freeze_order = 'hard'
online_spectral_policy_hybrid_mix_hard_after_ema_stop = True
online_spectral_policy_hybrid_update_hard_after_ema_stop = False

online_spectral_policy_order_history_enabled = True
online_spectral_policy_order_history_top_candidates = 3
online_spectral_policy_order_history_include_priority = True

head_signal_probe_stop_iter = 35000
