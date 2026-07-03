# OnlineSpectralOrderDistribution block64 non-permute run with try19
# head-signal stability probing.
#
# This keeps the distribution training schedule and rerank budget from the
# warmup5k/anneal15k update20 base. The try19 block below is diagnostic/probe
# only: it orients and records head orders during training, but does not add a
# head-signal loss or change the distribution policy update itself.

_base_config = 'config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_fast_b256_update20_top32_loss1024_warmup5k_anneal15k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-order-distribution-loss-rerank-fast-b256-update20-start5k-top32-loss1024-warmup5k-anneal15k-try19-headstable-b64-nonpermute-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-order-distribution-update20-warmup5k-anneal15k-try19-headstable-b64-nonpermute-50000-iters'

# Try19 head-signal stability probe.
head_signal_probe_enabled = True
head_signal_probe_interval = 20
head_signal_probe_out_dir = 'Report/head_singal_Stable/distribution_try19_update20_warmup5k_anneal15k/online_training_probe'
head_signal_probe_heads = 'all'
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
