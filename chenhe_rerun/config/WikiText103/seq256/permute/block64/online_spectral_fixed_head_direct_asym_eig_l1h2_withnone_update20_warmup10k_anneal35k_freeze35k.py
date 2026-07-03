# Fixed-head clean single-head asymmetric order discovery, L1H2.
#
# Compared with the distribution version, this does not maintain priority_ema.
# The try19 bridge writes the loss-oriented direct_asym_eig order directly into
# online_spectral_policy_cached_order, and training samples that cached order
# according to the 10k..35k anneal probability. After 35k, updates stop and the
# last cached order is used fixed until 50k.
#
# Method:
# - with_none + L1H2
# - A -> eig(A) -> right eigenvector of largest-real eigenvalue
# - sort eigenvector.real to get one raw order
# - compare raw order vs reverse(raw order) by current-model linear_profile_loss
# - no EMA, no Gumbel, no multi-head vote, no leave-one-out consensus

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-fixed-head-direct-asym-eig-l1h2-withnone-update20-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-fixed-head-direct-asym-eig-l1h2-withnone-update20-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

compile = False

aogpt_train_mode = 'OnlineSpectralFixedHeadOrder'

# Disable the old online spectral fixed-head updater; the direct_asym_eig
# head-signal bridge below is the only order update path.
online_spectral_policy_probe_batches = 0
online_spectral_policy_probe_include_train_step_attention = False
online_spectral_policy_loss_rerank_enabled = False

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 10000
online_spectral_policy_update_stop_iter = 35000

online_spectral_policy_anneal_start_iter = 10000
online_spectral_policy_anneal_end_iter = 35000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'fixed_top1'
online_spectral_policy_try19_bridge_order_field = 'loss_profile_selected_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 1e-4
online_spectral_policy_try19_bridge_max_candidates = 1
online_spectral_policy_try19_bridge_start_iter = 10000
online_spectral_policy_try19_bridge_stop_iter = 35000

online_spectral_policy_order_history_enabled = True
online_spectral_policy_order_history_top_candidates = 1
online_spectral_policy_order_history_include_priority = False

head_signal_probe_enabled = True
head_signal_probe_start_iter = 10000
head_signal_probe_stop_iter = 35000
head_signal_probe_interval = 20
head_signal_probe_out_dir = 'Report/head_singal_Stable/fixed_head_direct_asym_eig_l1h2_withnone_update20_warmup10k_anneal35k_freeze35k_permute/online_training_probe'

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
