# Layer-level distribution try: L0 all-head mean pairwise-max Fiedler axis.
#
# Method:
# - use all heads in layer 0 by averaging their block attention matrices;
# - build an undirected affinity W=max(A,A.T), diag(W)=0;
# - recover one axis by sorting the graph Laplacian Fiedler vector;
# - compare axis vs reverse(axis) with current-model linear_profile_loss;
# - update priority_ema with the loss-oriented order.
#
# No original L2R/tau/PPL/history is used for candidate recovery or direction.
# Original-frame tau remains diagnostic-only.
#
# Schedule matches the current clean distribution bridge family:
# - 0..10000: pure random base, no EMA/probe updates.
# - 10000..35000: every 20 steps, update priority_ema from the L0 layer axis.
# - 10000..35000: train anneals from random to EMA MAP.
# - 35000..50000: stop EMA updates and train fixed with the current EMA MAP.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_direct_asym_eig_l1h2_withnone_map_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

# All-head mean for layer 0. The recovery source below treats this matrix as a
# layer-level affinity signal rather than a single directional head.
head_signal_probe_heads = '0:mean'
head_signal_probe_out_dir = 'Report/head_singal_Stable/layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_permute/online_training_probe'

# The latest layer-level method: W=max(A,A.T), Laplacian Fiedler vector gives an
# axis; current-model loss-profile chooses axis vs reverse.
head_signal_probe_candidate_source = 'pairwise_max_fiedler'
head_signal_probe_top_m = 1
head_signal_probe_candidate_loss_profile_max_rank = 1
head_signal_probe_candidate_loss_profile_include_reverse = True
head_signal_probe_orientation_rule = 'linear_profile_candidate'
head_signal_probe_candidate_loss_profile_enabled = True
head_signal_probe_candidate_loss_profile_score = 'linear_profile'
head_signal_probe_candidate_loss_profile_low_confidence_fallback = 'none'

# Match the offline diagnostic that worked on 10k random base: drop the none
# token before block aggregation.
head_signal_probe_export_type = 'without_none'

# Bridge the oriented Fiedler order into the distribution priority EMA.
online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'oriented_distribution'
online_spectral_policy_try19_bridge_order_field = 'loss_profile_selected_order_current'
online_spectral_policy_try19_bridge_score_field = 'loss_profile_score_gap'
online_spectral_policy_try19_bridge_min_candidate_score = 1e-4
online_spectral_policy_try19_bridge_max_candidates = 1

# Distribution-only diagnostic: log the exact L0 layer-mean matrix consumed by
# pairwise_max_fiedler in current-L2R and original-L2R frames.
online_spectral_policy_log_input_attn = True
online_spectral_policy_log_input_attn_interval = 1000
online_spectral_policy_log_input_attn_prefix = 'layermean_pairwise_max_fiedler_l0_distribution_input_attn'
online_spectral_policy_log_input_attn_out_dir = 'Report/head_singal_Stable/layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_permute/input_attn'
online_spectral_policy_log_input_attn_save_latest = True
online_spectral_policy_log_input_attn_cmap = 'coolwarm'
online_spectral_policy_log_input_attn_vmax_percentile = 99.0
