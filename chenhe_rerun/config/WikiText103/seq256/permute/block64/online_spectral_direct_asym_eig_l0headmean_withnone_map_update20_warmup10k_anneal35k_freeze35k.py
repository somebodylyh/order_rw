# L0 all-head mean counterpart of the clean direct_asym_eig distribution run.
#
# This is matched to the L1-head-mean try. Only the averaged attention layer
# changes from layer 1 to layer 0; the distribution method and schedule remain
# unchanged.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_direct_asym_eig_l1headmean_withnone_map_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-direct-asym-eig-l0headmean-withnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-direct-asym-eig-l0headmean-withnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

head_signal_probe_out_dir = 'Report/head_singal_Stable/direct_asym_eig_l0headmean_withnone_map_update20_warmup10k_anneal35k_freeze35k_permute/online_training_probe'

# Average all heads in layer 0 into one block-attention matrix before
# direct_asym_eig. This produces exactly one raw order, then the current-model
# linear_profile_loss rule chooses order vs reverse.
head_signal_probe_heads = '0:mean'

# Distribution-only diagnostic: log the exact averaged attention matrix consumed
# by direct_asym_eig in both current-L2R and original-L2R frames.
online_spectral_policy_log_input_attn = True
online_spectral_policy_log_input_attn_interval = 1000
online_spectral_policy_log_input_attn_prefix = 'direct_asym_eig_l0headmean_distribution_input_attn'
online_spectral_policy_log_input_attn_out_dir = 'Report/head_singal_Stable/direct_asym_eig_l0headmean_withnone_map_update20_warmup10k_anneal35k_freeze35k_permute/input_attn'
online_spectral_policy_log_input_attn_save_latest = True
online_spectral_policy_log_input_attn_cmap = 'coolwarm'
online_spectral_policy_log_input_attn_vmax_percentile = 99.0
