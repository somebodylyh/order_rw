# Continuous-coordinate EMA ablation for the latest layer-level method.
#
# Difference from the rank-EMA baseline:
# - recover the same L0 layer-mean pairwise-max Fiedler axis;
# - orient axis vs reverse(axis) with current-model linear_profile_loss;
# - min-max normalize the oriented Fiedler coordinates into [0, 1] priority;
# - update priority_ema directly from that continuous priority vector;
# - produce the hard training order only with argsort(priority_ema).
#
# This tests whether preserving Fiedler coordinate gaps is better than the
# current order -> rank priority -> EMA path.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-continuous-minmax-ema-update20-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-continuous-minmax-ema-update20-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

# Keep the same distribution training path, but change the bridge update from
# rank-priority EMA to oriented continuous Fiedler-coordinate EMA.
online_spectral_policy_try19_bridge_mode = 'continuous_fiedler_minmax'
online_spectral_policy_continuous_priority_normalization = 'minmax'
online_spectral_policy_continuous_priority_eps = 1e-8

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_28'
head_signal_probe_out_dir = _report_root + '/online_training_probe'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_prefix = 'continuous_minmax_fiedler_l0_distribution_input_attn'
