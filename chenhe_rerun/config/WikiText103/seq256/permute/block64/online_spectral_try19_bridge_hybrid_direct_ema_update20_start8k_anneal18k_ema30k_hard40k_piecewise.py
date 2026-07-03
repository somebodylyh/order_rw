# Permuted try19 hybrid schedule with a piecewise hard-order ratio.
#
# Schedule:
# - 0..8000: pure random training orders, no head-signal bridge/probe cost.
# - 8000..18000: policy usage anneals 0 -> 1; hard ratio anneals 0.10 -> 0.35.
# - 18000..30000: EMA and hard both update; train with EMA/hard mix at 0.35 hard.
# - 30000..40000: freeze priority_ema/MAP; continue updating hard order and
#   decay hard ratio 0.35 -> 0.0.
# - 40000..50000: EMA-only fixed-order training; no hard update/probe.
#
# Orders and priorities are current-frame block ids. Original L2R/tau/PPL are
# diagnostics only and are not used to choose direction.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_try19_bridge_hybrid_direct_ema_update20_start8k_anneal18k_freeze28k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-try19-bridge-hybrid-direct-ema-update20-start8k-anneal18k-ema30k-hard40k-piecewise-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-try19-bridge-hybrid-direct-ema-update20-start8k-anneal18k-ema30k-hard40k-piecewise-b64-permute-50000-iters'

online_spectral_policy_update_stop_iter = 30000
online_spectral_policy_try19_bridge_stop_iter = 40000

online_spectral_policy_hybrid_direct_prob_schedule = 'piecewise'
online_spectral_policy_hybrid_direct_prob_points = '0:0.0,7999:0.0,8000:0.10,18000:0.35,30000:0.35,40000:0.0,50000:0.0'

# Linear fields are kept as a readable fallback if the piecewise schedule is
# disabled in an ablation.
online_spectral_policy_hybrid_direct_prob_start = 0.1
online_spectral_policy_hybrid_direct_prob_end = 0.0
online_spectral_policy_hybrid_direct_prob_anneal_start_iter = 8000
online_spectral_policy_hybrid_direct_prob_anneal_end_iter = 50000

online_spectral_policy_hybrid_freeze_order = 'ema'
online_spectral_policy_hybrid_mix_hard_after_ema_stop = True
online_spectral_policy_hybrid_update_hard_after_ema_stop = True

head_signal_probe_stop_iter = 40000
head_signal_probe_out_dir = 'Report/head_singal_Stable/try19_bridge_hybrid_direct_ema_update20_start8k_anneal18k_ema30k_hard40k_piecewise_permute/online_training_probe'
