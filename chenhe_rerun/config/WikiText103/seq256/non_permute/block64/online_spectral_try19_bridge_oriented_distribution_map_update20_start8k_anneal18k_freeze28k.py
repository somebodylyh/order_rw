# Scheme B variant: start order policy at 8k and freeze the MAP order at 28k.
#
# Schedule:
# - 0..8000: pure random training orders, no head-signal bridge/probe cost.
# - 8000..18000: linearly anneal policy usage from 0 to 1.
# - 18000..28000: train from the try19-oriented distribution MAP, updating every 20 steps.
# - 28000..50000: freeze updates and train with the current MAP order.
#
# The method remains no-prior B: current attention proposes head orders, current
# reveal/profile loss + same-probe cross-head consensus orients them, and only
# the oriented candidates enter priority_ema. Original L2R/tau/PPL/history are
# diagnostics only.

_base_config = 'config/WikiText103/seq256/non_permute/block64/online_spectral_try19_bridge_oriented_distribution_map_update20_warmup5k_anneal15k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-try19-bridge-oriented-distribution-map-update20-start8k-anneal18k-freeze28k-b64-nonpermute-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-try19-bridge-oriented-distribution-map-update20-start8k-anneal18k-freeze28k-b64-nonpermute-50000-iters'

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 8000
online_spectral_policy_update_stop_iter = 28000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_anneal_start_iter = 8000
online_spectral_policy_anneal_end_iter = 18000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0

online_spectral_policy_try19_bridge_start_iter = 8000
online_spectral_policy_try19_bridge_stop_iter = 28000

# Avoid spending the try19 probe budget before the order policy is allowed to update.
head_signal_probe_start_iter = 8000
head_signal_probe_interval = 20
head_signal_probe_out_dir = 'Report/head_singal_Stable/try19_bridge_oriented_distribution_map_update20_start8k_anneal18k_freeze28k/online_training_probe'
