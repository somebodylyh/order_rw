# Scheme C: try19 hard-order / EMA-order hybrid.
#
# Schedule:
# - 0..8000: pure random training orders, no head-signal bridge/probe cost.
# - 8000..18000: anneal policy usage from 0 to 1 while hard-order usage inside
#   the policy grows from 0.1 to 0.4.
# - 18000..28000: full hybrid policy, updating every 20 steps:
#     * hard_order: same-probe try19 consensus order, fixed-head-like.
#     * ema_order: priority_ema MAP order from the same oriented candidates.
#     * train samples use hard_order with probability 0.4, otherwise ema_order.
# - 28000..50000: freeze updates and train with the last blended hybrid order.
#
# No-prior rule: original L2R/R2L, original tau, validation PPL, target-position
# anchors, and historical sign are diagnostics only. Direction comes from
# current attention, current reveal/profile loss, and same-probe consensus.

_base_config = 'config/WikiText103/seq256/non_permute/block64/online_spectral_try19_bridge_oriented_distribution_map_update20_start8k_anneal18k_freeze28k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-try19-bridge-hybrid-direct-ema-update20-start8k-anneal18k-freeze28k-b64-nonpermute-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-try19-bridge-hybrid-direct-ema-update20-start8k-anneal18k-freeze28k-b64-nonpermute-50000-iters'

online_spectral_policy_try19_bridge_mode = 'hybrid_direct_ema'

# Keep the EMA side deterministic/MAP-like; the hybrid sampler performs the
# hard-vs-EMA sample mixture directly.
online_spectral_policy_distribution_sample_mode = 'map'
online_spectral_policy_distribution_per_sample = False
online_spectral_policy_random_mix_prob = 0.0

online_spectral_policy_hybrid_direct_prob_start = 0.1
online_spectral_policy_hybrid_direct_prob_end = 0.4
online_spectral_policy_hybrid_direct_prob_anneal_start_iter = 8000
online_spectral_policy_hybrid_direct_prob_anneal_end_iter = 18000
online_spectral_policy_hybrid_per_sample = True
online_spectral_policy_hybrid_freeze_order = 'hybrid'

head_signal_probe_out_dir = 'Report/head_singal_Stable/try19_bridge_hybrid_direct_ema_update20_start8k_anneal18k_freeze28k/online_training_probe'
