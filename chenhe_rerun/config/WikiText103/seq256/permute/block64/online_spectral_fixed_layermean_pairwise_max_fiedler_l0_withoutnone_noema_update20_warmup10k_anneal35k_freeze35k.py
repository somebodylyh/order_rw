# Fixed-order ablation for the latest layer-level distribution method.
#
# This is the "no order EMA" counterpart of:
# online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py
#
# Method kept the same:
# - use all heads in layer 0 by averaging their block attention matrices;
# - drop the None/predictor token before block aggregation;
# - build W=max(A,A.T), diag(W)=0;
# - recover one axis with the graph Laplacian Fiedler vector;
# - compare axis vs reverse(axis) with current-model linear_profile_loss.
#
# Training policy changed:
# - no priority_ema;
# - no MAP/Gumbel distribution sampling;
# - each accepted probe order is written directly to cached_order;
# - training anneals from random to that current cached hard order;
# - after 35k, probe updates stop and the last cached hard order is used fixed.
#
# Original-frame tau/L2R/PPL/history remain diagnostic-only.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-update20-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-update20-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

compile = False

aogpt_train_mode = 'OnlineSpectralFixedHeadOrder'

# Disable the old online spectral fixed-head updater; all updates come from the
# head-signal bridge below.
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

# This fixed variant intentionally does not maintain priority_ema.
online_spectral_policy_use_ema = False
online_spectral_policy_ema_decay = 0.0
online_spectral_policy_freeze_to_map_order_after_stop = False

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

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_26'
head_signal_probe_out_dir = _report_root + '/online_training_probe'

# L0 all-head mean, matching the latest distribution try.
head_signal_probe_heads = '0:mean'
head_signal_probe_batches = 32
head_signal_probe_batch_size = 16
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
head_signal_probe_prefix_k = 16
head_signal_probe_prefix_weight = 0.7
head_signal_probe_full_weight = 0.3
head_signal_probe_export_type = 'without_none'

head_signal_probe_candidate_source = 'pairwise_max_fiedler'
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

# Log the exact L0 layer-mean matrix consumed by pairwise_max_fiedler.
online_spectral_policy_log_input_attn = True
online_spectral_policy_log_input_attn_interval = 1000
online_spectral_policy_log_input_attn_prefix = 'fixed_layermean_pairwise_max_fiedler_l0_noema_input_attn'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_save_latest = True
online_spectral_policy_log_input_attn_cmap = 'coolwarm'
online_spectral_policy_log_input_attn_vmax_percentile = 99.0
