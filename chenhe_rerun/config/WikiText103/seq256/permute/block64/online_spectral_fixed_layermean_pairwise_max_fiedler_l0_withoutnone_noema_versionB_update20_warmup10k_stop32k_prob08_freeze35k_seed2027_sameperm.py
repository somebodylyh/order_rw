# Version B schedule for the latest L0 layer-mean pairwise-max Fiedler method.
#
# Method is inherited from the fixed/no-EMA seed2027 config:
# - L0 all-head mean attention, without-none block aggregation;
# - W=max(A,A.T), graph Laplacian Fiedler vector recovers one axis;
# - current-model linear_profile_loss chooses order vs reverse;
# - no priority/order EMA, each accepted probe order becomes cached hard order.
#
# Schedule B:
# - 0..10000: pure random, no teacher/probe updates.
# - 10000..32000: update every 20 steps; learned-order probability 0.0 -> 0.8.
# - 32000..35000: stop teacher updates; keep training cached order while
#   learned-order probability 0.8 -> 1.0.
# - 35000..50000: deterministic fixed cached hard order.
#
# Original L2R/tau/PPL remain diagnostic-only.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

_warmup_iter = 10000
_update_stop_iter = 32000
_freeze_iter = 35000

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-versionB-update20-warmup10k-stop32k-prob08-freeze35k-seed2027-sameperm-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-versionB-update20-warmup10k-stop32k-prob08-freeze35k-seed2027-sameperm-b64-permute-50000-iters'

# Keep the no-EMA fixed-hard-order policy explicit.
online_spectral_policy_use_ema = False
online_spectral_policy_ema_decay = 0.0
online_spectral_policy_freeze_to_map_order_after_stop = False

online_spectral_policy_update_start_iter = _warmup_iter
online_spectral_policy_update_stop_iter = _update_stop_iter
online_spectral_policy_update_every = 20

online_spectral_policy_try19_bridge_start_iter = _warmup_iter
online_spectral_policy_try19_bridge_stop_iter = _update_stop_iter
head_signal_probe_start_iter = _warmup_iter
head_signal_probe_stop_iter = _update_stop_iter
head_signal_probe_interval = 20

# Piecewise learned-order probability. This avoids pushing the teacher-update
# window all the way to prob=1.0, then lets the cached order become fixed.
online_spectral_policy_anneal_start_iter = _warmup_iter
online_spectral_policy_anneal_end_iter = _freeze_iter
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_prob_schedule = 'piecewise'
online_spectral_policy_prob_points = '0:0.0,9999:0.0,10000:0.0,32000:0.8,35000:1.0,50000:1.0'

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_30_fixed_layermean_noema_versionB_seed2027_stop32k_prob08_freeze35k'
head_signal_probe_out_dir = _report_root + '/online_training_probe'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_prefix = 'fixed_layermean_pairwise_max_fiedler_l0_noema_versionB_seed2027_input_attn'
