# Seed2027 fixed/no-EMA layer-level order policy with a later warmup.
#
# Diff from:
# online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py
#
# Only the schedule is changed:
# - 0..20000: pure random warmup, no probe/order updates.
# - 20000..35000: update every 20 steps and anneal random -> current hard order.
# - 35000..50000: stop updates and train fixed with the last cached hard order.
#
# Method, seed, data permutation, teacher, loss-profile orientation, and logging
# style are kept aligned with the seed2027/sameperm fixed no-EMA run.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

_warmup_iter = 20000
_freeze_iter = 35000

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-update20-warmup20k-anneal35k-freeze35k-seed2027-sameperm-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-update20-warmup20k-anneal35k-freeze35k-seed2027-sameperm-b64-permute-50000-iters'

online_spectral_policy_update_start_iter = _warmup_iter
online_spectral_policy_update_stop_iter = _freeze_iter
online_spectral_policy_anneal_start_iter = _warmup_iter
online_spectral_policy_anneal_end_iter = _freeze_iter
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0

online_spectral_policy_try19_bridge_start_iter = _warmup_iter
online_spectral_policy_try19_bridge_stop_iter = _freeze_iter

head_signal_probe_start_iter = _warmup_iter
head_signal_probe_stop_iter = _freeze_iter
head_signal_probe_interval = 20

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup20k_anneal35k_freeze35k_seed2027_sameperm'
head_signal_probe_out_dir = _report_root + '/online_training_probe'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_prefix = 'fixed_layermean_pairwise_max_fiedler_l0_noema_warmup20k_seed2027_input_attn'
online_spectral_policy_order_history_path = _report_root + '/online_spectral_policy_order_history.jsonl'
