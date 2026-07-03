# Second-seed temperature-sampling order-usage ablation for the latest L0
# layer-level distribution teacher. The data permutation and probe seed are
# fixed to make this comparable with the seed2027 run.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2028
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-temp-sampling-t2to01-update20-warmup15k-sample35k-freeze35k-seed2028-sameperm-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-temp-sampling-t2to01-update20-warmup15k-sample35k-freeze35k-seed2028-sameperm-b64-permute-50000-iters'

_warmup_iter = 15000
_sample_end_iter = 35000

online_spectral_policy_order_usage_mode = 'temperature_sampling'
online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = _warmup_iter
online_spectral_policy_update_stop_iter = _sample_end_iter
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_anneal_start_iter = _warmup_iter
online_spectral_policy_anneal_end_iter = _sample_end_iter
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_distribution_sample_mode = 'gumbel'
online_spectral_policy_distribution_per_sample = True
online_spectral_policy_random_mix_prob = 0.0
online_spectral_policy_sample_temperature = 0.1
online_spectral_policy_temperature_sampling_start_temperature = 2.0
online_spectral_policy_temperature_sampling_end_temperature = 0.1
online_spectral_policy_temperature_sampling_schedule = 'linear'

online_spectral_policy_try19_bridge_enabled = True
online_spectral_policy_try19_bridge_mode = 'oriented_distribution'
online_spectral_policy_try19_bridge_max_candidates = 1
online_spectral_policy_try19_bridge_start_iter = _warmup_iter
online_spectral_policy_try19_bridge_stop_iter = _sample_end_iter

head_signal_probe_start_iter = _warmup_iter
head_signal_probe_stop_iter = _sample_end_iter
head_signal_probe_interval = 20
head_signal_probe_seed = 24681357

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/temperature_sampling_l0_pairwise_max_fiedler_withoutnone_t2to01_update20_warmup15k_sample35k_freeze35k_seed2028_sameperm'
head_signal_probe_out_dir = _report_root + '/online_training_probe'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_prefix = 'temperature_sampling_l0_pairwise_max_fiedler_seed2028_distribution_input_attn'
online_spectral_policy_order_history_path = _report_root + '/online_spectral_policy_order_history.jsonl'
