# Seed ablation for continuous-coordinate EMA.
#
# Diff from the base continuous-minmax config:
# - seed = 2027 changes model initialization / training randomness.
# - permute_seed = 42 is intentionally unchanged.
# - head_signal_probe_seed is intentionally unchanged.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_continuous_minmax_ema_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2027
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-continuous-minmax-ema-update20-warmup10k-anneal35k-freeze35k-seed2027-sameperm-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-continuous-minmax-ema-update20-warmup10k-anneal35k-freeze35k-seed2027-sameperm-b64-permute-50000-iters'

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_29'
head_signal_probe_out_dir = _report_root + '/online_training_probe'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_prefix = 'continuous_minmax_fiedler_l0_seed2027_distribution_input_attn'
