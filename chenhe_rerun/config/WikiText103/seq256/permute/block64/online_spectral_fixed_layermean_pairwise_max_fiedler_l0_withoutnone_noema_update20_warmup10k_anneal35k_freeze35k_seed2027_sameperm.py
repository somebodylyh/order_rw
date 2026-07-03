# Seed ablation for the fixed/no-EMA layer-level order policy.
#
# Diff from the base fixed/no-EMA config:
# - seed = 2027 changes model initialization / training randomness.
# - permute_seed = 42 is intentionally unchanged, so the data block permutation
#   is matched to the default run.
# - head_signal_probe_seed is inherited unchanged, so probe sampling remains
#   comparable across runs.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_fixed_layermean_pairwise_max_fiedler_l0_withoutnone_noema_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2027
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-update20-warmup10k-anneal35k-freeze35k-seed2027-sameperm-b64-permute-block-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-online-spectral-fixed-layermean-pairwise-max-fiedler-l0-withoutnone-noema-update20-warmup10k-anneal35k-freeze35k-seed2027-sameperm-b64-permute-50000-iters'

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/try_27'

head_signal_probe_out_dir = _report_root + '/online_training_probe'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_prefix = 'fixed_layermean_pairwise_max_fiedler_l0_noema_seed2027_input_attn'
