# Seed ablation for the seq80/block1 layer-level distribution method.
#
# Diff from the base L0 layer-mean pairwise-max Fiedler config:
# - seed = 2027 changes model initialization / training randomness.
# - permute_seed = 42 is intentionally unchanged, so the data permutation is
#   matched to the base seq80 permuted run.
# - head_signal_probe_seed is intentionally unchanged, so probe sampling is
#   comparable across runs.

_base_config = 'config/WikiText103/seq80/permute/block1/online_spectral_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2027
permute_seed = 42

out_dir = 'out/base/permute/seq80/block1/out-wikitext103-seq80-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-map-update20-warmup10k-anneal35k-freeze35k-seed2027-sameperm-b1-permute-50000-iters'

wandb_log = True
wandb_project = 'AOGPT-order-block-seq80-final'
wandb_run_name = 'seq80-online-spectral-layermean-pairwise-max-fiedler-l0-withoutnone-map-update20-warmup10k-anneal35k-freeze35k-seed2027-sameperm-b1-permute-50000-iters'

_report_root = 'Report/language/wikitext103/order_teacher_distribution/head_signal_stability/distribution_training_tries/seq80_layermean_pairwise_max_fiedler_l0_withoutnone_map_update20_warmup10k_anneal35k_freeze35k_seed2027_sameperm_permute'

head_signal_probe_out_dir = _report_root + '/online_training_probe'
online_spectral_policy_log_input_attn_out_dir = _report_root + '/input_attn'
online_spectral_policy_log_input_attn_prefix = 'seq80_layermean_pairwise_max_fiedler_l0_seed2027_distribution_input_attn'
