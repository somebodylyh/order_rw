# try40: same as try39, but with a different training seed.
# The data permutation seed intentionally remains permute_seed=42.

_base_try_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try39_fromscratch_autohead_eigloss_withoutnone_directed_ribbon_seed2028_sameperm_lazyinit_warmup10k_anneal35k_fixed50k_wandb.py'
with open(_base_try_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2029
permute_seed = 42
head_asym_selector_seed = 24681357

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try40-seed2029-sameperm-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup10k-anneal35k-fixed50k-b64-permute-block'
wandb_run_name = 'seq256-try40-seed2029-sameperm-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup10k-anneal35k-fixed50k-b64-permute'

attn_mlp_policy_log_input_attn_prefix = 'try40_seed2029_autohead_eigloss_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/try_40/input_attn'
head_asym_selector_out_dir = 'Report/MLP_loss_training/try_40/head_asym_selector'
