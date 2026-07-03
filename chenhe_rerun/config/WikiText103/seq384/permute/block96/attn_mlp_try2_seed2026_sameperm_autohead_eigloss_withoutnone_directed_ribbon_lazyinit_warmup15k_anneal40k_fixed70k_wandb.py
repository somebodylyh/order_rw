# WikiText103 seq384 permuted block96 try_2.
#
# Same method and schedule as try_1, but with a different training seed.
# The data permutation seed stays fixed for comparability.

_base_config = 'config/WikiText103/seq384/permute/block96/attn_mlp_try1_fromscratch_autohead_eigloss_withoutnone_directed_ribbon_lazyinit_warmup15k_anneal40k_fixed70k_wandb.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2026
permute_seed = 42
head_asym_selector_seed = 24681357

attn_mlp_policy_directed_ribbon_flow_sign = 1.0

out_dir = 'out/base/permute/seq384/block96/out-wikitext103-seq384-try2-seed2026-sameperm-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup15k-anneal40k-fixed70k-b96-permute-block'
wandb_run_name = 'seq384-try2-seed2026-sameperm-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup15k-anneal40k-fixed70k-b96-permute'

attn_mlp_policy_log_input_attn_prefix = 'seq384_block96_try2_seed2026_autohead_eigloss_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/seq384_block96_try_2_seed2026/input_attn'

head_asym_selector_out_dir = 'Report/MLP_loss_training/seq384_block96_try_2_seed2026/head_asym_selector'
