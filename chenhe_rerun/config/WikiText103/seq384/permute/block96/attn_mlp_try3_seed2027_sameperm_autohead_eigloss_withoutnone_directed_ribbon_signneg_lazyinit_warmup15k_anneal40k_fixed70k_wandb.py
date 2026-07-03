# WikiText103 seq384 permuted block96 try_3.
#
# Same method and schedule as try_1, but with a different training seed.
# The data permutation seed stays fixed for comparability. This variant keeps
# directed_ribbon flow_sign=-1.0 explicitly for comparison against try_2's
# flow_sign=+1.0 run.

_base_config = 'config/WikiText103/seq384/permute/block96/attn_mlp_try1_fromscratch_autohead_eigloss_withoutnone_directed_ribbon_lazyinit_warmup15k_anneal40k_fixed70k_wandb.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2027
permute_seed = 42
head_asym_selector_seed = 24681357

attn_mlp_policy_directed_ribbon_flow_sign = -1.0

out_dir = 'out/base/permute/seq384/block96/out-wikitext103-seq384-try3-seed2027-sameperm-autohead-eigloss-withoutnone-directed-ribbon-signneg-lazyinit-warmup15k-anneal40k-fixed70k-b96-permute-block'
wandb_run_name = 'seq384-try3-seed2027-sameperm-autohead-eigloss-withoutnone-directed-ribbon-signneg-lazyinit-warmup15k-anneal40k-fixed70k-b96-permute'

attn_mlp_policy_log_input_attn_prefix = 'seq384_block96_try3_seed2027_autohead_eigloss_withoutnone_signneg_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/seq384_block96_try_3_seed2027_signneg/input_attn'

head_asym_selector_out_dir = 'Report/MLP_loss_training/seq384_block96_try_3_seed2027_signneg/head_asym_selector'
