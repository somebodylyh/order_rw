# WikiText103 seq384 permuted block96 try_6.
#
# Fixed L0H4 variant matching try_5, but with directed-ribbon flow_sign=-1.0.
# try_5 remains the flow_sign=+1.0 counterpart.

_base_config = 'config/WikiText103/seq384/permute/block96/attn_mlp_try5_seed2026_sameperm_fixedl0h4_eigloss_withoutnone_directed_ribbon_flowpos_lazyinitoff_warmup15k_anneal40k_fixed70k_wandb.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

attn_mlp_policy_directed_ribbon_flow_sign = -1.0

out_dir = 'out/base/permute/seq384/block96/out-wikitext103-seq384-try6-seed2026-sameperm-fixedl0h4-eigloss-withoutnone-directed-ribbon-signneg-lazyinitoff-warmup15k-anneal40k-fixed70k-b96-permute-block'
wandb_run_name = 'seq384-try6-seed2026-sameperm-fixedl0h4-eigloss-withoutnone-directed-ribbon-signneg-lazyinitoff-warmup15k-anneal40k-fixed70k-b96-permute'

attn_mlp_policy_log_input_attn_prefix = 'seq384_block96_try6_seed2026_fixedl0h4_eigloss_withoutnone_signneg_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/seq384_block96_try_6_seed2026_fixedl0h4_signneg/input_attn'

head_asym_selector_out_dir = 'Report/MLP_loss_training/seq384_block96_try_6_seed2026_fixedl0h4_signneg/head_asym_selector'
