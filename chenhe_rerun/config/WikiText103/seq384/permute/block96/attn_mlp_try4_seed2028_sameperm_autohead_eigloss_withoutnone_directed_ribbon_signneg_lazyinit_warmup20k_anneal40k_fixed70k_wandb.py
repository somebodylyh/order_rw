# WikiText103 seq384 permuted block96 try_4.
#
# Same no-prior autohead Attn-MLP method, with a longer Random warmup.
# The data permutation seed stays fixed for comparability.
#
# Schedule:
# - 0..20k: pure Random warmup; Attn-MLP lazy init is still pending.
# - iter 20k: select one head from current-frame without_none attention.
# - 20k..40k: train MLP/order with directed-ribbon loss and MLP LR anneal.
# - 40k..70k: fixed cached MLP order.

_base_config = 'config/WikiText103/seq384/permute/block96/attn_mlp_try1_fromscratch_autohead_eigloss_withoutnone_directed_ribbon_lazyinit_warmup15k_anneal40k_fixed70k_wandb.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2028
permute_seed = 42
head_asym_selector_seed = 24681357

attn_mlp_policy_directed_ribbon_flow_sign = -1.0

attn_mlp_policy_lazy_init_iter = 20000
attn_mlp_policy_start_iter = 20000
attn_mlp_policy_anneal_start_iter = 20000
attn_mlp_policy_anneal_end_iter = 40000
attn_mlp_policy_update_stop_iter = 40000
attn_mlp_policy_lr_anneal_start_iter = 20000
attn_mlp_policy_lr_anneal_end_iter = 40000
attn_mlp_policy_loss_stop_start_iter = 30000
head_asym_selector_target_iter = 20000

out_dir = 'out/base/permute/seq384/block96/out-wikitext103-seq384-try4-seed2028-sameperm-autohead-eigloss-withoutnone-directed-ribbon-signneg-lazyinit-warmup20k-anneal40k-fixed70k-b96-permute-block'
wandb_run_name = 'seq384-try4-seed2028-sameperm-autohead-eigloss-withoutnone-directed-ribbon-signneg-lazyinit-warmup20k-anneal40k-fixed70k-b96-permute'

attn_mlp_policy_log_input_attn_prefix = 'seq384_block96_try4_seed2028_autohead_eigloss_withoutnone_signneg_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/seq384_block96_try_4_seed2028_warmup20k_signneg/input_attn'

head_asym_selector_out_dir = 'Report/MLP_loss_training/seq384_block96_try_4_seed2028_warmup20k_signneg/head_asym_selector'
