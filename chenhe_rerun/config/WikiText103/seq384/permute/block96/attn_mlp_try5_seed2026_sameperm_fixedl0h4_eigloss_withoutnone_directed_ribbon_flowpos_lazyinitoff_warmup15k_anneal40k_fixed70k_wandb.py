# WikiText103 seq384 permuted block96 try_5.
#
# Based on try_2, but disables automatic head selection and uses a fixed
# current-frame attention head: L0H4.
#
# Schedule is unchanged from try_2:
# - 0..15k: pure Random warmup
# - 15k..40k: train Attn-MLP/order
# - 40k..70k: freeze cached MLP order and continue backbone training

_base_config = 'config/WikiText103/seq384/permute/block96/attn_mlp_try2_seed2026_sameperm_autohead_eigloss_withoutnone_directed_ribbon_lazyinit_warmup15k_anneal40k_fixed70k_wandb.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

# Fixed-head variant: keep try_2 seed/permutation/sign, but do not run selector.
head_asym_selector_enabled = False
head_asym_selector_assign_to_attn_mlp_policy = False
head_asym_selector_init_attn_mlp_policy = False
head_asym_selector_save_all_head_maps = False

attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = 4
attn_mlp_policy_export_type = 'without_none'

# With selector disabled, initialize the policy at startup using the fixed head.
# It still cannot affect the block order before attn_mlp_policy_start_iter=15000.
attn_mlp_policy_lazy_init_enabled = False
attn_mlp_policy_lazy_init_iter = 0

out_dir = 'out/base/permute/seq384/block96/out-wikitext103-seq384-try5-seed2026-sameperm-fixedl0h4-eigloss-withoutnone-directed-ribbon-flowpos-lazyinitoff-warmup15k-anneal40k-fixed70k-b96-permute-block'
wandb_run_name = 'seq384-try5-seed2026-sameperm-fixedl0h4-eigloss-withoutnone-directed-ribbon-flowpos-lazyinitoff-warmup15k-anneal40k-fixed70k-b96-permute'

attn_mlp_policy_log_input_attn_prefix = 'seq384_block96_try5_seed2026_fixedl0h4_eigloss_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/seq384_block96_try_5_seed2026_fixedl0h4/input_attn'

head_asym_selector_out_dir = 'Report/MLP_loss_training/seq384_block96_try_5_seed2026_fixedl0h4/head_asym_selector'
