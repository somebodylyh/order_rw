# WikiText103 seq384 permuted block96 try_7.
#
# Oracle head-selection ablation based on try_2.
#
# The only methodological change is at the 15k head-selection step:
# - recover a direct_asym_eig order for every head
# - compare raw/reverse orientation by original-frame Kendall tau
# - choose the head whose best orientation has the highest original tau
#
# After this oracle/prior head choice, the downstream Attn-MLP training path is
# unchanged: current-frame attention input, directed-ribbon loss, 15k..40k MLP
# updates, and fixed cached order through 70k. Original tau/order is not used
# inside the later MLP loss or order refresh.

_base_config = 'config/WikiText103/seq384/permute/block96/attn_mlp_try2_seed2026_sameperm_autohead_eigloss_withoutnone_directed_ribbon_lazyinit_warmup15k_anneal40k_fixed70k_wandb.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

head_asym_selector_selection_rule = 'oracle_direct_asym_eig_original_tau'
head_asym_selector_wandb_prefix = 'head_asym_selector_oracle_tau'

out_dir = 'out/base/permute/seq384/block96/out-wikitext103-seq384-try7-seed2026-sameperm-oraclehead-directasymtau-withoutnone-directed-ribbon-flowpos-lazyinit-warmup15k-anneal40k-fixed70k-b96-permute-block'
wandb_run_name = 'seq384-try7-seed2026-sameperm-oraclehead-directasymtau-withoutnone-directed-ribbon-flowpos-lazyinit-warmup15k-anneal40k-fixed70k-b96-permute'

attn_mlp_policy_log_input_attn_prefix = 'seq384_block96_try7_seed2026_oraclehead_directasymtau_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/seq384_block96_try_7_seed2026_oraclehead_directasymtau/input_attn'

head_asym_selector_out_dir = 'Report/MLP_loss_training/seq384_block96_try_7_seed2026_oraclehead_directasymtau/head_asym_selector'
