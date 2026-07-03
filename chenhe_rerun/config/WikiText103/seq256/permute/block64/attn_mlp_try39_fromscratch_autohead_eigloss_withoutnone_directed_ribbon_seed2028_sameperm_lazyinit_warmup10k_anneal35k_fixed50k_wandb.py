# try39: same training recipe as try35, but the 10k head selector changes from
# pure asymmetry argmax to asym-top-k + direct_asym_eig loss selection.
# Training seed differs from try35/36/37; data permutation seed stays fixed.

_base_try_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try35_fromscratch_autohead_withoutnone_directed_ribbon_lazyinit_warmup10k_anneal35k_fixed50k_wandb.py'
with open(_base_try_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

seed = 2028
permute_seed = 42
head_asym_selector_seed = 24681357

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try39-seed2028-sameperm-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup10k-anneal35k-fixed50k-b64-permute-block'
wandb_run_name = 'seq256-try39-seed2028-sameperm-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup10k-anneal35k-fixed50k-b64-permute'

attn_mlp_policy_log_input_attn_prefix = 'try39_seed2028_autohead_eigloss_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/try_39/input_attn'

head_asym_selector_out_dir = 'Report/MLP_loss_training/try_39/head_asym_selector'
head_asym_selector_selection_rule = 'asym_topk_direct_asym_eig_loss'
head_asym_selector_asym_top_k = 8
head_asym_selector_direct_asym_eig_mode = 'raw_right_largest_real_real'
head_asym_selector_loss_batches = 4
head_asym_selector_loss_batch_size = 16
head_asym_selector_loss_candidate_batch_size = 8
head_asym_selector_loss_prefix_k = 16
head_asym_selector_loss_score = 'linear_profile'
head_asym_selector_loss_exp_tau = 16.0
