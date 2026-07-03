# try42: same trainable Attn-MLP recipe as try40, but removes hard head
# selection. The policy input is the layer-1 average over all attention heads:
# use_global=False, layer=1, head=-1.
#
# This is a no-prior layer-average ablation. Original-frame order/tau remain
# diagnostics only and are not used for selection, training, or early stop.

_base_try_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try40_fromscratch_autohead_eigloss_withoutnone_directed_ribbon_seed2029_sameperm_lazyinit_warmup10k_anneal35k_fixed50k_wandb.py'
with open(_base_try_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

# Controlled ablation against try40: keep train/data seeds fixed.
seed = 2029
permute_seed = 42

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try42-seed2029-sameperm-l1headavg-withoutnone-directed-ribbon-noheadselect-warmup10k-anneal35k-fixed50k-b64-permute-block'
wandb_run_name = 'seq256-try42-seed2029-sameperm-l1headavg-withoutnone-directed-ribbon-noheadselect-warmup10k-anneal35k-fixed50k-b64-permute'

# No selector: use the average of all heads in layer 1.
head_asym_selector_enabled = False
head_asym_selector_assign_to_attn_mlp_policy = False
head_asym_selector_init_attn_mlp_policy = False
head_asym_selector_out_dir = 'Report/MLP_loss_training/try_42/head_asym_selector_disabled'

attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 1
attn_mlp_policy_head = -1
attn_mlp_policy_export_type = 'without_none'

# With the selector disabled, initialize the policy at startup. It is still not
# used for ordering before start_iter because start_prob=0 and policy attention
# collection is gated by attn_mlp_policy_start_iter.
attn_mlp_policy_lazy_init_enabled = False
attn_mlp_policy_lazy_init_iter = 0

attn_mlp_policy_log_input_attn_prefix = 'try42_seed2029_l1headavg_withoutnone_input_attn'
attn_mlp_policy_log_input_attn_out_dir = 'Report/MLP_loss_training/try_42/input_attn'
