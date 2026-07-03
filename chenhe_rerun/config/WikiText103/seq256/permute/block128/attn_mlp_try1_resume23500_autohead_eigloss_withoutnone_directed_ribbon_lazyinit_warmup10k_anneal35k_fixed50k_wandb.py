# Resume helper for seq256/block128 Attn-MLP try_1.
#
# This does not define a new experimental method. It only resumes the
# interrupted try_1 run from out_dir/ckpt.pt while keeping the same schedule,
# data permutation, head-selection result, MLP loss, and fixed-order phase.

_base_config = 'config/WikiText103/seq256/permute/block128/attn_mlp_try1_fromscratch_autohead_eigloss_withoutnone_directed_ribbon_lazyinit_warmup10k_anneal35k_fixed50k_wandb.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

init_from = 'resume'
resume_optimizer_state = True

# The 10k selector chose L0H7 without_none. These assignments make the resume
# config self-documenting; the checkpoint also restores them from
# attn_mlp_policy_state.
attn_mlp_policy_layer = 0
attn_mlp_policy_head = 7
attn_mlp_policy_export_type = 'without_none'

wandb_run_name = 'seq256-try1-autohead-eigloss-withoutnone-directed-ribbon-lazyinit-warmup10k-anneal35k-fixed50k-b128-permute-resume23500'
