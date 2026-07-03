# WikiText103 seq256 permuted block64 try_24:
# Same no-prior trainable Attn-MLP setup as try23, but the attention export
# semantics are switched from with_none to without_none.
#
# Controlled change relative to try23:
# - Direct MLP attention input uses the same L0H7 head.
# - Head-profile consensus still uses L1H2, L2H7, and L2H6.
# - Schedule, loss weights, MLP capacity, and no-prior target gate are unchanged.
# - Both attention paths aggregate the post-None 256 token positions:
#   layer_attn[:, :, 1:, 1:] -> block64.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, runtime target acceptance, or selection. Tau is diagnostic
# only after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try23_fromscratch_l0h7_singlehead_warmup10k_anneal40k_fixed10k_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try24-fromscratch-l0h7-singlehead-withoutnone-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try24-fromscratch-l0h7-singlehead-withoutnone-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute'

# Main ablation: remove the reveal-time None position from both attention
# aggregation paths. The final MLP input remains 3 x 64 x 64.
attn_mlp_policy_export_type = 'without_none'
head_signal_probe_export_type = 'without_none'
