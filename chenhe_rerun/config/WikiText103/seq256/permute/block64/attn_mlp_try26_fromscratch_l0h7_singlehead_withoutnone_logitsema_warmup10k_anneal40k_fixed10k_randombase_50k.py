# WikiText103 seq256 permuted block64 try_26:
# Same no-prior trainable Attn-MLP setup as try25, but switch both attention
# aggregation paths from with_none to without_none.
#
# Controlled change relative to try25:
# - Direct MLP attention input remains single-head L0H7.
# - Head-profile consensus still uses L1H2, L2H7, and L2H6.
# - Schedule, loss weights, MLP capacity, target gate, and logits EMA are
#   unchanged.
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

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try26-fromscratch-l0h7-singlehead-withoutnone-logitsema-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try26-fromscratch-l0h7-singlehead-withoutnone-logitsema-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute'

# Logits-EMA smoothing inherited from the archived try25 branch. It affects
# cached-order generation only: raw logits -> z-score -> EMA -> argsort_desc.
attn_mlp_policy_logits_ema_enabled = True
attn_mlp_policy_logits_ema_decay = 0.95
attn_mlp_policy_logits_ema_normalize = 'zscore'
attn_mlp_policy_logits_ema_start_iter = 10000

# Main ablation: keep try25 logits EMA smoothing, but remove the reveal-time
# None position from both attention aggregation paths. The final MLP input
# remains 3 x 64 x 64.
attn_mlp_policy_export_type = 'without_none'
head_signal_probe_export_type = 'without_none'
