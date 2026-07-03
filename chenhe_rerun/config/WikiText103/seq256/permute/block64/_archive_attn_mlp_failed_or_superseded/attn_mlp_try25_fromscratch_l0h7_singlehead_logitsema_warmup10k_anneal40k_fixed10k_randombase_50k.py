# WikiText103 seq256 permuted block64 try_25:
# Same no-prior trainable Attn-MLP setup as try23, but cached-order generation
# uses EMA-smoothed normalized logits before argsort.
#
# Controlled change relative to try23:
# - Direct MLP attention input remains L0H7 with with_none aggregation.
# - Head-profile consensus still uses L1H2, L2H7, and L2H6.
# - Schedule, loss weights, MLP capacity, and no-prior target gate are unchanged.
# - Only the cache/inference order source changes:
#   raw logits -> z-score -> EMA -> argsort_desc.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, runtime target acceptance, or selection. Tau is diagnostic
# only after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try23_fromscratch_l0h7_singlehead_warmup10k_anneal40k_fixed10k_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try25-fromscratch-l0h7-singlehead-logitsema-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try25-fromscratch-l0h7-singlehead-logitsema-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute'

# Main ablation: smooth the continuous MLP output before the discrete argsort.
# This does not change MLP training loss and does not use original/tau signals.
attn_mlp_policy_logits_ema_enabled = True
attn_mlp_policy_logits_ema_decay = 0.95
attn_mlp_policy_logits_ema_normalize = 'zscore'
attn_mlp_policy_logits_ema_start_iter = 10000
