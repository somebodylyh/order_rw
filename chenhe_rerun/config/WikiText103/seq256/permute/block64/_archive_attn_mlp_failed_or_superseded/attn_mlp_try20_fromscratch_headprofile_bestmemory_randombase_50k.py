# WikiText103 seq256 permuted block64 try_20:
# try15/try17 head-profile no-prior MLP-order method, random-base aligned,
# plus same-frame best-memory retention for head-profile targets.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, runtime target acceptance, or best-memory selection. Tau is
# logged only as a diagnostic after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try17_fromscratch_headprofile_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try20-fromscratch-headprofile-bestmemory-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try20-fromscratch-headprofile-bestmemory-randombase-50k-b64-permute'

# Same-frame target stability:
#
# At each head-profile refresh, score the new target, the cached target, and
# the retained best-memory target on the same current train-split probe batches
# under the same current backbone. A new target replaces the retained target
# only if it improves the current-frame linear_profile score by this margin.
#
# This keeps the useful mid-training order candidate available while avoiding
# direct OriginalL2R/tau/validation-based selection.
attn_mlp_policy_head_profile_accept_gate_enabled = True
attn_mlp_policy_head_profile_accept_antialigned_only = False
attn_mlp_policy_head_profile_accept_alignment_threshold = 0.0
attn_mlp_policy_head_profile_accept_margin = 0.001
attn_mlp_policy_head_profile_best_memory_enabled = True
attn_mlp_policy_head_profile_best_memory_use_for_fixed = False

# Reduce noise in the same-frame candidate/current/best comparison without
# changing the random base backbone schedule or optimizer settings.
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
