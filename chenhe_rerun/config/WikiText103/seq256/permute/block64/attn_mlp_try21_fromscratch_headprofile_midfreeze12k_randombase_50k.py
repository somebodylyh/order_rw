# WikiText103 seq256 permuted block64 try_21:
# try20/try15-style no-prior head-profile Attn-MLP, random-base aligned,
# but freeze the policy update loop in the middle of training.
#
# Motivation:
# try19 showed a strong mid-training order around 11k-12k, followed by later
# degradation while the backbone kept improving. This config tests whether
# freezing the MLP/order at that window preserves the useful order and gives
# the backbone a longer fixed-order adaptation phase.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, runtime target acceptance, or selection. Tau is diagnostic
# only after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try17_fromscratch_headprofile_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try21-fromscratch-headprofile-midfreeze12k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try21-fromscratch-headprofile-midfreeze12k-randombase-50k-b64-permute'

# Evidence-driven mid-training freeze point.
#
# In try19, the useful target appeared around 11k and the cached-order tau
# peaked near 12k before later drift. update_stop_iter stops attention
# collection, MLP policy optimizer steps, and cached-order refreshes from this
# iteration onward. The parameters are not toggled to requires_grad=False, but
# the MLP receives no more optimizer updates after the stop.
attn_mlp_policy_update_stop_iter = 12000

# Make the frozen cached order fully active immediately after the policy is
# stopped, so 12k-50k is a long fixed-order backbone adaptation phase.
attn_mlp_policy_anneal_end_iter = 12000

# Same-frame target stability inherited from the archived try20 branch:
# score candidate/current/best on the same train-split probe batches, accept a
# new head-profile target only by current-frame loss-profile score, and never
# use original order or tau for the decision.
attn_mlp_policy_head_profile_accept_gate_enabled = True
attn_mlp_policy_head_profile_accept_antialigned_only = False
attn_mlp_policy_head_profile_accept_alignment_threshold = 0.0
attn_mlp_policy_head_profile_accept_margin = 0.001
attn_mlp_policy_head_profile_best_memory_enabled = True
attn_mlp_policy_head_profile_best_memory_use_for_fixed = False

# Reduce noise in the same-frame candidate/current/best comparison without
# changing the random-base backbone schedule or optimizer settings.
head_signal_probe_loss_batches = 4
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8
