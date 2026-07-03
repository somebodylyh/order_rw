# WikiText103 seq256 permuted block64 try_19:
# try15/try17 head-profile no-prior MLP-order method, random-base aligned,
# plus a current-frame-only target acceptance gate.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, or runtime target acceptance. Tau is logged only as a
# diagnostic after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try17_fromscratch_headprofile_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try19-fromscratch-headprofile-acceptgate-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try19-fromscratch-headprofile-acceptgate-randombase-50k-b64-permute'

# Current-frame-only stability gate for head-profile target refresh.
#
# If a new consensus Q is aligned with the cached Q, accept it normally.
# If it is anti-aligned, compare the new target order against the cached order
# using the same train-split loss-profile score used by try15. Accept the new
# target only if it improves that current-frame score by this margin.
#
# This does not use OriginalL2R, tau, validation PPL, or position anchors.
attn_mlp_policy_head_profile_accept_gate_enabled = True
attn_mlp_policy_head_profile_accept_antialigned_only = True
attn_mlp_policy_head_profile_accept_alignment_threshold = 0.0
attn_mlp_policy_head_profile_accept_margin = 0.001
