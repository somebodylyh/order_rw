# WikiText103 seq256 permuted block64 try_23:
# Same no-prior trainable Attn-MLP method as try22, but the attention feature
# fed to the MLP is restricted to one directional head: layer 0, head 7.
#
# Schedule requested:
# - 0-10k: pure Random backbone warmup while MLP can shadow-train from current
#          internal signals.
# - 10k-40k: anneal from Random to MLP order.
# - 40k-50k: fixed cached MLP order for backbone adaptation.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, runtime target acceptance, or selection. Tau is diagnostic
# only after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try21_fromscratch_headprofile_midfreeze12k_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try23-fromscratch-l0h7-singlehead-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try23-fromscratch-l0h7-singlehead-warmup10k-anneal40k-fixed10k-randombase-50k-b64-permute'

# Parameter retune inherited from the archived try22 branch. No new training
# signal or original-order prior is added here; this is capacity/update/noise
# tuning used by the retained single-head line.
attn_mlp_policy_update_every = 2
attn_mlp_policy_ema_decay = 0.90

attn_mlp_policy_hidden_dims = '1536,1536'
attn_mlp_policy_lr = 6e-5
attn_mlp_policy_weight_decay = 0.005

attn_mlp_policy_head_profile_weight = 3.0
attn_mlp_policy_head_profile_every = 500
attn_mlp_policy_head_profile_accept_margin = 0.0002

attn_mlp_policy_pg_weight = 0.025
attn_mlp_policy_prefix_reward_weight = 0.025
attn_mlp_policy_move_pref_weight = 0.75
attn_mlp_policy_attn_pair_weight = 0.35

attn_mlp_policy_std_floor_weight = 0.02
attn_mlp_policy_entropy_floor_weight = 0.05
attn_mlp_policy_entropy_ceiling_weight = 0.01

head_signal_probe_batches = 12
head_signal_probe_loss_batches = 8
head_signal_probe_loss_batch_size = 16
head_signal_probe_candidate_batch_size = 8

# MLP input attention source.
# `head >= 0` means a single head is used. This replaces the previous
# layer-2 all-head mean input (`head=-1`) with the directional L0H7 head.
attn_mlp_policy_use_global = False
attn_mlp_policy_layer = 0
attn_mlp_policy_head = 7

# Requested schedule.
attn_mlp_policy_start_iter = 10000
attn_mlp_policy_start_prob = 0.0
attn_mlp_policy_end_prob = 1.0
attn_mlp_policy_anneal_start_iter = 10000
attn_mlp_policy_anneal_end_iter = 40000
attn_mlp_policy_update_stop_iter = 40000

# Keep shadow MLP training during the 0-10k random warmup so the policy can
# already learn from L0H7 attention/loss features before it affects backbone
# orders.
attn_mlp_policy_collect_warmup_attention = True

# Keep the inherited try22 loss weights/capacity/probe settings, but restrict
# the head-profile consensus auxiliary target to the three heads requested by
# the user. This keeps the direct MLP attention input at L0H7 while making the
# auxiliary pairwise target a focused consensus over L1H2, L2H7, and L2H6.
head_signal_probe_heads = '1:2,2:7,2:6'
