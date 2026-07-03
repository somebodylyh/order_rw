# WikiText103 seq256 permuted block64 try_22:
# Same no-prior head-profile Attn-MLP method as try20/try21, but with a
# parameter retune. No new training signal or original-order prior is added.
#
# Goal:
# Make the MLP adapt faster and score head-profile targets more robustly before
# the fixed-order phase. try21 froze correctly, but the retained order was only
# moderate. This run treats that as a hyperparameter issue.
#
# No-prior rule:
# OriginalL2R, original tau, oracle permutations, validation PPL, and
# hand-written original-order labels are not used for training, reranking,
# early stopping, runtime target acceptance, or selection. Tau is diagnostic
# only after an order has already been produced.

_base_config = 'config/WikiText103/seq256/permute/block64/attn_mlp_try21_fromscratch_headprofile_midfreeze12k_randombase_50k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-try22-fromscratch-headprofile-paramtuned-midfreeze15k-randombase-50k-b64-permute-block'

wandb_log = True
wandb_project = 'AOGPT-order-block-64-final'
wandb_run_name = 'seq256-try22-fromscratch-headprofile-paramtuned-midfreeze15k-randombase-50k-b64-permute'

# Parameter-only changes relative to try21:
# 1. Give the policy 3k more update time before freezing.
# 2. Let the MLP follow the current feature matrix faster.
# 3. Increase MLP capacity and learning rate moderately.
# 4. Make head-profile supervision stronger and less noisy.
# 5. Keep the same no-prior acceptance rule, but lower the improvement margin
#    because the probe estimate is now less noisy.

attn_mlp_policy_update_stop_iter = 15000
attn_mlp_policy_anneal_end_iter = 15000

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
