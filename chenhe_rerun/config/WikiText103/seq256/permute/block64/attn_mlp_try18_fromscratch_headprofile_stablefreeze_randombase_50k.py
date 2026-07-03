# WikiText103 seq256 permuted block64 try_18:
# try17 head-profile no-prior Attn-MLP, but with a stability-first schedule.
#
# This config is intentionally runnable with the current train.py. It does not
# introduce new original-order, tau, oracle, validation-PPL, or hand-written
# order signals. It only changes when the no-prior head-profile target is
# refreshed and when the learned cached order is frozen.

_base_try17_config = (
    'config/WikiText103/seq256/permute/block64/'
    'attn_mlp_try17_fromscratch_headprofile_randombase_50k.py'
)
with open(_base_try17_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = (
    'out/base/permute/seq256/block64/'
    'out-wikitext103-seq256-try18-fromscratch-headprofile-stablefreeze-'
    'randombase-50k-b64-permute-block'
)

# Stability changes relative to try17:
#
# 1. Keep using the MLP cached order every step according to the normal
#    5k->20k annealing schedule; do not only "apply" the order every 1000 steps.
# 2. Refresh the expensive/noisy head-profile target every 2500 steps instead
#    of every 1000. This should reduce target flips like the try17 13k and
#    23k/24k events, while still refreshing at the 20k full-MLP boundary.
# 3. Freeze MLP/cache updates at 22k, i.e. after annealing reaches 100% at 20k
#    and after a short 2k consolidation window. This is a fixed no-prior
#    schedule, not a tau/PPL-triggered selection rule.

attn_mlp_policy_head_profile_every = 2500
attn_mlp_policy_update_stop_iter = 22000

# Slightly smoother feature EMA. MLP parameter updates remain every 4 steps so
# the policy can still learn quickly before the 22k stop point; the stabilization
# is applied to the target/input/order refresh path, not by starving optimization.
attn_mlp_policy_ema_decay = 0.98
attn_mlp_policy_update_every = 4

# Keep the same 50k total budget for baseline comparability.
max_iters = 50000
lr_decay_iters = 50000
