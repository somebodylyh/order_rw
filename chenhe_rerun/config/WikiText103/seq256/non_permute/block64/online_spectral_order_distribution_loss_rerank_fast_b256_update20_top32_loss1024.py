# Faster-batch, lighter-rerank variant of online_spectral_order_distribution_loss_rerank.py.
#
# Compared with fast_b256_update5:
# - attention probe: 16 x 256 = 4096 samples
# - loss rerank:     32 candidates x (4 x 256 = 1024 samples)
# - update every 20 steps instead of every 5 steps

_base_config = 'config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-order-distribution-loss-rerank-fast-b256-update20-top32-loss1024-no-random-mix-freeze40k-map-b64-nonpermute-50000-iters'
wandb_run_name = 'seq256-online-spectral-order-distribution-loss-rerank-fast-b256-update20-top32-loss1024-no-random-mix-freeze40k-map-b64-nonpermute-50000-iters'
compile = False

online_spectral_policy_update_every = 20
online_spectral_policy_update_stop_iter = 40000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_probe_batches = 16
online_spectral_policy_probe_batch_size = 256

online_spectral_policy_loss_rerank_top_k = 32
online_spectral_policy_loss_rerank_batches = 4
online_spectral_policy_loss_rerank_batch_size = 256
online_spectral_policy_loss_rerank_candidate_batch_size = 1

online_spectral_policy_random_mix_prob = 0.0
