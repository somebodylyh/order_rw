# Faster-batch variant of online_spectral_order_distribution_loss_rerank.py.
#
# Keeps the same policy sample budgets as the original config:
# - attention probe: 32 x 128 = 4096 samples
# - loss rerank:     16 x 128 = 2048 samples per candidate

_base_config = 'config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-order-distribution-loss-rerank-fast-b128-no-random-mix-freeze40k-map-b64-nonpermute-50000-iters'
wandb_run_name = 'seq256-online-spectral-order-distribution-loss-rerank-fast-b128-no-random-mix-freeze40k-map-b64-nonpermute-50000-iters'
compile = False
online_spectral_policy_update_stop_iter = 40000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_probe_batches = 32
online_spectral_policy_probe_batch_size = 128

online_spectral_policy_loss_rerank_batches = 16
online_spectral_policy_loss_rerank_batch_size = 128
online_spectral_policy_loss_rerank_candidate_batch_size = 1

online_spectral_policy_random_mix_prob = 0.0
