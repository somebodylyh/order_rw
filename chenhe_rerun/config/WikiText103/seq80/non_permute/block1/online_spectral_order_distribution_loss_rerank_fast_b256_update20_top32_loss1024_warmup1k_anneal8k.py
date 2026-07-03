# WikiText103 seq80 non-permuted token-level online spectral distribution
# loss-rerank config. This mirrors the seq256/block64 fast_b256 warmup1k
# anneal8k setup, but inherits the seq80/block1 token-level base config.
#
# Schedule:
# - 0..1000: pure random training orders, no online probe/rerank updates
# - 1000..8000: linearly anneal policy usage from 0 to 1
# - 8000..40000: train from learned distribution, still updating online
# - 40000..50000: freeze updates and train with the current MAP order
#
# Rerank budget:
# - attention probe: 16 x 256 = 4096 samples
# - loss rerank:     32 candidates x (4 x 256 = 1024 samples)

_base_config = 'config/WikiText103/seq80/non_permute/block1/online_spectral_order_distribution.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq80/block1/out-wikitext103-seq80-online-spectral-order-distribution-loss-rerank-fast-b256-update20-start1k-top32-loss1024-warmup1k-anneal8k-no-random-mix-freeze40k-map-b1-nonpermute-50000-iters'
wandb_run_name = 'seq80-online-spectral-order-distribution-loss-rerank-fast-b256-update20-start1k-top32-loss1024-warmup1k-anneal8k-no-random-mix-freeze40k-map-b1-nonpermute-50000-iters'
compile = False
eval_batch_size = 64

online_spectral_policy_anneal_start_iter = 1000
online_spectral_policy_anneal_end_iter = 8000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 1000
online_spectral_policy_update_stop_iter = 40000
online_spectral_policy_freeze_to_map_order_after_stop = True

online_spectral_policy_probe_batches = 16
online_spectral_policy_probe_batch_size = 256

online_spectral_policy_loss_rerank_enabled = True
online_spectral_policy_loss_rerank_top_k = 32
online_spectral_policy_loss_rerank_batches = 4
online_spectral_policy_loss_rerank_batch_size = 256
online_spectral_policy_loss_rerank_candidate_batch_size = 1

online_spectral_policy_random_mix_prob = 0.0
