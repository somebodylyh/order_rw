# Faster-batch, lighter-rerank variant for fixed-head online spectral order.
#
# This mirrors the seq256/block64 distribution fast_b256 warmup1k anneal8k
# setup, but uses OnlineSpectralFixedHeadOrder. After updates stop, fixed-head
# training keeps using the last cached order recovered before the stop point.
#
# Schedule:
# - 0..1000: pure random training orders, no online probe/rerank updates
# - 1000..8000: linearly anneal cached-order usage from 0 to 1
# - 8000..40000: train from the fixed-head cached order, still updating online
# - 40000..50000: freeze updates and train with the last cached order
#
# Rerank budget:
# - attention probe: 16 x 256 = 4096 samples
# - loss rerank:     32 candidates x (4 x 256 = 1024 samples)

_base_config = 'config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order_loss_rerank.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-fixed-head-loss-rerank-fast-b256-update20-start1k-top32-loss1024-warmup1k-anneal8k-freeze40k-cached-b64-nonpermute-50000-iters'
wandb_run_name = 'seq256-online-spectral-fixed-head-loss-rerank-fast-b256-update20-start1k-top32-loss1024-warmup1k-anneal8k-freeze40k-cached-b64-nonpermute-50000-iters'
compile = False

online_spectral_policy_anneal_start_iter = 1000
online_spectral_policy_anneal_end_iter = 8000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0

online_spectral_policy_update_every = 20
online_spectral_policy_update_start_iter = 1000
online_spectral_policy_update_stop_iter = 40000

online_spectral_policy_probe_batches = 16
online_spectral_policy_probe_batch_size = 256

online_spectral_policy_loss_rerank_top_k = 32
online_spectral_policy_loss_rerank_batches = 4
online_spectral_policy_loss_rerank_batch_size = 256
online_spectral_policy_loss_rerank_candidate_batch_size = 1

# Not used by fixed-head sampling, but set to 0 so shared logs do not imply
# distribution-style random mixing.
online_spectral_policy_random_mix_prob = 0.0
