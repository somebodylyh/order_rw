# EMA-MAP distribution variant.
#
# Same schedule/budget as the warmup1k/anneal8k distribution run:
# - 0..1000: pure random training orders, no online probe/rerank updates
# - 1000..8000: linearly anneal policy usage from 0 to 1
# - 8000..40000: train from the current EMA MAP order, still updating EMA online
# - 40000..50000: freeze updates and continue with the final EMA MAP order
#
# Difference from the base fast_b256 warmup1k run:
# - no Gumbel/temperature sampling during policy training
# - maintain priority_ema exactly as before, but use argsort(priority_ema)

_base_config = 'config/WikiText103/seq256/non_permute/block64/online_spectral_order_distribution_loss_rerank_fast_b256_update20_top32_loss1024_warmup1k_anneal8k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/nonpermute/seq256/block64/out-wikitext103-seq256-online-spectral-order-distribution-ema-map-loss-rerank-fast-b256-update20-start1k-top32-loss1024-warmup1k-anneal8k-freeze40k-map-b64-nonpermute-50000-iters'
wandb_run_name = 'seq256-online-spectral-order-distribution-ema-map-loss-rerank-fast-b256-update20-start1k-top32-loss1024-warmup1k-anneal8k-freeze40k-map-b64-nonpermute-50000-iters'

online_spectral_policy_distribution_sample_mode = 'map'
online_spectral_policy_distribution_per_sample = False
online_spectral_policy_random_mix_prob = 0.0
# Ignored when distribution_sample_mode='map'; kept only for config compatibility.
online_spectral_policy_sample_temperature = 1.0
