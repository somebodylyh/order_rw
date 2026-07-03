# L0H7 counterpart of the clean single-head direct_asym_eig run.

_base_config = 'config/WikiText103/seq256/permute/block64/online_spectral_direct_asym_eig_l1h2_withnone_map_update20_warmup10k_anneal35k_freeze35k.py'
with open(_base_config, 'r', encoding='utf-8') as _handle:
    exec(_handle.read())

out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-direct-asym-eig-l0h7-withnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-block-50000-iters'

wandb_run_name = 'seq256-online-spectral-direct-asym-eig-l0h7-withnone-map-update20-warmup10k-anneal35k-freeze35k-b64-permute-50000-iters'

head_signal_probe_out_dir = 'Report/head_singal_Stable/direct_asym_eig_l0h7_withnone_map_update20_warmup10k_anneal35k_freeze35k_permute/online_training_probe'
head_signal_probe_heads = '0:7'
