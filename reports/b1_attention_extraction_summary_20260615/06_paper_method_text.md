# 06 - Paper Method Text

## Main B1 diagnostic method text

For attention diagnostics, we adopt a collaborator-aligned predictor-frame block extraction convention. Specifically, for each probe sequence we run the AO-GPT model with a randomized block reveal order and collect the causal self-attention tensor for each layer and head. We use the predictor-aligned attention frame `attn[:-1, :-1]`, which matches the model's prefix-to-next-token prediction layout, aggregate the resulting 256 by 256 token matrix into a 64 by 64 block matrix by averaging over the four query tokens and four key tokens in each block pair, and zero the diagonal.

For each head, we then group per-sample block matrices into probe batches and average within each batch. The graph consumed by the order readout is `B = A^T`, with the diagonal zeroed again after batch averaging. This convention keeps the attention diagnostic aligned with the predictor frame used by the model and avoids relying on the legacy `[None]`-source handling used in earlier internal analyses.

Unless otherwise stated, all post-alignment attention diagnostics use this B1/predictor-aligned extraction convention. Under this convention, the order-bearing signal remains strong: in the clean-base B1 scan the best head reaches `tau=1.000000` at 10k, 50k, and 60k, and in the latest continuous B1 all-head tracking the final best head is `L0H4` with `tau=0.957589` (`block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`).

The current frozen-controller training runs use the legacy B0 controller extraction path. The B1/predictor diagnostic is used to verify that the order-bearing signal is robust to the extraction convention. A B1 frozen-hook rerun would be needed before claiming B1-controller acceleration.

## Short method note for captions or appendix

Post-alignment attention diagnostics use the B1/predictor-frame extraction: token-level attention is taken from `attn[:-1, :-1]`, averaged into 64 block-pair cells, transposed to form `B=A^T`, and diagonal-zeroed before CDL/readout evaluation. Legacy frozen-g_beta controller runs used B0 extraction and are reported separately.
