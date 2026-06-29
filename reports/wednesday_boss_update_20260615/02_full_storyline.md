# Full Storyline

## Layer 1 - Motivation

AO-GPT / any-order autoregressive training allows a model to reveal blocks in arbitrary orders. The simplest training policy is random reveal order, but random order may be inefficient because it ignores structure in text. If the model's own attention already contains order information, we can read that structure out and use it as a training order controller.

The goal here is not to build a diffusion-language-model result, not to claim a solved image story, and not to optimize every possible evaluation order. The target is a text-side training-acceleration story: derive an order controller from AO-GPT attention and use it to reach a target canonical-order validation loss faster.

## Layer 2 - Mechanism Hypothesis

Hypothesis:

> Random-order AO-GPT attention contains sparse order-bearing heads. These heads encode a physical-L2R-like reveal order under the clean-permutation protocol.

Careful language:

- We can say order-bearing, L2R-like, sparse heads.
- We should not say all heads encode order.
- We should not say the model discovers orders beyond L2R.
- We should not say no-PE / PE-free order discovery.
- We should not say this is a universal permutation-robust likelihood gain.

The clean-permutation protocol matters: a fixed random block permutation is generated and held fixed; the model sees model-coordinate order, while diagnostics can inverse-remap to physical coordinates. This isolates whether attention recovers physical sequence structure instead of merely reading a per-sample permutation.

## Layer 3 - Method

Pipeline:

```text
random-order AO-GPT checkpoint
  -> selected per-head block attention graph B^{l,h}
  -> CDL teacher rollout
  -> train g_beta with pairwise ranking
  -> freeze g_beta
  -> hook back into AO-GPT training
  -> evaluate canonical-order validation loss
```

Definitions:

- `B^{l,h}` is a per-head, block-level directed attention graph extracted from one layer-head pair.
- CDL is an offline teacher / order extractor. It turns a graph into a reveal order through greedy `C-D+L` style decoding.
- `g_beta` is a learned readout trained to imitate teacher ordering from selected-head `B`. It is not online CDL at deployment.
- The frozen hook deploys a fixed `g_beta` inside AO-GPT training to choose the batch-global canonical order.
- The order is batch-global, not per-sample adaptive.
- The primary evaluation target is canonical-order validation loss, especially `val_ori_l2r_block`, not robustness to all random orders.

Current protocol state:

- Current verified `g_beta` sanity and frozen-hook acceleration are legacy B0 controller-path results.
- B1/predictor-aligned diagnostics are the post-alignment attention-signal robustness evidence.
- The program direction is B1 replacing B0; until B1 controller runs finish, claims should distinguish "B1 target protocol" from "current verified B0 acceleration evidence."

## Layer 4 - Evidence

### 1. Signal exists

B0 clean-base 10k scan shows sparse order-bearing attention heads:

- 9/32 heads with `|tau|>0.9`.
- Best head `L0H0 tau=+1.000`, pairwise tau=1.000.
- Source: `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json`.

This supports "sparse order-bearing heads", not dense all-head collapse.

### 2. Signal survives B1

B1/predictor-aligned diagnostics show the signal is not a B0 extraction artifact:

- Clean-base B1 ladder: `L0H0 tau=1.000000` at 10k, 50k, and 60k; `|tau|>0.9` count is 7/32 at 10k and 6/32 at 50k/60k.
- Latest continuous B1 tracking: final 60k best `L0H4 tau=0.957589`, `mean_pairwise_tau=0.930060`; `L0H4` is top in 5768/6001 tracked steps.
- Sources: `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json`; `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`.

Protocol caveat: result-bearing B1 files use `none_mode=predictor`, so write `B1/predictor-aligned diagnostic`.

### 3. `g_beta` reads `B`

Legacy B0 `g_beta` sanity confirms `g_beta` is not a constant L2R prior:

- Real selected-head B: `tau_vs_l2r=0.967500`, `tau_vs_teacher=0.949325`.
- Gaussian B: `tau_vs_l2r=-0.003353`.
- Entry-shuffled B: `tau_vs_l2r=0.016488`.
- Row/col shuffled B: `tau_vs_l2r=0.012748`.
- Gaussian family pairwise tau: `0.000313`.
- Zero B gives `tau_vs_l2r=1.0`, but margin is 0, so this is CDL tie-breaking.
- Source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`.

Boundary: this is B0 controller-path evidence; no B1 `g_beta` sanity was found.

### 4. Hook accelerates

Legacy B0 frozen `g_beta` improves canonical-order validation:

- seed123 L0H2 recovery @50k: from10k 86.0%, from20k 83.0%, from40k 58.7%.
- seed42 L0H4 recovery @50k: from10k 106.1%, from20k 101.5%, from40k 66.9%.
- Conservative step saving across from10k/from20k and both seeds: 33-42%.
- Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md` and the listed `eval_curve.tsv` files.

Boundary: acceleration is current verified B0 legacy hook path. It is not yet B1 hook acceleration.

### 5. Robustness

Granularity and scale diagnostics:

- B0 aggregation granularity 32/64/128 blocks: heavy tau > 0.96.
- B0 training granularity 32/64/128 groups: per-head `|tau|` around 1.0 in all three.
- B0 317M diagnostic: 6/256 heads with `|tau|>0.9`, best `L0H10 tau=0.955357`.
- Sources: `analyses/block_granularity_scan_results/scan_step5000_M100.json`; `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`; `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.

Boundary: no 317M frozen hook and no B1 317M scan were found in the verified package.

### 6. Boundaries

- B1 hook acceleration: pending / not established.
- B1 `g_beta` sanity: pending / not established.
- val_unstructured trade-off: frozen hook improves canonical-order loss but degrades arbitrary random-order evaluation, as expected for order specialization.
- Label-free audition: not fully closed end to end; current pipeline uses CDL teacher for `g_beta` training.
- CDL teacher matched seed123 run: marked in-progress in the verified package; no updated final result was incorporated here.

