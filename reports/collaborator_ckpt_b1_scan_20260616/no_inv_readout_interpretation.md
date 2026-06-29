# No-Inv B1 Readout Diagnostic

Date: 2026-06-16

Checkpoint: `/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt`

Main source:

- `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json`

## Question

Can raw model-frame B1 attention read out the original L2R order without using `inv_perm` during extraction or CDL?

## Protocol

- Build B1 maps in model-block coordinates only.
- Do not use `inv_perm` or physical remap in B construction.
- Run CDL on the raw model-frame B.
- Use checkpoint `block_perm` only after readout to score whether the returned model-block order corresponds to original physical L2R.
- Compare full B1 with `[None]` kept/folded versus content-only B1 with predictor row/column 0 removed.

## Key Results

### Full B1 with `[None]` kept

Top anchored posthoc tau:

- `L1H1`: posthoc tau vs ori-L2R = `0.942460`; best cyclic tau = `0.955357`.
- `L0H7`: posthoc tau vs ori-L2R = `0.835317`; best cyclic tau = `0.848214`.

Perfect cyclic order, but wrong anchor:

- `L0H1`: posthoc tau = `0.019841`; best cyclic tau = `1.000000`; first physical labels = `[38, 39, 40, 41, 42, 43, 44, 45]`.
- `L0H2`: posthoc tau = `0.019841`; best cyclic tau = `1.000000`; first physical labels = `[38, 39, 40, 41, 42, 43, 44, 45]`.
- `L0H3`: posthoc tau = `0.019841`; best cyclic tau = `1.000000`; first physical labels = `[38, 39, 40, 41, 42, 43, 44, 45]`.

Interpretation: keeping `[None]` can anchor CDL to model block 0. In this collaborator checkpoint, model block 0 corresponds to physical block 38, so these heads recover a perfect physical cyclic L2R order but not the anchored `0..63` ori-L2R order.

### Content-only B1

- `L0H3`: posthoc tau vs ori-L2R = `1.000000`; first physical labels = `[0, 1, 2, 3, 4, 5, 6, 7]`.
- `L0H1/L0H2/L0H4`: posthoc tau = `0.654762`; best cyclic tau = `1.000000`; first physical labels = `[6, 7, 8, 9, 10, 11, 12, 13]`.
- `L0H5`: posthoc tau = `0.700397`; best cyclic tau = `0.894841`.

Interpretation: a simple CDL readout from raw model-frame content attention can recover the model-block ordering that corresponds to original L2R for at least one selected head (`L0H3`), without using `inv_perm` inside extraction or readout. The permutation is used only for posthoc scoring.

## Bottom Line

The earlier physical-frame high tau is not purely a `[None]` artifact, but the old physical-frame diagnostic did use `inv_perm`.

The stronger label-free statement supported by this no-inv test is:

> Raw model-frame B1 attention contains a latent model-block ordering that corresponds to original L2R. For `L0H3`, content-only CDL recovers this order exactly under posthoc validation. With `[None]` kept, several heads recover a perfect cyclic L2R order, but `[None]` can anchor the cycle at model block 0 rather than physical block 0.

Current caveat:

> Full `[None]`-kept CDL does not always recover the anchored ori-L2R start without either choosing the right head (`L1H1`) or resolving the cyclic anchor. Therefore the most robust no-inv readout is currently content-only `L0H3`; the full-none version supports cyclic order but has an anchor ambiguity.
