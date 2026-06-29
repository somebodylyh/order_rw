# Label-Free Selector Audit

Run date: 2026-06-26

## Goal

Separate oracle audit from a label-free carrier selection protocol.

The selector must not use:

- tau vs physical L2R;
- first block equals physical block 0;
- prefix overlap with physical L2R;
- oracle strong/weak pass labels;
- downstream test NLL to choose among heads.

I implemented `scripts/label_free_selector_audit.py`, which reads saved `A_with_none_lh_mean.npy` tensors and computes graph-intrinsic features only. Oracle JSON is loaded only for post-hoc reporting.

## Inputs

| ckpt | A source | oracle source |
|---|---|---|
| new 5k | `reports/5k_signal_provenance_audit_20260626/old_diag_jun25_5k/A_with_none_lh_mean.npy` | `.../all_head_methods_none_separated_65.json` |
| new 10k | `reports/10k_signal_carrier_layer_multiseed_20260626/seed123_new/A_with_none_lh_mean.npy` | `.../all_head_methods_none_separated_65.json` |

Methods scanned: `C-D+L`, `L`.

## Selector Variants

### Variant A: margin + destroyed-control gap

This version uses:

- rollout margin;
- destroyed-control margin gap;
- rollout entropy;
- top-k mass;
- sink / column-mass penalties;
- tie fraction.

Result: it failed on both new 5k and new 10k.

| ckpt | top label-free pick | post-hoc tau | top-8 oracle strong hits |
|---|---|---:|---:|
| new 5k | L2H6 C-D+L | -0.707 | 0 |
| new 10k | L2H6 C-D+L | -0.707 | 0 |

Interpretation: margin/destroyed-gap alone can select a stable, non-random ordering source with the wrong physical orientation. This is a useful failure mode and should not be hidden.

### Variant B: structure-only sharpness / non-sink baseline

This version uses:

- low row entropy;
- high top-k mass;
- low sink argmax concentration;
- low column mass concentration;
- low tie fraction.

It does not use destroyed-control margin.

| ckpt | top structure-only pick | top-4 post-hoc selected | top-8 oracle strong hits |
|---|---|---|---:|
| new 5k | L1H7 C-D+L | L1H7 C-D+L, L1H7 L, L1H0 C-D+L, L1H0 L | 4 |
| new 10k | L1H7 C-D+L | L1H7 C-D+L, L1H7 L, L1H0 C-D+L, L1H0 L | 4 |

Post-hoc, all top-4 structure-only candidates have tau=1.000 and oracle `strong_pass`.

## Important Caveat

The structure-only selector also ranks reverse-order carriers highly:

- new 5k rank 5-8 includes L0H5/L1H2 with post-hoc tau near -1;
- new 10k rank 5-8 includes L1H2/L0H5 with post-hoc tau=-1.

So structure-only is promising as a label-free carrier detector, but it does not fully resolve orientation/sign ambiguity. It can find a stable ordering source, but not guarantee that the source is physical L2R without an additional label-free orientation criterion or a predeclared intervention evaluation.

## Current Conclusion

The strongest clean result is:

Using only graph-intrinsic sharpness/non-degeneracy features, the label-free selector recovers the same new-5k and new-10k L1 carriers found by the oracle audit. The L2R tau is revealed only after selection.

The negative result is equally important:

A minimal destroyed-margin selector is insufficient, because it selects L2H6, a stable but wrong-orientation carrier. This confirms that destroyed-control gap should not be based only on rollout margin.

## Recommended Next Protocol

For a paper-ready label-free selector, use structure-only as the base gate, then add one of:

1. cross-sample / bootstrap order consistency, computed before any L2R reveal;
2. g_beta validation learnability against the graph-derived teacher;
3. paired intervention evaluation with a predeclared top-k rule, reporting all top-k results rather than selecting the best test NLL.

Do not use tau, physical-first, prefix overlap, or test NLL to choose the head.

For the current new checkpoint, a defensible next experiment is:

- label-free select top-4 by structure-only score;
- train/evaluate g_beta or continuation for all top-4, or average them by a predeclared ensemble;
- only then reveal tau/intervention metrics.
