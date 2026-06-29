# 5k Signal Provenance Audit

Run date: 2026-06-26

## Question

The current new 5k random-baseline run looked weak under the matched strict65 CDL audit, but earlier notes claimed that 5k already had strong order signal. This audit separates four possibilities:

1. the old 5k signal was real but came from a different checkpoint;
2. the old and current diagnostics measured different signals;
3. the old signal came from a specific data/protocol/model setting;
4. the current diagnostic configuration hid a signal that another valid diagnostic can see.

## Provenance Of Earlier 5k Signal Claims

| source | checkpoint / protocol | metric | reported 5k result | same as current L0 consensus CDL-vs-L2R? |
|---|---|---|---:|---|
| `large_random_baseline_16l16h1024d/head_scan_step5000.json` | 317M, 16L16H, chunks, seed 42, B0 per-head scan | sparse per-head CDL tau | best L0H10 tau=+0.955; 6/256 heads with abs(tau)>0.9 | no: different model scale and sparse per-head metric |
| `analyses/block_granularity_scan_results/scan_step5000_M100.json` | `clean_base_random_perm/ckpt_step5000.pt`, B0 granularity scan | heavy/aggregate tau | 32blk heavy tau=0.9997; 64blk heavy tau=0.9999 | no: heavy aggregate, not strict65 per-head/consensus |
| `clean_base_none65_stability/step5000` | `clean_base_random_perm/ckpt_step5000.pt`, old none-separated 65 scan, M=8 | all-layer per-head method search | L1H2/L1H4/L1H0/L1H1/L2H0 tau=1.000 strong-pass | no: all-layer selected heads, old ckpt missing locally |
| current matched audit | jun08/jun25 5k, strict65, M=500, bs_mean=4 | L0-only consensus and L0 per-head tau | old consensus=-0.1235, new consensus=-0.0196 | yes: this is the current weak metric |

Important correction: the Jun13 verified small-model table's strongest clean-base per-head claim is mostly 10k, not 5k: `clean_base_random_perm/head_scan_10k.json` had 9/32 heads with abs(tau)>0.9 and best L0H0 tau=1.000.

## Old Diagnostic Re-run On Available 5k Checkpoints

I re-ran the legacy none-separated 65 head/method search:

```text
scripts/search_none_separated_65_heads.py
M=20, batch_size=8, seed=0, perm_orientation=phys_to_model
methods: C-D+L, L
```

| ckpt | old diagnostic best candidate | gate | interpretation |
|---|---|---|---|
| `random_baseline_continuous_jun08_seed2/ckpt_step5000.pt` | L2H6 L tau=0.952, phys0_rank=1 | weak_pass | signal exists but misses strict first-block gate |
| `overnight_20260625_random_baseline/ckpt_step5000.pt` | L1H7 L tau=1.000, phys0_rank=0 | strong_pass | old diagnostic sees a strong 5k per-head signal |

The new jun25 5k checkpoint is therefore not simply "signal absent." It has a strong old-diagnostic, all-layer, per-head none-separated signal, while the current matched audit reported weak L0-only consensus/static CDL-vs-L2R.

## Current Diagnostic Result For Same 5k Checkpoints

From `reports/old_new_ckpt_matched_audit_20260626/summary.md`:

| ckpt | current diagnostic | consensus tau vs L2R | top L0 head tau | note |
|---|---|---:|---:|---|
| jun08 old 5k | strict65, L0 all heads, M=500, bs_mean=4 | -0.1235 | H0 -0.3135 | L0 consensus weak/negative |
| jun25 new 5k | strict65, L0 all heads, M=500, bs_mean=4 | -0.0196 | H6 +0.0771 | L0 consensus weak |

This current diagnostic only scans L0. The old diagnostic that found strong jun25 5k signal found it in L1H0/L1H7, so a direct contradiction is avoided: the layer/head selection changed.

## 2x2 Audit Status

| checkpoint | old diagnostic | current diagnostic |
|---|---|---|
| historical `clean_base_random_perm/ckpt_step5000.pt` | saved result: strong-pass heads, several tau=1.000 | cannot rerun: checkpoint file is absent locally |
| jun08 old 5k | re-run result: L2H6 tau=0.952 weak_pass | current result: L0 consensus=-0.1235 |
| jun25 new 5k | re-run result: L1H0/L1H7 tau=1.000 strong_pass | current result: L0 consensus=-0.0196 |

The available 2x2 evidence points to diagnostic/signal-definition mismatch more than pure checkpoint absence. In particular:

- old diagnostic = all-layer selected per-head method search;
- current diagnostic = L0-only consensus/per-head summary;
- old diagnostic rolls out specific methods (`L`, `C-D+L`) from a mean 65-node graph;
- current diagnostic emphasizes consensus CDL tau across L0 heads and fixed M=500 batch-mean extraction;
- old claims mix per-head, heavy aggregate, 317M-scale, and strict65 discovery evidence.

## g_beta Provenance

The existing `l0_dynamic_gbeta_step5k_M2000_listmle_s123` run measures teacher-match training behavior, not CDL-vs-L2R static alignment. Its best epoch was 11 with validation pairwise accuracy about 0.751, and later epochs stayed around 0.75. That is a separate signal category and should not be used as evidence that static L2R consensus is high.

## Conclusion

Do not write "5k static signal is generally weak." The more accurate statement is:

Current new-ckpt 5k is weak under the L0-only strict65 consensus CDL-vs-L2R diagnostic, but the legacy all-layer none-separated per-head diagnostic finds strong 5k order-bearing heads in the same new checkpoint. The earlier "5k has signal" records were not one homogeneous metric; they included sparse per-head, heavy aggregate, different model scale, and old none-separated head/method search results.

The next clean experiment should report three channels separately:

1. L0 consensus/static CDL-vs-L2R;
2. all-layer selected per-head none-separated discovery;
3. g_beta teacher-match against the full CDL teacher.

For from5k intervention, the defensible claim is not "static L0 consensus is strong at 5k." It is: "some 5k checkpoints contain selected order-bearing heads under the old all-layer discovery protocol; g_beta/from5k should be evaluated by same-checkpoint paired continuation and teacher-match metrics."
