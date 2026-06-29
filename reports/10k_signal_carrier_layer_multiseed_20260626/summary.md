# 10k Signal Carrier Multiseed Ablation

Run date: 2026-06-26

## Protocol

Existing random-baseline 10k checkpoints scanned:

- seed42: `random_baseline_continuous_jun05/ckpt_step10000.pt`
- seed123_old: `random_baseline_continuous_jun08_seed2/ckpt_step10000.pt`
- seed2: `random_baseline_continuous_jun11_seed2_headtrack/ckpt_step10000.pt`
- seed124: `random_baseline_b1_headscan_seed124/ckpt_step10000.pt`
- seed123_new: `overnight_20260625_random_baseline/ckpt_step10000.pt`

Diagnostic:

```text
scripts/search_none_separated_65_heads.py
M=20, batch_size=8, fwd_batch=8, seed=0
perm_orientation=phys_to_model
methods: C-D+L, L
```

## Results

| run | best | strong rows | weak rows | strong layers | unique strong heads | late pass | early strong heads |
|---|---|---:|---:|---|---|---|---|
| `seed42` | L0H4 L tau=1.000 | 10 | 1 | L0 | L0H1, L0H2, L0H4, L0H5, L0H6 | L3H6 L tau=0.717 weak | L0H1, L0H2, L0H4, L0H5, L0H6 |
| `seed123_old` | L0H2 C-D+L tau=1.000 | 4 | 4 | L0 | L0H1, L0H2 | L2H3/H4/H5/H6 L weak, max tau=0.977 | L0H1, L0H2 |
| `seed2` | L1H6 L tau=1.000 | 12 | 1 | L0,L1 | L0H4, L0H5, L0H6, L0H7, L1H0, L1H2, L1H6 | - | L0H4, L0H5, L0H6, L0H7, L1H0, L1H2, L1H6 |
| `seed124` | L0H0 L tau=1.000 | 8 | 3 | L0 | L0H0, L0H1, L0H5, L0H6 | L2H1/H3 L weak, max tau=0.956 | L0H0, L0H1, L0H5, L0H6 |
| `seed123_new` | L1H7 C-D+L tau=1.000 | 7 | 2 | L0,L1,L2 | L0H1, L1H0, L1H7, L2H7 | L2H7 L tau=1.000 strong; L2H7 C-D+L tau=0.762 weak | L0H1, L1H0, L1H7 |

`strong rows` counts head-method rows, so one head can contribute two rows when both `C-D+L` and `L` pass.

## Layer Distribution

Strong-pass row counts:

| layer | strong rows |
|---|---:|
| L0 | 30 |
| L1 | 10 |
| L2 | 1 |
| L3 | 0 |

Across these five 10k random-baseline checkpoints:

- 5/5 have strong-pass selected heads.
- 4/5 have strong-pass heads only in L0/L1.
- 1/5 has an L2 strong-pass head: seed123_new has L2H7 L tau=1.000.
- 0/5 have L3 strong-pass heads.
- Every L2/L3 pass or near-pass co-occurs with L0/L1 strong-pass heads in the same checkpoint.

## Comparison To 5k

The 10k pattern is even more early-layer dominated than the 5k sample:

- At 5k, one run had L2 weak-pass without any L0/L1 gate pass.
- At 10k, all late-layer candidates are accompanied by L0/L1 strong-pass candidates.
- L3 remains weak-pass only; no L3 strong-pass was observed at either 5k or 10k.

## Interpretation

At 10k, the selected-head order signal is robustly present and mostly carried by L0, with some L1 and rare L2. Late-layer candidates do appear, but they do not appear to be the only carrier; they are accompanied by stronger early-layer heads.

Practical rule:

For 10k teacher-source selection, scan all layers to avoid missing rare L2 cases, but expect the best source to usually be in L0/L1. L3 should be treated as a near-miss/secondary diagnostic unless a future scan shows a strong-pass gate.
