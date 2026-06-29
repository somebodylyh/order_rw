# 5k Signal Carrier Multiseed Ablation

Run date: 2026-06-26

## Protocol

I extended the previous 5k signal-carrier scan to additional existing random-baseline checkpoints:

- seed42: `random_baseline_continuous_jun05/ckpt_step5000.pt`
- seed2: `random_baseline_continuous_jun11_seed2_headtrack/ckpt_step5000.pt`
- seed124: `random_baseline_b1_headscan_seed124/ckpt_step5000.pt`

Diagnostic:

```text
scripts/search_none_separated_65_heads.py
M=20, batch_size=8, fwd_batch=8, seed=0
perm_orientation=phys_to_model
methods: C-D+L, L
```

The combined table below also includes the four compile/no-compile 5k repeats and the two jun08/jun25 provenance scans, all under the same old all-layer none-separated 65 diagnostic.

## Combined Layer Results

| run | best | strong rows | weak rows | strong layers | unique strong heads | late weak heads |
|---|---|---:|---:|---|---|---|
| `compile_a` | L0H1 L tau=1.000 | 10 | 1 | L0,L1 | L0H1, L0H6, L1H2, L1H3, L1H7 | L3H3 L tau=0.507 |
| `compile_b` | L1H2 L tau=1.000 | 6 | 2 | L0,L1 | L0H1, L1H2, L1H4 | - |
| `nocompile_a` | L1H2 C-D+L tau=1.000 | 10 | 0 | L0,L1 | L0H1, L0H6, L1H2, L1H6, L1H7 | - |
| `nocompile_b` | L0H1 L tau=1.000 | 6 | 1 | L0,L1 | L0H1, L1H2, L1H7 | - |
| `jun08_seed123_old` | L2H6 L tau=0.952 | 0 | 2 | - | - | L2H6 C-D+L tau=0.959; L2H6 L tau=0.952 |
| `jun25_seed123_new` | L1H7 L tau=1.000 | 4 | 2 | L1 | L1H0, L1H7 | L3H1 C-D+L tau=0.983; L3H1 L tau=0.833 |
| `seed42` | L0H4 L tau=1.000 | 6 | 1 | L0 | L0H2, L0H4, L0H6 | - |
| `seed2` | L1H0 L tau=1.000 | 8 | 4 | L0,L1,L2 | L0H6, L0H7, L1H0, L1H6, L2H0 | L2H6 C-D+L tau=0.553; L3H3 L tau=0.941; L3H5 C-D+L tau=0.753 |
| `seed124` | L0H3 C-D+L tau=0.996 | 1 | 0 | L0 | L0H3 | - |

`strong rows` counts head-method rows, so one head can contribute two rows when both `C-D+L` and `L` pass.

## Layer Distribution

Across these nine available 5k random-baseline trajectories:

- 8/9 have at least one strong-pass selected head.
- 7/9 have strong-pass heads only in L0/L1.
- 1/9 has a strong-pass L2 head: seed2 has L2H0 with tau=1.000 under both `C-D+L` and `L`.
- 0/9 have a strong-pass L3 head.
- Several runs have late-layer weak-pass candidates, including L3H1, L3H3, and L3H5.

Strong-pass row counts by layer:

| layer | strong rows |
|---|---:|
| L0 | 21 |
| L1 | 28 |
| L2 | 2 |
| L3 | 0 |

## Interpretation

The answer is yes, the carrier can move later than L0/L1, but in the current sample it only reaches L2 as a strong-pass signal once. I do not see evidence yet that it reliably moves to L3.

The stable pattern is:

- most 5k order-bearing signal is early-layer, mainly L0/L1;
- head identity drifts substantially across seed/trajectory;
- L2 can carry the signal in at least one seed;
- L3 currently appears only as weak-pass / near-miss, not strong-pass.

So the practical diagnostic rule should be:

Run an all-layer selected-head scan before choosing a 5k teacher source. Do not assume L0-only is sufficient, but also do not expect late layers to dominate at 5k unless a scan shows it for that checkpoint.
