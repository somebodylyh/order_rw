# 5k Signal Carrier Layer Ablation

Run date: 2026-06-26

## Protocol

Same four checkpoints as the compile/no-compile 5k ablation:

- `compile_a`
- `compile_b`
- `nocompile_a`
- `nocompile_b`

Diagnostic:

```text
scripts/search_none_separated_65_heads.py
M=20, batch_size=8, fwd_batch=8, seed=0
perm_orientation=phys_to_model
methods: C-D+L, L
```

This is the legacy all-layer none-separated 65-node per-head/method search. It asks whether any selected head/method can recover physical L2R from the independent None/BOS start.

## Results

| run | best candidate | strong-pass count | weak-pass count | strong layers | strong heads |
|---|---|---:|---:|---|---|
| `compile_a` | L0H1 L, tau=1.000 | 10 | 1 | L0, L1 | L0H1, L0H6, L1H2, L1H3, L1H7 |
| `compile_b` | L1H2 L, tau=1.000 | 6 | 2 | L0, L1 | L0H1, L1H2, L1H4 |
| `nocompile_a` | L1H2 C-D+L, tau=1.000 | 10 | 0 | L0, L1 | L0H1, L0H6, L1H2, L1H6, L1H7 |
| `nocompile_b` | L0H1 L, tau=1.000 | 6 | 1 | L0, L1 | L0H1, L1H2, L1H7 |

## Interpretation

In this same-config 5k ablation, the order-bearing signal is present in every run under the all-layer per-head diagnostic. It is not erased by disabling `torch.compile`, and it is not absent from the new 5k trajectory.

The carrier is not fixed to one exact head. Stable motifs are:

- L0H1 appears in all four runs;
- L1H2 appears in all four runs;
- L1H7 appears in three of four runs;
- L0H6 appears in two of four runs;
- L1H3, L1H4, and L1H6 appear in one run each.

So the right statement is:

The 5k order signal is sparse and selected-head dependent. In these repeats it is mostly carried by L0/L1 heads, with head identity drifting across trajectories. A diagnostic that only reports L0 consensus CDL-vs-L2R can look weak even when an all-layer selected-head diagnostic finds strong order-bearing heads.

This supports using an all-layer/head teacher-source scan before training g_beta from 5k, rather than assuming the historical L0 head set or L0 consensus is the relevant source in every run.
