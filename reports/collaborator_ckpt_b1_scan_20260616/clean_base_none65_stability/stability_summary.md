# Clean-base single-training 65-node stability

Protocol: None-separated 65-node block graph; node0=None/BOS, content block i=x_{4i}:x_{4i+3}.
Run: `clean_base_random_perm` checkpoint ladder. Lightweight scan: M=8, batch_size=8, control seeds 0..4, methods L / C-D+L / none_edge.

| step | strong total | strong L | strong C-D+L | strong none_edge | weak | best | tau | first | phys0 rank | p4 | ctrl |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | L0H2 C-D+L | 0.191 | 38 | 8 | 1 | 0.063 |
| 1000 | 0 | 0 | 0 | 0 | 0 | L2H4 none_edge | 0.142 | 3 | 17 | 1 | 0.069 |
| 5000 | 10 | 5 | 5 | 0 | 2 | L1H2 L | 1.000 | 0 | 0 | 4 | 0.050 |
| 10000 | 10 | 5 | 5 | 0 | 2 | L1H2 L | 1.000 | 0 | 0 | 4 | 0.050 |
| 20000 | 16 | 8 | 8 | 0 | 1 | L1H4 L | 1.000 | 0 | 0 | 4 | 0.052 |
| 30000 | 15 | 7 | 8 | 0 | 4 | L0H1 L | 1.000 | 0 | 0 | 4 | 0.053 |
| 40000 | 15 | 8 | 7 | 0 | 1 | L1H2 L | 1.000 | 0 | 0 | 4 | 0.055 |
| 50000 | 14 | 7 | 7 | 0 | 1 | L0H0 L | 1.000 | 0 | 0 | 4 | 0.057 |
| 60000 | 16 | 8 | 8 | 0 | 2 | L0H0 L | 1.000 | 0 | 0 | 4 | 0.054 |

## Best readouts by step

| step | best L | L tau/gate | best C-D+L | C-D+L tau/gate | best none_edge | none_edge tau/gate |
|---:|---|---|---|---|---|---|
| 0 | L0H7 | 0.082/fail | L0H2 | 0.191/fail | L2H0 | 0.178/fail |
| 1000 | L1H1 | 0.073/fail | L2H4 | 0.118/fail | L2H4 | 0.142/fail |
| 5000 | L1H2 | 1.000/strong_pass | L1H4 | 1.000/strong_pass | L1H3 | 0.477/fail |
| 10000 | L1H2 | 1.000/strong_pass | L0H1 | 1.000/strong_pass | L0H3 | 0.211/fail |
| 20000 | L1H4 | 1.000/strong_pass | L1H4 | 1.000/strong_pass | L1H3 | 0.412/fail |
| 30000 | L0H1 | 1.000/strong_pass | L0H0 | 1.000/strong_pass | L2H4 | 0.518/weak_pass |
| 40000 | L1H2 | 1.000/strong_pass | L1H0 | 1.000/strong_pass | L2H4 | 0.373/fail |
| 50000 | L0H0 | 1.000/strong_pass | L1H0 | 1.000/strong_pass | L1H3 | 0.323/fail |
| 60000 | L0H0 | 1.000/strong_pass | L1H1 | 1.000/strong_pass | L1H3 | 0.415/fail |

## 5k-60k stable strong head/methods

- L1H0 C-D+L
- L1H0 L
- L1H1 C-D+L
- L1H1 L
- L1H2 C-D+L
- L1H2 L
- L1H4 C-D+L
- L1H4 L

## 20k-60k stable strong head/methods

- L0H0 C-D+L
- L0H0 L
- L1H0 C-D+L
- L1H0 L
- L1H1 C-D+L
- L1H1 L
- L1H2 C-D+L
- L1H2 L
- L1H4 C-D+L
- L1H4 L
- L2H0 C-D+L
- L2H0 L
- L2H4 C-D+L

## Interpretation

- Step 0 and 1k fail the strict None-start gate.
- From 5k onward the trained single-run checkpoints have strong-pass heads; from 20k onward the strong set is stable across both L and C-D+L readouts.
- `none_edge` alone never strong-passes; the evidence is not just None row sorting. The passing signal comes from transition rollout after None selects the start.
- This supports head-specific unanchored block-level discovery for this internal single-training ladder, while the collaborator ckpt remains a separate cross-ckpt check.
