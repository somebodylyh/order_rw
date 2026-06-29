# Direct Policy Signal Quality

**Date:** 2026-06-25  
**Checkpoint:** random baseline, step 20k  
**Dataset:** 2,000 L0 all-head model-frame strict65 batch-mean graphs  
**Evaluation:** held-out validation and test splits, 200 graphs each  
**Training performed:** none

## Test Results

| Method | tau vs sequential CDL | tau vs layout path | Pairwise accuracy | Prefix@8 vs teacher | Cross-sample tau |
|---|---:|---:|---:|---:|---:|
| sequential CDL consensus | 1.000 | 0.990 | - | 1.000 | 0.982 |
| new g_beta | **0.985** | **0.993** | **0.997** | **0.973** | 0.989 |
| initial CDL one-shot | 0.027 | 0.022 | 0.511 | 0.266 | 0.087 |
| source mass | 0.058 | 0.052 | 0.526 | 0.108 | 0.038 |
| readiness | 0.059 | 0.053 | 0.527 | 0.140 | 0.010 |

Validation results are nearly identical:

| Method | tau vs sequential CDL | tau vs layout path | Pairwise accuracy |
|---|---:|---:|---:|
| new g_beta | 0.984 | 0.993 | 0.996 |
| initial CDL one-shot | 0.031 | 0.025 | 0.512 |
| source mass | 0.058 | 0.051 | 0.526 |
| readiness | 0.059 | 0.053 | 0.526 |

## Destroyed-Graph Audit

| Method | Destroyed tau vs teacher | Destroyed pairwise accuracy | Destroyed order tau vs normal |
|---|---:|---:|---:|
| new g_beta | 0.045 | 0.522 | 0.045 |
| initial CDL one-shot | -0.007 | 0.496 | 0.003 |
| source mass | 0.058 | 0.526 | **1.000** |
| readiness | 0.051 | 0.523 | 0.612 |

The structure-preserving destroy shuffles target identity within each source
row while preserving row value multisets. Therefore `source_mass`, which is
only a row sum, is mathematically invariant to this destroy. The measured
order correlation of 1.000 confirms that it contains no target-assignment
topology under this audit.

## Conclusion

Under the same L0 all-head, model-frame strict65 input protocol:

1. The initial one-shot CDL reduction does not approximate the final sequential
   CDL order. Its test Kendall tau is 0.027 and pairwise accuracy is 0.511.
2. Row-mass and readiness heuristics are also near random with respect to the
   sequential teacher and layout path.
3. The full sequential CDL consensus is highly stable across samples and
   closely matches the layout path.
4. The learned g_beta reproduces this sequential signal on held-out graphs
   with tau about 0.985 and pairwise accuracy about 0.997.

The useful signal is therefore not available from these simple initial-state
marginals. The evidence supports either sequential dynamics or learned
nonlinear multi-head/topological fusion as necessary for recovering the
teacher order.

