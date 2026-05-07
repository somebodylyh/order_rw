# Graph-RW Order Policy Design

**Date:** 2026-05-07
**Status:** Spec — pending review before implementation
**Supersedes (in spirit, not in code):** the ON-based curriculum line (mixed025-NN, cotrain_aogpt_on)

## Motivation

The previous line treated the train-only global block attention as an undirected
proximity graph `W = 0.5·(A + Aᵀ)` and compressed it into a single deterministic
NN-greedy path, which an Order Network (ON) was then trained to imitate. Two
empirical results made this design questionable:

1. `mixed025-NN` showed early/mid-stage gains but suffered from U-shape
   degradation under a fixed deterministic teacher (mean Δ vs random = +0.0185,
   degradation 3k→7k), suggesting that the attention-derived signal is useful
   but should not be compressed into a single fixed NN-greedy path.
2. NLL-based GRPO and BC fine-tuning failed to make ON beat the random N=16
   baseline (best val_reward = 0.0107, KL ≈ 0 throughout). This motivates
   avoiding NLL/RL-based ON updates in the first Graph-RW version.

The new line keeps attention's **directionality**: if `A[i,j]` is "block i attends
to block j", then j is a context source / prerequisite for i, and the natural
order graph is the transpose `B = Aᵀ`. On this directed graph we define an
**explicit stochastic order policy** `π_RW(σ|B)`. Because `π_RW` is already a
directly sampleable distribution, **ON is removed from the first implementation**:
distilling `π_RW` into ON would only add imitation error without solving a
problem that exists in v1. ON is deferred to future settings that require
per-sample amortization or a learnable refinement of `π_RW`.

## Architecture Overview

```
A_global → B = A_globalᵀ → π_RW(σ|B) → σ → AOGPT → A_global^new
                                              ↑                ↓
                                              └── EMA refresh ──┘
```

- `A_global ∈ ℝ^{N×N}`: train-only global block attention
- `A_global[i,j]`: block i attends to block j
- `B = A_globalᵀ`, so `B[u,v]` means "u is a prerequisite / context source for v"
- `π_RW(σ|B)`: explicit stochastic order policy on the directed graph

### Hard constraints

- No L2R / original-position / index-based prior in generation. L2R is
  evaluation reference only.
- `[None]` signal handling unchanged: `extract_real_attention.py:247`'s
  `A_block += none_block * 0.1` stays as-is.
- N=64 is the main scale (matches cotrain_large).
- Random tie-break with seeded RNG; never argsort fallback to block id.
- Old ON / cotrain code is untouched. New module is fully isolated.

### Round-0 attention source

`A_global = mean(A_train_n64_10k, axis=0)` where
`probe_results/A_train_n64_10k.npy` is the train-only attention extracted from
the random-permutation block-64 50000-iter AOGPT checkpoint
(`~/ych/.../random-b64-permute-block-50000-iters/ckpt.pt`, 47.09M params,
`block_perm[:8] = [38, 24, 48, 63, 10, 29, 36, 51]`). No mixed025 / cotrain
contamination.

## Phase 1 — Offline Diagnostics

Pure numpy, no AOGPT training. Sample K=5000 orders per stochastic policy from
the round-0 `B` and report graph-score / order-distribution statistics. Decision
gate is qualitative; no fixed thresholds beyond legality.

### 1.1 Common construction

```
W = A_global; np.fill_diagonal(W, 0)
B = W.T                                  # B[u,v] = A_global[v,u]
out[u]    = Σ_v B[u,v]
in[u]     = Σ_v B[v,u]
source[u] = out[u] - α_dep · in[u]       # α_dep = 0.5
```

Comment in code: `future_t(v)` measures **unrevealed prerequisite mass for v**,
not v's outgoing future influence. `B` is **not row-normalized** in v1; raw edge
magnitudes are normalized only via softmax temperature.

### 1.2 Start distribution p₀ (shared by all stochastic policies)

```
p₀(u) = softmax(source / τ_start)        # τ_start = 1.0
σ_0 ~ p₀                                 # never argmax — argmax causes collapse
```

### 1.3 Main candidate: directed_progressive_rw

For step t, with revealed set `S_t`, unrevealed `U_t`, last node `σ_{t-1}`,
and candidate `v ∈ U_t`:

```
support_t(v) = Σ_{u ∈ S_t}            B[u, v]      # revealed support for v
future_t(v)  = Σ_{u ∈ U_t, u ≠ v}     B[u, v]      # unrevealed prerequisite mass
local_t(v)   = B[σ_{t-1}, v]                       # last-step directed edge

score_t(v) = β_sup · support_t(v)
           - β_fut · future_t(v)
           + β_src · source(v)
           + β_loc · local_t(v)

p_t(v) = softmax(score_t / τ_step)  over v ∈ U_t
σ_t   ~ p_t
```

Defaults (v1 fixed, no sweep):

| Param | Default |
|---|---|
| β_sup | 1.0 |
| β_fut | 0.5 |
| β_src | 0.2 |
| β_loc | 0.5 |
| τ_start | 1.0 |
| τ_step | 1.0 |
| α_dep | 0.5 |

logprob: `log π_RW(σ) = log p₀(σ_0) + Σ_t log p_t(σ_t)`.

### 1.4 Ablation: directed_self_avoiding_rw

Drop set-aware terms; only follow last-step directed edge:

```
score_t(v) = B[σ_{t-1}, v]
p_t(v)     = softmax(score_t / τ_step) over v ∈ U_t
```

Same p₀. Tests whether set-aware support / future terms add signal.

### 1.5 Baselines: pagerank_source, pagerank_uniform

```
P = row_normalize(B)
r = α · Pᵀ · r + (1 - α) · q             # α_pr = 0.85
```

- `pagerank_source`: `q(u) = softmax(source / τ_start)`
- `pagerank_uniform`: `q(u) = uniform(N)`

For each, two output modes:

- **deterministic** (`σ = argsort(-r)` with seeded random tie-breaking):
  **diagnostic only, never used to train AOGPT.** Deterministic global ranking
  would collapse to a single order family.
- **stochastic**: use `r` as the start distribution
  (`p₀(u) = softmax(r / τ_start)`), then walk by `self_avoiding_rw`.

### 1.6 Tie-break

For deterministic argmax/argsort operations, add tiny seeded noise for
tie-breaking:

```
score = score + 1e-9 · rng.standard_normal(score.shape)
```

For softmax/multinomial sampling, do not add tie-breaking noise unless all
candidate scores are exactly identical; sampling randomness already prevents
index-based fallback. Per-sample seed = `seed_base + sample_idx`. eps = 1e-9
(1e-12 risks being absorbed under float32).

### 1.7 Phase-1 metrics (per policy)

Output: `probe_results/phase1_diagnostic/{policy}.json`.

| Field | Computation |
|---|---|
| `legal_rate` | fraction of valid permutations (must be 1.0) |
| `tau_vs_l2r_mean / std` | Kendall τ vs `[0..N-1]` over K=5000 — **diagnostic only**, no fixed threshold |
| `first_node_entropy` | entropy of σ_0 over K=5000 (N/A for deterministic) |
| `pairwise_tau_mean` | mean τ on 1000 random σ_i, σ_j pairs (lower = more diverse family) |
| `mean_directed_score` | `mean_t B[σ_t, σ_{t+1}]` along sampled walks |
| `mean_progressive_support` | `mean_t support_t(σ_t) / |S_t|` |
| `policy_step_entropy` | `H_t = -Σ p_t(v) log p_t(v)`, reported per-step and as early/mid/late means |
| `logprob_mean / std / p10 / p90` | π_RW logprob distribution on its own samples |
| `cross_logprob` | progressive samples scored under self_avoiding policy and vice versa |

Add a **random_permutation_baseline** that samples K=5000 uniform σ and reports
`mean_directed_score`, `mean_progressive_support`, `tau_vs_l2r`,
`pairwise_tau_mean`. This anchors the lower bound on graph-score.

### 1.8 τ-vs-L2R interpretation (diagnostic, not gate)

| `tau_vs_l2r` | Reading |
|---|---|
| ≈ 0 | random-like order family |
| 0.2 – 0.6 | L2R-leaning but nontrivial |
| > 0.7 | possibly too close to L2R / inspect collapse |
| < 0 | reverse-order tendency / inspect direction convention |

Not a go/no-go condition. The teacher does not need to be far from L2R.

### 1.9 Phase-1 → Phase-2 go/no-go

1. `legal_rate = 1.0` for all policies (hard).
2. `progressive_rw.first_node_entropy > 0` (does not collapse, deterministic
   baselines exempt).
3. `progressive_rw.pairwise_tau_mean` not close to 1 (real order family).
4. `progressive_rw.mean_directed_score >` random_permutation_baseline.
5. `progressive_rw.mean_progressive_support >` self_avoiding_rw or
   random_permutation_baseline.

If 1–5 hold, proceed to Phase 2 with `directed_progressive_rw` as the training
policy.

## Phase 2 — AOGPT Training with Graph-RW Policy (no ON)

### 2.1 Backbone

- Start ckpt: `~/ych/.../random-b64-permute-block-50000-iters/ckpt.pt`.
- Script: `block_lo_arm_order_network/train_aogpt_graph_rw.py`, derived from
  `train_aogpt_with_on.py` with **all ON code removed** (BC, refresh, sampling,
  ckpt I/O).
- `batch_size = 4`, `grad_accum = 4` (effective 16). `lr = 3e-5`, AdamW, cosine.
- Pilot: 3000 steps × 1 seed × 2 refresh.
- Full (after pilot passes): 9000 steps × 3 seeds × 6 refresh.

### 2.2 Order schedule (per batch)

```
α(step) = linear warmup 0 → 0.7 over [0, 1500], constant 0.7 thereafter
for each sample in batch:
    if rng.random() < α(step):
        σ ~ π_RW(σ | B_global)            # graph-RW order
    else:
        σ ~ Uniform(S_N)                  # unstructured-order regularizer
    train AOGPT under σ
```

The 30% unstructured-order mixing is a **regularizer / capability-retention
diagnostic**, not a target and not an upper bound.

### 2.3 Validation (variance control)

```
val_rw_order = (1/M) · Σ_{m=1..M} L(x_val, σ^{(m)}),
               σ^{(m)} ~ π_RW(σ | B_r),  M = 3, fixed seeds [42, 123, 456]
val_unstructured_order = (1/M) Σ_m L(x_val, σ^{(m)}_uniform),  fixed seeds
val_ar = L(x_val, AR mode)
val_l2r = L(x_val, [0..N-1])              # diagnostic only
```

`val_rw_order` is the primary metric. `val_unstructured_order` is auxiliary,
checking that random-order capability does not collapse. `val_ar` is the
transfer baseline. `val_l2r` is diagnostic only.

After refresh, `val_rw_order` is evaluated under the **current round graph
B_r**, not the round-0 graph. Save `eval_order_seeds = [42, 123, 456]` and order
hashes alongside each eval point for reproducibility.

If M=3 is too expensive, fall back to M=1 with seed 42 (curve will be noisier).

## Phase 3 — Refresh Loop

```
every R = 1500 steps:
    A_curr_chunks = extract_attention(model_current, REFRESH_SUBSET)
    A_curr        = mean(A_curr_chunks, axis=0)
    A_global     ← 0.9 · A_global + 0.1 · A_curr     # EMA β = 0.9
    B            = A_global.T
    # π_RW updates implicitly through new B
```

`REFRESH_SUBSET` = fixed 2000 train chunk indices, sampled once at training
start with seed 42 and saved to
`probe_results/graph_rw_pilot/seed_42/refresh_subset_indices.npy`. Round-to-round
A_curr changes then reflect AOGPT evolution only, not data sampling noise.

### 3.1 Refresh-boundary diagnostics

Each refresh writes `probe_results/graph_rw_pilot/seed_42/round_{r}/diag.json`:

| Field | Computation |
|---|---|
| `A_global_drift` | `‖A_global_new - A_global_old‖_F / ‖A_global_old‖_F` |
| `A_global_top3_overlap` | mean per-row top-3 Jaccard vs previous round |
| `pi_RW_first_node_entropy` | over 5000 σ |
| `pi_RW_pairwise_tau` | over 1000 random σ pairs |
| `pi_RW_tau_vs_l2r` | over 5000 σ — diagnostic only |
| `pi_RW_tau_round_r_vs_r-1` | over 5000 σ sampled with shared seeds across rounds |
| `policy_step_entropy_mean / early / mid / late` | step-wise H_t — collapse detector beyond first-node entropy |
| `mean_directed_score` | same as Phase 1 |
| `mean_progressive_support` | same as Phase 1 |
| `val_rw_order` / `val_unstructured_order` / `val_ar` / `val_l2r` | full eval at refresh boundary |

### 3.2 Run matrix

| Run | Refresh | Seeds | Steps | Purpose |
|---|---|---|---|---|
| `graph_rw_pilot` | yes | 1 (42) | 3000 | pipeline validation, 2 refreshes |
| `graph_rw_full` | yes | 3 (42/123/456) | 9000 | main experiment |
| `graph_rw_fixed` | **no** (B locked at round 0) | 1 (42) | 9000 | refresh ablation |
| `graph_rw_random` | n/a (α = 0) | 1 (42) | 9000 | unstructured-order baseline, **rerun under same script and same eval pipeline** — do not reuse cotrain_large random runs |

### 3.3 Go / no-go for Phase 3

**Primary:**

- `val_rw_order` improves under graph-RW training.
- `val_ar` improves or at least does not degrade.
- `pi_RW_tau_round_r_vs_r-1` stabilizes (refresh converges, not oscillates).
- `A_global_drift` decreases or remains small.

**Secondary:**

- `val_unstructured_order` does not collapse (capability retention via the 30%
  random mix).

**`graph_rw_full` vs `graph_rw_fixed`:**

| Result | Interpretation |
|---|---|
| full > fixed | refresh loop useful, attention-policy closed loop established |
| full ≈ fixed | round-0 global RW policy is sufficient; refresh not necessary |
| full < fixed | refresh injects noise; inspect β, refresh interval, subset size |

## Code Organization

### New files

```
block_lo_arm_order_network/
├── directed_graph_policy.py
│   ├── build_directed_graph(A_global) → B
│   ├── compute_source(B, alpha_dep) → source, out_deg, in_deg
│   ├── progressive_rw_step(B, S, U, last, betas, tau) → p_t
│   ├── self_avoiding_rw_step(B, U, last, tau) → p_t
│   ├── pagerank(B, q, alpha_pr, max_iter) → r           # caller provides q
│   ├── sample_order(B, policy, params, seed) → order, logprob
│   │     # one order — used inside the training loop
│   ├── sample_orders(B, policy, params, K, seed_base)
│   │     → orders, logprobs, diagnostics                # K orders — Phase 1
│   └── policy_step_entropy(B, S, U, last, params)       # diagnostic helper
│
├── run_phase1_diagnostic.py
│   inputs:  probe_results/A_train_n64_10k.npy
│   policies: progressive_rw, self_avoiding_rw,
│             pagerank_source (det + stoch),
│             pagerank_uniform (det + stoch),
│             random_permutation_baseline
│   outputs: probe_results/phase1_diagnostic/{policy}.json,
│            probe_results/phase1_diagnostic/summary.tsv,
│            probe_results/phase1_diagnostic/sampled_orders.npz
│
├── train_aogpt_graph_rw.py
│   - derives from train_aogpt_with_on.py, all ON code removed
│   - α-mixed batch order, M=3 fixed-seed val_rw_order
│   - refresh every 1500 steps, EMA β=0.9, fixed REFRESH_SUBSET
│   - saves refresh_subset_indices.npy, eval_order_seeds.json,
│     round_{r}/diag.json, A_global_round{r}.npy
│
├── run_graph_rw_pilot.sh             # 1 seed × 3000 step × 2 refresh
└── run_graph_rw_full.sh              # 3 seeds × 9000 step × 6 refresh + ablations
```

### Reused / unmodified files

| File | Status |
|---|---|
| `attention_curriculum_orders.py` | unchanged (legacy ON line uses it) |
| `cotrain_aogpt_on.py`, `train_aogpt_with_on.py` | unchanged (cotrain_large running) |
| `order_diagnostics.py` | reused (`_kendall_tau`, `evaluate_order_diagnostics`) |
| `extract_real_attention.py` | reused at refresh time, `[None]` handling unchanged |
| `extract_train_A.py` | not invoked (round-0 reads disk) |
| `dp_solver.py` | not invoked |

### Smoke / unit tests

```
tests/test_directed_graph_policy.py
  test_B_construction:           B[u,v] == A_global[v,u]
  test_progressive_rw_legal:     5000 sampled orders are 100% valid permutations
  test_progressive_rw_seedable:  same seed → bit-identical output
  test_tie_break_no_id_bias:     all-zero B → σ_0 frequency near uniform (χ² test)
  test_pagerank_converges:       ‖r_{k+1} - r_k‖_1 < 1e-6 within 50 iter
  test_logprob_consistency:      sample_order's logprob matches recomputation
  test_sample_one_vs_K:          sample_order(seed=s) == sample_orders(K=1, seed_base=s)[0]

tests/test_train_aogpt_graph_rw_smoke.py
  50-step pilot: α schedule fires, π_RW samples enter batch,
  refresh triggers at step 25 and B changes, all output files written.
  Loss values not asserted; pipeline-only.
```

## Output Layout

```
probe_results/
├── phase1_diagnostic/
│   ├── progressive_rw.json
│   ├── self_avoiding_rw.json
│   ├── pagerank_source_det.json
│   ├── pagerank_source_stoch.json
│   ├── pagerank_uniform_det.json
│   ├── pagerank_uniform_stoch.json
│   ├── random_permutation.json
│   ├── summary.tsv
│   └── sampled_orders.npz
│
├── graph_rw_pilot/
│   └── seed_42/
│       ├── ckpt_step3000.pt
│       ├── refresh_subset_indices.npy
│       ├── eval_order_seeds.json
│       ├── round_{0,1,2}/diag.json
│       ├── A_global_round{0,1,2}.npy
│       └── eval_curve.tsv
│
└── graph_rw_full/
    ├── seed_{42,123,456}/...
    ├── fixed_seed_42/...                   # graph_rw_fixed
    ├── random_seed_42/...                  # graph_rw_random (α=0)
    └── log.txt
```

## Commit Plan (5 commits, code separated from results)

| # | Commit message | Trigger |
|---|---|---|
| 1 | `add directed_graph_policy.py + unit tests` | unit tests all pass |
| 2 | `add run_phase1_diagnostic.py` | script lands |
| 3 | `add phase1 diagnostic outputs` | Phase 1 run complete, summary.tsv written, go/no-go passes |
| 4 | `add train_aogpt_graph_rw.py + smoke test` | smoke test passes |
| 5 | `add graph_rw pilot results, then full + ablation` | pilot then full runs land |

Each commit is reviewed in a worktree before merging back. Code and experiment
outputs are kept separable in commit messages even when bundled, so results can
be re-run without reverting code.

## Paper Framing

We interpret the train-only global block attention `A_global` as a directed
dependency graph `B = A_globalᵀ`, where `B[u,v]` indicates that u is a
prerequisite / context source for v. On this directed graph we define an
explicit stochastic order policy `π_RW(σ|B)`: start from
`σ_0 ~ softmax(source / τ_start)`, then at each step pick the next block by a
context-supported random walk

```
score_t(v) = β_sup · support_t(v)
           - β_fut · future_t(v)
           + β_src · source(v)
           + β_loc · local_t(v)
p_t(v)     = softmax(score_t / τ_step)
```

This upgrades "global attention → reveal order" from a single deterministic
NN-greedy path to a directly sampleable order distribution. AOGPT is trained
on `σ ~ π_RW`; every R steps we re-extract attention from the current AOGPT,
EMA-update `A_global`, and implicitly refresh `π_RW`, forming an
attention-policy closed loop.

This line does **not** rely on ON. Since `π_RW` is an explicit, directly
sampleable policy, introducing ON at this stage would only distill an already
available distribution and add imitation error without solving a problem in v1.
ON is deferred to future settings requiring per-sample amortization
(`A_s → π_φ(σ|A_s)`), fusing additional features (hidden states, loss stats),
or learning a policy that improves over hand-defined `π_RW`.
