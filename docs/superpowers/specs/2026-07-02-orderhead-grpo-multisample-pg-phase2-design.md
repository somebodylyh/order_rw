# OrderHead GRPO-Style Multi-Sample PG — Phase-2 Design

**Date:** 2026-07-02
**Branch:** `v3-joint-orderhead-phase1` (Phase-2 work continues here or a new branch)
**Status:** implemented — Tasks 1–4 complete (2026-07-02); GPU micro-smoke pending
**Depends on:** Phase-1 decoupled path (spec `2026-07-02-orderhead-cdl-pretrain-decoupled-init-design.md`)

## Purpose (one sentence)

Replace single-action EMA-baseline REINFORCE with **same-state multi-sample
normalized advantage** to test whether the production OrderHead has exploitable
local order variation, now that wiring and PG scale are stable.

## Why (Phase-1 evidence)

The Phase-1 GPU smoke established, on the production `train_clean_aogpt.py`
freeze→unfreeze→PG path:

- **wiring works**: Stage 2 frozen argsort → unfreeze exactly at the target step
  → Stage 3 PG active, gβ trains + persists across save/resume, CDL absent.
- **scale is fixed**: with `--pg-logp-normalize length` + conservative HPs
  (`--lam-pg 1e-2 --orderhead-lr 3e-5 --pg-adv-clip 0.1 --pg-tau 0.05`), loss is
  stable ~4.0 over 250 steps (vs blow-up to 42 at `λ=1, sum-logp, τ=1`).
- **but no signal**: entropy climbs toward the 64-PL max (~204), advantage is
  noise around 0 (±0.1, clipped), gβ drifts, `train_obj 4.05 > ori_l2r 3.87`.

The failure is now **algorithmic, not wiring/scale/frame/CDL**:

$$A_{\text{batch}} = b_{\text{EMA}} - \ell(\sigma)$$

one action $\sigma$ per step, a lagging cross-batch EMA baseline → no knowledge of
how $\sigma_1$ compares to $\sigma_2$ **at the same state $B$**. Continuing to tune
single-action REINFORCE has low expected value.

## Core design

At a state $B$ (batch-mean, single-head L1H7, same as Phase-1), draw **K** orders
and use their **within-state normalized advantage** (no baseline):

$$\sigma_1,\dots,\sigma_K \sim \text{PL}(s,\tau),\qquad r_k = -\ell(\sigma_k; x)$$
$$\hat A_k = \frac{r_k - \operatorname{mean}_j r_j}{\operatorname{std}_j r_j + \varepsilon}$$
$$\mathcal L_{\text{GRPO}} = -\frac1K\sum_{k=1}^{K}\operatorname{stopgrad}(\hat A_k)\,\log p_\beta(\sigma_k\mid B)\;-\;\beta H$$

Advantage over Phase-1 REINFORCE:
1. compares multiple sampled orders **at the same $B$**;
2. no lagging EMA baseline (baseline = the group mean);
3. advantage is self-normalized (scale-stable);
4. directly answers: *does the current OrderHead neighborhood contain a better order?*

`log p_β(σ_k | B)` all come from **one** gβ forward (shared scores $s$); only the
K reward evaluations $\ell(\sigma_k)$ need forwards.

## Compute (must be controlled)

Each step is **K LM forwards** for the rewards (`@torch.no_grad`, batchable as a
stacked `K` mini-batch) + **1 gβ forward** (grad) + backward through gβ only.
The **backbone** LM update uses **one** order per step (design decision below), so
its grad forward count is unchanged.

- Micro-smoke: `K=4`, 100–250 steps. Not a fair-PPL run.
- **Do not compare GRPO vs REINFORCE per optimizer-step.** Report
  **forward-equivalent steps** (or tokens/FLOPs-matched). Phase-2 smoke claims
  *mechanism* only (does multi-sample normalized advantage give a stable
  directional update), not a PPL win.

## Decision: which order trains the backbone?

The K samples are for gβ's advantage. The backbone still needs one order/step for
its LM update. Options:
- **(recommended) greedy argsort `σ* = argsort(-s)`** — a stable curriculum for
  the backbone, decoupled from gβ's exploration; matches the frozen-warmup order
  at `τ→0`.
- one of the K samples (`σ_1`) — backbone sees an exploratory order (noisier).

Recommend **greedy** so backbone curriculum stays stable while gβ explores; the K
reward forwards remain `no_grad`. (Flag for review.)

## CLI (extends Phase-1)

- `--pg-update {reinforce, grpo}` (default `reinforce` — Phase-1 behavior intact).
- `--pg-k INT` (default `4`) — samples per state (grpo).
- reuse `--pg-tau`, `--pg-beta`, `--lam-pg`, `--orderhead-lr`, `--pg-logp-normalize`.
- **Phase-2 micro-smoke defaults**: `--pg-update grpo --pg-k 4 --pg-tau 0.05
  --pg-beta 0 --pg-logp-normalize length --orderhead-lr 3e-5 --lam-pg 1e-2`.

## REINFORCE no-entropy sanity (control, NOT main line)

Keep one small control to confirm the entropy bonus was the drift driver, not the
main investment:
`--pg-update reinforce --pg-beta 0 --pg-tau 0.02/0.05 --lam-pg 1e-2/3e-2`,
100–250 steps, check entropy no longer climbs to the max. Not a primary result.

### REINFORCE no-entropy control — run config

```bash
python -m block_lo_arm_order_network.train_clean_aogpt \
  --run-kind frozen_beta \
  --frozen-beta-ckpt reports/uniform_label_free_v1/nodewise_K1000.pt \
  --resume-ckpt <backbone_ckpt> \
  --batch-mean-probes 4 \
  --unfreeze-orderhead-at-step <start_step> \
  --pg-update reinforce --pg-tau 0.05 --pg-beta 0 \
  --pg-logp-normalize length --orderhead-lr 3e-5 --lam-pg 1e-2 \
  --pg-adv-clip 0.1 \
  --max-steps <start_step+250> --batch-size 8 \
  --device cuda --wandb-mode online
```

**What to watch:**
- Entropy should NOT drift to the max (~204) when β=0 — if it still does, the drift
  source is the REINFORCE advantage noise, not the entropy bonus.
- Compare entropy trend against the Phase-1 β=3e-3 run — if β=0 entropy is
  significantly lower, the entropy bonus was the primary drift driver.
- `train_obj` vs `ori_l2r` gap — any improvement over Phase-1?
- This is a sanity control, not a primary result. 100–250 steps only.

### Implementation note

The REINFORCE path is the `else` branch in the Phase-B `if pg_on:` block and is
**completely unchanged** from Phase-1. `--pg-beta 0` simply zeros out the entropy
term in `L_pg = -(A_batch * logp) - beta * entropy`. No code changes needed.

## Diagnostics to record (the real question)

| metric | purpose |
|---|---|
| **`std_k r_k`** (reward spread across K at a state) | **the gate**: if ≈0, sampled orders are indistinguishable → GRPO cannot help |
| best-of-K vs mean r | is there a positive sample in the local neighborhood? |
| $\hat A_k$ distribution | healthy (not all ~0, not exploding)? |
| entropy trend | still drifting to uniform? |
| OrderHead L1 delta | is gβ moving, and is the move advantage-driven (not entropy)? |
| `train_obj` vs `ori_l2r` gap | any directional improvement? |
| τ(top sampled order, greedy/L2R) | local swaps vs random churn |

**Primary gate:** `std_k r_k`. If same-state K orders have ~equal reward, no PG
variant (GRPO included) can extract signal — that would itself be an informative
NEGATIVE (the OrderHead neighborhood is flat at this backbone).

## Pitfalls to avoid (locked)

1. **compute-matched** must be redefined (K forwards/step) — no per-step PPL claim.
2. **entropy bonus small**: `β = 0` or `1e-4`, not `3e-3` (it dominated the
   length-normalized PG signal in Phase-1).
3. **τ ≠ 1**: high temperature is the known bad region; use `τ ∈ {0.02, 0.05}`.
4. **batch-level first**: keep the Phase-1 batch-level provider (`B_batch → K
   orders`). Do NOT also introduce a true group-level provider now — too many
   variables at once.

## Red lines carried over from Phase-1 (unchanged)

CDL only in Stage-1 init (no CDL in the loop); `B` detached before gβ (no PG grad
to backbone); advantage `stopgrad`; LM loss updates backbone, GRPO updates gβ
only; weight continuity across unfreeze; OrderHead persists across save/resume;
`B_PG == B_frozen` construction; model-frame → physical → token conversion reused.

## Files (sketch)

- **edit** `batch_readout/orderhead_pg.py`: add `grpo_advantage(rewards) -> Â
  (K,)` (pure, unit-testable) and a K-sample helper.
- **edit** `train_clean_aogpt.py` Phase-B branch: when `--pg-update grpo`, sample
  K orders from the shared scores, K `no_grad` reward forwards (stacked), compute
  `Â_k`, `L_GRPO`; backbone LM on the greedy order; log the diagnostics.
- **CLI**: `--pg-update`, `--pg-k`.
- **tests**: `grpo_advantage` normalization (mean 0, unit std, ε-safe on
  zero-variance rewards → zeros); K-sample logp grad routes to gβ only; a
  `std_k r_k` diagnostic is logged.

## Success criteria (micro-smoke)

- entropy does NOT drift to the max (β small + advantage signal);
- `std_k r_k` measurably > 0 at typical states (else report flat-neighborhood NEG);
- gβ update is advantage-driven (delta correlates with |Â|, not entropy);
- stable directional update over 250 steps (no blow-up, `train_obj` non-increasing
  on the fixed-order eval).
Fair-PPL comparison and any "beats L2R" claim are OUT of this micro-smoke.

## Task structure (for writing-plans)

1. `grpo_advantage` pure fn + tests.
2. K-sample PG in the Phase-B branch (grpo path) + `--pg-update/--pg-k` CLI +
   greedy-backbone decision + diagnostics logging.
3. `std_k r_k` / best-of-K diagnostics + micro-smoke runner.
4. REINFORCE no-entropy sanity control (config only) + docs.
