# P7 — Loss-Trained Reveal Policy (summary, 2026-07-01)

Goal (user's framing): stop supervising the order MLP from features; instead
**update the MLP parameters directly from AO-GPT NLL**. Turn the deterministic
argsort readout (gbeta) into a **Plackett-Luce probability policy** and train it
with **policy gradient** (AO-NLL reward, EMA baseline), frozen AO-GPT.

## Path

1. **Predictability probe** (`p7_predictability_probe.py`, M=40, held-out): the
   hill-climb best order σ* is NOT learnable from any per-block feature — B τ*=0.34,
   H 0.24, **content −0.02, one-step −0.03**; none beats L2R. Supervised feature→σ*
   route is dead.
2. **Cold PG loss-only** (`p7_reveal_policy.py`): PG from a fresh MLP over the full
   64! order space does not move train NLL (exploration too sparse) — inconclusive
   without a good init.
3. **gbeta-policy** (`p7_gbeta_policy.py`, the user's plan): **reuse the deployed
   gbeta parameters** as the PL policy init (at low temp PL==argsort(gbeta)==deployed
   order, a built-in anchor — no distillation needed), AO-NLL PG at higher temp
   fine-tunes the shared gbeta params. **batch-mean readout** (the regime gbeta was
   trained in; the signal lives in batch-mean B, not per-sample). Frozen AO-GPT,
   20k ckpt (gbeta_K1000_from10k_L1H7), single-head L1H7 strict65, physical frame.

## Result — MECHANISM WORKS ("能更新")

Policy gradient from AO-GPT NLL **does update the shared gbeta parameters and
improves the reveal order**:
- per-sample version: policy_vs_deployed = **−0.081**
- batch-mean version (M=64, test 20): policy_vs_deployed = **−0.017**

So the core question is answered: **converting gbeta (argsort) → PL probability
policy and training it with AO-GPT loss is a working closed loop that updates the
MLP.** batch-mean is the correct regime (matches how gbeta was trained; the earlier
per-sample "deployed=3.92" was a val-frame mismatch vs the recorded batch-mean
`val_beta_order`≈3.69).

## Honest boundaries

- The gbeta consensus order is ~0.5 nat **worse than L2R** on the 20k model; local
  PG dents it (−0.02) but does not cross the gap to L2R (3.29) / oracle (3.23).
- **Structural tension:** gbeta = batch-mean **global consensus** (one order for
  all texts) → cannot be per-sample. The real headroom over L2R is **per-sample**
  (oracle), which a global order cannot reach; per-sample gbeta B is too noisy
  (consensus τ 0.31) to give good per-sample orders.
- At 20k, L2R is already near-optimal (oracle−L2R ≈ 0.06); large context-dependent
  headroom lives at earlier ckpts (10k: 0.19).

## One-line

Turning gbeta from a deterministic argsort into a Plackett-Luce policy and training
it by AO-GPT NLL policy gradient **reliably updates the MLP and improves the order
(mechanism established)** — a viable "train the order controller from downstream
loss" loop. It does not by itself beat L2R, because the exploitable per-sample
context-dependent signal is not present in the (global, position-dominated) gbeta
readout.

## Code
`p7_predictability_probe.py`, `p7_reveal_policy.py`, `p7_gbeta_policy.py`;
results `runs/p7/**`. Full order-line context: `P7_CONTEXT_ORDER_FINDINGS.md`.
