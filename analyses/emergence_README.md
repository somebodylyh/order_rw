# Spec A — Order-Signal Emergence Characterization (Results)

**Date:** 2026-06-27 · **Branch:** `attn-order-alternating` · **Compute:** no new training
**Inputs:** `runs/handoff_overnight/seed{2,42,123}/` (every-200-step `tau_table.npz`,
`eval_curve.tsv`, ckpts at 1000/2000/10000)
**Code:** `analyses/emergence_characterization.py`, `analyses/plot_emergence.py`
**Outputs:** `runs/emergence/seed{2,42,123}/{concentration.csv,summary.json,*.png}`

## Question

Pillar ③ falsified the cross-layer **handoff**. So: **how is the carrier layer/head set
*selected*, and is that emergence stable?** (Stability sense 1 — is the selection
contingent — is what A2 sets up for Spec B; full reproducibility needs GPU and is Spec B.)

## Headline result

The order/L2R template is **present diffusely and redundantly from very early** (it is
*not* constructed at the event). A short, **intrinsic** winner-take-all **pruning event**
(~step 1400–1600) then makes a **contingent** selection — *which* already-L2R-like heads
survive is not predictable from early τ — leaving the seed-locked, layer-local redundant
carrier of Pillar ③.

## Per-seed table

| seed | winner (tier) | pruning onset/mid/comp | A2 verdict | A3 | A4 pattern (pre-τ@1000) |
|------|---------------|------------------------|-----------|----|--------------------------|
| 2    | L1 {0,3,5,7} (strong) | 1000 / **1400** / 1600 | **contingent** | intrinsic | **sharpening** (0.94) |
| 42   | L0 {0,3,5,6,7} (weak)  | 1200 / **1600** / 1800 | **contingent** | intrinsic | **sharpening** (0.92) |
| 123  | L1 {0,5,6,7} (strong) | 800 / **1400** / 1600  | **contingent** | intrinsic | **sharpening** (0.95) |

## A1 — pruning event (diffuse_count, not entropy)

The event is timed on **`diffuse_count`** = number of (layer,head) cells with `|τ|≥0.7`,
which collapses **32 → ~10** over ~step 1000→1600 (matching the Pillar-1/2 strong_pass
story). **`|τ|-mass entropy` and layer-mass entropy barely move** (≈3.46→3.30 and
≈1.386→1.36): pruned heads keep *moderate* `|τ|`, so the order *mass* never concentrates
much — **the event is heads losing strong order signal, not mass relocating to one layer.**
(Entropy is reported as a secondary/robustness signal; the deviation from the spec's
"entropy primary" is empirical — entropy is too flat to time the event.)

## A2 — winner is contingent, not early-predictable (the key result)

For every seed, early-`|τ|` rank does **not** predict final carrier membership: early→final
ROC-AUC hovers near 0.5 (seed2 {0:.56, 200:.50, 600:.75, 1000:.44}), and the **winning
layer is not the highest-τ-mass layer early** — it only reaches rank 1 around step 600–1000
(seed2 layer-rank {0:3, 200:4, 600:4, 1000:1}). Verdict **contingent** for all three.

## A4 — Pattern A (sharpening), not a structural switch

Re-extracting the winning layer's carrier-head B at ckpts {1000, 2000, 10000}: at **step
1000 (pre-pruning) the winning heads already read `τ≈0.87–0.99`** — they are already
L2R-like *before* the event and merely sharpen/stabilize afterward (strong seeds → 1.0;
seed42's weak carriers stay ~0.7–0.85). The B structure does **not** switch.

## Synthesis (why A2-contingent and A4-sharpening agree)

The apparent tension is the mechanism. In Phase A **almost every head in every layer is
already L2R-like** (`diffuse_count≈32`, all `|τ|≥0.7`). So "being L2R-like early" cannot
distinguish winners from losers — the *losers are L2R-like early too, then get pruned*.
Hence A4 sees the survivors as already-sharp (sharpening), while A2 sees no early
predictability (contingent). The order template is a **diffuse, redundant prior present
from the start**; the pruning event is a **contingent winner-take-all selection of which
redundant copy survives**, intrinsic (LR flat, no loss-transition), settling into one
seed-specific layer.

This unifies the line: ③ said the carrier is layer-local/seed-locked/redundant (not
handed off); A says that carrier is **selected by contingent symmetry-breaking among a
pre-existing diffuse L2R prior**, not built or inherited.

## Pre-registered Spec B fork → resolved branch

Observed: **A2 = contingent**, **A3 = intrinsic**, **A4 = sharpening**. Therefore Spec B:

- **(from A2=contingent)** Test **pruning-window contingency**: branch from a **pre-pruning
  checkpoint (~step 800–1000, before onset)** and continue training with a **different
  dataloader order / micro-noise / different seed of the data RNG**; ask whether a
  *different* layer/head set wins. Contingency predicts the winner can move; early-bias
  (rejected here) would have predicted it cannot.
- **(from A3=intrinsic)** No dedicated LR/warmup arm needed (LR is flat in-window); the
  transition is intrinsic, so Spec B's perturbations target the *data/noise* in the
  pruning window, not the schedule.
- **(from A4=sharpening)** Pillar ⑤ is reframed: the order template is **already present
  early**, so ⑤ asks *what* this early diffuse L2R prior is (position vs content vs
  partial-order) and why it is L2R-like under any-order training — not "what transition
  creates the order code" (there is no such creation; only selection).

## Limitations

- One run per seed → A2 measures **within-run predictability**, not **reproducibility**
  (that is exactly Spec B). 3 seeds is a floor.
- A4 brackets the event with ckpts at 1000 (pre) / 2000 (post); **transition-step B
  (1200–1400) is not observed** (no ckpt). The pre/post bracket is sufficient to classify
  sharpening vs switch but not to watch the transition itself.
- A3 is correlational (temporal association), not causal.
