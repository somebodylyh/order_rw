# P6 / Version B — Online Co-adaptive H-Residual Controller

> Spec: `docs/superpowers/specs/2026-06-30-p6-online-coadaptive-controller-design.md`
> Predecessor: P5 Version A (`p5_utility_controller.py`, frozen direct-NLL routing).

**Question:** does hidden state H become a useful *sample-specific* utility
signal when the reveal-order controller is wired into AO-GPT training
(co-adaptation), lowering val NLL above a B-only controller? Claim is "H **may
become** useful in the loop," NOT "H already contains recoverable order info."

**Locked from Version A:** no `argsort(z)` (direct-NLL learns a router, not a
ranker); **cosine logits** (unnormalized logits saturate softmax as ‖z‖ grows).

## Status (2026-06-30)

**Infra (Task 1-5) + B0 frozen sanity DONE on CPU.** B2 (joint co-adaptation,
needs GPU) not started.

### B0 frozen sanity — code-path PASS

> RED LINE: B0 has NO co-adaptation → H is frozen-posthoc → a B0 H-null is
> expected **by construction** and is NOT a scientific result. B0 only proves the
> online chain runs: candidate pool → B/H extract → cosine routing → Σ p_k L_k →
> backward.

seed123 step10k, multi-head B[1,2,3,4], K=6, H=L1, cosine τ=0.3, M=64.
headroom 0.0182, σ_B NLL 3.7675.

| arm | hard_sel | soft_exp | fixed_ord | entropy | max_p | res_ratio |
|-----|----------|----------|-----------|---------|-------|-----------|
| b_only | 3.7505 | 3.7919 | 3.7505 | 1.39 | 0.39 | — |
| real | 3.7505 | 3.7963 | 3.7505 | 1.44 | 0.37 | 0.46 |
| shuffle | 3.7505 | 3.7971 | 3.7505 | 1.44 | 0.37 | 0.53 |
| zero | 3.7505 | 3.7919 | 3.7505 | 1.39 | 0.39 | 0.0005 |
| mean | 3.7505 | 3.7919 | 3.7505 | 1.39 | 0.39 | 0.018 |

- ✅ loss_route decreases (b_only/real/shuffle/mean; zero flat = degenerate
  control: H=0 adds only a uniform constant, nothing to learn).
- ✅ selected NLL (3.7505) ≤ σ_B (3.7675) — routes off σ_B onto near-optimal phys.
- ✅ **entropy not collapsed (1.39, max_p 0.39)** — cosine logits fixed the
  Version-A saturation (was entropy→0/max_p→1). ln(6)=1.79 ceiling.
- ✅ selection sensible (phys, near-optimal).
- ✅ B+H ≈ B-only (H null) — expected by construction, not a result.
- ✅ residual ratio bounded (<1).

**Carry to B2:** hard-selected NLL saturates on phys across all arms (no
discrimination, as in Version A) → **B2 discrimination must come from fixed-order
val NLL (model quality)**, confirming spec §6's choice of it as the primary
metric.

## Code

- `analyses/p6_online_controller.py`:
  - `cosine_logits` / `routing_p` (Task 2, scale-invariant)
  - `routing_loss_from_scores` (Task 3; L_k constant w.r.t. controller)
  - `controller_scores` + `detach_h` (Task 4/5)
  - `h_mode_list` (real/shuffle/zero/mean)
  - `p6_candidate_pool` (Task 1, fixed protocol, deduped via priority_matrix)
  - `build_dataset_p6`, `_yl`
  - `train_controller_online` (B0 controller-only loop)
  - `eval_fixed_order_nll` / `eval_hard_selected_nll` / `eval_soft_expected_nll`
    (separately named — risk #3)
  - `run_b0_sanity`
- Tests: `block_lo_arm_order_network/tests/test_p6_routing.py` (10 tests).
- Output: `runs/p6/seed123/b0_sanity/b0.json`.

## Next — B2 (GPU)

Joint co-adaptation from early ckpt (5k→20k). First GPU smoke: 500-1000 steps,
K=6, arms B-only / B+H / shuffled-H, H=L1, detach_h=True; watch entropy / memory
/ fixed-order val NLL before scaling to 2k. See spec §16-18.
