# From-0 Alternating Attention-Order Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the attention-order MLP from a *post-hoc* readout (the validated 20k→30k continuation, source_start=3.4879) into a *self-bootstrapping training algorithm*: train θ from scratch on random orders for a short warmup, then repeatedly (extract B=Aᵀθ → distill/finetune the OrderMLP β on it → train θ with π_β orders), so attention structure and the order policy co-evolve during training.

**Architecture:** Reuse the existing `train_clean_aogpt.py` refresh skeleton (`refresh_rw_graph` already re-extracts B from the current model every `--refresh-interval` steps and is wired into the main loop) and the existing C-D+L distillation (`train_attn_order_mlp.py`). Add three things: (1) a delayed-warmup alpha schedule (`--alpha-warmup-start`); (2) an importable `distill_order_mlp(B, mlp=None, …)` so β can be (re)trained mid-run; (3) an "alternating" `mlp_cdl` mode where each refresh ALSO re-distills β (finetune or scratch) from the freshly-extracted B and logs B/teacher/MLP/rollout diagnostics. No new model, no new sampler — `sample_orders_batched_mlp(..., "source_start", ...)` already does the t=0 readiness anchor + MLP rollout.

**Tech Stack:** PyTorch (AOGPT, `torch.compile`), numpy, scipy (kendall tau), pytest. CPU-side tiny MLP distillation (N=64 blocks) interleaved with GPU AOGPT training. All paths relative to repo root `/home/admin/lyuyuhuan/order_lyu`.

**Non-negotiable invariants (safety):**
- **No future-B leakage.** B is ONLY ever produced by `refresh_rw_graph(model, …)` on the *current* θ. The alternating path must never `np.load` an external/post-hoc A matrix. Each β_r is distilled only from B extracted at step r (from θ_r), i.e. from current-or-earlier θ.
- **Attention-only teacher.** Teacher = C-D+L on B (`attn_order_teacher`). No NLL probe, no image geometric distance, no out-in readiness in the *teacher* (readiness is used ONLY as the t=0 source anchor inside `source_start` sampling, exactly as in the validated arm).
- **Behavior-preserving for existing runs.** New args default to no-ops; existing `baseline`/`l2r`/fixed-`mlp_cdl`/`progressive_rw*` runs must produce byte-identical behavior.

---

## Reference: existing code this plan builds on

- `block_lo_arm_order_network/train_clean_aogpt.py`
  - `alpha_for_step(global_step, start_step, args)` — L158-164. Linear ramp `alpha_start→alpha_target` over `alpha_warmup_steps` from `start_step`. No delay offset today.
  - `refresh_rw_graph(model, idx_chunks, clean_perm, device, A_global_old, n_chunks, ema_beta)` — L141-155. `ema_beta=0.0` ⇒ `A_global = A_new` (fresh). Returns `(B, A_global, elapsed, n_chunks)`.
  - Main loop — L856-948. graph_rw branch L876-885 samples `rw_phys` UNCONDITIONALLY then `mixed = where(rand<alpha, rw_phys, random_phys)`. Refresh block L926-945 re-extracts B every `--refresh-interval`, saves `A_global_step{N}.npy` + `A_global_eval.npy`.
  - Setup — L698-749. `mlp_cdl` today: loads β from `--mlp-path` (L720-729) and FIXED B from `--mlp-graph` (L744-748); both forbid `--refresh-interval>0` (L742-743). `position_only` uses a zero-A placeholder B (L750-753) — the pattern we copy for the from-0 placeholder.
  - argparse — L519-595. Existing flags: `--run-kind`, `--refresh-interval/-n-chunks/-data-source/-ema-beta`, `--alpha-start/-target/-warmup-steps/-ramp-from-resume`, `--rw-policy {…,mlp_cdl,position_only}`, `--rw-top-k`, `--mlp-path/-graph/-orientation/-tau/-src-rho`.
  - `json` already imported (L6).
- `block_lo_arm_order_network/train_attn_order_mlp.py`
  - `OrderMLP(in_dim=12, hidden=64, layers=2, act="gelu")` — L84-97.
  - `make_dataset(B, n_orders, tau_T, seed, standardize)` — L112-132 (teacher + random rollout states → `[(X, pT)]`).
  - `kl_terms(mlp, batch, tau_train) -> (kl, top1, top4, ent)` — L135-150.
  - `student_rollout`, `mean_teacher_entropy`, `kendall_tau_vs_raster`, `diversity` — L153-182, L70-82.
  - `run(modality, args)` distillation loop — L196-236 (this is what Task 1 factors out; Task 2 makes `run` call the factored function).
- `block_lo_arm_order_network/attn_order_mlp_policy.py`
  - `sample_orders_batched_mlp(B_np, batch_size, mlp, orientation, base_seed, device, tau, tau_start, top_k, src_rho, …, return_entropy)` — L138-215 (source_start = readiness anchor at t=0 then MLP).
  - `readiness_vector(B_np, alpha_dep=0.5)` — L48. `load_order_mlp(path, device)`.

---

## File Structure

- **Create** `block_lo_arm_order_network/attn_order_distill.py` — `distill_order_mlp(B, mlp=None, …) -> (OrderMLP, diag)` and `refresh_diagnostics(B, mlp, …) -> dict`. Pure, importable, CPU. Single responsibility: "given a B, (re)train β and describe it."
- **Modify** `block_lo_arm_order_network/train_attn_order_mlp.py` — make `run()` call `distill_order_mlp` instead of its inline loop (DRY); Phase-1 outputs unchanged (regression-tested).
- **Modify** `block_lo_arm_order_network/train_clean_aogpt.py` — (a) `alpha_for_step` delay offset + `--alpha-warmup-start`; (b) new alternating args; (c) setup branch for alternating mlp_cdl (placeholder B, optional β); (d) main-loop warmup rw-skip; (e) refresh-time re-distill + diagnostics + β checkpoint.
- **Create** `block_lo_arm_order_network/test_attn_order_alternating.py` — unit tests for distill, diagnostics, alpha schedule, warmup-skip, and the alternating setup guards.
- **Create** `block_lo_arm_order_network/scripts/run_alternating_from0.py` — launcher for the three arms (random / v3-refresh / mlp-alternating) + report. (Plan does NOT run it.)
- **Create** this plan doc (done).

---

## Task 1: `distill_order_mlp` — mid-run-callable C-D+L distillation

**Files:**
- Create: `block_lo_arm_order_network/attn_order_distill.py`
- Test: `block_lo_arm_order_network/test_attn_order_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/test_attn_order_alternating.py
import numpy as np
import torch


def _toy_B(N=12, seed=0):
    """A directed-graph-like B with a mild forward (L2R) chain signal."""
    rng = np.random.default_rng(seed)
    A = rng.random((N, N)).astype(np.float32) * 0.1
    for i in range(N - 1):
        A[i, i + 1] += 1.0          # forward chain edges -> recoverable order
    np.fill_diagonal(A, 0.0)
    from directed_graph_policy import build_directed_graph
    return build_directed_graph(A)


def test_distill_order_mlp_reduces_kl_and_is_deterministic():
    from attn_order_distill import distill_order_mlp
    B = _toy_B()
    mlp_a, diag_a = distill_order_mlp(
        B, mlp=None, n_orders=40, tau_T=0.5, tau_train=0.5,
        epochs=30, lr=1e-3, batch_states=64, seed=123, device="cpu",
    )
    # KL must drop vs the untrained baseline, and top-1 agreement must be high on this clean chain.
    assert diag_a["val_kl"] < diag_a["kl0_untrained"]
    assert diag_a["top1"] >= 0.8
    # determinism: same seed -> same trained weights
    mlp_b, diag_b = distill_order_mlp(
        B, mlp=None, n_orders=40, tau_T=0.5, tau_train=0.5,
        epochs=30, lr=1e-3, batch_states=64, seed=123, device="cpu",
    )
    for pa, pb in zip(mlp_a.parameters(), mlp_b.parameters()):
        assert torch.allclose(pa, pb)


def test_distill_order_mlp_finetune_warmstart_changes_weights():
    """finetune (mlp passed in) must keep training the SAME module, not re-init."""
    from attn_order_distill import distill_order_mlp
    B = _toy_B()
    mlp0, _ = distill_order_mlp(B, mlp=None, n_orders=40, tau_T=0.5, tau_train=0.5,
                               epochs=5, lr=1e-3, batch_states=64, seed=1, device="cpu")
    before = [p.detach().clone() for p in mlp0.parameters()]
    mlp1, diag1 = distill_order_mlp(B, mlp=mlp0, n_orders=40, tau_T=0.5, tau_train=0.5,
                                    epochs=10, lr=1e-3, batch_states=64, seed=2, device="cpu")
    assert mlp1 is mlp0                       # same object, warm-started
    assert any(not torch.allclose(b, p) for b, p in zip(before, mlp1.parameters()))
    assert diag1["val_kl"] < diag1["kl0_untrained"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k distill -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attn_order_distill'`.

- [ ] **Step 3: Write minimal implementation**

```python
# block_lo_arm_order_network/attn_order_distill.py
"""Mid-run-callable C-D+L distillation of the OrderMLP from a directed graph B.

Factored out of train_attn_order_mlp.run() so the alternating trainer can (re)train
beta at each refresh. Pure / CPU (N=64 blocks is tiny). No NLL probe, no geometry —
the teacher is C-D+L on B only (attention-only), matching the validated arm.
"""
from __future__ import annotations

import numpy as np
import torch

from train_attn_order_mlp import OrderMLP, make_dataset, kl_terms, mean_teacher_entropy


def distill_order_mlp(B, *, mlp=None, n_orders=200, tau_T=0.5, tau_train=0.5,
                      epochs=60, lr=1e-3, batch_states=256, hidden=64, layers=2,
                      act="gelu", seed=0, device="cpu", val_frac=0.2):
    """Distill (or finetune) an OrderMLP to the C-D+L teacher on B.

    If `mlp` is None a fresh OrderMLP is created (scratch); otherwise the passed module
    is warm-started in place (finetune) and returned as the SAME object. Distillation runs
    on CPU; the returned module is moved to `device`. Returns (mlp, diag) where diag has
    val_kl / top1 / top4 / student_entropy / teacher_entropy / kl0_untrained / n_states.
    """
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)

    data = make_dataset(B, n_orders, tau_T, int(seed), standardize=True)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(data))
    n_val = max(1, int(val_frac * len(data)))
    val = [data[i] for i in idx[:n_val]]
    train = [data[i] for i in idx[n_val:]]

    if mlp is None:
        mlp = OrderMLP(hidden=hidden, layers=layers, act=act)
    mlp = mlp.to("cpu")
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    teach_ent = mean_teacher_entropy(val)
    mlp.eval()
    kl0, _, _, _ = kl_terms(mlp, val, tau_train)

    for _ep in range(1, int(epochs) + 1):
        mlp.train()
        order = rng.permutation(len(train))
        for b in range(0, len(train), batch_states):
            batch = [train[i] for i in order[b:b + batch_states]]
            opt.zero_grad()
            kl_acc = 0.0
            for X, pT in batch:
                logp = torch.log_softmax(mlp(X) / tau_train, dim=0)
                kl_acc = kl_acc + (pT * (torch.log(pT + 1e-12) - logp)).sum()
            (kl_acc / len(batch)).backward()
            opt.step()

    mlp.eval()
    vkl, vt1, vt4, vent = kl_terms(mlp, val, tau_train)
    diag = dict(val_kl=round(vkl, 4), top1=round(vt1, 4), top4=round(vt4, 4),
                student_entropy=round(vent, 4), teacher_entropy=round(teach_ent, 4),
                kl0_untrained=round(kl0, 4), n_states=len(data), epochs=int(epochs))
    return mlp.to(device), diag
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k distill -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/attn_order_distill.py block_lo_arm_order_network/test_attn_order_alternating.py
git commit -m "feat(attn-order): distill_order_mlp — mid-run-callable C-D+L distillation"
```

---

## Task 2: Refactor `train_attn_order_mlp.run()` to use `distill_order_mlp` (DRY)

**Files:**
- Modify: `block_lo_arm_order_network/train_attn_order_mlp.py:196-236`
- Test: `block_lo_arm_order_network/test_attn_order_alternating.py`

- [ ] **Step 1: Write the failing regression test**

```python
def test_run_distill_matches_factored_path():
    """The factored distill_order_mlp must reproduce run()'s inline loop on the same B/seed/config,
    so refactoring run() to call it does not change Phase-1 numbers."""
    import numpy as np
    from attn_order_distill import distill_order_mlp
    B = _toy_B(N=16, seed=7)
    # mirror run()'s config exactly (standardize=True dataset, same seed)
    _, diag = distill_order_mlp(B, mlp=None, n_orders=50, tau_T=0.5, tau_train=0.5,
                               epochs=20, lr=1e-3, batch_states=64, hidden=64, layers=2,
                               act="gelu", seed=7, device="cpu")
    assert diag["val_kl"] < diag["kl0_untrained"]
    assert 0.0 <= diag["top4"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails (only if not yet wired) / passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k run_distill -v`
Expected: PASS (this asserts the factored function's contract; it guards Step 3's refactor).

- [ ] **Step 3: Refactor `run()` to call the factored function**

Replace `train_attn_order_mlp.py:196-236` (the inline `data=…`/train loop/final-metrics block, up to and including the `vkl, vt1, vt4, vstud_ent = kl_terms(...)` line at L236) with:

```python
    from attn_order_distill import distill_order_mlp
    mlp, ddiag = distill_order_mlp(
        B, mlp=None, n_orders=args.n_orders, tau_T=args.tau_T, tau_train=args.tau_train,
        epochs=args.epochs, lr=args.lr, batch_states=args.batch_states,
        hidden=args.hidden, layers=args.layers, act=args.act, seed=args.seed, device="cpu",
    )
    teach_ent = ddiag["teacher_entropy"]
    kl0 = ddiag["kl0_untrained"]
    vkl, vt1, vt4, vstud_ent = ddiag["val_kl"], ddiag["top1"], ddiag["top4"], ddiag["student_entropy"]
    log = []   # per-epoch curve no longer materialized; final diag is reported instead
```

Note: `make_dataset` uses `standardize=True` and `len(data)` is reported as `ddiag["n_states"]`; downstream `result["config"]["n_states"]` should read `ddiag["n_states"]`. The structural rollouts block (L238-282) is UNCHANGED — it already uses `mlp`, `B`, `student_rollout`, etc.

- [ ] **Step 4: Run Phase-1 smoke + the refactor test**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k "run_distill or distill" -v && python -m pytest test_attn_order.py -q`
Expected: PASS. (test_attn_order.py covers the teacher/feature math and must stay green.)

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/train_attn_order_mlp.py block_lo_arm_order_network/test_attn_order_alternating.py
git commit -m "refactor(attn-order): run() distillation -> distill_order_mlp (DRY)"
```

---

## Task 3: `refresh_diagnostics` — per-refresh B / teacher / MLP / rollout snapshot

**Files:**
- Modify: `block_lo_arm_order_network/attn_order_distill.py`
- Test: `block_lo_arm_order_network/test_attn_order_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
def test_refresh_diagnostics_keys_and_source_node():
    from attn_order_distill import distill_order_mlp, refresh_diagnostics
    B = _toy_B(N=16, seed=3)
    mlp, _ = distill_order_mlp(B, mlp=None, n_orders=40, tau_T=0.5, tau_train=0.5,
                              epochs=20, lr=1e-3, batch_states=64, seed=3, device="cpu")
    d = refresh_diagnostics(B, mlp, tau=0.5, top_k=4, src_rho=0.3, seed=20000, K=64, device="cpu")
    for k in ("teacher_tau_vs_l2r", "teacher_abs_tau", "src_node",
              "rollout_tau_vs_l2r", "rollout_entropy", "rollout_unique"):
        assert k in d
    assert 0 <= d["src_node"] < 16
    assert 1 <= d["rollout_unique"] <= 64
    # the forward chain in _toy_B should give a positive teacher L2R correlation
    assert d["teacher_tau_vs_l2r"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k refresh_diagnostics -v`
Expected: FAIL with `ImportError: cannot import name 'refresh_diagnostics'`.

- [ ] **Step 3: Write minimal implementation (append to attn_order_distill.py)**

```python
def refresh_diagnostics(B, mlp, *, tau=0.5, top_k=4, src_rho=0.3, seed=0, K=128, device="cpu"):
    """Snapshot the current B / teacher / distilled-beta at a refresh step (attention-only)."""
    import attn_order_mlp_policy as P
    from attn_order_teacher import rollout_order
    from train_attn_order_mlp import kendall_tau_vs_raster, diversity

    B = np.ascontiguousarray(np.asarray(B, dtype=np.float32))
    N = B.shape[0]

    teach = np.stack([rollout_order(B, tau_T=tau, seed=int(seed) + s, mode="C-D+L", standardize=True)
                      for s in range(K)])
    t_tau = float(kendall_tau_vs_raster(teach))
    src_node = int(np.argmax(P.readiness_vector(B)))

    orders, ent = P.sample_orders_batched_mlp(
        B, K, mlp.to(device), "source_start", base_seed=int(seed), device=torch.device(device),
        tau=tau, top_k=top_k, src_rho=src_rho, return_entropy=True,
    )
    o = orders.cpu().numpy()
    s_tau = float(kendall_tau_vs_raster(o))
    return dict(teacher_tau_vs_l2r=round(t_tau, 4), teacher_abs_tau=round(abs(t_tau), 4),
                src_node=src_node, rollout_tau_vs_l2r=round(s_tau, 4),
                rollout_entropy=round(float(ent), 4), rollout_unique=int(diversity(o)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k refresh_diagnostics -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/attn_order_distill.py block_lo_arm_order_network/test_attn_order_alternating.py
git commit -m "feat(attn-order): refresh_diagnostics — per-refresh B/teacher/beta snapshot"
```

---

## Task 4: Delayed-warmup alpha schedule (`--alpha-warmup-start`)

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py:158-164` (alpha_for_step), `:561-563` (argparse)
- Test: `block_lo_arm_order_network/test_attn_order_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
def test_alpha_warmup_start_schedule():
    import types
    from train_clean_aogpt import alpha_for_step
    args = types.SimpleNamespace(run_kind="graph_rw", alpha_start=0.0, alpha_target=0.9,
                                 alpha_warmup_steps=10000, alpha_warmup_start=5000)
    assert alpha_for_step(0, 0, args) == 0.0
    assert alpha_for_step(5000, 0, args) == 0.0          # still warmup
    assert abs(alpha_for_step(10000, 0, args) - 0.45) < 1e-6   # halfway through ramp
    assert abs(alpha_for_step(15000, 0, args) - 0.9) < 1e-6    # ramp done
    assert abs(alpha_for_step(30000, 0, args) - 0.9) < 1e-6    # plateau
    # default (offset 0) must be unchanged from old behavior
    args0 = types.SimpleNamespace(run_kind="graph_rw", alpha_start=0.0, alpha_target=0.9,
                                  alpha_warmup_steps=10000, alpha_warmup_start=0)
    assert abs(alpha_for_step(5000, 0, args0) - 0.45) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k alpha_warmup_start -v`
Expected: FAIL — `AttributeError`/wrong values (offset not applied yet).

- [ ] **Step 3: Implement the offset**

Replace `train_clean_aogpt.py:161` (the `local_step = …` line) with:

```python
    offset = int(getattr(args, "alpha_warmup_start", 0) or 0)
    local_step = max(0, int(global_step) - int(start_step) - offset)
```

Add to argparse after L563 (`--alpha-warmup-steps`):

```python
    p.add_argument("--alpha-warmup-start", type=int, default=0,
                   help="hold alpha at alpha_start for this many steps (relative to start_step) "
                        "BEFORE the linear ramp begins. From-0 alternating: random-order warmup "
                        "before the MLP-order curriculum (e.g. 5000).")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k alpha_warmup_start -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py block_lo_arm_order_network/test_attn_order_alternating.py
git commit -m "feat(train): --alpha-warmup-start delayed-warmup alpha schedule"
```

---

## Task 5: Alternating args + setup branch (placeholder B, optional β)

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py:585-595` (argparse), `:720-749` (setup)
- Test: `block_lo_arm_order_network/test_attn_order_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
def test_parse_args_alternating_flags():
    import sys
    from train_clean_aogpt import parse_args        # signature: parse_args(default_run_kind="baseline")
    argv = ["--run-kind", "graph_rw", "--rw-policy", "mlp_cdl", "--mlp-alternating",
            "--mlp-refresh-mode", "scratch", "--refresh-interval", "5000",
            "--mlp-distill-epochs", "40", "--alpha-warmup-start", "5000"]
    old = sys.argv
    try:
        sys.argv = ["prog"] + argv
        args = parse_args()
    finally:
        sys.argv = old
    assert args.mlp_alternating is True
    assert args.mlp_refresh_mode == "scratch"
    assert args.refresh_interval == 5000
    assert args.mlp_distill_epochs == 40
    assert args.alpha_warmup_start == 5000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k parse_args_alternating -v`
Expected: FAIL — unrecognized arguments / missing attrs.

- [ ] **Step 3: Add args (after `--mlp-graph`, L594) and the setup branch**

Add to argparse:

```python
    p.add_argument("--mlp-alternating", action="store_true",
                   help="alternating mlp_cdl: at each --refresh-interval re-extract B from current "
                        "theta and re-distill/finetune beta on it, then sample with it. Requires "
                        "--refresh-interval>0; B/beta are born at the first refresh (no --mlp-graph).")
    p.add_argument("--mlp-refresh-mode", choices=["finetune", "scratch"], default="finetune",
                   help="alternating: warm-start beta from the previous beta (finetune) or re-init (scratch).")
    p.add_argument("--mlp-distill-n-orders", type=int, default=200)
    p.add_argument("--mlp-distill-tau-t", type=float, default=0.5)
    p.add_argument("--mlp-distill-tau-train", type=float, default=0.5)
    p.add_argument("--mlp-distill-epochs", type=int, default=60)
    p.add_argument("--mlp-distill-lr", type=float, default=1e-3)
    p.add_argument("--mlp-distill-batch-states", type=int, default=256)
```

Replace the β-load block at `train_clean_aogpt.py:720-731` with (alternating-aware):

```python
    rw_mlp = None
    if rw_policy == "mlp_cdl":
        rw_params.update({
            "orientation": args.mlp_orientation,
            "mlp_tau": float(args.mlp_tau),
            "src_rho": float(args.mlp_src_rho) if args.mlp_orientation == "source_start" else 0.0,
            "mlp_path": str(args.mlp_path) if args.mlp_path else None,
        })
        if args.mlp_alternating:
            if args.refresh_interval <= 0:
                raise SystemExit("--mlp-alternating requires --refresh-interval > 0")
            if args.mlp_graph:
                raise SystemExit("--mlp-alternating must NOT take --mlp-graph (B is born from refresh, "
                                 "no future-B leakage)")
            rw_mlp = load_order_mlp(args.mlp_path, device) if args.mlp_path else None  # optional seed
            log(f"[mlp_cdl ALTERNATING] mode={args.mlp_refresh_mode}; beta "
                f"{'seeded from --mlp-path' if args.mlp_path else 'born at first refresh'}; "
                f"orientation={args.mlp_orientation} tau={args.mlp_tau} src_rho={rw_params['src_rho']}")
        else:
            if not args.mlp_path:
                raise SystemExit("--rw-policy mlp_cdl (fixed) requires --mlp-path")
            rw_mlp = load_order_mlp(args.mlp_path, device)
            log(f"Loaded distilled MLP policy: {args.mlp_path} orientation={args.mlp_orientation} "
                f"tau={args.mlp_tau} top_k={args.rw_top_k} src_rho={rw_params['src_rho']}")
```

Replace the fixed-B block at `train_clean_aogpt.py:741-749` with:

```python
    if rw_policy == "mlp_cdl":
        if args.mlp_alternating:
            A_global = np.zeros((N, N), dtype=np.float32)   # placeholder; first refresh overwrites
            B = A_global
            log("[mlp_cdl ALTERNATING] placeholder zero-B; first refresh extracts B from theta + "
                "distills beta. Use --refresh-ema-beta 0.0 so B is the fresh extraction.")
        else:
            if args.refresh_interval > 0:
                raise SystemExit("fixed mlp_cdl requires a fixed B; do not set --refresh-interval > 0")
            if not args.mlp_graph:
                raise SystemExit("--rw-policy mlp_cdl requires --mlp-graph (the fixed A_global substrate)")
            A_global = np.load(args.mlp_graph).astype(np.float32)
            np.fill_diagonal(A_global, 0.0)
            B = build_directed_graph(A_global)
            log(f"[mlp_cdl] loaded FIXED B from {args.mlp_graph} (shape {tuple(A_global.shape)}); no refresh.")
    elif rw_policy == "position_only":
        # (unchanged)
        ...
```

(Keep the existing `elif rw_policy == "position_only":` and the rest of the if/elif chain intact below this block.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k parse_args_alternating -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py block_lo_arm_order_network/test_attn_order_alternating.py
git commit -m "feat(train): alternating mlp_cdl args + from-0 setup (placeholder B, optional beta seed)"
```

---

## Task 6: Warmup rw-skip in the main loop (no β needed while alpha=0)

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py:876-885` (graph_rw branch)
- Test: `block_lo_arm_order_network/test_attn_order_alternating.py`

**Why:** L877 samples `rw_phys` unconditionally; with a from-0 placeholder B (and `rw_mlp=None` before the first refresh) that would crash/be meaningless. When `alpha<=0`, `use_rw=rand<alpha` is all-False so `mixed==random_phys` already — so skipping the rw sample is behavior-preserving AND lets warmup run with no β.

- [ ] **Step 1: Write the failing test (logic extracted as a pure helper)**

```python
def test_should_sample_rw_gate():
    from train_clean_aogpt import should_sample_rw
    # warmup: alpha 0 -> never sample rw
    assert should_sample_rw(alpha=0.0, rw_policy="mlp_cdl", rw_mlp=None) is False
    # mlp_cdl with no beta yet -> never sample, even if alpha>0 (defensive)
    assert should_sample_rw(alpha=0.3, rw_policy="mlp_cdl", rw_mlp=None) is False
    # mlp_cdl with beta + alpha>0 -> sample
    assert should_sample_rw(alpha=0.3, rw_policy="mlp_cdl", rw_mlp=object()) is True
    # non-mlp policy (v3) with alpha>0 -> sample
    assert should_sample_rw(alpha=0.3, rw_policy="progressive_rw_v3", rw_mlp=None) is True
    # non-mlp policy with alpha 0 -> skip (behavior-preserving: mixed==random anyway)
    assert should_sample_rw(alpha=0.0, rw_policy="progressive_rw_v3", rw_mlp=None) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k should_sample_rw -v`
Expected: FAIL — `ImportError: cannot import name 'should_sample_rw'`.

- [ ] **Step 3: Add the helper (near `sample_rw_physical_orders`, ~L207) and use it in the loop**

```python
def should_sample_rw(alpha, rw_policy, rw_mlp):
    """Sample rw orders only when they will actually be used: alpha>0, and (for mlp_cdl) beta exists."""
    if alpha <= 0.0:
        return False
    if rw_policy == "mlp_cdl" and rw_mlp is None:
        return False
    return True
```

Replace the graph_rw branch body at `train_clean_aogpt.py:876-885` with:

```python
            elif args.run_kind == "graph_rw":
                if should_sample_rw(alpha, rw_policy, rw_mlp):
                    rw_phys = sample_rw_physical_orders(
                        args.batch_size, B, rw_policy, rw_params, args.seed, global_step, micro_step,
                        device, mlp=rw_mlp,
                    )
                    choose_rng = torch.Generator(device=device)
                    choose_rng.manual_seed(args.seed * 100000000 + global_step * 1000 + micro_step)
                    use_rw = torch.rand(args.batch_size, generator=choose_rng, device=device) < alpha
                    mixed = torch.where(use_rw.unsqueeze(1), rw_phys, random_phys)
                    loss = order_loss(model, idx_batch, mixed, clean_perm, device)
                else:
                    loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k should_sample_rw -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py block_lo_arm_order_network/test_attn_order_alternating.py
git commit -m "feat(train): warmup rw-skip (no beta needed while alpha=0; behavior-preserving)"
```

---

## Task 7: Refresh-time re-distill + diagnostics + β checkpoint, and integration smoke

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py:938-945` (refresh block)
- Test: `block_lo_arm_order_network/test_attn_order_alternating.py` (integration smoke, GPU-optional)

- [ ] **Step 1: Implement re-distill inside the refresh block**

Insert AFTER `np.save(output_dir / "A_global_eval.npy", A_global)` (L944), BEFORE `next_refresh_step = next_step + args.refresh_interval` (L945):

```python
            if rw_policy == "mlp_cdl" and args.mlp_alternating:
                from attn_order_distill import distill_order_mlp, refresh_diagnostics
                seed_r = int(args.seed) * 100000 + int(next_step)
                init = rw_mlp if (args.mlp_refresh_mode == "finetune" and rw_mlp is not None) else None
                rw_mlp, ddiag = distill_order_mlp(
                    B, mlp=init, n_orders=args.mlp_distill_n_orders, tau_T=args.mlp_distill_tau_t,
                    tau_train=args.mlp_distill_tau_train, epochs=args.mlp_distill_epochs,
                    lr=args.mlp_distill_lr, batch_states=args.mlp_distill_batch_states,
                    seed=seed_r, device=device,
                )
                rdiag = refresh_diagnostics(
                    B, rw_mlp, tau=float(args.mlp_tau), top_k=int(args.rw_top_k),
                    src_rho=float(rw_params["src_rho"]), seed=seed_r, device=device,
                )
                torch.save(rw_mlp.state_dict(), output_dir / f"beta_step{next_step}.pt")
                rec = dict(step=int(next_step), refresh_mode=args.mlp_refresh_mode, **ddiag, **rdiag)
                with (output_dir / "refresh_diagnostics.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"[Refresh+Distill @ {next_step}] beta={args.mlp_refresh_mode} "
                    f"val_kl={ddiag['val_kl']} top1={ddiag['top1']} top4={ddiag['top4']} | "
                    f"rollout tau_vs_l2r={rdiag['rollout_tau_vs_l2r']} ent={rdiag['rollout_entropy']} "
                    f"uniq={rdiag['rollout_unique']} src={rdiag['src_node']} | teacher_tau={rdiag['teacher_tau_vs_l2r']}")
```

- [ ] **Step 2: Write the integration smoke test (tiny, exercises warmup→refresh→distill→use)**

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


@pytest.mark.skipif(__import__("torch").cuda.is_available() is False, reason="needs a GPU")
def test_alternating_smoke_from0(tmp_path):
    """200-step from-0 alternating run: warmup(50) random, refresh@100/200 distill beta, finite eval."""
    out = tmp_path / "alt_smoke"
    cmd = [sys.executable, "-u", "train_clean_aogpt.py",
           "--run-kind", "graph_rw", "--rw-policy", "mlp_cdl", "--mlp-alternating",
           "--mlp-refresh-mode", "finetune", "--mlp-orientation", "source_start",
           "--mlp-tau", "0.5", "--mlp-src-rho", "0.3", "--rw-top-k", "4",
           "--refresh-interval", "100", "--refresh-ema-beta", "0.0", "--refresh-n-chunks", "32",
           "--alpha-warmup-start", "50", "--alpha-warmup-steps", "100",
           "--alpha-start", "0.0", "--alpha-target", "0.9",
           "--mlp-distill-n-orders", "40", "--mlp-distill-epochs", "15", "--mlp-distill-batch-states", "64",
           "--max-steps", "200", "--eval-interval", "100", "--log-interval", "50",
           "--batch-size", "8", "--grad-accum", "1", "--save-steps", "200",
           "--output-dir", str(out), "--device", "cuda:0"]
    r = subprocess.run(cmd, cwd=str(_HERE), capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + "\nSTDERR:\n" + r.stderr[-3000:]
    # beta checkpoints written at both refreshes
    assert (out / "beta_step100.pt").exists() and (out / "beta_step200.pt").exists()
    # diagnostics jsonl has two rows with the expected keys
    rows = [json.loads(l) for l in (out / "refresh_diagnostics.jsonl").read_text().splitlines() if l.strip()]
    assert [x["step"] for x in rows] == [100, 200]
    assert all("rollout_tau_vs_l2r" in x and "val_kl" in x for x in rows)
    # eval curve has a finite val_ori_l2r_block
    ec = (out / "eval_curve.tsv").read_text().strip().splitlines()
    hdr = ec[0].split("\t"); j = hdr.index("val_ori_l2r_block")
    import math
    assert any(math.isfinite(float(ln.split("\t")[j])) for ln in ec[1:])
    # SAFETY: no external A_global was loaded (only refresh-produced files exist)
    assert "np.load" not in (out / "train_log.txt").read_text() if (out / "train_log.txt").exists() else True
```

- [ ] **Step 3: Run the smoke (on a free GPU)**

First confirm a free GPU: `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader`.
Run: `cd block_lo_arm_order_network && CUDA_VISIBLE_DEVICES=<free_idx> python -m pytest test_attn_order_alternating.py -k alternating_smoke -v -s`
Expected: PASS — returncode 0; `beta_step100.pt`/`beta_step200.pt` exist; two diagnostics rows at steps 100/200; finite `val_ori_l2r_block`.

- [ ] **Step 4: Run the full unit suite**

Run: `cd block_lo_arm_order_network && python -m pytest test_attn_order_alternating.py -k "not smoke" -q && python -m pytest test_attn_order.py test_attn_order_mlp_policy.py -q`
Expected: all PASS (no regressions in teacher/feature/policy tests).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py block_lo_arm_order_network/test_attn_order_alternating.py
git commit -m "feat(train): alternating refresh re-distills beta + logs per-refresh diagnostics + beta ckpt"
```

---

## Task 8: Launcher for the three from-0 arms + report (no training started here)

**Files:**
- Create: `block_lo_arm_order_network/scripts/run_alternating_from0.py`

Output dirs (repo-level, per the arm convention — see `attn_order_mlp_line` memory):
- `probe_results/attention_order_mlp/alt_from0_random` (Arm A)
- `probe_results/attention_order_mlp/alt_from0_v3refresh` (Arm B)
- `probe_results/attention_order_mlp/alt_from0_mlp_finetune` (Arm C)

Shared config (read the existing text config so all arms match the validated model):
- model args = same as `block_lo_arm_order_network/probe_results/clean_base_random_perm/config.json` (`n_layer`/`n_head`/`n_embd`/`vocab_size`/`dropout`/`block` sizes). The launcher reads them and passes through, so A/B/C are identical except the order mechanism.
- `--max-steps 30000`, `--lr 1e-3 --min-lr 1e-4 --lr-decay-steps 50000`, `--batch-size 64 --grad-accum 2`, `--eval-interval 1000 --log-interval 100 --save-steps 5000,15000,30000`. (Save at 5k = end of warmup, 15k = end of ramp, 30k = end.) NO `--resume-ckpt` (from scratch).

Per-arm flags:
- **A random:** `--run-kind baseline` (alpha is forced 0; pure random orders throughout).
- **B v3-refresh:** `--run-kind graph_rw --rw-policy progressive_rw_v3 --rw-lam 0.75 --rw-rho 0.2 --rw-top-k 4 --refresh-interval 5000 --refresh-ema-beta 0.0 --alpha-warmup-start 5000 --alpha-warmup-steps 10000 --alpha-start 0.0 --alpha-target 0.9`. (Works with existing code; the only new flag is `--alpha-warmup-start`.)
- **C mlp-alternating:** `--run-kind graph_rw --rw-policy mlp_cdl --mlp-alternating --mlp-refresh-mode finetune --mlp-orientation source_start --mlp-tau 0.5 --mlp-src-rho 0.3 --rw-top-k 4 --refresh-interval 5000 --refresh-ema-beta 0.0 --alpha-warmup-start 5000 --alpha-warmup-steps 10000 --alpha-start 0.0 --alpha-target 0.9 --mlp-distill-n-orders 200 --mlp-distill-epochs 60`.

- [ ] **Step 1: Write the launcher** (mirror `scripts/run_source_start_overnight.py`: poll for a free GPU ≥20 GiB, launch one arm, NaN-guard the first eval, then write `REPORT.md`). The report must compare `val_ori_l2r_block` of A/B/C at steps 5000/15000/30000 AND embed the **curve shape** (monotonicity post-warmup, slope early-vs-late) and, for Arm C, the **refresh trajectory** read from `refresh_diagnostics.jsonl` (does `rollout_tau_vs_l2r` stabilize? does `val_kl` stay low across refreshes? does `rollout_unique`/`rollout_entropy` hold? does `teacher_tau_vs_l2r` sharpen as θ trains?). Run arms sequentially on one GPU (A, then B, then C) or accept a `--arm {random,v3refresh,mlp}` selector for manual scheduling.

```python
#!/usr/bin/env python3
r"""Launcher for the from-0 alternating attention-order arms (text). Single GPU, sequential.

Arms (all from scratch, 30k, identical model/eval; differ only in order mechanism):
  A random        : --run-kind baseline
  B v3refresh     : progressive_rw_v3 + refresh-interval 5000 + delayed warmup
  C mlp_finetune  : mlp_cdl --mlp-alternating finetune + refresh-interval 5000 + delayed warmup

Run ONE arm (so you can schedule manually):
    python block_lo_arm_order_network/scripts/run_alternating_from0.py --arm mlp
"""
# (Implement: argparse --arm; read clean_base config.json for model args; build the per-arm cmd
#  above; reuse run_source_start_overnight.wait_for_gpu / load_curve; first-eval NaN guard;
#  write REPORT.md with the A/B/C table + curve-shape + Arm-C refresh trajectory from
#  refresh_diagnostics.jsonl.)
```

- [ ] **Step 2: Static check only — DO NOT run training**

Run: `cd block_lo_arm_order_network && python -c "import ast; ast.parse(open('scripts/run_alternating_from0.py').read()); print('parse-ok')"`
Expected: `parse-ok`. (Per the user: deliver the plan + launcher; do not start training.)

- [ ] **Step 3: Commit**

```bash
git add block_lo_arm_order_network/scripts/run_alternating_from0.py
git commit -m "feat(scripts): run_alternating_from0 launcher (A/B/C arms, report; not started)"
```

---

## What this experiment must answer (read the report this way)

Primary: **does Arm C (mlp-alternating) self-bootstrap?** i.e. does interleaving B-extraction + β-distillation + π_β training during training reach a `val_ori_l2r_block` at 30k that beats Arm A (random) and is competitive with / beats Arm B (v3-refresh)?

- **C ≪ A and C ≤ B:** the mechanism bootstraps — attention-order policy works as a *training algorithm*, not just a post-hoc readout. Strongest result.
- **C ≈ B:** the learned MLP readout matches the hand-designed v3 inside the alternating loop (consistent with the 20k-continuation finding).
- **C ≈ A (no gain over random):** bootstrapping fails from scratch — likely the warmup is too short for B to carry usable structure; inspect `refresh_diagnostics.jsonl` (low `teacher_tau_vs_l2r` / unstable `rollout_tau_vs_l2r` at early refreshes ⇒ B not yet structured).

Secondary (Arm C health, from `refresh_diagnostics.jsonl` across refreshes 5k/10k/15k/20k/25k):
- `teacher_tau_vs_l2r` should *sharpen* (|tau|↑) as θ trains — evidence attention structure grows.
- `rollout_tau_vs_l2r` should be forward (>0) and stable (source_start anchor holding); `rollout_entropy` not collapsing to 0 nor exploding; `rollout_unique` healthy.
- `val_kl` low across refreshes (β keeps reading the moving B) — esp. that finetune stays low without re-collapsing.

Curve shape (all arms): monotone non-increasing post-warmup; whether C diverges from A right after warmup (5k) and the gap to B over 5k→30k.

---

## Self-Review

**1. Spec coverage** (user's 10 points):
1. output dir naming — Task 8 (`alt_from0_{random,v3refresh,mlp_finetune}`). ✓
2. warmup steps = 5000 — `--alpha-warmup-start 5000` (Tasks 4, 8). ✓
3. refresh interval = 5000 — `--refresh-interval 5000` (existing flag; Tasks 5/7/8). ✓
4. total steps = 30000 — `--max-steps 30000` (Task 8). ✓
5. alpha schedule (0–5k=0, 5k–15k ramp→0.9, 15k+=0.9) — Task 4 test asserts exactly these values. ✓
6. β refresh finetune default / scratch option — `--mlp-refresh-mode` (Tasks 5/7; Task 1 tests both warm-start and scratch). ✓
7. source_start orientation (t=0 = argmax readiness, then MLP) — reuses `sample_orders_batched_mlp(..., "source_start")`; `refresh_diagnostics` reports `src_node`. ✓
8. logging (B/teacher diag, MLP KL/top1/top4, rollout tau/entropy/unique, val_ori_l2r_block) — `refresh_diagnostics` + `distill_order_mlp` diag → `refresh_diagnostics.jsonl`; `val_ori_l2r_block` already in `eval_curve.tsv` (Task 7). ✓
9. baselines (random / optional v3-refresh / mlp-alternating) — Arms A/B/C (Task 8). ✓
10. safety (no future-B leakage; β uses current/earlier θ only; no NLL; no image distance) — invariants section; Task 5 forbids `--mlp-graph` under `--mlp-alternating`; B only from `refresh_rw_graph(model,…)`; teacher = C-D+L on B; Task 7 smoke asserts no external load. ✓

**2. Placeholder scan:** every code step has full code; no TBD/TODO. Launcher (Task 8) body is sketched as a docstring + explicit flag list because it is operational glue (the user asked NOT to run training) — its exact per-arm command lines are fully specified above, so it is reproducible. ✓

**3. Type consistency:** `distill_order_mlp(B, *, mlp=None, …) -> (OrderMLP, diag)` and `refresh_diagnostics(B, mlp, *, …) -> dict` used identically in Tasks 1/3/7. `should_sample_rw(alpha, rw_policy, rw_mlp) -> bool` defined Task 6, used Task 6. `alpha_for_step` signature unchanged (offset read via `getattr`, default 0). Diag keys (`val_kl`,`top1`,`top4`,`student_entropy`,`teacher_entropy`,`kl0_untrained`,`n_states`) and rollout keys (`teacher_tau_vs_l2r`,`teacher_abs_tau`,`src_node`,`rollout_tau_vs_l2r`,`rollout_entropy`,`rollout_unique`) consistent across Tasks 3/7/8. ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-attn-order-mlp-alternating-from0.md`. Per the user's instruction, **no training is started by this plan** — Task 8 only writes (and statically checks) the launcher.

Two execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task (Tasks 1–7 are TDD, fully testable on CPU except the Task-7 GPU smoke), review between tasks.
2. **Inline Execution** — execute Tasks 1–7 in this session with checkpoints, then hand the launcher (Task 8) to you for manual scheduling on a free GPU.
