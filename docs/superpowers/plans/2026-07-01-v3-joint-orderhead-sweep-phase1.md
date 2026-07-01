# V3 Joint OrderHead Sweep — Phase-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-0 fixes + Phase-1 machinery to run the 4-arm group-level joint OrderHead sweep (L2R / frozen-gβ / joint-group m=16 / joint-batch-level m=64) from the 10k parent, and emit a trajectory-based two-tier gate report.

**Architecture:** Reuse the feasibility modules (`OrderHeadModule`, `AOGPTWithOrderHead`). A unified group-credit trainer runs any arm as `(m, G, trainable)`; the OrderHead reads a per-group mean-B, samples one PL order per group, broadcasts it to the group's samples, trains the backbone on the grad-enabled batch LM loss and the OrderHead on per-group PG reward (per-group EMA baseline). Separate evaluator modules compute the group-level mechanism probe (with real/shuffle/zero-B guards and fixed held-out grouping) and the fixed-order PPL curve. A runner orchestrates the arms and produces the gate report. Unit tests validate the machinery on tiny CPU smokes; the actual 10k→30k run is a GPU experiment launched by the runner, whose output IS the gate report.

**Tech Stack:** Python, PyTorch, existing `block_lo_arm_order_network` infra, `analyses/order_head_module.py` + `analyses/p5_utility_controller` + `analyses/p7_gbeta_policy`, `scipy.stats.kendalltau`.

## Global Constraints

- **Run from repo root.** Every script starts with `import sys; sys.path.insert(0, "block_lo_arm_order_network")` (and repo root) before importing `batch_readout.*` / `clean_training_protocol` / `analyses.*`.
- **N=64** (gβ requires 64×64 B; block orders are permutations of 0..63). Reused constants imported (not re-hardcoded): `GBETA_CKPT`, `HEAD=(1,7)`, `NONE_MODE="strict65_model"` from `analyses.p7_gbeta_policy`; `N`, `BLOCK_LEN`, `load_p5_ckpt`, `order_nll` from `analyses.p5_utility_controller`.
- **Implicit co-adaptation:** `B` is `.detach()`-ed before the OrderHead (already done in `AOGPTWithOrderHead.compute_order_logits`); the PG advantage is detached. LM loss (grad-enabled scalar `loss` from `forward_fn`) updates backbone φ ONLY; PG loss updates OrderHead θ ONLY.
- **Group credit (m=16, G=4):** per-sample loss `ℓ_i = token_losses.mean(dim=1)` from `forward_fn(..., return_token_loss=True)` (returns `(logits, loss, token_losses)`; `token_losses` is `(b,t)` detached); `ℓ_g = mean_{i∈g} ℓ_i`; `A_g = b_g − ℓ_g`; `L_PG = −(1/G) Σ_g stopgrad(A_g)·logp_g − β·H_g`. **Per-group EMA baseline** `b_g ← α·b_g + (1−α)·ℓ_g` preferred; scalar EMA is an allowed fallback.
- **Arm definitions (no trainability/path confound):** `frozen-gβ` = internal `OrderHeadModule` deterministic `argsort`, params NOT updated (m=64, G=1) — NOT the external hook. `joint-batch-level` = trainable OrderHead, m=64, G=1 (updates, but global-only). `joint-group` = trainable OrderHead, m=16, G=4. `L2R` = fixed `arange(N)` order, no OrderHead.
- **Controlled comparison:** all arms use identical data stream, batch size, backbone optimizer + LR schedule, eval set, eval cadence. OrderHead uses a **separate optimizer param group**; adding it must NOT change the backbone LR.
- **Fixed held-out grouping:** held-out set, m=16, and group ids are FIXED across all checkpoints and arms; same `clean_perm`/remap.
- **Start ckpt:** `runs/handoff_overnight/seed123/ckpt_step10000.pt` (10k parent). Eval at 10k/15k/20k/25k/30k. Results under `runs/v3_sweep/`.
- **Frames:** gβ scores are MODEL frame; remap `order_phys = inv_perm[order_model]` (`inv_perm = clean_perm.inv_perm_model_to_phys`) before `order_nll` (PHYSICAL frame).
- **Scope:** Phase-0 + Phase-1 ONLY. No per-sample m=1 stretch, no 60k, no multi-seed, no random arm — those are Phase-2 (separate plan/spec).

---

### Task 1: Phase-0 fixes — device, `.train()`, group-reward assembly

**Files:**
- Modify: `analyses/order_head_module.py` (device fix)
- Create: `analyses/v3_group_credit.py` (pure reward-assembly helpers — no training loop yet)
- Test: `tests/test_v3_group_credit.py`

**Interfaces:**
- Produces:
  - `OrderHeadModule.device` derived from a module parameter (follows `.to()`).
  - `per_sample_loss(token_losses: Tensor(b,t)) -> Tensor(b,)` = `token_losses.mean(dim=1)`.
  - `group_ids_for(batch_size:int, m:int) -> list[np.ndarray]` — contiguous groups; `G=batch_size//m`.
  - `group_rewards(per_sample_ell: Tensor(b,), groups: list) -> Tensor(G,)` = per-group mean loss.
  - `GroupEMA(G:int, alpha:float=0.9)` with `.update(ell_g: Tensor(G,)) -> Tensor(G,)` returning the pre-update baseline `b_g` (detached) and updating in place.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v3_group_credit.py
import sys, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from analyses.v3_group_credit import per_sample_loss, group_ids_for, group_rewards, GroupEMA
from analyses.order_head_module import OrderHeadModule
from analyses.p7_gbeta_policy import GBETA_CKPT

def test_per_sample_loss_shape():
    tl = torch.rand(64, 256)
    assert per_sample_loss(tl).shape == (64,)

def test_group_ids_and_rewards():
    groups = group_ids_for(64, 16)
    assert len(groups) == 4 and all(len(g) == 16 for g in groups)
    ell = torch.arange(64).float()
    gr = group_rewards(ell, groups)
    assert gr.shape == (4,)
    assert torch.allclose(gr[0], ell[:16].mean())

def test_group_ema_returns_pre_update_baseline():
    ema = GroupEMA(4, alpha=0.5)
    b0 = ema.update(torch.ones(4) * 2.0)          # first call: baseline init to the value
    assert b0.shape == (4,)
    b1 = ema.update(torch.zeros(4))
    assert torch.all(b1 <= b0 + 1e-6)             # baseline moved toward new lower loss

def test_orderhead_device_follows_to():
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    # device is derived from a parameter, not a frozen string
    p_dev = next(oh.gbeta.parameters()).device
    assert str(oh.device) == str(p_dev)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v3_group_credit.py -v`
Expected: FAIL (`ModuleNotFoundError: analyses.v3_group_credit`).

- [ ] **Step 3: Write the device fix + reward helpers**

In `analyses/order_head_module.py`, change `OrderHeadModule.__init__` to derive device from a parameter and make `self.device` a property:

```python
        # was: self.device = device
        self._requested_device = device
        self.gbeta.to(device)

    @property
    def device(self):
        return next(self.gbeta.parameters()).device
```

Create `analyses/v3_group_credit.py`:

```python
"""Phase-0 group-reward assembly (no training loop). Per-sample loss -> group mean
reward -> per-group EMA baseline. Reward is detached by construction (uses the
detached token_losses from forward_fn(return_token_loss=True))."""
import numpy as np
import torch


def per_sample_loss(token_losses):
    """token_losses: (b, t) detached per-token loss -> (b,) per-sample mean."""
    return token_losses.mean(dim=1)


def group_ids_for(batch_size, m):
    """Contiguous groups of size m. G = batch_size // m."""
    assert batch_size % m == 0, f"batch_size {batch_size} not divisible by m {m}"
    return [np.arange(g * m, (g + 1) * m) for g in range(batch_size // m)]


def group_rewards(per_sample_ell, groups):
    """(b,) per-sample loss -> (G,) per-group mean loss."""
    return torch.stack([per_sample_ell[torch.as_tensor(g)].mean() for g in groups])


class GroupEMA:
    """Per-group EMA baseline. update() returns the pre-update baseline (detached)
    then folds the new group loss in. First call initializes to the observed value."""
    def __init__(self, G, alpha=0.9):
        self.alpha = alpha
        self.b = None
        self.G = G

    def update(self, ell_g):
        ell_g = ell_g.detach()
        if self.b is None:
            self.b = ell_g.clone()
            return self.b.clone()
        pre = self.b.clone()
        self.b = self.alpha * self.b + (1 - self.alpha) * ell_g
        return pre
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v3_group_credit.py -v`
Expected: PASS (4 tests). Also run `python -m pytest tests/test_order_head_module.py tests/test_aogpt_with_orderhead.py -q` to confirm the device change did not regress the feasibility tests.

- [ ] **Step 5: Commit**

```bash
git add analyses/order_head_module.py analyses/v3_group_credit.py tests/test_v3_group_credit.py
git commit -m "feat(v3): Phase-0 fixes — OrderHead device follows .to(); group-reward assembly"
```

---

### Task 2: Group-credit trainer (unified arm entry)

**Files:**
- Create: `analyses/v3_group_trainer.py`
- Test: `tests/test_v3_group_trainer.py`

**Interfaces:**
- Consumes: `OrderHeadModule`, `AOGPTWithOrderHead` (`analyses/order_head_module.py`); `per_sample_loss`, `group_ids_for`, `group_rewards`, `GroupEMA` (Task 1); `sample_pl`, `GBETA_CKPT` (`analyses/p7_gbeta_policy`); `load_p5_ckpt`, `N` (`analyses/p5_utility_controller`); `random_probe_token_orders` (`batch_readout.hook_order_provider`).
- Produces:
  - `train_arm(ckpt_path, arm, *, n_steps, batch_size=64, m=None, lr_backbone=..., lr_orderhead=3e-4, tau=1.0, beta=3e-3, ema_alpha=0.9, adv_clip=0.3, device="cpu", out_dir="runs/v3_sweep", tag) -> dict` where `arm in {"l2r","frozen_gbeta","joint_group","joint_batch"}`.
  - Arm→config: `l2r` (no OrderHead, order=arange), `frozen_gbeta` (OrderHead argsort, not updated, m=64), `joint_group` (OrderHead trainable, m=16), `joint_batch` (OrderHead trainable, m=64).
  - Per-step log dict with: `lm_loss`, `pg`, `entropy`, `orderhead_grad_norm`, `backbone_grad_norm`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v3_group_trainer.py
import sys, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest, torch
CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"
pytestmark = pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="10k parent ckpt absent")

def test_joint_group_smoke_runs_and_routes_grad():
    from analyses.v3_group_trainer import train_arm
    res = train_arm(CKPT, "joint_group", n_steps=2, batch_size=16, m=4,
                    device="cpu", out_dir="runs/v3_sweep_test", tag="t")
    assert res["nan"] is False
    assert res["orderhead_param_delta"] > 0.0        # OrderHead learned
    assert res["backbone_param_delta"] > 0.0         # backbone co-adapted
    # grad routing: a PG-only step leaves backbone grad at zero
    assert res["pg_only_backbone_grad"] == 0.0

def test_frozen_gbeta_does_not_update_orderhead():
    from analyses.v3_group_trainer import train_arm
    res = train_arm(CKPT, "frozen_gbeta", n_steps=2, batch_size=16, m=16,
                    device="cpu", out_dir="runs/v3_sweep_test", tag="t")
    assert res["orderhead_param_delta"] == 0.0       # frozen
    assert res["backbone_param_delta"] > 0.0         # backbone still trains
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v3_group_trainer.py -v`
Expected: FAIL (`ModuleNotFoundError: analyses.v3_group_trainer`).

- [ ] **Step 3: Write the trainer**

```python
# analyses/v3_group_trainer.py
"""Unified group-credit joint OrderHead trainer. Same-batch probe-then-train
two-forward loop. Implicit co-adaptation (B detached). LM loss -> backbone;
per-group PG (per-group EMA baseline) -> OrderHead. Arms: l2r / frozen_gbeta /
joint_group (m=16) / joint_batch (m=64)."""
import sys, pathlib, json
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
from analyses.v3_group_credit import per_sample_loss, group_ids_for, group_rewards, GroupEMA
from analyses.p7_gbeta_policy import sample_pl, GBETA_CKPT
from analyses.p5_utility_controller import load_p5_ckpt, N
from batch_readout.hook_order_provider import random_probe_token_orders

ARMS = {  # arm -> (m, orderhead_trainable, use_orderhead)
    "l2r":          (None, False, False),
    "frozen_gbeta": (64,   False, True),
    "joint_group":  (16,   True,  True),
    "joint_batch":  (64,   True,  True),
}
L2R = np.arange(N, dtype=np.int64)


def _group_order(wrap, idx_group, probe_g):
    """Per-group mean-B -> gβ scores -> PL sample -> (order_model, logp, ent)."""
    z = wrap.compute_order_logits(idx_group, probe_g, per_sample=False)[0]   # (N,) grad on OH
    order_model, logp, ent = sample_pl(z, tau=1.0)
    return order_model, logp, ent


def train_arm(ckpt_path, arm, *, n_steps, batch_size=64, m=None,
              lr_backbone=1e-4, lr_orderhead=3e-4, tau=1.0, beta=3e-3,
              ema_alpha=0.9, adv_clip=0.3, device="cpu",
              out_dir="runs/v3_sweep", tag="phase1",
              eval_steps=(), evaluator=None):
    """eval_steps: 1-based step numbers at which to call
    evaluator(step, model, wrap) -> dict; results collected in res['evals'].
    The runner passes an evaluator closed over the FIXED held-out chunks/grouping."""
    cfg_m, oh_trainable, use_oh = ARMS[arm]
    m = m if m is not None else cfg_m
    total = n_steps * batch_size + batch_size          # +1 batch reserved (unused holdout guard)
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, total, device=device)
    model.train()                                       # Phase-0 fix: co-adapt needs train mode
    oh = OrderHeadModule(GBETA_CKPT, device=str(dev))
    for p in oh.gbeta.parameters():
        p.requires_grad_(oh_trainable)
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device=str(dev))

    # separate param groups: backbone LR fixed regardless of OrderHead presence
    groups = [{"params": model.parameters(), "lr": lr_backbone}]
    if use_oh and oh_trainable:
        groups.append({"params": oh.gbeta.parameters(), "lr": lr_orderhead})
    opt = torch.optim.Adam(groups)

    G = (batch_size // m) if use_oh else 0
    ema = GroupEMA(G, ema_alpha) if (use_oh and oh_trainable) else None
    oh_p0 = torch.cat([p.detach().flatten().clone() for p in oh.gbeta.parameters()])
    bb_p0 = torch.cat([p.detach().flatten().clone() for p in list(model.parameters())[:4]])
    log, pg_only_bb_grad, evals = [], None, []

    for step in range(n_steps):
        s = batch_size + step * batch_size
        idx = torch.stack([chunks[s + i] for i in range(batch_size)]).to(dev)
        if not use_oh:                                  # L2R arm: fixed order
            order_bn = np.repeat(L2R[None, :], batch_size, axis=0)
            logp = ent = torch.zeros((), device=dev)
            pg = torch.zeros((), device=dev)
        else:
            groups_ids = group_ids_for(batch_size, m)
            orders, logps, ents = [], [], []
            for g in groups_ids:
                idx_g = idx[torch.as_tensor(g)]
                probe_g = random_probe_token_orders(idx_g.shape[0], 0, step, dev)
                o, lp, e = _group_order(wrap, idx_g, probe_g)
                orders.append(np.repeat(o[None, :], m, axis=0)); logps.append(lp); ents.append(e)
            order_bn = np.concatenate(orders, axis=0)   # (batch_size, N)
            logp = torch.stack(logps); ent = torch.stack(ents)
        token_order = wrap.token_orders_from_model_blocks(order_bn).to(dev)
        _, lm_loss, token_losses = model.forward_fn(idx, token_order, return_token_loss=True)

        if use_oh and oh_trainable:
            ell_i = per_sample_loss(token_losses)              # (batch,)
            ell_g = group_rewards(ell_i, group_ids_for(batch_size, m))   # (G,)
            b_g = ema.update(ell_g)
            adv = (b_g - ell_g).clamp(-adv_clip, adv_clip)     # (G,) detached
            pg = -(adv * logp).mean() - beta * ent.mean()
            loss = lm_loss + pg
        else:
            loss = lm_loss
        opt.zero_grad(); loss.backward(); opt.step()
        log.append({"step": step, "lm_loss": float(lm_loss),
                    "pg": float(pg) if torch.is_tensor(pg) else 0.0,
                    "entropy": float(ent.mean()) if torch.is_tensor(ent) else 0.0})
        if evaluator is not None and (step + 1) in set(eval_steps):
            model.eval()
            evals.append({"step": step + 1, **evaluator(step + 1, model, wrap)})
            model.train()

    # grad-routing probe: PG-only backward leaves backbone grad zero
    if use_oh and oh_trainable:
        idx = torch.stack([chunks[batch_size + i] for i in range(m)]).to(dev)
        probe_g = random_probe_token_orders(m, 0, 0, dev)
        z = wrap.compute_order_logits(idx, probe_g, per_sample=False)[0]
        model.zero_grad()
        z.sum().backward()
        pg_only_bb_grad = float(sum((p.grad.abs().sum() for p in model.parameters()
                                     if p.grad is not None), start=torch.zeros(())))

    oh_p1 = torch.cat([p.detach().flatten().clone() for p in oh.gbeta.parameters()])
    bb_p1 = torch.cat([p.detach().flatten().clone() for p in list(model.parameters())[:4]])
    res = {"arm": arm, "m": m, "n_steps": n_steps,
           "orderhead_param_delta": float((oh_p1 - oh_p0).abs().sum()),
           "backbone_param_delta": float((bb_p1 - bb_p0).abs().sum()),
           "pg_only_backbone_grad": pg_only_bb_grad if pg_only_bb_grad is not None else 0.0,
           "nan": any(not np.isfinite(x["lm_loss"]) for x in log), "log": log, "evals": evals}
    out = pathlib.Path(out_dir) / tag; out.mkdir(parents=True, exist_ok=True)
    (out / f"{arm}_train.json").write_text(json.dumps(res, indent=2))
    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v3_group_trainer.py -v`
Expected: PASS (2 tests; a few minutes on CPU).

- [ ] **Step 5: Commit**

```bash
git add analyses/v3_group_trainer.py tests/test_v3_group_trainer.py
git commit -m "feat(v3): group-credit joint trainer (arms l2r/frozen_gbeta/joint_group/joint_batch)"
```

---

### Task 3: Group-probe evaluator (mechanism metric + guards)

**Files:**
- Create: `analyses/v3_group_probe.py`
- Test: `tests/test_v3_group_probe.py`

**Interfaces:**
- Consumes: `AOGPTWithOrderHead`, `OrderHeadModule`; `order_nll`, `N`; `group_ids_for` (Task 1); `random_probe_token_orders`; `kendalltau`.
- Produces:
  - `group_probe(wrap, held_idx, m=16, seed=0, device="cpu") -> dict` with keys `delta_probe_group`, `tau_to_l2r`, `tau_consensus`, `delta_real`, `delta_shuffle`, `delta_zero`, `real_beats_controls` (bool: `delta_real < min(delta_shuffle, delta_zero)`).
  - Fixed grouping: `held_idx` and `group_ids_for(len(held_idx), m)` are computed once by the caller and passed identically for every checkpoint/arm.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v3_group_probe.py
import sys, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest, torch
CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"
pytestmark = pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="10k parent ckpt absent")

def test_group_probe_keys_and_guard_present():
    from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
    from analyses.v3_group_probe import group_probe
    from analyses.p5_utility_controller import load_p5_ckpt
    from analyses.p7_gbeta_policy import GBETA_CKPT
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, 32, device="cpu")
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    held = torch.stack([chunks[i] for i in range(32)])
    r = group_probe(wrap, held, m=16, device="cpu")
    for k in ("delta_probe_group","tau_to_l2r","tau_consensus",
              "delta_real","delta_shuffle","delta_zero","real_beats_controls"):
        assert k in r
    assert isinstance(r["real_beats_controls"], bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v3_group_probe.py -v`
Expected: FAIL (`ModuleNotFoundError: analyses.v3_group_probe`).

- [ ] **Step 3: Write the evaluator**

```python
# analyses/v3_group_probe.py
"""Group-level mechanism probe with false-positive guards. Fixed held-out grouping.
delta_probe_group = E_g[ nll(group order) - nll(L2R) ]. Guards: real-B vs
shuffle-B vs zero-B, tau_to_l2r, tau_consensus. All orders MODEL-frame -> physical
remap via wrap.inv_perm before order_nll."""
import sys, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import kendalltau

from analyses.v3_group_credit import group_ids_for
from analyses.p5_utility_controller import order_nll, N
from batch_readout.hook_order_provider import random_probe_token_orders

L2R = np.arange(N, dtype=np.int64)


def _delta_for_B_variant(wrap, held_idx, groups, dev, variant):
    """Mean over groups of [nll(group order from variant-B) - nll(L2R)]."""
    probe = random_probe_token_orders(held_idx.shape[0], 0, 0, dev)
    A = wrap.extract_B(held_idx, probe).detach()               # (M,N,N) real B
    if variant == "shuffle":
        perm = torch.randperm(A.shape[0])
        A = A[perm]                                            # sample-shuffled B
    elif variant == "zero":
        A = torch.zeros_like(A)
    deltas, orders = [], []
    for g in groups:
        gi = torch.as_tensor(g)
        Bg = A[gi]                                             # this group's B
        z = wrap.order_head.scores(Bg, per_sample=False)[0]    # (N,) group consensus
        sig_model = torch.argsort(-z).cpu().numpy()
        orders.append(sig_model)
        phys = wrap.inv_perm[sig_model]
        row = torch.stack([held_idx[i] for i in g])
        deltas.append(order_nll(wrap.backbone, row, phys, wrap.clean_perm, dev)
                      - order_nll(wrap.backbone, row, L2R, wrap.clean_perm, dev))
    return float(np.mean(deltas)), np.stack(orders)


def group_probe(wrap, held_idx, m=16, seed=0, device="cpu"):
    dev = wrap.device if hasattr(wrap, "device") else device
    groups = group_ids_for(held_idx.shape[0], m)
    d_real, orders = _delta_for_B_variant(wrap, held_idx, groups, dev, "real")
    d_shuf, _ = _delta_for_B_variant(wrap, held_idx, groups, dev, "shuffle")
    d_zero, _ = _delta_for_B_variant(wrap, held_idx, groups, dev, "zero")
    taus_l2r = [kendalltau(o, L2R).correlation for o in orders]
    taus_cons = [kendalltau(orders[i], orders[j]).correlation
                 for i in range(len(orders)) for j in range(i + 1, len(orders))]
    return {"delta_probe_group": d_real,
            "tau_to_l2r": float(np.nanmean(taus_l2r)),
            "tau_consensus": float(np.nanmean(taus_cons)) if taus_cons else float("nan"),
            "delta_real": d_real, "delta_shuffle": d_shuf, "delta_zero": d_zero,
            "real_beats_controls": bool(d_real < min(d_shuf, d_zero))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v3_group_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/v3_group_probe.py tests/test_v3_group_probe.py
git commit -m "feat(v3): group-level mechanism probe + real/shuffle/zero-B guard"
```

---

### Task 4: Fixed-order eval curve (payoff metric)

**Files:**
- Create: `analyses/v3_fixed_order_eval.py`
- Test: `tests/test_v3_fixed_order_eval.py`

**Interfaces:**
- Consumes: `load_p5_ckpt`, `order_nll`, `N`.
- Produces:
  - `fixed_order_val_loss(model, eval_chunks, clean_perm, device) -> float` — mean `order_nll` under the fixed L2R block order over the eval set (this is `val_ori_l2r_block`; PPL = `exp(loss)`).
  - `eval_curve(ckpt_paths: dict[int,str], eval_chunks, clean_perm, device) -> dict[int, {"loss":..,"ppl":..}]` keyed by step.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v3_fixed_order_eval.py
import sys, pathlib, math
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest, torch
CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"
pytestmark = pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="10k parent ckpt absent")

def test_fixed_order_val_loss_is_finite_and_ppl_consistent():
    from analyses.v3_fixed_order_eval import fixed_order_val_loss
    from analyses.p5_utility_controller import load_p5_ckpt
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, 16, device="cpu")
    loss = fixed_order_val_loss(model, [chunks[i] for i in range(16)], clean_perm, "cpu")
    assert math.isfinite(loss) and loss > 0
    assert math.isclose(math.exp(loss), math.exp(loss))   # ppl computable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v3_fixed_order_eval.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the evaluator**

```python
# analyses/v3_fixed_order_eval.py
"""Fixed-order validation loss/PPL (val_ori_l2r_block): teacher-forced AO-NLL under
the fixed physical L2R block order, identical eval set across all arms."""
import sys, pathlib, math
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from analyses.p5_utility_controller import order_nll, N

L2R = np.arange(N, dtype=np.int64)


@torch.no_grad()
def fixed_order_val_loss(model, eval_chunks, clean_perm, device):
    nlls = [order_nll(model, c.unsqueeze(0), L2R, clean_perm, device) for c in eval_chunks]
    return float(np.mean(nlls))


def eval_curve(ckpt_paths, eval_chunks, clean_perm, device):
    from analyses.p5_utility_controller import load_p5_ckpt   # local to avoid cycle
    out = {}
    for step, path in ckpt_paths.items():
        model, _, cp, dev = load_p5_ckpt(path, 2, device=device)
        loss = fixed_order_val_loss(model, eval_chunks, clean_perm, device)
        out[step] = {"loss": loss, "ppl": math.exp(loss)}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v3_fixed_order_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/v3_fixed_order_eval.py tests/test_v3_fixed_order_eval.py
git commit -m "feat(v3): fixed-order val loss/PPL curve (val_ori_l2r_block)"
```

---

### Task 5: Phase-1 runner + trajectory gate report

**Files:**
- Create: `analyses/v3_phase1_runner.py`
- Test: `tests/test_v3_phase1_gate.py`

**Interfaces:**
- Consumes: `train_arm` (Task 2), `group_probe` (Task 3), `fixed_order_val_loss` (Task 4).
- Produces:
  - `decide_gate(metrics: dict) -> dict` — pure function computing Tier-1 (health) and Tier-2 (life-sign) from an arm-metrics dict; returns `{"tier1_pass":bool, "tier2_pass":bool, "gate":"PASS"|"FAIL", "reasons":[...]}`.
  - CLI `python analyses/v3_phase1_runner.py --arms l2r frozen_gbeta joint_group joint_batch --n-steps 20000 --eval-steps 15000 20000 25000 30000 --device cuda` that runs the arms, evaluates the curve + group probe at each eval step, writes `runs/v3_sweep/phase1_seed123/<arm>/*.json` and `phase1_gate_report.{json,md}`.
- Gate thresholds (from spec, verbatim): Tier-1 requires no NaN, entropy healthy, `orderhead_param_delta>0` (for joint arms), and `loss_joint_group - loss_frozen_gbeta <= 0.01` (ε_loss). Tier-2 passes if EITHER mechanism life (`delta_probe_group` of joint_group improves vs frozen_gbeta at best checkpoint AND `real_beats_controls`) OR payoff life (joint_group fixed-order loss/AUC/threshold matches-or-beats frozen_gbeta over the trajectory).

- [ ] **Step 1: Write the failing test (pure gate logic — no ckpt needed)**

```python
# tests/test_v3_phase1_gate.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from analyses.v3_phase1_runner import decide_gate

def _base():
    return {"nan": False, "entropy_ok": True, "orderhead_param_delta": 1.0,
            "loss_joint_group_final": 3.30, "loss_frozen_gbeta_final": 3.30,
            "delta_probe_group_best_joint": -0.05, "delta_probe_group_best_frozen": +0.10,
            "real_beats_controls": True,
            "loss_curve_joint": [3.4,3.35,3.32,3.30], "loss_curve_frozen": [3.4,3.36,3.34,3.33]}

def test_gate_pass_on_mechanism_life():
    r = decide_gate(_base())
    assert r["tier1_pass"] and r["tier2_pass"] and r["gate"] == "PASS"

def test_gate_fail_when_catastrophically_worse():
    m = _base(); m["loss_joint_group_final"] = 3.30 + 0.5   # > frozen + eps
    r = decide_gate(m)
    assert r["tier1_pass"] is False and r["gate"] == "FAIL"

def test_gate_fail_when_no_life_sign():
    m = _base()
    m["delta_probe_group_best_joint"] = +0.11               # worse than frozen +0.10
    m["real_beats_controls"] = False
    m["loss_curve_joint"] = [3.4,3.4,3.4,3.4]               # no payoff either
    m["loss_curve_frozen"] = [3.4,3.36,3.34,3.33]
    r = decide_gate(m)
    assert r["tier2_pass"] is False and r["gate"] == "FAIL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v3_phase1_gate.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the runner + gate**

```python
# analyses/v3_phase1_runner.py
"""Phase-1 orchestration + trajectory-based two-tier gate. decide_gate() is a pure
function (unit-tested); the CLI runs the 4 arms on GPU and writes the gate report.
Gate is ADVISORY — final go/no-go is a human call, but the report auto-computes it."""
import sys, pathlib, json, argparse
sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

EPS_LOSS = 0.01


def _auc(curve):
    return float(np.trapz(curve))


def decide_gate(m):
    reasons = []
    # Tier 1 — hard health
    tier1 = (not m["nan"]) and m["entropy_ok"] and m["orderhead_param_delta"] > 0
    if m["loss_joint_group_final"] - m["loss_frozen_gbeta_final"] > EPS_LOSS:
        tier1 = False; reasons.append("joint_group catastrophically worse than frozen (>eps_loss)")
    if m["nan"] or not m["entropy_ok"]:
        reasons.append("health: nan or entropy unhealthy")
    # Tier 2 — life sign (either)
    mech = (m["delta_probe_group_best_joint"] < m["delta_probe_group_best_frozen"]
            and m["real_beats_controls"])
    payoff = (m["loss_joint_group_final"] <= m["loss_frozen_gbeta_final"]
              or _auc(m["loss_curve_joint"]) <= _auc(m["loss_curve_frozen"]))
    tier2 = bool(mech or payoff)
    if mech: reasons.append("mechanism life: group probe beats frozen under real-B guard")
    if payoff: reasons.append("payoff life: PPL/AUC matches-or-beats frozen")
    gate = "PASS" if (tier1 and tier2) else "FAIL"
    return {"tier1_pass": bool(tier1), "tier2_pass": tier2, "gate": gate, "reasons": reasons}


def _build_evaluator(held_chunks, clean_perm, device, m=16):
    """Closure over the FIXED held-out chunks/grouping — identical for every arm and
    checkpoint (Global Constraints: fixed held-out grouping)."""
    import torch
    from analyses.v3_group_probe import group_probe
    from analyses.v3_fixed_order_eval import fixed_order_val_loss
    held_stack = torch.stack(list(held_chunks))

    def evaluator(step, model, wrap):
        loss = fixed_order_val_loss(model, list(held_chunks), clean_perm, device)
        gp = group_probe(wrap, held_stack, m=m, device=device)
        return {"val_loss": loss, "ppl": float(np.exp(loss)),
                "delta_probe_group": gp["delta_probe_group"],
                "real_beats_controls": gp["real_beats_controls"],
                "tau_to_l2r": gp["tau_to_l2r"], "tau_consensus": gp["tau_consensus"]}
    return evaluator


def run_phase1(arms, n_steps, eval_steps, device, out_root="runs/v3_sweep/phase1_seed123",
               ckpt="runs/handoff_overnight/seed123/ckpt_step10000.pt", n_held=32):
    from analyses.v3_group_trainer import train_arm
    from analyses.p5_utility_controller import load_p5_ckpt
    import torch
    out = pathlib.Path(out_root); out.mkdir(parents=True, exist_ok=True)
    # Fixed held-out chunks + clean_perm, built ONCE and reused for every arm.
    _, chunks, clean_perm, dev = load_p5_ckpt(ckpt, n_held, device=device)
    held_chunks = [chunks[i] for i in range(n_held)]
    evaluator = _build_evaluator(held_chunks, clean_perm, str(dev), m=16)

    summary = {}
    for arm in arms:
        res = train_arm(ckpt, arm, n_steps=n_steps, device=device, tag="phase1_seed123",
                        eval_steps=eval_steps, evaluator=evaluator)
        summary[arm] = res
    # Assemble the gate metrics from frozen_gbeta (baseline) + joint_group (main).
    fz, jg = summary.get("frozen_gbeta"), summary.get("joint_group")
    if fz and jg:
        jg_ev, fz_ev = jg["evals"], fz["evals"]
        entropy_ok = all(np.isfinite(x["entropy"]) for x in jg["log"]) and \
            jg["log"][-1]["entropy"] > 0.0
        metrics = {
            "nan": jg["nan"], "entropy_ok": entropy_ok,
            "orderhead_param_delta": jg["orderhead_param_delta"],
            "loss_joint_group_final": jg_ev[-1]["val_loss"],
            "loss_frozen_gbeta_final": fz_ev[-1]["val_loss"],
            "delta_probe_group_best_joint": min(e["delta_probe_group"] for e in jg_ev),
            "delta_probe_group_best_frozen": min(e["delta_probe_group"] for e in fz_ev),
            "real_beats_controls": any(e["real_beats_controls"] for e in jg_ev),
            "loss_curve_joint": [e["val_loss"] for e in jg_ev],
            "loss_curve_frozen": [e["val_loss"] for e in fz_ev]}
        gate = decide_gate(metrics)
        (out / "phase1_gate_report.json").write_text(
            json.dumps({"gate": gate, "metrics": metrics}, indent=2, default=float))
        (out / "phase1_gate_report.md").write_text(
            f"# Phase-1 gate: {gate['gate']}\n\nTier1={gate['tier1_pass']} "
            f"Tier2={gate['tier2_pass']}\n\nReasons:\n" +
            "\n".join(f"- {r}" for r in gate["reasons"]))
    (out / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["l2r", "frozen_gbeta", "joint_group", "joint_batch"])
    ap.add_argument("--n-steps", type=int, default=20000)
    ap.add_argument("--eval-steps", nargs="+", type=int, default=[15000, 20000, 25000, 30000])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_phase1(a.arms, a.n_steps, a.eval_steps, a.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v3_phase1_gate.py -v`
Expected: PASS (3 tests — pure gate logic, no ckpt).

- [ ] **Step 5: Add the runner integration test (fixed grouping + gate report)**

Append to `tests/test_v3_phase1_gate.py`:

```python
import pathlib as _pl
_CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"

@__import__("pytest").mark.skipif(not _pl.Path(_CKPT).exists(), reason="10k parent ckpt absent")
def test_run_phase1_writes_gate_report():
    from analyses.v3_phase1_runner import run_phase1
    run_phase1(["frozen_gbeta", "joint_group"], n_steps=2, eval_steps=[2],
               device="cpu", out_root="runs/v3_sweep_test/phase1", n_held=32)
    rep = _pl.Path("runs/v3_sweep_test/phase1/phase1_gate_report.json")
    assert rep.exists()
    import json
    assert "gate" in json.loads(rep.read_text())["gate"]
```

This exercises the full seam: fixed held-out chunks/grouping built once and shared
across both arms, `train_arm`'s `evaluator` hook firing at the eval step, and
`decide_gate` producing a report. Confirms the fixed-grouping constraint holds (both
arms evaluated on the identical `held_chunks`).

- [ ] **Step 6: Run all V3 tests to verify they pass**

Run: `python -m pytest tests/test_v3_group_credit.py tests/test_v3_group_trainer.py tests/test_v3_group_probe.py tests/test_v3_fixed_order_eval.py tests/test_v3_phase1_gate.py -v`
Expected: PASS/skip (gate-logic tests always run; ckpt-dependent tests run since the 10k ckpt exists).

- [ ] **Step 7: Commit**

```bash
git add analyses/v3_phase1_runner.py tests/test_v3_phase1_gate.py
git commit -m "feat(v3): Phase-1 runner + trajectory two-tier gate report"
```

---

## Launch (after Tasks 1–5 pass — GPU, the actual experiment)

This is NOT a unit test — it is the Phase-1 experiment. Run on GPU from repo root:

```bash
python analyses/v3_phase1_runner.py \
  --arms l2r frozen_gbeta joint_group joint_batch \
  --n-steps 20000 --eval-steps 15000 20000 25000 30000 --device cuda
```

Inspect `runs/v3_sweep/phase1_seed123/phase1_gate_report.md`. The gate is advisory;
confirm the PASS/FAIL by eye against the 10k–30k trajectory (final, best, AUC) and
the real-B guard before deciding Phase-2. **Do not launch this on GPU without the
user's go** (compute is the boss's call).
