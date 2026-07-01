# OrderHead Internalization Feasibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the external frozen `gβ` hook with an internal deterministic `OrderHead`, then run a short same-batch probe-then-train joint smoke (per-sample `[B,T]` orders as the main arm, batch-level broadcast as a diagnostic sanity arm) — no full V3 sweep, no lagged single-forward controller, no multi-seed.

**Architecture:** An `OrderHeadModule` wraps the deployed `gβ` MLP as a model component with batch-mean and per-sample readouts plus `argsort` / PL sampling APIs. An `AOGPTWithOrderHead` wrapper extracts head-L1H7 attention `B` from the frozen-or-trainable backbone, **detaches** it into the OrderHead, and builds model→physical→model-token reveal orders. A smoke driver runs the two-forward joint loop (LM loss → backbone φ, PG loss → OrderHead θ) for both arms and logs feasibility metrics.

**Tech Stack:** Python, PyTorch, existing `block_lo_arm_order_network` infra (`FrozenBetaHook`, `extract_selected_head_A_for_batch`, `pl_argsort`, `sample_pl`), `analyses/p5_utility_controller` + `analyses/p7_gbeta_policy`, `scipy.stats.kendalltau`. CPU is sufficient for D1 and a short smoke.

## Global Constraints

- **Run from repo root.** All scripts start with `import sys; sys.path.insert(0, "block_lo_arm_order_network")` before importing `batch_readout.*` / `clean_training_protocol`.
- **Reused constants (verbatim):** `GBETA_CKPT = "reports/uniform_label_free_v1/nodewise_K1000.pt"`, `HEAD = (1, 7)`, `NONE_MODE = "strict65_model"` — import from `analyses.p7_gbeta_policy`.
- **Frames:** `gβ` produces **MODEL-frame** block scores. Bit-match (D1) compares in MODEL frame. Any `order_nll` / L2R comparison is **PHYSICAL frame**: remap `order_phys = inv_perm[order_model]` where `inv_perm = clean_perm.inv_perm_model_to_phys` (`inv_perm[model_block] = physical_block`), then `physical_blocks_to_model_token_order(...)` inside `order_nll`.
- **Gradient routing (hard requirement):** `B` is `.detach()`-ed before the OrderHead; the PG advantage `A` is detached (`lm_loss.detach()`). LM loss updates backbone φ only; PG loss updates OrderHead θ only. No PG gradient may reach the backbone.
- **D1 uses deterministic `argsort` only** (via `pl_argsort`), never PL sampling. Low-temp PL only concentrates near `argsort`, not equal.
- **Per-sample orders stay `[B, T]`** — never collapse per-sample to batch-level in the main arm. `orders.shape == idx.shape` is required by `AOGPT_block.forward` (`mode=None`).
- **Scope guard:** implement only Tasks 1–4. Do NOT add the full arm matrix, lagged single-forward controller, multi-seed eval, step-to-threshold/PPL curves, or the confound-guarded probe harness — those are the deferred V3 sweep.
- Results land under `runs/v3_feasibility/`.

---

### Task 1: `OrderHeadModule` (internal gβ wrapper) + batch-mean bit-match

**Files:**
- Create: `analyses/order_head_module.py`
- Test: `tests/test_order_head_module.py`

**Interfaces:**
- Consumes: `FrozenBetaHook` (`block_lo_arm_order_network/batch_readout/integration_hook.py`), `pl_argsort` (`batch_readout/pl_sampling.py`), `sample_pl` + `GBETA_CKPT` (`analyses/p7_gbeta_policy.py`).
- Produces:
  - `OrderHeadModule(gbeta_ckpt=GBETA_CKPT, device="cpu")` — `nn.Module` holding `self.gbeta`.
  - `OrderHeadModule.scores(A: Tensor(Bs,N,N), per_sample: bool) -> Tensor(rows,N)` (rows=`Bs` if per_sample else 1), MODEL frame, grad-enabled through `self.gbeta`.
  - `OrderHeadModule.argsort_order(A, per_sample=False) -> LongTensor(rows,N)` (model-frame block orders).
  - `OrderHeadModule.sample_pl_order(A, per_sample, tau) -> (np.ndarray(rows,N), Tensor(rows,), Tensor(rows,))` = `(orders, logps, entropies)`.
  - Module-level helper `bmatrix_from_A(A, per_sample) -> Tensor` (transpose, optional batch-mean, diagonal zeroed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_order_head_module.py
import sys, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from batch_readout.integration_hook import FrozenBetaHook
from analyses.order_head_module import OrderHeadModule
from analyses.p7_gbeta_policy import GBETA_CKPT

def test_batchmean_argsort_bitmatches_frozenbetahook():
    torch.manual_seed(0)
    N = 64                                         # gβ requires 64x64 B (verified)
    A = torch.rand(8, N, N)                        # synthetic per-sample attention
    hook = FrozenBetaHook(GBETA_CKPT, mode="argsort", device="cpu")
    ext_sigma = hook.step(A)                       # (64,) model-frame block order
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    int_sigma = oh.argsort_order(A, per_sample=False)[0]   # (64,)
    assert int_sigma.shape == ext_sigma.shape
    assert torch.equal(int_sigma, ext_sigma), "internal batch-mean argsort must bit-match the hook"

def test_per_sample_scores_shape():
    torch.manual_seed(0)
    A = torch.rand(6, 64, 64)
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    z_bm = oh.scores(A, per_sample=False)
    z_ps = oh.scores(A, per_sample=True)
    assert z_bm.shape == (1, 64)
    assert z_ps.shape == (6, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_order_head_module.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analyses.order_head_module'`.

- [ ] **Step 3: Write minimal implementation**

```python
# analyses/order_head_module.py
"""Internal OrderHead: wraps the deployed gβ MLP as a model module.

Batch-mean readout reproduces the external FrozenBetaHook (bit-match target);
per-sample readout is the V3-headline config. MODEL-frame scores throughout —
callers remap MODEL→PHYSICAL before any order_nll (see AOGPTWithOrderHead).
"""
import sys
sys.path.insert(0, "block_lo_arm_order_network")
import numpy as np
import torch
import torch.nn as nn

from batch_readout.integration_hook import FrozenBetaHook
from batch_readout.pl_sampling import pl_argsort
from analyses.p7_gbeta_policy import sample_pl, GBETA_CKPT


def bmatrix_from_A(A, per_sample):
    """A: (Bs,N,N) -> B. per_sample: (Bs,N,N); else batch-mean (1,N,N). Diagonal zeroed.
    Matches FrozenBetaHook.step: B = mean over batch of A^T, diag=0."""
    B = A.transpose(1, 2)
    B = B.clone() if per_sample else B.mean(dim=0, keepdim=True)
    n = B.shape[-1]
    d = torch.arange(n, device=B.device)
    B[:, d, d] = 0.0
    return B


class OrderHeadModule(nn.Module):
    def __init__(self, gbeta_ckpt=GBETA_CKPT, device="cpu"):
        super().__init__()
        hook = FrozenBetaHook(gbeta_ckpt, mode="argsort", device=device)
        self.gbeta = hook.model        # nn.Module, MODEL-frame scores; params ARE gβ
        self.device = device

    def scores(self, A, per_sample):
        B = bmatrix_from_A(A.to(self.device).float(), per_sample)
        return self.gbeta(B)           # (rows, N), grad-enabled through gβ

    def argsort_order(self, A, per_sample=False):
        z = self.scores(A, per_sample)
        return pl_argsort(z.detach().cpu())        # (rows, N) model-frame block orders

    def sample_pl_order(self, A, per_sample, tau):
        z = self.scores(A, per_sample)
        orders, logps, ents = [], [], []
        for r in range(z.shape[0]):
            o, lp, e = sample_pl(z[r], tau)
            orders.append(o); logps.append(lp); ents.append(e)
        return np.stack(orders), torch.stack(logps), torch.stack(ents)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_order_head_module.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add analyses/order_head_module.py tests/test_order_head_module.py
git commit -m "feat(v3): internal OrderHeadModule + batch-mean gβ bit-match"
```

---

### Task 2: `AOGPTWithOrderHead` wrapper (extract → detach → order + frame remap)

**Files:**
- Modify: `analyses/order_head_module.py` (append the wrapper class)
- Test: `tests/test_aogpt_with_orderhead.py`

**Interfaces:**
- Consumes: `OrderHeadModule` (Task 1); `extract_selected_head_A_for_batch`, `random_probe_token_orders` (`batch_readout/hook_order_provider.py`); `HEAD`, `NONE_MODE` (`analyses/p7_gbeta_policy`); `load_p5_ckpt`, `BLOCK_LEN`, `N` (`analyses/p5_utility_controller`); `physical_blocks_to_model_token_order` (`clean_training_protocol`).
- Produces:
  - `AOGPTWithOrderHead(backbone, order_head, clean_perm, device="cpu")` — `nn.Module`.
  - `.extract_B(idx_batch, probe) -> Tensor(Bs,N,N)` (MODEL frame, no detach here).
  - `.compute_order_logits(idx_batch, probe, per_sample) -> Tensor(rows,N)` — **detaches `B`**, returns grad-enabled scores through `order_head` only.
  - `.token_orders_from_model_blocks(order_model_BN: array(Bs,N)) -> LongTensor(Bs,T)` (model-block → physical-block → model-token).
  - `.inv_perm` (np.ndarray, `[model]=phys`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aogpt_with_orderhead.py
import sys, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import pytest
from batch_readout.integration_hook import FrozenBetaHook
from batch_readout.hook_order_provider import random_probe_token_orders
from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
from analyses.p5_utility_controller import load_p5_ckpt, N
from analyses.p7_gbeta_policy import GBETA_CKPT

CKPT = "block_lo_arm_order_network/probe_results/gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt"

@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="backbone ckpt not present")
def test_wrapper_extract_argsort_matches_hook():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=8, device="cpu")
    idx = torch.stack([chunks[i] for i in range(8)])
    probe = random_probe_token_orders(idx.shape[0], 0, 0, dev)
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    A = wrap.extract_B(idx, probe)                       # (8,N,N)
    hook = FrozenBetaHook(GBETA_CKPT, mode="argsort", device="cpu")
    ext = hook.step(A)                                   # (N,)
    internal = oh.argsort_order(A, per_sample=False)[0]  # (N,)
    assert torch.equal(internal, ext)

def test_per_sample_orders_keep_BN_shape_and_remap():
    # backbone unused by the remap path (pure index + block-perm arithmetic)
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    wrap = AOGPTWithOrderHead(backbone=None, order_head=oh,
                              clean_perm=_make_clean_perm(), device="cpu")
    order_model = np.stack([np.random.permutation(64) for _ in range(5)])  # (5,64)
    tok = wrap.token_orders_from_model_blocks(order_model)
    assert tok.shape[0] == 5 and tok.ndim == 2          # (5, T)

def _make_clean_perm():
    # reuse a real clean_perm from a tiny ckpt load; skip if the ckpt is absent
    if pathlib.Path(CKPT).exists():
        _, _, clean_perm, _ = load_p5_ckpt(CKPT, M=2, device="cpu")
        return clean_perm
    pytest.skip("no clean_perm source")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aogpt_with_orderhead.py -v`
Expected: FAIL with `ImportError: cannot import name 'AOGPTWithOrderHead'`.

- [ ] **Step 3: Write minimal implementation (append to `analyses/order_head_module.py`)**

```python
from batch_readout.hook_order_provider import (
    extract_selected_head_A_for_batch, random_probe_token_orders,  # noqa: F401
)
from analyses.p7_gbeta_policy import HEAD, NONE_MODE
from analyses.p5_utility_controller import BLOCK_LEN, N  # noqa: F401
from clean_training_protocol import physical_blocks_to_model_token_order


class AOGPTWithOrderHead(nn.Module):
    """Backbone + internal OrderHead. B is detached before the OrderHead so no
    policy gradient reaches the backbone (implicit co-adaptation)."""
    def __init__(self, backbone, order_head, clean_perm, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.order_head = order_head
        self.clean_perm = clean_perm
        self.inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()  # [model]=phys
        self.device = device

    def extract_B(self, idx_batch, probe):
        A = extract_selected_head_A_for_batch(
            self.backbone, idx_batch.to(self.device), HEAD, self.clean_perm,
            self.device, probe, none_mode=NONE_MODE)
        return A.to(self.device).float()

    def compute_order_logits(self, idx_batch, probe, per_sample):
        A = self.extract_B(idx_batch, probe).detach()   # DETACH — no grad to backbone
        return self.order_head.scores(A, per_sample)     # grad on order_head only

    def token_orders_from_model_blocks(self, order_model_BN):
        phys = self.inv_perm[np.asarray(order_model_BN, dtype=np.int64)]   # (Bs,N) model->phys
        toks = [physical_blocks_to_model_token_order(
                    torch.from_numpy(phys[r:r + 1]), self.clean_perm, BLOCK_LEN)[0]
                for r in range(phys.shape[0])]
        return torch.stack(toks)                                          # (Bs,T)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aogpt_with_orderhead.py -v`
Expected: PASS (ckpt-dependent test may `skip` if the ckpt is absent; the shape/remap test passes).

- [ ] **Step 5: Commit**

```bash
git add analyses/order_head_module.py tests/test_aogpt_with_orderhead.py
git commit -m "feat(v3): AOGPTWithOrderHead wrapper (detach B + model->phys->token remap)"
```

---

### Task 3: D2 joint smoke driver (per-sample main + batch-level control)

**Files:**
- Create: `analyses/v3_feasibility_smoke.py`
- Test: (covered by Task 4)

**Interfaces:**
- Consumes: `AOGPTWithOrderHead`, `OrderHeadModule` (Tasks 1–2); `load_p5_ckpt`, `order_nll`, `N` (`analyses/p5_utility_controller`); `sample_pl` (`analyses/p7_gbeta_policy`); `random_probe_token_orders`; `kendalltau`.
- Produces:
  - `run_smoke(ckpt_path, arm, n_steps=40, batch_size=16, tau=1.0, beta=3e-3, lam_pg=1.0, ema_decay=0.9, adv_clip=0.3, lr=3e-4, held=32, device="cpu", out_dir="runs/v3_feasibility") -> dict` where `arm in {"per_sample", "batch_level"}`.
  - `denoising_metrics(wrap, held_idx, clean_perm, dev) -> {"tau_consensus": float, "delta_probe": float}` (per-sample only).
  - Writes `<out_dir>/<arm>_smoke.json` with the per-step log + summary.

- [ ] **Step 1: Write the implementation**

```python
# analyses/v3_feasibility_smoke.py
"""D2 joint co-adaptation smoke. Same-batch probe-then-train two-forward path.
per_sample arm = V3 headline config ([B,T] orders); batch_level arm = broadcast
sanity/diagnostic. Feasibility sign checks only — NOT a scientific verdict."""
import sys, json, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
import numpy as np
import torch
from scipy.stats import kendalltau

from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
from analyses.p5_utility_controller import load_p5_ckpt, order_nll, N
from analyses.p7_gbeta_policy import sample_pl, GBETA_CKPT
from batch_readout.hook_order_provider import random_probe_token_orders

L2R = np.arange(N, dtype=np.int64)


def _sample_orders(z, tau):
    orders, logps, ents = [], [], []
    for r in range(z.shape[0]):
        o, lp, e = sample_pl(z[r], tau)
        orders.append(o); logps.append(lp); ents.append(e)
    return np.stack(orders), torch.stack(logps), torch.stack(ents)


def denoising_metrics(wrap, held_idx, dev):
    """Per-sample consensus τ + Δ_probe vs L2R (uses no_grad order_nll)."""
    probe = random_probe_token_orders(held_idx.shape[0], 0, 0, dev)
    sig_model = wrap.order_head.argsort_order(
        wrap.extract_B(held_idx, probe), per_sample=True)          # (M,64) model-frame
    sig_model = sig_model.cpu().numpy()
    taus = [kendalltau(sig_model[i], sig_model[j]).correlation
            for i in range(len(sig_model)) for j in range(i + 1, len(sig_model))]
    tau_consensus = float(np.nanmean(taus)) if taus else float("nan")
    deltas = []
    for i in range(held_idx.shape[0]):
        row = held_idx[i:i + 1]
        phys = wrap.inv_perm[sig_model[i]]
        deltas.append(order_nll(wrap.backbone, row, phys, wrap.clean_perm, dev)
                      - order_nll(wrap.backbone, row, L2R, wrap.clean_perm, dev))
    return {"tau_consensus": tau_consensus, "delta_probe": float(np.mean(deltas))}


def run_smoke(ckpt_path, arm, n_steps=40, batch_size=16, tau=1.0, beta=3e-3,
              lam_pg=1.0, ema_decay=0.9, adv_clip=0.3, lr=3e-4, held=32,
              device="cpu", out_dir="runs/v3_feasibility"):
    assert arm in ("per_sample", "batch_level")
    per_sample = arm == "per_sample"
    M = n_steps * batch_size + held
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    oh = OrderHeadModule(GBETA_CKPT, device=str(dev))
    for p in oh.gbeta.parameters():
        p.requires_grad_(True)
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device=str(dev))
    held_idx = torch.stack([chunks[i] for i in range(held)]).to(dev)

    before = denoising_metrics(wrap, held_idx, dev) if per_sample else None
    oh_p0 = torch.cat([p.detach().flatten().clone() for p in oh.gbeta.parameters()])

    opt = torch.optim.Adam(
        list(model.parameters()) + list(oh.gbeta.parameters()), lr=lr)
    ema, log = None, []
    for step in range(n_steps):
        s = held + step * batch_size
        idx = torch.stack([chunks[s + i] for i in range(batch_size)]).to(dev)
        probe = random_probe_token_orders(idx.shape[0], 0, step, dev)
        z = wrap.compute_order_logits(idx, probe, per_sample)     # (rows,N) grad on OH
        order_bn, logp, ent = _sample_orders(z, tau)              # rows = B or 1
        if not per_sample:                                       # broadcast one order to all rows
            order_bn = np.repeat(order_bn, batch_size, axis=0)
        token_order = wrap.token_orders_from_model_blocks(order_bn).to(dev)  # (B,T)
        _, lm_loss = model.forward_fn(idx, token_order)           # grad on backbone (scalar batch loss)
        r = lm_loss.detach()
        ema = r if ema is None else ema_decay * ema + (1 - ema_decay) * r
        # Feasibility simplification: batch-scalar advantage (forward_fn returns a
        # batch-mean loss). Per-sample credit assignment is a V3-sweep refinement.
        adv = (ema - r).clamp(-adv_clip, adv_clip)
        pg = -(adv * logp.mean()) - beta * ent.mean()
        loss = lm_loss + lam_pg * pg
        opt.zero_grad(); loss.backward(); opt.step()
        log.append({"step": step, "lm_loss": float(lm_loss), "pg": float(pg),
                    "adv": float(adv), "entropy": float(ent.mean()),
                    "logp": float(logp.mean())})

    oh_p1 = torch.cat([p.detach().flatten().clone() for p in oh.gbeta.parameters()])
    after = denoising_metrics(wrap, held_idx, dev) if per_sample else None
    result = {
        "arm": arm, "n_steps": n_steps, "batch_size": batch_size,
        "orderhead_param_delta": float((oh_p1 - oh_p0).abs().sum()),
        "entropy_first": log[0]["entropy"], "entropy_last": log[-1]["entropy"],
        "nan": any(not np.isfinite(x["lm_loss"]) for x in log),
        "denoise_before": before, "denoise_after": after, "log": log,
    }
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{arm}_smoke.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/handoff_overnight/seed123/ckpt_step10000.pt",
                    help="10k parent lineage; any load_p5_ckpt-compatible ckpt works for the smoke")
    ap.add_argument("--arm", choices=["per_sample", "batch_level"], default="per_sample")
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(json.dumps(run_smoke(a.ckpt, a.arm, n_steps=a.n_steps, device=a.device)
                     ["orderhead_param_delta"]))
```

- [ ] **Step 2: Verify the driver imports cleanly**

Run: `python -c "import sys; sys.path.insert(0,'block_lo_arm_order_network'); import analyses.v3_feasibility_smoke as m; print('ok', hasattr(m,'run_smoke'))"`
Expected: `ok True`.

- [ ] **Step 3: Commit**

```bash
git add analyses/v3_feasibility_smoke.py
git commit -m "feat(v3): D2 joint smoke driver (per-sample + batch-level arms)"
```

---

### Task 4: D2 smoke test — feasibility sign checks + gradient attribution

**Files:**
- Create: `tests/test_v3_feasibility_smoke.py`

**Interfaces:**
- Consumes: `run_smoke` (Task 3), `AOGPTWithOrderHead` / `OrderHeadModule` (Tasks 1–2), `load_p5_ckpt`, `random_probe_token_orders`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v3_feasibility_smoke.py
import sys, pathlib, math
sys.path.insert(0, "block_lo_arm_order_network")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import pytest
from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
from analyses.p5_utility_controller import load_p5_ckpt
from analyses.p7_gbeta_policy import GBETA_CKPT
from batch_readout.hook_order_provider import random_probe_token_orders

CKPT = "block_lo_arm_order_network/probe_results/gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt"
pytestmark = pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="backbone ckpt absent")

def test_grad_routing_pg_does_not_touch_backbone():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=8, device="cpu")
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    for p in oh.gbeta.parameters(): p.requires_grad_(True)
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    idx = torch.stack([chunks[i] for i in range(8)])
    probe = random_probe_token_orders(8, 0, 0, dev)
    z = wrap.compute_order_logits(idx, probe, per_sample=True)   # B detached inside
    # PG-only surrogate: depends on OrderHead params, not backbone
    pg = z.sum()
    model.zero_grad(); [p.grad and p.grad.zero_() for p in oh.gbeta.parameters()]
    pg.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters()), \
        "PG term must not produce backbone gradient (B is detached)"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in oh.gbeta.parameters()), \
        "PG term must update OrderHead params"

def test_smoke_runs_and_params_move():
    from analyses.v3_feasibility_smoke import run_smoke
    res = run_smoke(CKPT, arm="per_sample", n_steps=3, batch_size=4, held=6,
                    device="cpu", out_dir="runs/v3_feasibility_test")
    assert res["nan"] is False
    assert res["orderhead_param_delta"] > 0.0
    assert math.isfinite(res["entropy_last"])
    assert 0.0 < res["entropy_last"]                       # entropy did not collapse to 0
    assert res["denoise_before"] is not None and res["denoise_after"] is not None

def test_per_sample_orders_are_diverse_BN():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=8, device="cpu")
    oh = OrderHeadModule(GBETA_CKPT, device="cpu")
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    idx = torch.stack([chunks[i] for i in range(8)])
    probe = random_probe_token_orders(8, 0, 0, dev)
    A = wrap.extract_B(idx, probe)
    orders = oh.argsort_order(A, per_sample=True).cpu().numpy()  # (8,N)
    assert orders.shape[0] == 8
    assert not np.all(orders == orders[0]), "per-sample orders must not be identical"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v3_feasibility_smoke.py -v`
Expected: FAIL (import of `run_smoke` fails before Task 3, or collection error) — if Task 3 already landed, the grad-routing test fails only on a real wiring bug.

- [ ] **Step 3: Make them pass**

No new implementation code — Tasks 1–3 provide everything. If `test_grad_routing_pg_does_not_touch_backbone` fails, the `.detach()` in `AOGPTWithOrderHead.compute_order_logits` is missing or misplaced; fix there. If entropy collapses to 0, lower `tau` sensitivity is not the issue for a 3-step smoke — inspect `sample_pl` logits.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v3_feasibility_smoke.py -v`
Expected: PASS (tests `skip` cleanly if `CKPT` is absent).

- [ ] **Step 5: Run the full test module + a real per-sample + batch-level smoke**

Run:
```bash
python -m pytest tests/test_order_head_module.py tests/test_aogpt_with_orderhead.py tests/test_v3_feasibility_smoke.py -v
python analyses/v3_feasibility_smoke.py --arm per_sample --n-steps 40 --device cpu
python analyses/v3_feasibility_smoke.py --arm batch_level --n-steps 40 --device cpu
```
Expected: tests PASS/skip; both smokes write `runs/v3_feasibility/{per_sample,batch_level}_smoke.json` with `nan=false`, `orderhead_param_delta > 0`, finite entropy.

- [ ] **Step 6: Commit**

```bash
git add tests/test_v3_feasibility_smoke.py
git commit -m "test(v3): D2 feasibility sign checks + PG->backbone grad-routing guard"
```

---

## Feasibility gate (read after Tasks 1–4)

Report per the spec's decision table:
- **D1 PASS** = Task 1 + Task 2 bit-match tests pass (τ ≥ 0.999 / exact, MODEL frame).
- **D2 healthy** = Task 4: both arms run (no NaN), OrderHead params move, entropy in band, PG→backbone grad == 0.
- **Denoising sign** (per-sample arm) = compare `denoise_after` vs `denoise_before` (`tau_consensus` vs 0.31 baseline; `delta_probe` shift). Soft evidence only — a flat sign on a 40-step CPU smoke is an informative negative, not a wiring failure.
- If all three hold → greenlight the separate V3 sweep spec. Otherwise isolate the failing layer (internalization / infra / per-sample variance) using the batch-level control as the contrast.
