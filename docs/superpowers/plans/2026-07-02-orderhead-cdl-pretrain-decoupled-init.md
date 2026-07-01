# OrderHead CDL-Pretrain → Frozen Warmup → PG Unfreeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the production `train_clean_aogpt.py` path, decouple the OrderHead's CDL initialization from its downstream fine-tuning: CDL-pretrain gβ (L0DynamicGBeta) offline, warm the backbone up with gβ frozen (argsort), then at a scheduled step unfreeze gβ and fine-tune it with LM-NLL policy gradient — CDL and PG never in the same loss.

**Architecture:** Stage 1 is a thin orchestrator over the existing two-step CDL pipeline (`build_l0_dynamic_gbeta_dataset` + `train_l0_dynamic_gbeta`). Stages 2–3 live in `train_clean_aogpt.py --run-kind frozen_beta`: Stage 2 is today's frozen behavior; Stage 3 adds a `--unfreeze-orderhead-at-step` schedule that flips gβ to trainable, inserts one optimizer param group, switches the order policy argsort→Plackett-Luce, and applies a `v3_group_credit` batch-level advantage.

**Tech Stack:** PyTorch, NumPy, pytest. Reuses `batch_readout.l0_dynamic_gbeta.L0DynamicGBeta`, `batch_readout.frozen_gbeta_hook.FrozenGBetaModelFrameBlockProvider`, `analyses.v3_group_credit`, `analyses.p7_gbeta_policy.sample_pl`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-02-orderhead-cdl-pretrain-decoupled-init-design.md` (commit `1f11f00`). Every task's requirements implicitly include the spec's Red Lines.
- **Core principle:** CDL initializes gβ ⇒ frozen argsort curriculum ⇒ LM-NLL PG fine-tuning. CDL appears ONLY in Stage 1.
- **gβ = `L0DynamicGBeta`**, input `B_raw (batch, H=8, 65, 65)`, output `(scores (B,64), aux)`; deployed 8-head batch-mean via `FrozenGBetaModelFrameBlockProvider` (`--batch-mean-probes 4`, `none_mode='model'`).
- **Source backbone (10k parent):** `block_lo_arm_order_network/probe_results/overnight_20260625_random_baseline/ckpt_step10000.pt`.
- **Stage-1 loss default:** `--loss-type pairwise_bce` (internally `soft_pairwise_bce_loss`).
- **Stage-3 is batch-level:** one `σ_batch`/`logp` per step; m=16 groups aggregate to a scalar `A_batch` (baseline variance reduction only). True group-level orders are OUT OF SCOPE (deferred `--orderhead-scope group`).
- **B_PG == B_frozen-provider:** the grad-enabled gβ input is constructed identically to the frozen provider's B (same `batch_mean_probes`, 8 heads, strict65, `none_mode='model'`, model frame, probe seeds); the ONLY difference is grad-enabled vs `no_grad`.
- **Order frame:** PL returns a model-frame block order; convert to token order via the SAME path the frozen loop uses (`model_blocks_to_physical_blocks` → `physical_blocks_to_model_token_order`). No P7 physical `order_nll` remap.
- **λ_PG (`--lam-pg`, default 1.0):** Phase A does not construct `L_PG` (≡0); Phase B `λ_PG>0`.
- **Backward compatible:** with `--unfreeze-orderhead-at-step` absent, `frozen_beta` behaves exactly as today.
- **Commit** after each task's tests pass. Do NOT push unless asked.

## File Structure

- **new** `analyses/gbeta_cdl_pretrain.py` — Stage-1 orchestrator + provenance sidecar (one responsibility: produce a gβ ckpt from a backbone ckpt via the existing CDL pipeline).
- **new** `block_lo_arm_order_network/batch_readout/orderhead_pg.py` — Stage-3 pure helpers: grad-enabled gβ scorer wrapper, PL sampling adapter, batch-level group-credit advantage. Isolated so the PG math is unit-testable without the 1900-line training loop.
- **edit** `block_lo_arm_order_network/train_clean_aogpt.py` — new CLI flags; Stage-1 auto-produce resolution; wire `orderhead_pg` into the `frozen_beta` step with the unfreeze schedule + one-time param-group insertion + logging.
- **new tests** under `block_lo_arm_order_network/tests/`:
  - `test_gbeta_cdl_pretrain.py` (Task 1)
  - `test_orderhead_pg.py` (Tasks 2–4 unit)
  - `test_orderhead_unfreeze_schedule.py` (Task 3–4 integration-lite)

---

### Task 1: CDL producer orchestrator

**Files:**
- Create: `analyses/gbeta_cdl_pretrain.py`
- Test: `block_lo_arm_order_network/tests/test_gbeta_cdl_pretrain.py`

**Interfaces:**
- Consumes (existing, verified):
  - `analyses.build_l0_dynamic_gbeta_dataset.build_l0_dynamic_gbeta_dataset(ckpt_path, M=2000, batch_mean_size=16, seed=2, device, split="train", ..., out_path=None, layers=(0,), heads=()) -> str` (returns `.npz` path)
  - `batch_readout.train_l0_dynamic_gbeta.train(dataset_path, out_dir, epochs=40, batch_size=32, lr=3e-4, ..., heads=8, loss_type="pairwise_bce", device) -> dict` (writes `<out_dir>/g_beta_best.pt` = `{"model_state_dict", "config"}`)
- Produces (later tasks / callers rely on):
  - `pretrain_gbeta_cdl(ckpt_10k, *, out_dir, loss_type="pairwise_bce", M=2000, heads=8, epochs=40, seed=0, device="cpu", dataset_path=None) -> str` returning the `g_beta_best.pt` path. Writes `<out_dir>/gbeta_provenance.json`.

- [ ] **Step 1: Write the failing test (provenance + shape via a stubbed pipeline)**

Use monkeypatch to stub the two heavy steps so the test is CPU-fast and asserts orchestration + provenance, not GPU training.

```python
# block_lo_arm_order_network/tests/test_gbeta_cdl_pretrain.py
import json, pathlib, sys
import numpy as np, torch, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta


def _write_fake_gbeta(out_dir):
    m = L0DynamicGBeta(heads=8, nodes=65)
    ck = pathlib.Path(out_dir) / "g_beta_best.pt"
    torch.save({"model_state_dict": m.state_dict(),
                "config": {"heads": 8, "nodes": 65}}, ck)
    return str(ck)


def test_pretrain_orchestrates_and_writes_provenance(tmp_path, monkeypatch):
    import analyses.gbeta_cdl_pretrain as mod

    calls = {}
    def fake_build(ckpt_path, **kw):
        calls["build"] = {"ckpt_path": ckpt_path, **kw}
        npz = tmp_path / "ds.npz"
        np.savez(npz, teacher_pairwise=np.zeros((2, 64, 64), dtype=np.float32))
        return str(npz)
    def fake_train(dataset_path, out_dir, **kw):
        calls["train"] = {"dataset_path": dataset_path, "out_dir": out_dir, **kw}
        _write_fake_gbeta(out_dir)
        return {"best_metrics": {"pairwise_acc": 0.9}}

    monkeypatch.setattr(mod, "build_l0_dynamic_gbeta_dataset", fake_build)
    monkeypatch.setattr(mod, "train_l0_dynamic_gbeta_train", fake_train)

    out = mod.pretrain_gbeta_cdl(
        "FAKE_10k.pt", out_dir=str(tmp_path), loss_type="pairwise_bce",
        M=8, heads=8, epochs=1, seed=0, device="cpu")

    assert pathlib.Path(out).name == "g_beta_best.pt"
    assert calls["train"]["loss_type"] == "pairwise_bce"
    assert calls["build"]["ckpt_path"] == "FAKE_10k.pt"
    prov = json.loads((tmp_path / "gbeta_provenance.json").read_text())
    assert prov["producer"] == "build_l0_dynamic_gbeta_dataset + train_l0_dynamic_gbeta"
    assert prov["source_ckpt"] == "FAKE_10k.pt"
    assert prov["loss_type"] == "pairwise_bce"


def test_reuse_dataset_skips_build(tmp_path, monkeypatch):
    import analyses.gbeta_cdl_pretrain as mod
    hit = {"build": 0}
    def fake_build(*a, **k):
        hit["build"] += 1
        return "SHOULD_NOT_BE_CALLED"
    def fake_train(dataset_path, out_dir, **kw):
        _write_fake_gbeta(out_dir); return {}
    monkeypatch.setattr(mod, "build_l0_dynamic_gbeta_dataset", fake_build)
    monkeypatch.setattr(mod, "train_l0_dynamic_gbeta_train", fake_train)

    npz = tmp_path / "pre.npz"; np.savez(npz, x=np.zeros(1))
    mod.pretrain_gbeta_cdl("FAKE.pt", out_dir=str(tmp_path),
                           dataset_path=str(npz), device="cpu")
    assert hit["build"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest block_lo_arm_order_network/tests/test_gbeta_cdl_pretrain.py -v`
Expected: FAIL — `ModuleNotFoundError: analyses.gbeta_cdl_pretrain`.

- [ ] **Step 3: Write minimal implementation**

```python
# analyses/gbeta_cdl_pretrain.py
"""Stage-1 CDL pretrain orchestrator: produce an L0DynamicGBeta gβ from a
backbone ckpt via the EXISTING two-step CDL pipeline (no CDL reimplemented).

  build_l0_dynamic_gbeta_dataset(ckpt) -> .npz  (CDL dynamic teacher)
  train_l0_dynamic_gbeta.train(.npz)   -> g_beta_best.pt  (L0DynamicGBeta)
"""
from __future__ import annotations
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyses.build_l0_dynamic_gbeta_dataset import build_l0_dynamic_gbeta_dataset
from batch_readout.train_l0_dynamic_gbeta import train as train_l0_dynamic_gbeta_train

DEFAULT_SOURCE_CKPT = str(
    ROOT / "block_lo_arm_order_network/probe_results/"
    "overnight_20260625_random_baseline/ckpt_step10000.pt"
)


def pretrain_gbeta_cdl(ckpt_10k, *, out_dir, loss_type="pairwise_bce",
                       M=2000, heads=8, epochs=40, seed=0, device="cpu",
                       dataset_path=None):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if dataset_path is None:
        dataset_path = build_l0_dynamic_gbeta_dataset(
            ckpt_path=ckpt_10k, M=M, seed=seed, device=device,
            out_path=str(out / "cdl_dataset.npz"))
    train_l0_dynamic_gbeta_train(
        dataset_path=dataset_path, out_dir=str(out), epochs=epochs,
        seed=seed, device=device, heads=heads, loss_type=loss_type)
    ckpt = out / "g_beta_best.pt"
    (out / "gbeta_provenance.json").write_text(json.dumps({
        "producer": "build_l0_dynamic_gbeta_dataset + train_l0_dynamic_gbeta",
        "source_ckpt": str(ckpt_10k), "loss_type": loss_type,
        "M": M, "heads": heads, "epochs": epochs, "seed": seed,
        "dataset_path": str(dataset_path),
    }, indent=2))
    return str(ckpt)


__all__ = ["pretrain_gbeta_cdl", "DEFAULT_SOURCE_CKPT"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest block_lo_arm_order_network/tests/test_gbeta_cdl_pretrain.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add analyses/gbeta_cdl_pretrain.py block_lo_arm_order_network/tests/test_gbeta_cdl_pretrain.py
git commit -m "feat(v3): Stage-1 CDL pretrain orchestrator (gbeta_cdl_pretrain)"
```

---

### Task 2: Stage-3 PG helpers (grad scorer + PL + batch advantage)

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/orderhead_pg.py`
- Test: `block_lo_arm_order_network/tests/test_orderhead_pg.py`

**Interfaces:**
- Consumes (existing, verified):
  - `batch_readout.frozen_gbeta_hook.extract_probe_averaged_model_frame_strict65(model, idx_batch, clean_perm, batch_mean_probes, seed, global_step, device) -> B_t (1, H=8, 65, 65)` (the SAME B the frozen provider feeds gβ; wrapped in `@torch.no_grad()` there — here we call it, then run gβ WITH grad on the detached B).
  - `analyses.p7_gbeta_policy.sample_pl(scores, tau) -> (order, logp, entropy)` (PL over a 1-D score vector; `order` is model-frame block indices).
  - `analyses.v3_group_credit.group_ids_for(batch_size, m)`, `per_sample_loss(token_losses)`, `group_rewards(ell_i, groups)`, `GroupEMA(n_groups, alpha)`.
- Produces:
  - `gbeta_scores_with_grad(gbeta_model, B_det) -> scores (64,)` — grad-enabled forward of L0DynamicGBeta on a single batch-mean B, returning the model-frame score vector.
  - `batch_advantage(token_losses, groups, ema, adv_clip) -> (A_batch: scalar tensor, ell_i: (Bs,))` — per-sample loss → per-group reward+EMA baseline → scalar mean advantage, detached & clamped.

- [ ] **Step 1: Write failing tests (pure math, tiny modules — CPU)**

```python
# block_lo_arm_order_network/tests/test_orderhead_pg.py
import pathlib, sys
import torch, pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta


def test_gbeta_scores_with_grad_shape_and_grad():
    from batch_readout.orderhead_pg import gbeta_scores_with_grad
    m = L0DynamicGBeta(heads=8, nodes=65)
    B = torch.randn(1, 8, 65, 65)
    B.requires_grad_(False)  # B is detached input
    scores = gbeta_scores_with_grad(m, B)
    assert scores.shape == (64,)
    assert scores.requires_grad          # grad flows to gβ params
    scores.sum().backward()
    assert any(p.grad is not None for p in m.parameters())


def test_batch_advantage_is_scalar_and_detached():
    from batch_readout.orderhead_pg import batch_advantage
    from analyses.v3_group_credit import group_ids_for, GroupEMA
    bs, m = 16, 8
    groups = group_ids_for(bs, m)
    ema = GroupEMA(len(groups), 0.9)
    token_losses = torch.rand(bs, 10)          # (Bs, T) per-token CE
    A, ell_i = batch_advantage(token_losses, groups, ema, adv_clip=0.3)
    assert A.dim() == 0                        # scalar advantage for ONE order
    assert not A.requires_grad                 # advantage detached
    assert ell_i.shape == (bs,)
    assert float(A.abs()) <= 0.3 + 1e-6        # clamped


def test_pl_logp_is_grad_connected_to_scores():
    from batch_readout.orderhead_pg import gbeta_scores_with_grad
    from analyses.p7_gbeta_policy import sample_pl
    m = L0DynamicGBeta(heads=8, nodes=65)
    scores = gbeta_scores_with_grad(m, torch.randn(1, 8, 65, 65))
    _order, logp, entropy = sample_pl(scores, tau=1.0)
    logp.backward()
    assert any(p.grad is not None for p in m.parameters())
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest block_lo_arm_order_network/tests/test_orderhead_pg.py -v`
Expected: FAIL — `ModuleNotFoundError: batch_readout.orderhead_pg`.

- [ ] **Step 3: Minimal implementation**

```python
# block_lo_arm_order_network/batch_readout/orderhead_pg.py
"""Stage-3 policy-gradient helpers for the internal OrderHead (L0DynamicGBeta).

Batch-level: one sampled order per step -> one logp. Groups only reduce baseline
variance (see spec). No CDL anywhere in this module.
"""
from __future__ import annotations
import torch
from analyses.v3_group_credit import per_sample_loss, group_rewards


def gbeta_scores_with_grad(gbeta_model, B_det):
    """Grad-enabled L0DynamicGBeta forward on ONE batch-mean B.

    B_det: (1, H=8, 65, 65) detached tensor built identically to the frozen
    provider's B. Returns model-frame score vector (64,) with grad on gβ params.
    """
    scores, _aux = gbeta_model(B_det, apply_head_dropout=False)  # (1, 64)
    return scores[0]


def batch_advantage(token_losses, groups, ema, adv_clip):
    """Per-sample CE -> per-group reward + EMA baseline -> scalar advantage.

    Returns (A_batch scalar detached+clamped, ell_i (Bs,)).
    """
    ell_i = per_sample_loss(token_losses.detach())      # (Bs,)
    ell_g = group_rewards(ell_i, groups)                # (G,)
    baseline = ema.update(ell_g)                        # (G,)
    adv_g = (baseline - ell_g)                          # (G,) lower loss -> +adv
    A = adv_g.mean().detach().clamp(-adv_clip, adv_clip)
    return A, ell_i


__all__ = ["gbeta_scores_with_grad", "batch_advantage"]
```

Note: `gbeta_scores_with_grad` calls gβ directly; the B construction (identical to the provider) is Task 3's responsibility at the call site.

- [ ] **Step 4: Run to verify pass**

Run: `pytest block_lo_arm_order_network/tests/test_orderhead_pg.py -v`
Expected: PASS (3 tests). If `sample_pl`'s signature differs, adapt the import to the actual `(order, logp, entropy)` return in `analyses/p7_gbeta_policy.py`.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/batch_readout/orderhead_pg.py block_lo_arm_order_network/tests/test_orderhead_pg.py
git commit -m "feat(v3): Stage-3 PG helpers (grad gβ scorer, batch-level advantage)"
```

---

### Task 3: CLI flags + B-source invariant + freeze→unfreeze schedule

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py` (parse_args near line 881–917; optimizer at 1264; main loop `frozen_beta` branch 1893–1907)
- Test: `block_lo_arm_order_network/tests/test_orderhead_unfreeze_schedule.py`

**Interfaces:**
- Consumes: `orderhead_pg.gbeta_scores_with_grad`; `extract_probe_averaged_model_frame_strict65`; `beta_provider.gbeta_module` (add a property exposing the underlying `L0DynamicGBeta` so it can be frozen/unfrozen and added to the optimizer).
- Produces: new args `--unfreeze-orderhead-at-step`, `--orderhead-lr`, `--lam-pg`, `--pg-tau`, `--pg-group-m`, `--pg-beta`, `--pg-adv-clip`, `--cdl-pretrain`, `--cdl-source-ckpt`; helper `maybe_unfreeze_orderhead(optimizer, gbeta_module, global_step, args, state) -> bool` (idempotent single insertion).

- [ ] **Step 1: Write failing tests (schedule invariants with a dummy optimizer)**

```python
# block_lo_arm_order_network/tests/test_orderhead_unfreeze_schedule.py
import pathlib, sys
import torch, pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta


def _fresh():
    gb = L0DynamicGBeta(heads=8, nodes=65)
    for p in gb.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    for g in opt.param_groups:
        g.setdefault("is_orderhead", False)
    return gb, opt


def test_no_orderhead_group_before_unfreeze():
    from train_clean_aogpt import maybe_unfreeze_orderhead
    gb, opt = _fresh()
    args = type("A", (), {"unfreeze_orderhead_at_step": 100, "orderhead_lr": 3e-4})
    state = {"unfrozen": False}
    did = maybe_unfreeze_orderhead(opt, gb, global_step=50, args=args, state=state)
    assert did is False
    assert all(not g.get("is_orderhead") for g in opt.param_groups)
    assert all(not p.requires_grad for p in gb.parameters())


def test_single_insertion_and_weight_continuity():
    from train_clean_aogpt import maybe_unfreeze_orderhead
    gb, opt = _fresh()
    before = {k: v.clone() for k, v in gb.state_dict().items()}
    args = type("A", (), {"unfreeze_orderhead_at_step": 100, "orderhead_lr": 3e-4})
    state = {"unfrozen": False}
    # fire at unfreeze, then again later -> must not double-insert
    did1 = maybe_unfreeze_orderhead(opt, gb, 100, args, state)
    did2 = maybe_unfreeze_orderhead(opt, gb, 101, args, state)
    assert did1 is True and did2 is False
    oh_groups = [g for g in opt.param_groups if g.get("is_orderhead")]
    assert len(oh_groups) == 1
    ids = [id(p) for g in oh_groups for p in g["params"]]
    assert len(ids) == len(set(ids))                 # no duplicate params
    assert all(p.requires_grad for p in gb.parameters())
    after = gb.state_dict()
    for k in before:                                  # weight continuity
        assert torch.equal(before[k], after[k])


def test_disabled_when_flag_none():
    from train_clean_aogpt import maybe_unfreeze_orderhead
    gb, opt = _fresh()
    args = type("A", (), {"unfreeze_orderhead_at_step": None, "orderhead_lr": 3e-4})
    state = {"unfrozen": False}
    assert maybe_unfreeze_orderhead(opt, gb, 10_000, args, state) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest block_lo_arm_order_network/tests/test_orderhead_unfreeze_schedule.py -v`
Expected: FAIL — `ImportError: cannot import name 'maybe_unfreeze_orderhead'`.

- [ ] **Step 3: Add the CLI flags**

In `parse_args` (after the existing `--frozen-beta-*` block, ~line 917):

```python
    p.add_argument("--unfreeze-orderhead-at-step", type=int, default=None,
                   help="frozen_beta: global step at which gβ is unfrozen and PG fine-tuning begins.")
    p.add_argument("--orderhead-lr", type=float, default=3e-4)
    p.add_argument("--lam-pg", type=float, default=1.0)
    p.add_argument("--pg-tau", type=float, default=1.0)
    p.add_argument("--pg-group-m", type=int, default=16)
    p.add_argument("--pg-beta", type=float, default=3e-3)
    p.add_argument("--pg-adv-clip", type=float, default=0.3)
    p.add_argument("--cdl-pretrain", action="store_true",
                   help="if set and --frozen-beta-ckpt absent, CDL-pretrain gβ before training.")
    p.add_argument("--cdl-source-ckpt", type=str, default=None,
                   help="backbone ckpt for --cdl-pretrain (default: 10k parent).")
```

- [ ] **Step 4: Implement `maybe_unfreeze_orderhead` (module-level, near `alpha_for_step`)**

```python
def maybe_unfreeze_orderhead(optimizer, gbeta_module, global_step, args, state):
    """Idempotently unfreeze gβ and insert ONE optimizer param group at the
    scheduled step. Returns True exactly on the firing step, else False."""
    at = getattr(args, "unfreeze_orderhead_at_step", None)
    if at is None or state.get("unfrozen") or global_step < int(at):
        return False
    for p in gbeta_module.parameters():
        p.requires_grad_(True)
    assert not any(g.get("is_orderhead") for g in optimizer.param_groups)
    optimizer.add_param_group({
        "params": [p for p in gbeta_module.parameters()],
        "lr": float(args.orderhead_lr), "weight_decay": 0.0,
        "is_orderhead": True,
    })
    state["unfrozen"] = True
    return True
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest block_lo_arm_order_network/tests/test_orderhead_unfreeze_schedule.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Wire CLI resolution for `--cdl-pretrain` (near the `frozen_beta` provider setup, ~line 1752)**

Before building `beta_provider`, resolve the ckpt:

```python
        if not args.frozen_beta_ckpt:
            if args.cdl_pretrain:
                from analyses.gbeta_cdl_pretrain import pretrain_gbeta_cdl, DEFAULT_SOURCE_CKPT
                src = args.cdl_source_ckpt or DEFAULT_SOURCE_CKPT
                args.frozen_beta_ckpt = pretrain_gbeta_cdl(
                    src, out_dir=str(pathlib.Path(args.output_dir) / "gbeta_cdl"),
                    device=str(device))
                log(f"[stage1] CDL-pretrained gβ -> {args.frozen_beta_ckpt}")
            else:
                raise ValueError("run-kind=frozen_beta requires --frozen-beta-ckpt "
                                 "or --cdl-pretrain")
```

This runs the producer strictly BEFORE the optimizer/training loop (satisfies the no-CDL-in-loop red line). `--frozen-beta-ckpt` always wins.

- [ ] **Step 7: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py block_lo_arm_order_network/tests/test_orderhead_unfreeze_schedule.py
git commit -m "feat(v3): unfreeze-orderhead schedule + CLI + CDL-pretrain resolution"
```

---

### Task 4: PG wiring in the training step + red lines + logging

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py` (the `frozen_beta` branch of the micro-step loop, 1893–1907; add `beta_provider.gbeta_module` property in `frozen_gbeta_hook.py`)
- Test: extend `block_lo_arm_order_network/tests/test_orderhead_pg.py`

**Interfaces:**
- Consumes: `orderhead_pg.gbeta_scores_with_grad`, `orderhead_pg.batch_advantage`, `analyses.p7_gbeta_policy.sample_pl`, `analyses.v3_group_credit.group_ids_for/GroupEMA`, `compute_token_ce` (line 390), `model_blocks_to_physical_blocks`, `physical_blocks_to_model_token_order`, `extract_probe_averaged_model_frame_strict65`.
- Produces: the Phase-B loss path and per-step log fields (`pg`, `entropy`, `pg_active`, `orderhead_trainable`, `cdl_loss=None`, `cdl_calls=0`).

- [ ] **Step 1: Write failing tests — B-source invariant + grad routing**

```python
# append to block_lo_arm_order_network/tests/test_orderhead_pg.py
def test_grad_scores_match_frozen_scores_before_unfreeze():
    """B_PG == B_frozen-provider: same B in => grad and no_grad scores allclose,
    identical argsort (spec B-source + order-frame invariant)."""
    from batch_readout.orderhead_pg import gbeta_scores_with_grad
    m = L0DynamicGBeta(heads=8, nodes=65).eval()
    B = torch.randn(1, 8, 65, 65)
    with torch.no_grad():
        s_frozen, _ = m(B, apply_head_dropout=False)
    s_grad = gbeta_scores_with_grad(m, B.detach())
    assert torch.allclose(s_frozen[0], s_grad, atol=1e-6)
    assert torch.equal(s_frozen[0].argsort(descending=True),
                       s_grad.argsort(descending=True))


def test_pg_loss_does_not_touch_backbone():
    """PG grad routes to gβ only; a detached-B ⇒ no path to backbone params.
    Emulate a 'backbone' param feeding B and assert it gets no grad."""
    from batch_readout.orderhead_pg import gbeta_scores_with_grad, batch_advantage
    from analyses.p7_gbeta_policy import sample_pl
    from analyses.v3_group_credit import group_ids_for, GroupEMA
    backbone_p = torch.nn.Parameter(torch.randn(1, 8, 65, 65))
    B_det = backbone_p.detach()                      # spec: B detached before gβ
    m = L0DynamicGBeta(heads=8, nodes=65)
    scores = gbeta_scores_with_grad(m, B_det)
    _order, logp, entropy = sample_pl(scores, tau=1.0)
    groups = group_ids_for(16, 8); ema = GroupEMA(len(groups), 0.9)
    A, _ = batch_advantage(torch.rand(16, 10), groups, ema, 0.3)
    L_pg = -(A * logp) - 3e-3 * entropy
    L_pg.backward()
    assert backbone_p.grad is None                   # backbone untouched
    assert any(p.grad is not None for p in m.parameters())
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest block_lo_arm_order_network/tests/test_orderhead_pg.py -v`
Expected: the two new tests fail only if helpers are wrong; if Tasks 2 helpers exist they should pass — if so, these tests are guardrails. If `gbeta_scores_with_grad` is missing they FAIL at import. (Write them RED first by running before Task 2 is merged, or treat as regression guards; either way they must pass at the end.)

- [ ] **Step 3: Add `gbeta_module` accessor on the provider**

In `batch_readout/frozen_gbeta_hook.py`, `FrozenGBetaModelFrameBlockProvider`:

```python
    @property
    def gbeta_module(self):
        """The underlying L0DynamicGBeta (for freeze/unfreeze + optimizer).
        FrozenGBetaModelFrameBlockProvider wraps FrozenGBetaModelFrameProvider
        as ``self._provider``, which holds the model as ``.model``."""
        return self._provider.model
```

(Verified: `FrozenGBetaModelFrameBlockProvider.__init__` sets `self._provider = FrozenGBetaModelFrameProvider(...)`, and that inner provider builds `self.model = L0DynamicGBeta(...)`.)

- [ ] **Step 4: Implement the Phase-A/Phase-B branch (replace the `frozen_beta` block at 1893–1907)**

```python
                elif args.run_kind == "frozen_beta":
                    pg_on = (args.unfreeze_orderhead_at_step is not None
                             and global_step >= args.unfreeze_orderhead_at_step)
                    if pg_on and alpha > 0.0:
                        # ---- Phase B: PL sample + LM-NLL PG (batch-level) ----
                        from batch_readout.orderhead_pg import (
                            gbeta_scores_with_grad, batch_advantage)
                        from batch_readout.frozen_gbeta_hook import (
                            extract_probe_averaged_model_frame_strict65)
                        from analyses.p7_gbeta_policy import sample_pl
                        # Signature: (aogpt_model, idx_batch, global_step, seed,
                        # batch_mean_probes, device) — NO clean_perm. Returns the
                        # per-sample strict65 graphs (Bsz, H=8, 65, 65).
                        B_all = extract_probe_averaged_model_frame_strict65(
                            model, idx_batch, global_step, args.seed,
                            args.batch_mean_probes, device)
                        # Batch-level single order (spec): consensus B = mean over
                        # batch → (1, H, 65, 65), detached.
                        # IMPLEMENTER: confirm this reduction matches how
                        # FrozenGBetaModelFrameProvider reduces (Bsz,64) scores to
                        # the single order it returns — read its model_frame_token_order
                        # and mirror it EXACTLY (mean-over-batch B vs per-sample then
                        # consensus). Verify in the GPU smoke (Step 6) that the first
                        # Phase-B pre-sample argsort == the last Phase-A frozen argsort
                        # on the same batch (the true B_PG == B_frozen check).
                        B_det = B_all.mean(dim=0, keepdim=True).detach()
                        scores = gbeta_scores_with_grad(pg_state["gbeta"], B_det)
                        sigma_model, logp, entropy = sample_pl(scores, tau=args.pg_tau)
                        sigma_phys = model_blocks_to_physical_blocks(
                            torch.as_tensor(sigma_model, device=device), clean_perm)
                        phys = sigma_phys.unsqueeze(0).expand(args.batch_size, -1)
                        token_orders = physical_blocks_to_model_token_order(
                            phys, clean_perm, BLOCK_LEN).to(device)
                        token_losses, _ = compute_token_ce(model, idx_batch, token_orders, device)
                        A_batch, _ell = batch_advantage(
                            token_losses, pg_state["groups"], pg_state["ema"], args.pg_adv_clip)
                        lm_loss = token_losses.mean()
                        L_pg = -(A_batch * logp) - args.pg_beta * entropy
                        loss = lm_loss + args.lam_pg * L_pg
                        pg_log = {"pg": float(L_pg.detach()), "entropy": float(entropy.detach()),
                                  "pg_active": True, "orderhead_trainable": True,
                                  "cdl_loss": None, "cdl_calls": 0}
                    elif alpha > 0.0:
                        # ---- Phase A: existing frozen argsort behavior ----
                        sigma = beta_provider.physical_order(model, idx_batch, global_step).to(device)
                        if args.frozen_beta_none_mode in ("model", "content", "strict65_model"):
                            sigma = model_blocks_to_physical_blocks(sigma, clean_perm)
                        phys = sigma.unsqueeze(0).expand(args.batch_size, -1)
                        choose_rng = torch.Generator(device=device)
                        choose_rng.manual_seed(args.seed * 100000000 + global_step * 1000 + micro_step)
                        use_beta = torch.rand(args.batch_size, generator=choose_rng, device=device) < alpha
                        mixed = torch.where(use_beta.unsqueeze(1), phys, random_phys)
                        loss = order_loss(model, idx_batch, mixed, clean_perm, device)
                    else:
                        loss = order_loss(model, idx_batch, random_phys, clean_perm, device)
```

Setup once before the loop (after `beta_provider` is built and the optimizer at 1264), guarded by `args.unfreeze_orderhead_at_step is not None`:

```python
    pg_state = None
    if args.run_kind == "frozen_beta" and args.unfreeze_orderhead_at_step is not None:
        from analyses.v3_group_credit import group_ids_for, GroupEMA
        _gb = beta_provider.gbeta_module
        for p in _gb.parameters():
            p.requires_grad_(False)                 # frozen through Phase A
        _groups = group_ids_for(args.batch_size, args.pg_group_m)
        pg_state = {"gbeta": _gb, "groups": _groups,
                    "ema": GroupEMA(len(_groups), 0.9)}
    unfreeze_state = {"unfrozen": False}
```

In the per-step body (top of the `for global_step` loop, before micro-steps), fire the unfreeze:

```python
        if pg_state is not None:
            maybe_unfreeze_orderhead(optimizer, pg_state["gbeta"], global_step, args, unfreeze_state)
```

- [ ] **Step 5: Run the unit + invariant tests**

Run: `pytest block_lo_arm_order_network/tests/test_orderhead_pg.py block_lo_arm_order_network/tests/test_orderhead_unfreeze_schedule.py -v`
Expected: PASS (all).

- [ ] **Step 6: CPU smoke of the full step wiring (tiny, no 566MB ckpt)**

Add a smoke that constructs a tiny AO-GPT-like stub exposing `forward_fn` and runs 2 Phase-A + 2 Phase-B steps, asserting: Phase A gβ param delta == 0; Phase B gβ param delta > 0; backbone (stub) receives LM grad; `cdl_calls` stays 0. Keep it under the existing tiny-model test fixtures if present; otherwise assert the loss-path pieces in isolation (Steps 1–5 already cover the math). Document that the true end-to-end run uses the 10k ckpt on GPU:

```bash
# manual smoke (GPU), not a unit test:
python -u block_lo_arm_order_network/train_clean_aogpt.py --run-kind frozen_beta \
  --cdl-pretrain --batch-mean-probes 4 --frozen-beta-none-mode model \
  --unfreeze-orderhead-at-step 10050 --max-steps 10100 --lam-pg 1.0 \
  --data-source continuous --train-bin <...> --val-bin <...> --seed 123 --device cuda:0
```

- [ ] **Step 7: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py \
        block_lo_arm_order_network/batch_readout/frozen_gbeta_hook.py \
        block_lo_arm_order_network/tests/test_orderhead_pg.py
git commit -m "feat(v3): Phase-B PG wiring in frozen_beta step (batch-level, no CDL)"
```

---

## Self-Review

**Spec coverage:**
- Stage 1 producer → Task 1. ✓
- Stage 2 frozen warmup (unchanged) → preserved in Task 4 Phase-A branch. ✓
- Stage 3 unfreeze + PL + group-credit PG → Tasks 2–4. ✓
- B_PG == B_frozen invariant → Task 4 Step 1 unit test covers grad/no_grad **gβ-forward** equivalence; the full B-**construction** match (loop's `B_det` vs the provider's B/order reduction) is an integration check verified in the Task 4 Step 6 GPU smoke (first Phase-B pre-sample argsort == last Phase-A frozen argsort). Called out for the implementer at the `B_det` reduction. ⚠ partial-unit + integration.
- Order-frame (model→physical→token, no P7 remap) → Task 4 Step 4 uses `model_blocks_to_physical_blocks` + `physical_blocks_to_model_token_order` (the frozen path). ✓
- Weight continuity + single param-group insertion → Task 3 tests. ✓
- λ_PG (Phase A not constructed) → Task 4 branch. ✓
- No-CDL red line (behavioral) → producer runs before loop (Task 3 Step 6); `cdl_calls=0` log (Task 4). ✓
- CLI resolution (never auto-trigger; ckpt overrides) → Task 3 Step 6. ✓
- Producer pairwise_acc from `teacher_pairwise` → **GAP**: Task 1 stubs training, so the learning-sanity/pairwise test is not exercised on real data. Mitigation: add a Task 1 optional `@pytest.mark.slow` real-data test (small M, CPU/GPU) computing `pairwise_acc` from the `.npz` `teacher_pairwise` per spec test 2; mark skipped by default. Add it as Task 1 Step 6.
- Backward compat (flag absent) → Task 3 `test_disabled_when_flag_none` + Task 4 branch leaves Phase-A path byte-for-byte. ✓

**Placeholder scan:** none (all steps have concrete code/commands). The one intentionally deferred item (real-data producer sanity) is called out as a marked-slow test, not a placeholder.

**Type consistency:** `pretrain_gbeta_cdl` returns `str`; `maybe_unfreeze_orderhead` returns `bool`; `gbeta_scores_with_grad` returns `(64,)` tensor; `batch_advantage` returns `(scalar, (Bs,))`. Call sites in Task 4 match. `sample_pl` return `(order, logp, entropy)` must be confirmed against `analyses/p7_gbeta_policy.py` at implementation time (Task 2 Step 4 note).

**Fix applied inline:** add Task 1 Step 6 (marked-slow real-data producer sanity) to close the pairwise_acc coverage gap.

- [ ] **Task 1 Step 6 (added): marked-slow real-data producer sanity**

```python
@pytest.mark.slow
def test_producer_learns_cdl_pairwise(tmp_path):
    """Real small-M run: gβ agrees with teacher_pairwise above chance.
    pairwise_acc = mean[ sign(s_i - s_j) == sign(Y_ij - 0.5) ] (spec test 2)."""
    import numpy as np, torch
    from analyses.gbeta_cdl_pretrain import pretrain_gbeta_cdl, DEFAULT_SOURCE_CKPT
    from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta
    ck = pretrain_gbeta_cdl(DEFAULT_SOURCE_CKPT, out_dir=str(tmp_path),
                            M=64, epochs=5, device="cpu")
    ds = np.load(tmp_path / "cdl_dataset.npz")
    Y = torch.from_numpy(ds["teacher_pairwise"]).float()     # (M,64,64)
    B = torch.from_numpy(ds["B_raw"]).float()                # (M,8,65,65)
    state = torch.load(ck, map_location="cpu")
    m = L0DynamicGBeta(heads=8, nodes=65); m.load_state_dict(state["model_state_dict"]); m.eval()
    with torch.no_grad():
        s, _ = m(B, apply_head_dropout=False)                # (M,64)
    sd = torch.sign(s.unsqueeze(2) - s.unsqueeze(1))
    yd = torch.sign(Y - 0.5)
    acc = (sd == yd).float().mean().item()
    assert acc > 0.55
```

Run (opt-in): `pytest block_lo_arm_order_network/tests/test_gbeta_cdl_pretrain.py -v -m slow`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-orderhead-cdl-pretrain-decoupled-init.md`.
