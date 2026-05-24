# Image MLP-refresh Alternating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-arm from-0 image self-bootstrapping trainer that, after a random-order warmup, every 3k steps re-extracts B from the current model, re-distills a C-D+L MLP order policy, and trains on its (parallel-sampled) orders — to test whether the order→attention→better-order loop rolls up on the 8×8 single-token VQ image substrate.

**Architecture:** New `scripts/train_vq64_alternating.py` (the existing `scripts/train_vq64_round2.py` is left untouched). It reuses round2's `AOGPT` construction, `_forward_with_block_orders`, `evaluate_7orders`, `get_lr`, and reuses the validated text-alternating modules (`attn_order_distill`, `attn_order_teacher`, `attn_order_features`, `attn_order_mlp_policy`) plus `extract_image_attention_e2.extract_a_global`. New code: from-0 (random-init, un-permuted) model setup, delayed-warmup alpha schedule, the refresh+distill block (mirroring `train_clean_aogpt.py:1008-1049`), an image refresh-diagnostics helper, and a `val_mlp_order` eval column.

**Tech Stack:** PyTorch, NumPy, pytest. Image AOGPT (nanogpt-learned-order). CPU for the tiny N=64 MLP distillation, GPU for model training + batched order sampling.

**Spec:** `docs/superpowers/specs/2026-05-24-image-mlp-refresh-alternating-design.md` — read §1 (positioning: feasibility, NOT cross-arm baseline) and §5/§6 (criteria + safety invariants) before reporting results.

---

## Reference implementations (read these; the plan mirrors them)

- **Text refresh+distill block to mirror:** `block_lo_arm_order_network/train_clean_aogpt.py:1008-1049` (refresh → `refresh_rw_graph` → `distill_order_mlp` → `refresh_diagnostics` → save `beta_step{N}.pt` → append `refresh_diagnostics.jsonl`). Also `:210-216` `should_sample_rw`, `:155-186` `alpha_for_step`/`--alpha-warmup-start`, `:935-1010` the train loop.
- **Image trainer to reuse:** `scripts/train_vq64_round2.py` — `load_baseline_model` (we will NOT use it; build from scratch instead), `_forward_with_block_orders` (`:153-181`), `evaluate_7orders` (`:346-388`), `get_lr` (`:137-144`), `main` loop (`:465-518`).
- **Attention extraction to reuse:** `block_lo_arm_order_network/extract_image_attention_e2.py` — `extract_a_global(model, data_tokens, tokens_per_image, n_images, m_passes, device)` returns (64,64) A with zero diagonal.
- **Distill / sampler / metrics:** `attn_order_distill.distill_order_mlp`, `attn_order_mlp_policy.sample_orders_batched_mlp` (orientation `"original"`), `train_attn_order_mlp.{locality_stats, top4_follow_and_edge}`.

## File structure

- Create: `block_lo_arm_order_network/attn_order_image_diag.py` — image per-refresh diagnostics (`image_refresh_diagnostics`).
- Create: `scripts/train_vq64_alternating.py` — the single-arm from-0 alternating trainer.
- Create: `block_lo_arm_order_network/test_vq64_alternating.py` — unit tests (run from `block_lo_arm_order_network/`).
- Test data/ckpt available: `block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/{val.bin,meta.pkl}`; an existing AOGPT ckpt `probe_results_image/e2_vq_round2_20k/cont_random/ckpt_step20000.pt` (used by extraction tests only).

**Invariants enforced by code (spec §6):** no `--a-block-path` / external A (the trainer does not accept one); teacher is C-D+L only (no Manhattan distance — never call the Bcov/distance samplers); orientation is `"original"` (no source_start, no reversed); during warmup (alpha==0 or beta is None) sample random orders only.

---

### Task 1: Image refresh-diagnostics helper

**Files:**
- Create: `block_lo_arm_order_network/attn_order_image_diag.py`
- Test: `block_lo_arm_order_network/test_vq64_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
# test_vq64_alternating.py
import sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from directed_graph_policy import build_directed_graph
from train_attn_order_mlp import OrderMLP
from attn_order_distill import distill_order_mlp
import attn_order_image_diag as D


def _local_A(N=64, grid=8, seed=0):
    """A 64x64 with locality: each node attends mostly to its 4-neighbours on an 8x8 grid."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 0.01, size=(N, N))
    for i in range(N):
        r, c = divmod(i, grid)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid and 0 <= nc < grid:
                A[i, nr * grid + nc] += 1.0
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)


def test_image_refresh_diagnostics_keys_and_locality():
    B = build_directed_graph(_local_A())
    mlp, _ = distill_order_mlp(B, mlp=None, n_orders=20, epochs=3, seed=0, device="cpu")
    rec = D.image_refresh_diagnostics(B, mlp, tau=0.5, top_k=4, seed=0, K=32, device="cpu")
    for k in ("p_le1", "p_le2", "top4_follow", "B_edge_ratio",
              "rollout_entropy", "rollout_unique", "teacher_p_le1", "teacher_top4_follow"):
        assert k in rec, f"missing key {k}"
    # on a locality graph the student rollout must beat the random floor on both metrics
    assert rec["p_le1"] > 0.10, rec["p_le1"]
    assert rec["B_edge_ratio"] > 1.3, rec["B_edge_ratio"]
    assert 0 < rec["rollout_unique"] <= 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_image_refresh_diagnostics_keys_and_locality -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'attn_order_image_diag'`.

- [ ] **Step 3: Write minimal implementation**

```python
# attn_order_image_diag.py
"""Per-refresh diagnostics for the IMAGE alternating arm (orientation='original', no
source_start, no Manhattan distance). Mirrors attn_order_distill.refresh_diagnostics
but reports image structural metrics (locality / B-edge-following) instead of L2R tau.
"""
from __future__ import annotations
import numpy as np
import torch

from attn_order_teacher import rollout_order
from train_attn_order_mlp import locality_stats, top4_follow_and_edge, diversity
import attn_order_mlp_policy as P


def image_refresh_diagnostics(B, mlp, *, tau=0.5, top_k=4, seed=0, K=128, grid=8, device="cpu"):
    """Snapshot teacher + distilled-beta order structure on the current B (attention-only).

    Returns student rollout metrics (p_le1/p_le2/top4_follow/B_edge_ratio/entropy/unique)
    plus the matched teacher metrics (teacher_p_le1/teacher_top4_follow), so a refresh row
    shows whether the loop is sharpening attention-derived structure.
    """
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))

    teach = np.stack([rollout_order(B, tau_T=tau, seed=int(seed) + s, mode="C-D+L", standardize=True)
                      for s in range(K)])
    t_loc = locality_stats(teach, grid=grid)
    t_t4, t_er = top4_follow_and_edge(teach, B)

    orders, ent = P.sample_orders_batched_mlp(
        B, K, mlp.to(device), "original", base_seed=int(seed), device=torch.device(device),
        tau=tau, top_k=top_k, return_entropy=True,
    )
    o = orders.cpu().numpy()
    s_loc = locality_stats(o, grid=grid)
    s_t4, s_er = top4_follow_and_edge(o, B)
    return dict(
        p_le1=round(s_loc["p_le1"], 4), p_le2=round(s_loc["p_le2"], 4),
        mean_manh=round(s_loc["mean_manh"], 4), top4_follow=round(s_t4, 4),
        B_edge_ratio=round(s_er, 4), rollout_entropy=round(float(ent), 4),
        rollout_unique=int(diversity(o)),
        teacher_p_le1=round(t_loc["p_le1"], 4), teacher_top4_follow=round(t_t4, 4),
        teacher_B_edge_ratio=round(t_er, 4),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_image_refresh_diagnostics_keys_and_locality -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/attn_order_image_diag.py block_lo_arm_order_network/test_vq64_alternating.py
git commit -m "feat(image-alt): C-D+L image refresh diagnostics (orientation=original)"
```

---

### Task 2: From-0 un-permuted attention extraction wrapper

Reuse `extract_a_global`; wrap it so the trainer can call it on a token array with a controlled seed, returning B directly. From-0 is **un-permuted** (physical = model coords) so no remap is involved.

**Files:**
- Modify: `scripts/train_vq64_alternating.py` (create the file with this helper first)
- Test: `block_lo_arm_order_network/test_vq64_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_vq64_alternating.py
import importlib.util

def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_vq64_alternating", str(_REPO / "scripts" / "train_vq64_alternating.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CKPT = _REPO / "probe_results_image/e2_vq_round2_20k/cont_random/ckpt_step20000.pt"
VAL  = _REPO / "block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/val.bin"

@pytest.mark.skipif(not (CKPT.exists() and VAL.exists()), reason="needs image ckpt+data")
def test_extract_B_from_model_shape_and_determinism():
    import pytest  # noqa
    T = _load_trainer()
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, model_args = T.build_or_load_model_for_extraction(str(CKPT), dev)
    data = np.memmap(str(VAL), dtype=np.uint16, mode="r")
    B1 = T.extract_B_from_model(model, data, tokens_per_image=64, n_images=20, m_passes=2,
                                device=dev, seed=123)
    B2 = T.extract_B_from_model(model, data, tokens_per_image=64, n_images=20, m_passes=2,
                                device=dev, seed=123)
    assert B1.shape == (64, 64)
    assert np.allclose(np.diag(B1), 0.0)
    assert np.isfinite(B1).all()
    assert np.allclose(B1, B2), "same seed must give identical B"
```

(Add `import pytest` at the top of the test file if not already present.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_extract_B_from_model_shape_and_determinism -q`
Expected: FAIL (`train_vq64_alternating.py` has no `extract_B_from_model` / file missing).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/train_vq64_alternating.py` with the imports + these two helpers (full trainer added in later tasks):

```python
#!/usr/bin/env python3
"""Single-arm from-0 image MLP-refresh alternating trainer (self-bootstrapping feasibility).
See docs/superpowers/specs/2026-05-24-image-mlp-refresh-alternating-design.md."""
from __future__ import annotations
import argparse, json, math, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))

from AOGPT import AOGPTConfig, AOGPT
from directed_graph_policy import build_directed_graph
from extract_image_attention_e2 import extract_a_global
from attn_order_distill import distill_order_mlp
from attn_order_image_diag import image_refresh_diagnostics
import attn_order_mlp_policy as P
# reuse round2 helpers without modifying it:
from train_vq64_round2 import _forward_with_block_orders, evaluate_7orders, get_lr

N_BLOCKS = 64
DEFAULT_MODEL_ARGS = dict(block_size=64, vocab_size=8192, n_layer=4, n_head=8,
                          n_embd=256, dropout=0.0, bias=False,
                          block_order_block_len=1, order_impl="block")


def build_or_load_model_for_extraction(ckpt_path, device):
    """Test/utility: load an existing AOGPT ckpt (used only by extraction tests)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    keys = list(DEFAULT_MODEL_ARGS.keys())
    model_args = {k: ma[k] for k in keys if k in ma}
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, model_args


def extract_B_from_model(model, data_tokens, tokens_per_image, n_images, m_passes, device, seed):
    """Extract A_global from the CURRENT model (un-permuted physical coords) and return B=A^T.

    Reuses extract_image_attention_e2.extract_a_global verbatim; seeds torch so the random
    AO orders (hence the snapshot) are reproducible.
    """
    was_training = model.training
    model.eval()
    torch.manual_seed(int(seed))
    A = extract_a_global(model, data_tokens, tokens_per_image, n_images, m_passes, device)
    if was_training:
        model.train()
    B = build_directed_graph(np.ascontiguousarray(A.astype(np.float64)))
    return B
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_extract_B_from_model_shape_and_determinism -q`
Expected: PASS (or SKIP if no GPU/data on the box — then run once on the GPU host).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_vq64_alternating.py block_lo_arm_order_network/test_vq64_alternating.py
git commit -m "feat(image-alt): from-0 un-permuted B extraction wrapper (reuses extract_a_global)"
```

---

### Task 3: Delayed-warmup alpha schedule

Mirror `train_clean_aogpt.py` `--alpha-warmup-start`: alpha is 0 until `warmup_start`, then ramps linearly to `alpha_max` over `ramp` steps, then holds.

**Files:**
- Modify: `scripts/train_vq64_alternating.py`
- Test: `block_lo_arm_order_network/test_vq64_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_vq64_alternating.py
def test_alpha_schedule_delayed_warmup():
    T = _load_trainer()
    # warmup_start=3000, ramp=10000, alpha_max=0.9  -> 0 until 3k, 0.9 at 13k, flat after
    f = lambda s: T.get_alpha_alt(s, warmup_start=3000, ramp=10000, alpha_max=0.9)
    assert f(0) == 0.0
    assert f(2999) == 0.0
    assert f(3000) == 0.0
    assert abs(f(8000) - 0.45) < 1e-6   # halfway through ramp
    assert abs(f(13000) - 0.9) < 1e-6
    assert abs(f(30000) - 0.9) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_alpha_schedule_delayed_warmup -q`
Expected: FAIL (`get_alpha_alt` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/train_vq64_alternating.py`:

```python
def get_alpha_alt(step, *, warmup_start, ramp, alpha_max):
    """0 until warmup_start; linear ramp to alpha_max over `ramp` steps; then flat."""
    if step < warmup_start:
        return 0.0
    if ramp <= 0:
        return alpha_max
    frac = (step - warmup_start) / float(ramp)
    return float(min(alpha_max, alpha_max * max(0.0, frac)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_alpha_schedule_delayed_warmup -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_vq64_alternating.py block_lo_arm_order_network/test_vq64_alternating.py
git commit -m "feat(image-alt): delayed-warmup alpha schedule"
```

---

### Task 4: Eval with `val_mlp_order` column

Wrap `evaluate_7orders` (8 fixed orders) and add a 9th: NLL under the current π_β order. Before the first refresh (`mlp is None`), `val_mlp_order` falls back to the random-order NLL (guard) and is flagged.

**Files:**
- Modify: `scripts/train_vq64_alternating.py`
- Test: `block_lo_arm_order_network/test_vq64_alternating.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_vq64_alternating.py
def test_evaluate_with_mlp_order_columns_and_guard():
    T = _load_trainer()
    import torch
    dev = "cpu"
    model = T.AOGPT(T.AOGPTConfig(**T.DEFAULT_MODEL_ARGS)).to(dev).eval()
    val_tokens = torch.randint(0, T.DEFAULT_MODEL_ARGS["vocab_size"], (8, 64), dtype=torch.long)
    B = build_directed_graph(_local_A())

    # before first refresh: mlp=None -> val_mlp_order present, equals val_random (guard)
    cols0 = T.evaluate_with_mlp(model, val_tokens, B, mlp=None, device=dev,
                                batch_size=4, max_eval_batches=2, step=0, tau=0.5, top_k=4)
    assert "val_mlp_order" in cols0
    assert abs(cols0["val_mlp_order"] - cols0["val_random"]) < 1e-6

    # after a (toy) distill: mlp present -> val_mlp_order is a finite NLL
    mlp, _ = distill_order_mlp(B, mlp=None, n_orders=20, epochs=3, seed=0, device=dev)
    cols1 = T.evaluate_with_mlp(model, val_tokens, B, mlp=mlp, device=dev,
                                batch_size=4, max_eval_batches=2, step=0, tau=0.5, top_k=4)
    assert np.isfinite(cols1["val_mlp_order"])
    assert set(["val_random", "val_raster", "val_mlp_order"]).issubset(cols1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_evaluate_with_mlp_order_columns_and_guard -q`
Expected: FAIL (`evaluate_with_mlp` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/train_vq64_alternating.py` (from-0 ⇒ `fixed_token_perm=None, inv_block_perm=None`):

```python
@torch.no_grad()
def evaluate_with_mlp(model, val_tokens, B, *, mlp, device, batch_size, max_eval_batches,
                      step, tau=0.5, top_k=4):
    """8 fixed-order NLLs (reused) + val_mlp_order. mlp=None -> val_mlp_order == val_random."""
    raster_order = torch.arange(N_BLOCKS, device=device)
    cols = {f"val_{k}": v for k, v in evaluate_7orders(
        model, val_tokens, B, raster_order, device, batch_size, max_eval_batches, step,
        fixed_token_perm=None, inv_block_perm=None).items()}

    if mlp is None:
        cols["val_mlp_order"] = cols["val_random"]
        return cols

    model.eval()
    V = val_tokens.shape[0]
    n_batches = min(max_eval_batches, math.ceil(V / batch_size))
    losses = []
    for bi in range(n_batches):
        s, e = bi * batch_size, min((bi + 1) * batch_size, V)
        x = val_tokens[s:e].to(device)
        mlp_orders = P.sample_orders_batched_mlp(
            B, x.shape[0], mlp, "original", base_seed=7_000_000 + step + bi,
            device=torch.device(device), tau=tau, top_k=top_k)
        loss = _forward_with_block_orders(model, x, mlp_orders,
                                          fixed_token_perm=None, inv_block_perm=None)
        losses.append(float(loss.item()))
    cols["val_mlp_order"] = float(np.mean(losses))
    return cols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py::test_evaluate_with_mlp_order_columns_and_guard -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_vq64_alternating.py block_lo_arm_order_network/test_vq64_alternating.py
git commit -m "feat(image-alt): val_mlp_order eval column with pre-refresh guard"
```

---

### Task 5: Main loop — argparse, from-0 setup, warmup, refresh+distill, training, logging

Wire the pieces into `main()`. Mirror `train_clean_aogpt.py:935-1052`: per-step order sampling (warmup→random; post-refresh→alpha-mixed MLP/random), refresh block at each `next_refresh_step`, beta save + `refresh_diagnostics.jsonl`, eval+TSV (with `val_mlp_order`), ckpt at save-steps. Reuse round2's `_forward_with_block_orders`, `get_lr`, data memmap loading (`train_vq64_round2.main:433-446`).

**Files:**
- Modify: `scripts/train_vq64_alternating.py`
- Test: covered by the 400-step smoke (Task 6); `main()` itself is integration glue.

- [ ] **Step 1: Add `parse_args()`**

```python
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train", required=True)
    p.add_argument("--data-val", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--warmup-start", type=int, default=3000, help="alpha=0 until this step (random warmup)")
    p.add_argument("--alpha-ramp", type=int, default=10000)
    p.add_argument("--alpha-max", type=float, default=0.9)
    p.add_argument("--refresh-interval", type=int, default=3000)
    p.add_argument("--first-refresh", type=int, default=3000, help="step of the first refresh (== warmup end)")
    p.add_argument("--extract-n-images", type=int, default=500)
    p.add_argument("--extract-m-passes", type=int, default=3)
    p.add_argument("--mlp-tau", type=float, default=0.5)
    p.add_argument("--mlp-top-k", type=int, default=4)
    p.add_argument("--mlp-refresh-mode", choices=["finetune", "scratch"], default="finetune")
    p.add_argument("--distill-n-orders", type=int, default=200)
    p.add_argument("--distill-epochs", type=int, default=60)
    p.add_argument("--distill-tau-t", type=float, default=0.5)
    p.add_argument("--distill-tau-train", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--max-eval-batches", type=int, default=16)
    p.add_argument("--save-steps", type=str, default="3000,15000,30000")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()
```

- [ ] **Step 2: Add `main()` — setup**

```python
def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    with open(args.meta, "rb") as f: meta = pickle.load(f)
    tokens_per_image = int(meta["tokens_per_image"]); assert tokens_per_image == N_BLOCKS
    train_mm = np.memmap(args.data_train, dtype=np.uint16, mode="r")
    val_mm = np.memmap(args.data_val, dtype=np.uint16, mode="r")
    n_train = len(train_mm) // N_BLOCKS
    n_val = min(2000, len(val_mm) // N_BLOCKS)
    val_tokens = torch.from_numpy(
        np.asarray(val_mm[:n_val * N_BLOCKS], dtype=np.int64).reshape(n_val, N_BLOCKS))

    model = AOGPT(AOGPTConfig(**DEFAULT_MODEL_ARGS)).to(device)   # FROM-0 random init
    model_args = dict(DEFAULT_MODEL_ARGS)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr,
                                           (args.beta1, args.beta2), args.device.split(":")[0])
    model.train()

    json.dump(vars(args), open(out / "config.json", "w"), indent=2)
    tsv = out / "eval_curve.tsv"
    header = ("step\ttrain_loss\talpha\tlr\tval_random\tval_raster\tval_hilbert\t"
              "val_Bcov_balanced\tval_distance_only_coverage\tval_rw_top4_eps0\t"
              "val_rw_eps015\tval_rw_topk8\tval_mlp_order\n")
    if not tsv.exists(): tsv.write_text(header)
    log_f = open(out / "train_log.txt", "a")
    def log(m): print(m, flush=True); log_f.write(m + "\n"); log_f.flush()
    save_steps = {int(x) for x in args.save_steps.split(",") if x}

    rw_mlp = None          # born at first refresh
    B = np.zeros((N_BLOCKS, N_BLOCKS), dtype=np.float64)   # placeholder until first refresh
    next_refresh = args.first_refresh
    log(f"[start] FROM-0 alternating max_steps={args.max_steps} warmup_start={args.warmup_start} "
        f"refresh_interval={args.refresh_interval} eff_batch={args.batch_size*args.grad_accum}")
```

- [ ] **Step 3: Add `main()` — train loop (warmup→random, post-refresh→alpha-mixed MLP), refresh block, eval, save**

```python
    def get_batch():
        idxs = rng.integers(0, n_train, size=args.batch_size)
        toks = np.stack([np.asarray(train_mm[i*N_BLOCKS:(i+1)*N_BLOCKS], dtype=np.int64) for i in idxs])
        return torch.from_numpy(toks).to(device, non_blocking=True)

    def run_eval(step, train_loss, lr, alpha):
        cols = evaluate_with_mlp(model, val_tokens, B, mlp=rw_mlp, device=device,
                                 batch_size=args.batch_size, max_eval_batches=args.max_eval_batches,
                                 step=step, tau=args.mlp_tau, top_k=args.mlp_top_k)
        with open(tsv, "a") as f:
            f.write(f"{step}\t{train_loss:.4f}\t{alpha:.4f}\t{lr:.2e}\t"
                    f"{cols['val_random']:.4f}\t{cols['val_raster']:.4f}\t{cols['val_hilbert']:.4f}\t"
                    f"{cols['val_Bcov_balanced']:.4f}\t{cols['val_distance_only_coverage']:.4f}\t"
                    f"{cols['val_rw_top4_eps0']:.4f}\t{cols['val_rw_eps015']:.4f}\t"
                    f"{cols['val_rw_topk8']:.4f}\t{cols['val_mlp_order']:.4f}\n")
        log(f"[eval] step={step} train={train_loss:.4f} a={alpha:.3f} "
            f"rnd={cols['val_random']:.4f} ras={cols['val_raster']:.4f} mlp={cols['val_mlp_order']:.4f}")

    t0 = time.time(); running = []
    for step in range(args.max_steps + 1):
        lr_now = get_lr(step, args)
        for g in optimizer.param_groups: g["lr"] = lr_now
        alpha = get_alpha_alt(step, warmup_start=args.warmup_start, ramp=args.alpha_ramp,
                              alpha_max=args.alpha_max)

        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum):
                x = get_batch()
                random_orders = torch.stack([torch.randperm(N_BLOCKS, device=device)
                                             for _ in range(args.batch_size)])
                use_mlp = (rw_mlp is not None) and (alpha > 0.0)
                if not use_mlp:
                    block_orders = random_orders
                else:
                    mlp_orders = P.sample_orders_batched_mlp(
                        B, args.batch_size, rw_mlp, "original",
                        base_seed=args.seed * 100000000 + step * 1000 + micro,
                        device=device, tau=args.mlp_tau, top_k=args.mlp_top_k)
                    crng = torch.Generator(device=device)
                    crng.manual_seed(args.seed * 7 + step * 1000 + micro)
                    pick = torch.rand(args.batch_size, generator=crng, device=device) < alpha
                    block_orders = torch.where(pick.unsqueeze(1), mlp_orders, random_orders)
                loss = _forward_with_block_orders(model, x, block_orders,
                                                  fixed_token_perm=None, inv_block_perm=None)
                (loss / args.grad_accum).backward()
                running.append(float(loss.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.eval_interval == 0:
            run_eval(step, float(np.mean(running[-100:])) if running else float("nan"), lr_now, alpha)

        # ---- refresh + distill (mirror train_clean_aogpt.py:1008-1049) ----
        if step > 0 and step >= next_refresh and step <= args.max_steps:
            B = extract_B_from_model(model, train_mm, tokens_per_image, args.extract_n_images,
                                     args.extract_m_passes, args.device, seed=args.seed + step)
            np.save(out / f"A_global_step{step}.npy", B.T)   # A = B^T
            init = rw_mlp if (args.mlp_refresh_mode == "finetune" and rw_mlp is not None) else None
            seed_r = args.seed * 100000 + step
            rw_mlp, ddiag = distill_order_mlp(
                B, mlp=init, n_orders=args.distill_n_orders, tau_T=args.distill_tau_t,
                tau_train=args.distill_tau_train, epochs=args.distill_epochs,
                seed=seed_r, device=str(device))
            rdiag = image_refresh_diagnostics(B, rw_mlp, tau=args.mlp_tau, top_k=args.mlp_top_k,
                                              seed=seed_r + 1, device=str(device))
            torch.save(rw_mlp.state_dict(), out / f"beta_step{step}.pt")
            with (out / "refresh_diagnostics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=int(step), refresh_mode=args.mlp_refresh_mode,
                                        **ddiag, **rdiag)) + "\n")
            log(f"[Refresh+Distill @ {step}] val_kl={ddiag['val_kl']} top1={ddiag['top1']} "
                f"top4={ddiag['top4']} | p_le1={rdiag['p_le1']} top4_follow={rdiag['top4_follow']} "
                f"B_edge={rdiag['B_edge_ratio']} ent={rdiag['rollout_entropy']}")
            next_refresh = step + args.refresh_interval

        if step > 0 and step in save_steps:
            torch.save({"model": model.state_dict(), "model_args": model_args, "step": step,
                        "config": vars(args)}, out / f"ckpt_step{step}.pt")
            log(f"[save] ckpt_step{step}.pt")

    log(f"[done] step {args.max_steps}; eval_curve={tsv}"); log_f.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-import the module (no run yet)**

Run: `cd block_lo_arm_order_network && python -c "import importlib.util,pathlib; s=importlib.util.spec_from_file_location('t', str(pathlib.Path('../scripts/train_vq64_alternating.py'))); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('ok', m.DEFAULT_MODEL_ARGS)"`
Expected: prints `ok {...}` with no import errors.

- [ ] **Step 5: Run the full unit suite**

Run: `cd block_lo_arm_order_network && python -m pytest test_vq64_alternating.py -q`
Expected: all PASS (extraction test may SKIP off-GPU).

- [ ] **Step 6: Commit**

```bash
git add scripts/train_vq64_alternating.py block_lo_arm_order_network/test_vq64_alternating.py
git commit -m "feat(image-alt): from-0 alternating main loop (warmup + refresh/distill + val_mlp_order)"
```

---

### Task 6: 400-step GPU smoke (gate before the 30k launch)

Tiny end-to-end run on a free GPU. Verifies: warmup→refresh transition, β born at first refresh, no NaN, `val_mlp_order` written, `refresh_diagnostics.jsonl` populated.

**Files:**
- Run only (writes to a throwaway `--output-dir`).

- [ ] **Step 1: Pick a free GPU**

Run: `nvidia-smi --query-gpu=index,memory.free --format=csv`
Expected: choose an index with ≥ ~4000 MiB free (smoke is tiny). Avoid the GPU chenhe is using.

- [ ] **Step 2: Run the smoke (warmup100 / refresh100 / 400 steps)**

```bash
cd /home/admin/lyuyuhuan/order_lyu && python scripts/train_vq64_alternating.py \
  --data-train block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/train.bin \
  --data-val   block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/val.bin \
  --meta       block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/meta.pkl \
  --output-dir probe_results_image/vq64_alt_from0_mlp_SMOKE \
  --max-steps 400 --warmup-start 100 --alpha-ramp 150 --first-refresh 100 --refresh-interval 100 \
  --extract-n-images 60 --extract-m-passes 2 --distill-epochs 10 --distill-n-orders 40 \
  --eval-interval 100 --save-steps 400 --device cuda:0 --seed 42
```

Expected: completes in a few minutes; log shows `[Refresh+Distill @ 100]` (and 200/300/400) with finite `val_kl`/`top1`.

- [ ] **Step 3: Verify smoke outputs**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu && \
  echo "--- eval_curve (expect 13 cols incl val_mlp_order) ---" && \
  head -1 probe_results_image/vq64_alt_from0_mlp_SMOKE/eval_curve.tsv && \
  tail -2 probe_results_image/vq64_alt_from0_mlp_SMOKE/eval_curve.tsv && \
  echo "--- refresh diagnostics ---" && \
  cat probe_results_image/vq64_alt_from0_mlp_SMOKE/refresh_diagnostics.jsonl && \
  echo "--- beta ckpts born ---" && ls probe_results_image/vq64_alt_from0_mlp_SMOKE/beta_step*.pt
```
Expected: header has `val_mlp_order`; no `nan` in the last rows; `refresh_diagnostics.jsonl` has ≥1 line with `val_kl`,`p_le1`,`top4_follow`,`B_edge_ratio`; `beta_step100.pt` exists.

- [ ] **Step 4: Clean up the smoke dir**

```bash
rm -rf /home/admin/lyuyuhuan/order_lyu/probe_results_image/vq64_alt_from0_mlp_SMOKE
```

- [ ] **Step 5: Commit (smoke is run-only; nothing to commit unless fixes were made)**

If Tasks 1-5 needed fixes during smoke, commit them with `fix(image-alt): ...`.

---

## After smoke passes (NOT part of this plan's auto-execution)

Launch the real single-arm 30k (user-gated; ~1hr on a free GPU):

```bash
cd /home/admin/lyuyuhuan/order_lyu && python scripts/train_vq64_alternating.py \
  --data-train block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/train.bin \
  --data-val   block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/val.bin \
  --meta       block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/meta.pkl \
  --output-dir probe_results_image/vq64_alt_from0_mlp \
  --device cuda:<free> --seed 42      # all other args default to the spec schedule
```

Report per spec §5: (1) `refresh_diagnostics.jsonl` KL↓/top1↑/B_edge↑/p_le1 trend across refreshes; (2) `val_mlp_order` trend; (3) `val_mlp_order` vs same-run `val_random`; (4) over-specialization (cross-order columns). Per §1, `cont_random`/`cont_raster` are historical reference only — no cross-arm win claim.

## Self-review notes (done)

- **Spec coverage:** single arm (T5) ✓; from-0 un-permuted (T2/T5 `fixed_token_perm=None`) ✓; warmup3k/refresh3k/30k (T5 defaults) ✓; C-D+L MLP distill + finetune (T5) ✓; GPU-batched parallel sampling (`sample_orders_batched_mlp`) ✓; plain orientation (T1/T4/T5 use `"original"`) ✓; `val_mlp_order` + 8 fixed cols (T4) ✓; per-refresh diagnostics incl. p_le1/top4/B_edge (T1) ✓; safety invariants (no external A, no manh, isolated file) ✓; tests + 400-step smoke gate (T6) ✓.
- **Placeholder scan:** none — all steps have concrete code/commands.
- **Type consistency:** `get_alpha_alt`, `extract_B_from_model`, `evaluate_with_mlp`, `image_refresh_diagnostics`, `DEFAULT_MODEL_ARGS`, `sample_orders_batched_mlp(..., "original")` used consistently across tasks.
