# P5 Attention-Scaffolded Utility Controller — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On ONE frozen AO-GPT checkpoint, test whether a hidden-state residual on top of an attention-derived order scaffold lowers downstream teacher-forced AO NLL (does H carry sample-specific ordering utility beyond attention topology?).

**Architecture:** All work is on a frozen model (no AO-GPT training). A per-sample attention scaffold B65 (reused from P2 `carrier_b65_per_text`) gives the B-only order σ_B and per-block B features; block-level hidden states H are pooled under σ_B's reveal context. Candidate reveal orders are scored by real downstream NLL (`forward_fn`); a soft pairwise utility teacher trains a B-only controller, which is frozen, then an H residual `z = z_B + α·g_H(H)` is trained on top. Success = held-out ΔNLL(B+H − B-only) < 0, lost under text-shuffled-H, not matched by zero/mean-H.

**Tech Stack:** Python, numpy, torch, pytest; frozen ckpt `runs/handoff_overnight/seed123/ckpt_step10000.pt`; CPU-feasible.

## Global Constraints

- **Setting unchanged:** fixed training layout; controller input is `[B, H]`.
- **Utility target only:** supervision = downstream teacher-forced AO NLL via `forward_fn`. NEVER physical-rank or CDL(B) targets (they structurally ignore H).
- **Reporting rule:** all ΔNLL/regret/shuffle-drop metrics are computed from the controller's HARD order `σ_pred = argsort(-z)` re-run through `forward_fn` — never from the training pairwise loss.
- **Headroom gate is necessary, not sufficient:** a positive gate + null B+H = "room exists, H not learned", not failure.
- **Fair baseline:** B-only controller trained on the SAME soft-utility teacher, then frozen; H residual trained on top with small learnable α.
- **Split on TEXTS** (train/val/test disjoint), report held-out-text metrics only.
- **Decisive control:** text-level shuffled-H (B from text i, H from text j). Plus zero-H / mean-H capacity controls.
- **τ is side analysis only**, never a success criterion.
- **Carrier (seed123):** `P2_CARRIERS[123] = (layer=0, heads=[1,2,3,4])`; Phase 0 uses the dominant head **L0H1** for the per-sample B65 scaffold.
- **Constants:** `SEQ_LEN=256`, `N=64` blocks, `BLOCK_LEN=4`.
- **Outputs:** `runs/p5/seed123/phase0.json` + `runs/p5/seed123/phase0_metrics.png`.

---

### Task 1: Order-NLL utility (downstream teacher-forced loss per reveal order)

**Files:**
- Create: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_order_nll.py`

**Interfaces:**
- Consumes: `clean_training_protocol.physical_blocks_to_model_token_order(phys_orders, clean_perm, block_len)`; `model.forward_fn(idx, token_orders) -> (logits, loss)`; `neural_readout.extract_b._load_model_and_chunks(ckpt, M, seed, device, split)`.
- Produces: `order_nll(model, idx_row, sigma_phys, clean_perm, device) -> float` (idx_row: (1,256) LongTensor; sigma_phys: (64,) int array, a physical-block order); `utility_pool(model, idx_row, sigmas, clean_perm, device) -> list[float]`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_order_nll.py
import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import order_nll, utility_pool, load_p5_ckpt

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")

def test_order_nll_matches_forward_fn_and_pool_orders():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=2, device="cpu")
    idx = chunks[0:1]
    l2r = np.arange(64)
    rev = np.arange(64)[::-1].copy()
    nll_l2r = order_nll(model, idx, l2r, clean_perm, dev)
    pool = utility_pool(model, idx, [l2r, rev], clean_perm, dev)
    assert isinstance(nll_l2r, float) and nll_l2r > 0
    assert abs(pool[0] - nll_l2r) < 1e-5      # pool reuses order_nll
    assert pool[0] != pool[1]                 # different orders -> different NLL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_order_nll.py -v`
Expected: FAIL (`ModuleNotFoundError: analyses.p5_utility_controller`).

- [ ] **Step 3: Write minimal implementation**

```python
# analyses/p5_utility_controller.py
"""P5: attention-scaffolded utility controller (Phase 0).

H improves downstream reveal-order utility beyond the attention scaffold; it does
NOT recover physical order. Supervision = downstream teacher-forced AO NLL (never
physical-rank/CDL(B), which structurally ignore H under fixed layout).
See docs/superpowers/specs/2026-06-29-p5-attention-scaffolded-utility-controller-design.md.
"""
import pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCK = ROOT / "block_lo_arm_order_network"
for _p in (str(ROOT), str(_BLOCK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from clean_training_protocol import physical_blocks_to_model_token_order  # noqa: E402

SEQ_LEN, N, BLOCK_LEN = 256, 64, 4


def load_p5_ckpt(ckpt_path, M, device="cpu"):
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    return model, chunks, clean_perm, dev


@torch.no_grad()
def order_nll(model, idx_row, sigma_phys, clean_perm, device):
    """Downstream teacher-forced AO NLL of revealing blocks in physical-block order
    sigma_phys (64,). Lower = better utility."""
    sigma = np.asarray(sigma_phys, dtype=np.int64)[None, :]            # (1,64)
    token_order = physical_blocks_to_model_token_order(
        torch.from_numpy(sigma), clean_perm, BLOCK_LEN).to(device)
    _, loss = model.forward_fn(idx_row.to(device), token_order)
    return float(loss)


def utility_pool(model, idx_row, sigmas, clean_perm, device):
    return [order_nll(model, idx_row, s, clean_perm, device) for s in sigmas]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_order_nll.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_order_nll.py
git commit -m "feat(p5): order-NLL utility (downstream teacher-forced loss per reveal order)"
```

---

### Task 2: Per-sample scaffold (B65 + σ_B + per-block B features)

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_scaffold.py`

**Interfaces:**
- Consumes: `physical_signal_source.carrier_b65_per_text(ckpt, layer, head, M, n_reveals, fixed_reveal_seed, device)`; `none_separated_block_graph.{rollout_by_method, discovery_metrics}`.
- Produces: `sample_scaffold(ckpt_path, M, layer=0, head=1, n_reveals=8, device="cpu") -> {"B": [B65...], "sigma_B": [(64,)...], "chunks": LongTensor, "clean_perm": ...}` (one entry per text); `block_b_features(B65) -> np.ndarray (64, 130)` (row+col of each non-None block).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_scaffold.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import sample_scaffold, block_b_features

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")

def test_scaffold_shapes_and_sigma_is_permutation():
    sc = sample_scaffold(CKPT, M=3, n_reveals=4)
    assert len(sc["B"]) == 3 and len(sc["sigma_B"]) == 3
    assert sc["B"][0].shape == (65, 65)
    sig = sc["sigma_B"][0]
    assert sorted(sig.tolist()) == list(range(64))      # a permutation of 64 blocks
    feat = block_b_features(sc["B"][0])
    assert feat.shape == (64, 130)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_scaffold.py -v`
Expected: FAIL (`ImportError: sample_scaffold`).

- [ ] **Step 3: Write minimal implementation**

Append to `analyses/p5_utility_controller.py` (add imports at top):

```python
from physical_signal_source import carrier_b65_per_text  # noqa: E402
from none_separated_block_graph import rollout_by_method  # noqa: E402


def block_b_features(B65):
    """Per non-None block features: its row (out-edges) and column (in-edges) of B65."""
    B = np.asarray(B65, dtype=np.float64)
    rows = B[1:, :]                                  # (64, 65) outgoing
    cols = B[:, 1:].T                                # (64, 65) incoming
    return np.concatenate([rows, cols], axis=1).astype(np.float32)   # (64, 130)


def sigma_from_B65(B65):
    """C-D+L rollout order of a B65 -> physical-block order (64,).

    rollout_by_method returns the order ndarray DIRECTLY (P2 wraps it as
    discovery_metrics(rollout_by_method(...))); it is NOT a dict."""
    order = rollout_by_method(np.asarray(B65), "C-D+L")
    return np.asarray(order, dtype=np.int64)


def sample_scaffold(ckpt_path, M, layer=0, head=1, n_reveals=8, fixed_reveal_seed=0,
                    device="cpu"):
    """Per-text B65 (seed123 carrier L0H1), its C-D+L order sigma_B, and the chunks."""
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    B_list, _tau = carrier_b65_per_text(
        ckpt_path, layer, head, M=M, n_reveals=n_reveals,
        fixed_reveal_seed=fixed_reveal_seed, device=device)
    sig = [sigma_from_B65(B) for B in B_list]
    return {"B": B_list, "sigma_B": sig, "chunks": chunks,
            "clean_perm": clean_perm, "model": model, "dev": dev}
```

Note: `rollout_by_method(...)["order"]` — confirm the key is `"order"`; if the rollout dict uses a different key for the produced order, use that key (it is the same dict `discovery_metrics` consumes).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_scaffold.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_scaffold.py
git commit -m "feat(p5): per-sample scaffold (B65 + sigma_B + per-block B features)"
```

---

### Task 3: Candidate-order pool (diverse, scaffold-centred)

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_candidates.py`

**Interfaces:**
- Produces: `candidate_orders(sigma_B, rng, n_random=4, n_noisy=4, noisy_swaps=3) -> {label: sigma}` with labels `sigma_B`, `phys`, `reverse_phys`, `random_0..`, `noisy_B_0..`, `local`. Each sigma is a (64,) permutation.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_candidates.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import candidate_orders

def test_candidate_pool_diverse_and_valid():
    rng = np.random.default_rng(0)
    sigB = rng.permutation(64)
    pool = candidate_orders(sigB, rng, n_random=4, n_noisy=4)
    assert "sigma_B" in pool and "phys" in pool and "reverse_phys" in pool
    assert sum(k.startswith("random_") for k in pool) == 4
    assert sum(k.startswith("noisy_B_") for k in pool) == 4
    for s in pool.values():
        assert sorted(s.tolist()) == list(range(64))     # all valid perms
    assert np.array_equal(pool["sigma_B"], sigB)
    # noisy_B is a small perturbation of sigma_B (not identical, not random)
    assert not np.array_equal(pool["noisy_B_0"], sigB)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_candidates.py -v`
Expected: FAIL (`ImportError: candidate_orders`).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def _swap_perturb(sigma, rng, n_swaps):
    s = np.asarray(sigma, dtype=np.int64).copy()
    for _ in range(n_swaps):
        i, j = rng.integers(0, len(s), size=2)
        s[i], s[j] = s[j], s[i]
    return s


def candidate_orders(sigma_B, rng, n_random=4, n_noisy=4, noisy_swaps=3):
    sigma_B = np.asarray(sigma_B, dtype=np.int64)
    phys = np.arange(64, dtype=np.int64)
    pool = {"sigma_B": sigma_B.copy(),
            "phys": phys.copy(),
            "reverse_phys": phys[::-1].copy(),
            "local": phys.copy()}                       # local = identity adjacency baseline
    for k in range(n_random):
        pool[f"random_{k}"] = rng.permutation(64).astype(np.int64)
    for k in range(n_noisy):
        pool[f"noisy_B_{k}"] = _swap_perturb(sigma_B, rng, noisy_swaps)
    return pool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_candidates.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_candidates.py
git commit -m "feat(p5): diverse candidate-order pool (scaffold/phys/random/noisy-B)"
```

---

### Task 4: Utility-headroom gate + best-candidate distribution

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_headroom.py`

**Interfaces:**
- Produces: `headroom_stats(nll_by_label, sigma_b_label="sigma_B", n_boot=1000, seed=0) -> {"abs_mean","abs_ci_low","abs_ci_high","rel_mean","gate_pass","best_dist"}`. Input `nll_by_label`: list over samples of `{label: nll}` dicts. `gate_pass` = abs CI excludes 0 (low>0). `best_dist` = fraction of samples whose min-NLL candidate is each label.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_headroom.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import headroom_stats

def test_headroom_positive_when_other_candidate_better():
    # sigma_B nll=1.0 but 'phys' is better (0.5) every sample -> positive headroom
    samples = [{"sigma_B": 1.0, "phys": 0.5, "random_0": 1.2} for _ in range(20)]
    r = headroom_stats(samples)
    assert r["abs_mean"] > 0.4 and r["gate_pass"] is True
    assert r["best_dist"]["phys"] == 1.0

def test_headroom_zero_when_sigmaB_is_best():
    samples = [{"sigma_B": 0.5, "phys": 1.0} for _ in range(20)]
    r = headroom_stats(samples)
    assert r["abs_mean"] <= 0.0 and r["gate_pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_headroom.py -v`
Expected: FAIL (`ImportError: headroom_stats`).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def headroom_stats(nll_by_label, sigma_b_label="sigma_B", n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    abs_h, rel_h, best_labels = [], [], []
    for d in nll_by_label:
        nb = d[sigma_b_label]
        best_label = min(d, key=lambda k: d[k])
        best = d[best_label]
        abs_h.append(nb - best)                       # >=0 by construction
        rel_h.append((nb - best) / (abs(nb) + 1e-9))
        best_labels.append(best_label)
    abs_h = np.asarray(abs_h)
    boot = np.array([rng.choice(abs_h, size=len(abs_h), replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    labels = sorted({l for d in nll_by_label for l in d})
    best_dist = {l: float(np.mean([bl == l for bl in best_labels])) for l in labels}
    return {"abs_mean": float(abs_h.mean()), "abs_ci_low": lo, "abs_ci_high": hi,
            "rel_mean": float(np.mean(rel_h)), "gate_pass": bool(lo > 0.0),
            "best_dist": best_dist}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_headroom.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_headroom.py
git commit -m "feat(p5): utility-headroom gate + best-candidate distribution monitor"
```

---

### Task 5: Block-level hidden-state extraction (H under σ_B reveal context)

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_hidden.py`

**Interfaces:**
- Consumes: `path_patch_handoff.capture_block_input(model, layer)`; `physical_blocks_to_model_token_order`.
- Produces: `block_hidden_states(model, idx_row, sigma_phys, clean_perm, layer, device) -> np.ndarray (64, d)` — per model-block mean-pooled residual entering `layer`, under the reveal order `sigma_phys`. (Model block m's tokens sit at the seq positions whose revealed model-token index // BLOCK_LEN == m; the leading [None] position is dropped.)

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_hidden.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import load_p5_ckpt, block_hidden_states

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")

def test_hidden_shape_and_determinism():
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, M=2, device="cpu")
    sig = np.arange(64)
    H1 = block_hidden_states(model, chunks[0:1], sig, clean_perm, layer=0, device=dev)
    H2 = block_hidden_states(model, chunks[0:1], sig, clean_perm, layer=0, device=dev)
    assert H1.shape[0] == 64 and H1.ndim == 2
    assert np.allclose(H1, H2)                          # deterministic, frozen model
    # different texts give different H
    H3 = block_hidden_states(model, chunks[1:2], sig, clean_perm, layer=0, device=dev)
    assert not np.allclose(H1, H3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_hidden.py -v`
Expected: FAIL (`ImportError: block_hidden_states`).

- [ ] **Step 3: Write minimal implementation**

Append (add `from analyses.path_patch_handoff import capture_block_input` to imports):

```python
@torch.no_grad()
def block_hidden_states(model, idx_row, sigma_phys, clean_perm, layer, device):
    """Per model-block mean-pooled residual entering `layer`, under reveal sigma_phys."""
    sigma = np.asarray(sigma_phys, dtype=np.int64)[None, :]
    token_order = physical_blocks_to_model_token_order(
        torch.from_numpy(sigma), clean_perm, BLOCK_LEN).to(device)   # (1,256) model-token order
    handle, store = capture_block_input(model, layer)
    try:
        model.forward_fn(idx_row.to(device), token_order)
    finally:
        handle.remove()
    x = store["x"][0].cpu().numpy()                     # (257, d) incl leading [None]
    revealed = x[1:]                                    # (256, d) reveal-ordered positions
    tok = token_order[0].cpu().numpy()                  # model-token index at each position
    block_of_pos = tok // BLOCK_LEN                      # (256,) model block per position
    d = revealed.shape[1]
    H = np.zeros((N, d), dtype=np.float64)
    for m in range(N):
        sel = revealed[block_of_pos == m]
        H[m] = sel.mean(axis=0) if len(sel) else 0.0
    return H.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_hidden.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_hidden.py
git commit -m "feat(p5): block-level hidden-state extraction under sigma_B reveal context"
```

---

### Task 6: Soft pairwise utility teacher + weighted-BCE loss

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_teacher.py`

**Interfaces:**
- Produces: `soft_pref(sigmas, nlls, T) -> np.ndarray (64,64)` where `P[i,j]` = soft prob block i revealed before j, using normalized temperature `w_k = softmax(-(nll_k - min nll)/T)`; `pairwise_loss(z, P) -> torch scalar` confidence-weighted BCE with `c_ij = |P_ij - 0.5|` (z: (64,) torch tensor of scores).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_teacher.py
import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import soft_pref, pairwise_loss

def test_soft_pref_follows_best_order():
    # two candidates; the much-lower-NLL one dominates -> P matches its order
    sA = np.array([0, 1, 2] + list(range(3, 64)))     # i before j ascending
    sB = sA[::-1].copy()
    P = soft_pref([sA, sB], [0.1, 10.0], T=0.5)        # sA hugely better
    assert P[0, 1] > 0.9 and P[1, 0] < 0.1             # 0 before 1 (per sA)

def test_pairwise_loss_lower_when_scores_match_teacher():
    P = np.full((64, 64), 0.5); P[0, 1] = 1.0; P[1, 0] = 0.0
    z_good = torch.zeros(64); z_good[0] = 5.0          # 0 scored well above 1
    z_bad = torch.zeros(64); z_bad[1] = 5.0
    assert float(pairwise_loss(z_good, P)) < float(pairwise_loss(z_bad, P))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_teacher.py -v`
Expected: FAIL (`ImportError: soft_pref`).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def _order_to_rank(sigma):
    """sigma is a reveal order (positions->block). Return rank[block] = reveal position."""
    rank = np.empty(N, dtype=np.int64)
    rank[np.asarray(sigma, dtype=np.int64)] = np.arange(N)
    return rank


def soft_pref(sigmas, nlls, T):
    nlls = np.asarray(nlls, dtype=np.float64)
    w = np.exp(-(nlls - nlls.min()) / max(T, 1e-9))
    w = w / w.sum()
    P = np.zeros((N, N), dtype=np.float64)
    for wk, sig in zip(w, sigmas):
        rank = _order_to_rank(sig)
        before = (rank[:, None] < rank[None, :]).astype(np.float64)   # i before j
        P += wk * before
    return P


def pairwise_loss(z, P):
    P = torch.as_tensor(P, dtype=torch.float32, device=z.device)
    diff = z[:, None] - z[None, :]                      # z_i - z_j
    p_hat = torch.sigmoid(diff)
    c = (P - 0.5).abs()
    eps = 1e-6
    bce = -(P * torch.log(p_hat + eps) + (1 - P) * torch.log(1 - p_hat + eps))
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    return (c * bce)[mask].mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_teacher.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_teacher.py
git commit -m "feat(p5): soft pairwise utility teacher + confidence-weighted BCE"
```

---

### Task 7: Controllers (B-only + scaffolded residual with learnable α)

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_controllers.py`

**Interfaces:**
- Produces: `BOnlyController(b_dim)` → `forward(B_feat (64,b_dim)) -> z (64,)`; `ScaffoldedController(g_B, h_dim, alpha_init=0.01)` → `forward(B_feat, H (64,h_dim)) -> z`, with frozen `g_B`, learnable residual MLP `g_H` and learnable scalar `α=softplus(a)`; method `residual_ratio(B_feat, H) -> float` = `||α·g_H(H)|| / ||g_B(B)||`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_controllers.py
import pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import BOnlyController, ScaffoldedController

def test_scaffolded_freezes_gB_and_small_init_residual():
    gB = BOnlyController(b_dim=130)
    for p in gB.parameters():
        p.requires_grad_(False)
    sc = ScaffoldedController(gB, h_dim=8, alpha_init=0.01)
    B = torch.randn(64, 130); H = torch.randn(64, 8)
    z = sc(B, H)
    assert z.shape == (64,)
    # gB is frozen inside the scaffolded controller
    assert all(not p.requires_grad for p in sc.g_B.parameters())
    assert any(p.requires_grad for p in sc.g_H.parameters())
    assert sc.residual_ratio(B, H) < 0.5               # residual is small at init
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_controllers.py -v`
Expected: FAIL (`ImportError: BOnlyController`).

- [ ] **Step 3: Write minimal implementation**

Append (add `import torch.nn as nn` at top):

```python
import torch.nn as nn  # noqa: E402


class BOnlyController(nn.Module):
    def __init__(self, b_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(b_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, B_feat):
        return self.net(B_feat).squeeze(-1)            # (64,)


class ScaffoldedController(nn.Module):
    def __init__(self, g_B, h_dim, hidden=64, alpha_init=0.01):
        super().__init__()
        self.g_B = g_B
        for p in self.g_B.parameters():
            p.requires_grad_(False)
        self.g_H = nn.Sequential(nn.Linear(h_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # softplus(a)=alpha_init  ->  a = log(exp(alpha)-1)
        self.a = nn.Parameter(torch.tensor(float(np.log(np.expm1(alpha_init)))))

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.a)

    def _delta(self, H):
        return self.g_H(H).squeeze(-1)

    def forward(self, B_feat, H):
        return self.g_B(B_feat) + self.alpha * self._delta(H)

    @torch.no_grad()
    def residual_ratio(self, B_feat, H):
        zb = self.g_B(B_feat)
        dh = self.alpha * self._delta(H)
        return float(dh.norm() / (zb.norm() + 1e-9))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_controllers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_controllers.py
git commit -m "feat(p5): B-only + scaffolded-residual controllers (frozen g_B, learnable small alpha)"
```

---

### Task 8: Dataset assembly + train loops (B-only stage1/freeze, residual stage3) + controls

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_train.py`

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces:
  - `build_dataset(ckpt_path, M, layer=0, head=1, n_reveals=8, K_rand=4, K_noisy=4, T=0.3, cand_seed=0, device="cpu") -> list[sample]` where each `sample = {"B_feat":(64,130), "H":(64,d), "P":(64,64), "sigma_B":(64,), "idx_row":Tensor, "nll_by_label":dict, "cands":dict}`.
  - `train_b_only(samples, epochs=200, lr=1e-2) -> BOnlyController`.
  - `train_residual(samples, g_B, h_dim, epochs=200, lr=1e-2, h_mode="real") -> ScaffoldedController` where `h_mode in {"real","shuffle","zero","mean"}` selects the H control (shuffle = roll H by one sample so B/text and H mismatch).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_train.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import build_dataset, train_b_only, train_residual

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")

def test_dataset_and_training_runs():
    samples = build_dataset(CKPT, M=6, n_reveals=4, K_rand=3, K_noisy=3)
    s0 = samples[0]
    assert s0["B_feat"].shape == (64, 130) and s0["P"].shape == (64, 64)
    gB = train_b_only(samples, epochs=20)
    sc = train_residual(samples, gB, h_dim=s0["H"].shape[1], epochs=20, h_mode="real")
    scs = train_residual(samples, gB, h_dim=s0["H"].shape[1], epochs=20, h_mode="shuffle")
    assert sc is not None and scs is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_train.py -v`
Expected: FAIL (`ImportError: build_dataset`).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
def build_dataset(ckpt_path, M, layer=0, head=1, n_reveals=8, K_rand=4, K_noisy=4,
                  T=0.3, cand_seed=0, device="cpu"):
    sc = sample_scaffold(ckpt_path, M, layer=layer, head=head, n_reveals=n_reveals,
                         device=device)
    model, chunks, clean_perm, dev = sc["model"], sc["chunks"], sc["clean_perm"], sc["dev"]
    rng = np.random.default_rng(cand_seed)
    samples = []
    for t in range(M):
        B65 = sc["B"][t]; sigma_B = sc["sigma_B"][t]
        cands = candidate_orders(sigma_B, rng, n_random=K_rand, n_noisy=K_noisy)
        labels = list(cands)
        nlls = utility_pool(model, chunks[t:t+1], [cands[l] for l in labels], clean_perm, dev)
        nll_by_label = dict(zip(labels, nlls))
        P = soft_pref([cands[l] for l in labels], nlls, T)
        H = block_hidden_states(model, chunks[t:t+1], sigma_B, clean_perm, layer, dev)
        samples.append({"B_feat": block_b_features(B65), "H": H, "P": P,
                        "sigma_B": sigma_B, "idx_row": chunks[t:t+1],
                        "nll_by_label": nll_by_label, "cands": cands,
                        "clean_perm": clean_perm, "model": model, "dev": dev})
    return samples


def _apply_h_mode(samples, h_mode):
    Hs = [s["H"] for s in samples]
    if h_mode == "real":
        return Hs
    if h_mode == "shuffle":
        return Hs[1:] + Hs[:1]                          # text-level mismatch (roll by 1)
    if h_mode == "zero":
        return [np.zeros_like(h) for h in Hs]
    if h_mode == "mean":
        m = np.mean(Hs, axis=0)
        return [m.copy() for _ in Hs]
    raise ValueError(h_mode)


def train_b_only(samples, epochs=200, lr=1e-2):
    g_B = BOnlyController(b_dim=samples[0]["B_feat"].shape[1])
    opt = torch.optim.Adam(g_B.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s in samples:
            z = g_B(torch.tensor(s["B_feat"]))
            loss = loss + pairwise_loss(z, s["P"])
        (loss / len(samples)).backward(); opt.step()
    return g_B


def train_residual(samples, g_B, h_dim, epochs=200, lr=1e-2, h_mode="real"):
    for p in g_B.parameters():
        p.requires_grad_(False)
    sc = ScaffoldedController(g_B, h_dim=h_dim)
    Hs = _apply_h_mode(samples, h_mode)
    opt = torch.optim.Adam([p for p in sc.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s, H in zip(samples, Hs):
            z = sc(torch.tensor(s["B_feat"]), torch.tensor(H))
            loss = loss + pairwise_loss(z, s["P"])
        (loss / len(samples)).backward(); opt.step()
    return sc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_train.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_train.py
git commit -m "feat(p5): dataset assembly + B-only/residual train loops + H controls"
```

---

### Task 9: Hard-order forward-NLL metrics + verdict

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Test: `block_lo_arm_order_network/tests/test_p5_metrics.py`

**Interfaces:**
- Produces:
  - `predict_order(controller, B_feat, H=None) -> sigma (64,)` = `argsort(-z)` (z from controller; B-only if H is None).
  - `eval_controller_nll(samples, controller, h_list=None) -> list[float]` — per-sample real downstream NLL of the predicted hard order (re-run `order_nll`).
  - `p5_metrics(samples, g_B, sc_real, sc_shuf, sc_zero, sc_mean) -> dict` with `delta_nll`, `regret_bonly`, `regret_bh`, `h_shuffle_drop`, `zero_match`, `mean_match`, and `verdict`.
  - `classify_p5(m) -> "utility_gain" | "no_gain"` (gain iff `delta_nll<0` AND shuffle recovers most gain AND zero/mean do NOT match B+H).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_metrics.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import classify_p5

def test_classify_requires_all_three():
    gain = {"delta_nll": -0.05, "h_shuffle_drop": 0.04, "zero_match": False, "mean_match": False}
    assert classify_p5(gain) == "utility_gain"
    # shuffle keeps the gain -> not content-driven -> no_gain
    nshuf = {"delta_nll": -0.05, "h_shuffle_drop": 0.0, "zero_match": False, "mean_match": False}
    assert classify_p5(nshuf) == "no_gain"
    # no delta -> no_gain
    nd = {"delta_nll": 0.01, "h_shuffle_drop": 0.04, "zero_match": False, "mean_match": False}
    assert classify_p5(nd) == "no_gain"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_metrics.py -v`
Expected: FAIL (`ImportError: classify_p5`).

- [ ] **Step 3: Write minimal implementation**

Append:

```python
@torch.no_grad()
def predict_order(controller, B_feat, H=None):
    B = torch.tensor(B_feat)
    z = controller(B) if H is None else controller(B, torch.tensor(H))
    return np.argsort(-z.cpu().numpy()).astype(np.int64)


@torch.no_grad()
def eval_controller_nll(samples, controller, h_list=None):
    out = []
    for i, s in enumerate(samples):
        H = None if h_list is None else h_list[i]
        sigma = predict_order(controller, s["B_feat"], H)
        out.append(order_nll(s["model"], s["idx_row"], sigma, s["clean_perm"], s["dev"]))
    return out


def _regret(samples, nll_pred):
    return float(np.mean([nll_pred[i] - min(s["nll_by_label"].values())
                          for i, s in enumerate(samples)]))


def p5_metrics(samples, g_B, sc_real, sc_shuf, sc_zero, sc_mean):
    n_b = eval_controller_nll(samples, g_B)
    n_bh = eval_controller_nll(samples, sc_real, _apply_h_mode(samples, "real"))
    n_sh = eval_controller_nll(samples, sc_shuf, _apply_h_mode(samples, "shuffle"))
    n_zero = eval_controller_nll(samples, sc_zero, _apply_h_mode(samples, "zero"))
    n_mean = eval_controller_nll(samples, sc_mean, _apply_h_mode(samples, "mean"))
    mb, mbh = float(np.mean(n_b)), float(np.mean(n_bh))
    m = {"nll_b_only": mb, "nll_bh": mbh,
         "delta_nll": mbh - mb,
         "regret_bonly": _regret(samples, n_b), "regret_bh": _regret(samples, n_bh),
         "h_shuffle_drop": float(np.mean(n_sh)) - mbh,
         "zero_match": bool(abs(float(np.mean(n_zero)) - mbh) < 0.005),
         "mean_match": bool(abs(float(np.mean(n_mean)) - mbh) < 0.005)}
    m["verdict"] = classify_p5(m)
    return m


def classify_p5(m):
    gain = m["delta_nll"] < 0
    content = m["h_shuffle_drop"] > 0.5 * abs(m["delta_nll"])     # shuffle loses most gain
    not_capacity = (not m["zero_match"]) and (not m["mean_match"])
    return "utility_gain" if (gain and content and not_capacity) else "no_gain"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analyses/p5_utility_controller.py block_lo_arm_order_network/tests/test_p5_metrics.py
git commit -m "feat(p5): hard-order forward-NLL metrics + utility-gain verdict"
```

---

### Task 10: Phase-0 driver + plot + README + memory

**Files:**
- Modify: `analyses/p5_utility_controller.py`
- Create: `analyses/plot_p5.py`, `analyses/p5_utility_controller_README.md`
- Modify: `/home/admin/.claude/projects/-home-admin-lyuyuhuan-order-lyu/memory/MEMORY.md`
- Test: `block_lo_arm_order_network/tests/test_p5_driver.py`

**Interfaces:**
- Produces: `run_phase0(ckpt_path, M=64, n_reveals=8, T=0.3, split=(0.7,0.15,0.15), epochs=200, layer=0, head=1, out_dir="runs/p5/seed123", device="cpu") -> dict` — builds the dataset, runs the headroom gate on the TEST split, and (if gate passes) trains B-only on TRAIN, freezes, trains residual variants, and writes `phase0.json`. `plot_p5(json_path, out_dir)` writes `phase0_metrics.png`.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_p5_driver.py
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import run_phase0
from analyses.plot_p5 import plot_p5

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")

def test_run_phase0_writes_outputs(tmp_path):
    r = run_phase0(CKPT, M=8, n_reveals=4, epochs=10, out_dir=str(tmp_path))
    assert (tmp_path / "phase0.json").exists()
    assert "headroom" in r and "gate_pass" in r["headroom"]
    plot_p5(str(tmp_path / "phase0.json"), str(tmp_path))
    assert (tmp_path / "phase0_metrics.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_driver.py -v`
Expected: FAIL (`ImportError: run_phase0`).

- [ ] **Step 3: Write minimal implementation**

Append to `analyses/p5_utility_controller.py` (add `import json as _json, csv as _csv` at top):

```python
def _split_idx(n, split, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(split[0] * n); n_va = int(split[1] * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def run_phase0(ckpt_path, M=64, n_reveals=8, T=0.3, split=(0.7, 0.15, 0.15),
               epochs=200, layer=0, head=1, out_dir="runs/p5/seed123", device="cpu"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_dataset(ckpt_path, M, layer=layer, head=head, n_reveals=n_reveals,
                            T=T, device=device)
    tr, va, te = _split_idx(len(samples), split)
    test = [samples[i] for i in te] or [samples[i] for i in tr]
    head_in = headroom_stats([s["nll_by_label"] for s in test])
    result = {"ckpt": ckpt_path, "layer": layer, "head": head, "headroom": head_in}
    if head_in["gate_pass"]:
        train = [samples[i] for i in tr]
        g_B = train_b_only(train, epochs=epochs)
        for p in g_B.parameters():
            p.requires_grad_(False)
        h_dim = samples[0]["H"].shape[1]
        sc_real = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="real")
        sc_shuf = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="shuffle")
        sc_zero = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="zero")
        sc_mean = train_residual(train, g_B, h_dim, epochs=epochs, h_mode="mean")
        result["metrics"] = p5_metrics(test, g_B, sc_real, sc_shuf, sc_zero, sc_mean)
        result["residual_ratio"] = sc_real.residual_ratio(
            torch.tensor(test[0]["B_feat"]), torch.tensor(test[0]["H"]))
    _json.dump(result, open(out / "phase0.json", "w"), indent=2, default=float)
    return result
```

Create `analyses/plot_p5.py`:

```python
"""Plot P5 Phase-0 utility metrics."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_p5(json_path, out_dir):
    r = json.load(open(json_path)); out = pathlib.Path(out_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    hd = r["headroom"]; bd = hd["best_dist"]
    ax1.bar(range(len(bd)), list(bd.values()))
    ax1.set_xticks(range(len(bd))); ax1.set_xticklabels(list(bd), rotation=60, fontsize=6)
    ax1.set_title(f"best-candidate dist | headroom_abs={hd['abs_mean']:.3f} "
                  f"gate={hd['gate_pass']}")
    if "metrics" in r:
        m = r["metrics"]
        bars = {"B-only": m["nll_b_only"], "B+H": m["nll_bh"]}
        ax2.bar(range(len(bars)), list(bars.values()))
        ax2.set_xticks(range(len(bars))); ax2.set_xticklabels(list(bars))
        ax2.set_title(f"ΔNLL={m['delta_nll']:.4f}  shuf_drop={m['h_shuffle_drop']:.4f}  "
                      f"verdict={m['verdict']}")
    else:
        ax2.text(0.5, 0.5, "headroom gate FAILED\n(no training)", ha="center")
    fig.tight_layout(); fig.savefig(out / "phase0_metrics.png", dpi=120); plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/order_lyu && python -m pytest block_lo_arm_order_network/tests/test_p5_driver.py -v`
Expected: PASS.

- [ ] **Step 5: Run the real Phase-0 smoke, then write README + memory**

Run (the real single-ckpt Phase 0; settings from the spec):

```bash
cd /home/admin/lyuyuhuan/order_lyu && python -c "
from analyses.p5_utility_controller import run_phase0
r = run_phase0('runs/handoff_overnight/seed123/ckpt_step10000.pt', M=64, n_reveals=8, epochs=200)
print('gate:', r['headroom']['gate_pass'], 'abs:', round(r['headroom']['abs_mean'],4))
print('metrics:', r.get('metrics'))
"
```

Then create `analyses/p5_utility_controller_README.md` (fill the Result line from the run): include the three verbatim spec sentences (red line + utility-not-recovery + claim), the headroom-gate result, and the Phase-0 verdict (`utility_gain` / `no_gain`, with the `headroom positive but no_gain = room exists, H not learned` reading). Add a one-line `MEMORY.md` pointer:

```
- [P5 Utility Controller Phase 0 2026-06-29](p5-utility-controller-phase0.md) — H 在 attention scaffold 上做 downstream-NLL utility residual(非 physical-order recovery); fixed layout; headroom gate + frozen B-only baseline + shuffled-H 控制; Phase0 verdict=<fill>
```

- [ ] **Step 6: Commit**

```bash
git add analyses/p5_utility_controller.py analyses/plot_p5.py analyses/p5_utility_controller_README.md \
        block_lo_arm_order_network/tests/test_p5_driver.py runs/p5 MEMORY.md
git commit -m "feat(p5): Phase-0 driver + plot + README + memory"
```

---

## Self-Review

**Spec coverage:** §0 claim/red-line → README (Task 10) + Global Constraints. §1 NLL utility → Task 1. §2 headroom gate (oracle-vs-learnable, best-cand dist) → Task 4 + driver gate (Task 10). §3 candidate diversity → Task 3. §4 soft teacher (normalized T) → Task 6 (T selected on val: the driver exposes `T`; the Phase-0 smoke uses the spec default — a T-sweep over val is a one-loop extension noted below). §5 architecture/frozen-baseline/extraction-context → Tasks 5 (H under σ_B), 7 (frozen g_B + small α), 2 (B65). §6 text-split → Task 10 `_split_idx`. §7 reporting rule (hard-order forward NLL) + controls + success → Tasks 9, 8 (`_apply_h_mode`). §8 Phase-0-only sequencing → Task 10 (`run_phase0`, no ckpt scan). §9 can/cannot, §11 task split → README (Task 10).

**Noted gap (T-selection):** the spec wants T chosen on validation; Task 10 runs a single spec-default T for the smoke. Selecting T on `va` is a small wrapper loop over `train_b_only` + held-out regret; if the smoke shows T-sensitivity, add a `select_T(train, val, Ts)` helper before the residual stage. Flagged rather than silently dropped.

**Placeholder scan:** README result line is filled from the real run in Task 10 Step 5 (run precedes write). No code placeholders.

**Type consistency:** `order_nll(model, idx_row, sigma_phys, clean_perm, device)` used identically in Tasks 1/5/9; `block_b_features -> (64,130)` matches `BOnlyController(b_dim=130)`; `block_hidden_states -> (64,d)` matches `ScaffoldedController(h_dim=d)`; `samples[*]` dict keys (`B_feat/H/P/sigma_B/idx_row/nll_by_label/model/clean_perm/dev`) consistent across Tasks 8/9/10; `pairwise_loss(z, P)` and `soft_pref -> (64,64)` consistent.
