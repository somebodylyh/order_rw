# NR-1 Neural Attention-to-Order Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate `g_β: R^{64×64} → R^{64}`, a Graph-Transformer readout that imitates the source-start C-D+L teacher's order from a per-sample attention graph `B = A^T`, with strict separation of attention-order matching (training + gates) from frozen-θ NLL (diagnostic only).

**Architecture:** Per-node tokens (`x_v = [B[v,:], B[:,v]] ∈ R^128`) → linear to `d=64` → 2-layer TransformerEncoder (no positional embedding) → scalar head producing `s_v`. Trained with pairwise logistic loss against teacher rank where rank 0 = earliest. All hyperparameter and checkpoint selection done by Kendall τ / pairwise precedence acc / Spearman ρ; frozen-θ NLL is computed once after selection as a diagnostic only.

**Tech Stack:** Python 3, PyTorch (existing), NumPy, SciPy (Kendall/Spearman), pytest. Reuses `block_lo_arm_order_network/train_clean_aogpt.py:extract_A_matrices` (with a seeding bugfix), `attn_order_teacher.rollout_order`, `attn_order_mlp_policy.readiness_vector`.

**Spec:** `docs/superpowers/specs/2026-05-28-neural-attention-to-order-readout-nr1-design.md`

---

## File structure

### Create

| Path | Responsibility |
|---|---|
| `block_lo_arm_order_network/neural_readout/__init__.py` | package marker |
| `block_lo_arm_order_network/neural_readout/extract_b.py` | per-sample `B = A^T` extraction with explicit seeding |
| `block_lo_arm_order_network/neural_readout/teacher_labels.py` | `B → (σ_T, rank, pairwise_Y)`, CDL-source-start teacher, rank-0=earliest convention |
| `block_lo_arm_order_network/neural_readout/dataset.py` | build / save / load `(B, σ_T, rank)` `.npz` dataset |
| `block_lo_arm_order_network/neural_readout/diversity_stats.py` | unique-σ, first-node entropy, distinct first-3, mean pairwise τ over teacher |
| `block_lo_arm_order_network/neural_readout/graph_transformer_readout.py` | `g_β` module: per-node features → 2L TransformerEncoder → scalar |
| `block_lo_arm_order_network/neural_readout/loss.py` | pairwise logistic loss with rank-0=earliest assertion |
| `block_lo_arm_order_network/neural_readout/eval_metrics.py` | Kendall τ / pairwise precedence acc / Spearman ρ |
| `block_lo_arm_order_network/neural_readout/train_nr1.py` | training loop; model selection by §5.1 gates only |
| `block_lo_arm_order_network/neural_readout/eval_frozen_nll.py` | diagnostic-only frozen-θ NLL gap, asserts post-selection |
| `block_lo_arm_order_network/neural_readout/ablation_inputs.py` | four input transforms: identity, reverse, sym, row-shuffle, B_global |
| `block_lo_arm_order_network/neural_readout/report.py` | aggregate metrics + diagnostics + apply §9/§10 interpretation |
| `block_lo_arm_order_network/tests/test_neural_readout_extract_seed.py` | extraction seeding (existing extract_A_matrices fix) |
| `block_lo_arm_order_network/tests/test_neural_readout_extract_b.py` | per-sample B wrapper |
| `block_lo_arm_order_network/tests/test_neural_readout_teacher.py` | teacher convention + direction tests |
| `block_lo_arm_order_network/tests/test_neural_readout_dataset.py` | round-trip save/load |
| `block_lo_arm_order_network/tests/test_neural_readout_diversity.py` | degenerate vs diverse |
| `block_lo_arm_order_network/tests/test_neural_readout_model.py` | shapes + param count + no PE |
| `block_lo_arm_order_network/tests/test_neural_readout_loss.py` | direction / monotonicity / minimum |
| `block_lo_arm_order_network/tests/test_neural_readout_metrics.py` | identity / reversed / known values |
| `block_lo_arm_order_network/tests/test_neural_readout_overfit.py` | overfit-tiny-batch sanity |
| `block_lo_arm_order_network/tests/test_neural_readout_selection_policy.py` | enforces no-NLL-in-selection |

### Modify

| Path | What changes |
|---|---|
| `block_lo_arm_order_network/train_clean_aogpt.py` | add `seed: int | None = None` to `extract_A_matrices`; when set, use `torch.Generator` for the internal `torch.randperm`. Default `None` preserves current behavior so all existing callers keep working. |

### Outputs (generated, not committed)

| Path | Produced by |
|---|---|
| `block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.npz` | dataset task |
| `block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz` | dataset task |
| `block_lo_arm_order_network/neural_readout/data/text_cross_<step>_1k.npz` | per cross-ckpt step |
| `block_lo_arm_order_network/neural_readout/checkpoints/g_beta_<run>.pt` | training |
| `analyses/neural_readout_nr1_2026-05-28/REPORT.md` | final report |

---

## Constants (cite from spec and existing code)

- `N = 64` (block count)
- `alpha_dep = 0.5` (verified against `configs/text/wikitext103_seq256_block64_v3_readiness.py:71` and `attn_order_mlp_policy.py:149`)
- Train ckpt: `block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt`
- Cross-ckpt set: `ckpt_step{10000,20000,30000,40000,50000,60000}.pt` in the same dir
- `block_perm.npy` / `inv_perm.npy` (seed=42 clean_perm) live in the ckpt dir
- Number of pairs per graph: `N*(N-1)/2 = 2016`

---

## Task 1: Fix `extract_A_matrices` seeding (PRE-REQUISITE)

**Files:**
- Modify: `block_lo_arm_order_network/train_clean_aogpt.py:71-138` (function `extract_A_matrices`)
- Test: `block_lo_arm_order_network/tests/test_neural_readout_extract_seed.py`

**Background:** Current implementation at `train_clean_aogpt.py:85` does `torch.randperm(N, device='cpu')` un-seeded — cross-run B extraction is not reproducible. This is the root cause of [[cdl_evolution_clean_base_20260527]]. The fix must remain backward-compatible with all existing callers (`scripts/extract_random_substrate_A.py`, `scripts/run_causal_hidden_probe.py`, etc.).

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_neural_readout_extract_seed.py
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

import pytest


@pytest.fixture(scope="module")
def loaded_model_chunks():
    """Load 5k ckpt + a few chunks once for all seeding tests."""
    from train_clean_aogpt import build_model, CleanPermutation, phys_to_model_idx_clean
    from training_utils import load_train_chunks, SEQ_LEN

    ckpt_dir = ROOT / "block_lo_arm_order_network/probe_results/clean_base_random_perm"
    ckpt = torch.load(ckpt_dir / "ckpt_step5000.pt", map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    model_args = ckpt["model_args"]
    model_args["block_size"] = SEQ_LEN

    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(model_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx_phys = load_train_chunks(n_chunks=None)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    return model, idx_model[:4], clean_perm, device


def test_same_seed_bit_identical(loaded_model_chunks):
    from train_clean_aogpt import extract_A_matrices
    model, idx_chunks, clean_perm, device = loaded_model_chunks
    A1 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=42)
    A2 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=42)
    np.testing.assert_array_equal(A1, A2)


def test_different_seed_different_A(loaded_model_chunks):
    from train_clean_aogpt import extract_A_matrices
    model, idx_chunks, clean_perm, device = loaded_model_chunks
    A1 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=1)
    A2 = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=2)
    # At least one element should differ
    assert not np.array_equal(A1, A2)


def test_seed_none_preserves_old_behavior(loaded_model_chunks):
    """seed=None must not crash and must produce a valid (n,64,64) array."""
    from train_clean_aogpt import extract_A_matrices
    model, idx_chunks, clean_perm, device = loaded_model_chunks
    A = extract_A_matrices(model, idx_chunks, clean_perm, device, seed=None)
    assert A.shape == (4, 64, 64)
    assert np.all(np.diagonal(A, axis1=1, axis2=2) == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_extract_seed.py -v`
Expected: `TypeError: extract_A_matrices() got an unexpected keyword argument 'seed'`.

- [ ] **Step 3: Modify `extract_A_matrices` to accept seed**

In `block_lo_arm_order_network/train_clean_aogpt.py:71`, change the signature and the `torch.randperm` call:

```python
@torch.no_grad()
def extract_A_matrices(model, idx_chunks, clean_perm, device, n_chunks=None, seed=None):
    """Extract NxN attention matrices from current model on given chunks.

    Args:
        seed: if int, use a torch.Generator seeded with `seed + i` for the i-th
              chunk's randperm so calls are bit-reproducible across runs.
              If None, falls back to the global torch RNG (original behavior).
    """
    if n_chunks is None:
        n_chunks = len(idx_chunks)
    n_chunks = min(n_chunks, len(idx_chunks))

    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    A_all = np.zeros((n_chunks, N, N), dtype=np.float32)
    model.eval()

    for i in range(n_chunks):
        tokens = idx_chunks[i:i+1].to(device)
        if seed is not None:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(int(seed) + int(i))
            rand_blocks = torch.randperm(N, generator=gen, device='cpu')
        else:
            rand_blocks = torch.randperm(N, device='cpu')
        token_order = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        ).to(device)
        # ... rest unchanged ...
```

(All lines after `token_order = ...` remain identical to the existing function body.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_extract_seed.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Smoke-run a backward-compat caller**

Run: `python -c "import sys; sys.path.insert(0, 'block_lo_arm_order_network'); from train_clean_aogpt import extract_A_matrices; print('import ok')"`
Expected: prints `import ok`. Confirms no syntax breakage in the file other callers import.

- [ ] **Step 6: Commit**

```bash
git add block_lo_arm_order_network/train_clean_aogpt.py \
        block_lo_arm_order_network/tests/test_neural_readout_extract_seed.py
git commit -m "fix(extract_A): add explicit seed argument for reproducible per-sample B extraction"
```

---

## Task 2: `extract_b` — per-sample B extraction wrapper

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/__init__.py`
- Create: `block_lo_arm_order_network/neural_readout/extract_b.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_extract_b.py`

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_neural_readout_extract_b.py
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_extract_b_shape_diag_reproducible():
    from neural_readout.extract_b import extract_per_sample_B

    ckpt_dir = ROOT / "block_lo_arm_order_network/probe_results/clean_base_random_perm"
    ckpt = ckpt_dir / "ckpt_step5000.pt"
    B1 = extract_per_sample_B(str(ckpt), M=4, seed=42, device="cuda:0" if torch.cuda.is_available() else "cpu")
    B2 = extract_per_sample_B(str(ckpt), M=4, seed=42, device="cuda:0" if torch.cuda.is_available() else "cpu")

    assert B1.shape == (4, 64, 64)
    assert B1.dtype == np.float32
    assert np.all(np.diagonal(B1, axis1=1, axis2=2) == 0.0)
    np.testing.assert_array_equal(B1, B2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_extract_b.py -v`
Expected: `ModuleNotFoundError: No module named 'neural_readout'`.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/__init__.py` (empty file).

Create `block_lo_arm_order_network/neural_readout/extract_b.py`:

```python
"""Per-sample B = A^T extraction from a ckpt (NR-1 §2.2).

Reuses extract_A_matrices (with explicit seeding) and applies B = A^T, diag=0.
"""
import sys, pathlib, numpy as np, torch
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from train_clean_aogpt import (
    extract_A_matrices, build_model, CleanPermutation, phys_to_model_idx_clean,
)
from training_utils import load_train_chunks, SEQ_LEN


def extract_per_sample_B(ckpt_path: str, M: int, seed: int, device: str = "cuda:0") -> np.ndarray:
    """Forward M wikitext-103 chunks through ckpt, return per-sample B = A^T.

    Shape (M, 64, 64), float32, diag=0. Bit-reproducible given (ckpt, M, seed).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN

    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_model(model_args).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Pull M eval-set chunks, deterministic via seed
    idx_phys = load_train_chunks(n_chunks=None)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    eval_indices = np.asarray(protocol["eval_indices"], dtype=np.int64)
    idx_eval = idx_model[eval_indices]

    rng = np.random.RandomState(seed)
    pick = rng.choice(len(idx_eval), size=min(M, len(idx_eval)), replace=False)
    chunks = idx_eval[pick]

    A = extract_A_matrices(model, chunks, clean_perm, dev, n_chunks=M, seed=seed)
    # B = A^T, diag = 0
    B = np.transpose(A, (0, 2, 1)).copy()
    for i in range(B.shape[0]):
        np.fill_diagonal(B[i], 0.0)
    return B.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_extract_b.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/__init__.py \
        block_lo_arm_order_network/neural_readout/extract_b.py \
        block_lo_arm_order_network/tests/test_neural_readout_extract_b.py
git commit -m "feat(neural-readout): per-sample B extraction with reproducible seeding"
```

---

## Task 3: `teacher_labels` — CDL-source-start teacher + rank/pairwise conversion

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/teacher_labels.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_teacher.py`

**Convention (load-bearing, asserted in code):** `rank[v]` = position of node `v` in the reveal order, so `rank[σ[0]] == 0` (earliest). Score convention: rank 0 = earliest ⇒ should get the **highest** score.

- [ ] **Step 1: Write the failing tests**

```python
# block_lo_arm_order_network/tests/test_neural_readout_teacher.py
import sys, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_rank_zero_is_earliest():
    """rank[sigma[0]] must equal 0 (earliest revealed)."""
    from neural_readout.teacher_labels import generate_teacher_label
    rng = np.random.default_rng(0)
    B = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    sigma, rank, _Y = generate_teacher_label(B, alpha_dep=0.5)
    assert rank[sigma[0]] == 0
    assert rank[sigma[-1]] == 63


def test_pairwise_direction():
    """Y[i,j] = 1 iff rank[i] < rank[j] (i revealed before j)."""
    from neural_readout.teacher_labels import generate_teacher_label
    rng = np.random.default_rng(1)
    B = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    sigma, rank, Y = generate_teacher_label(B, alpha_dep=0.5)
    for i in range(64):
        for j in range(64):
            if i == j:
                continue
            assert int(Y[i, j]) == int(rank[i] < rank[j]), (i, j, rank[i], rank[j], Y[i, j])


def test_source_start_uses_readiness_argmax():
    """First revealed node must equal argmax(out(v) - alpha * in(v))."""
    from neural_readout.teacher_labels import generate_teacher_label
    from attn_order_mlp_policy import readiness_vector
    rng = np.random.default_rng(2)
    B = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    expected_start = int(np.argmax(readiness_vector(B, alpha_dep=0.5)))
    sigma, _rank, _Y = generate_teacher_label(B, alpha_dep=0.5)
    assert sigma[0] == expected_start


def test_toy_chain_direction():
    """Strictly forward 'chain' B: i->i+1 with weight 1, else 0. With source-start,
    teacher should roll out 0,1,2,...,N-1 because node 0 has highest readiness."""
    from neural_readout.teacher_labels import generate_teacher_label
    N = 8
    B = np.zeros((N, N), dtype=np.float32)
    for i in range(N - 1):
        B[i, i + 1] = 1.0
    sigma, _rank, _Y = generate_teacher_label(B, alpha_dep=0.5)
    # source(v) = out(v) - 0.5 * in(v); node 0: out=1, in=0 -> 1.0 (max)
    # forward rollout follows the chain
    assert sigma[0] == 0
    np.testing.assert_array_equal(sigma, np.arange(N))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_teacher.py -v`
Expected: `ModuleNotFoundError: No module named 'neural_readout.teacher_labels'`.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/teacher_labels.py`:

```python
"""CDL-source-start teacher — generate (sigma_T, rank, pairwise Y) from B.

Convention (load-bearing):
  - rank[v] = position of v in the reveal order
  - rank[sigma[0]] == 0 (earliest revealed = rank 0)
  - pairwise Y[i, j] = 1 iff rank[i] < rank[j]  (i revealed before j)
  - score convention (used in loss.py): earliest should get HIGHEST score.
"""
import sys, pathlib, numpy as np
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from attn_order_teacher import rollout_order
from attn_order_mlp_policy import readiness_vector


def generate_teacher_label(B: np.ndarray, alpha_dep: float = 0.5):
    """Return (sigma, rank, pairwise_Y) for one graph B.

    sigma   : (N,) int64, reveal order
    rank    : (N,) int64, rank[v] = position of v in sigma; rank 0 = earliest
    pairwise_Y : (N, N) uint8, Y[i, j] = 1 iff rank[i] < rank[j]
    """
    B = np.asarray(B, dtype=np.float64)
    assert B.ndim == 2 and B.shape[0] == B.shape[1], f"B must be square, got {B.shape}"
    assert np.all(np.diag(B) == 0.0), "B must have zero diagonal"

    N = B.shape[0]
    r = readiness_vector(B, alpha_dep=alpha_dep)
    start = int(np.argmax(r))

    sigma = rollout_order(B, mode="C-D+L", greedy=True, start=start)
    sigma = np.asarray(sigma, dtype=np.int64)
    assert sigma.shape == (N,)

    # rank: inverse permutation; rank[sigma[t]] = t (so rank 0 = earliest)
    rank = np.empty(N, dtype=np.int64)
    rank[sigma] = np.arange(N, dtype=np.int64)

    # Pairwise Y[i, j] = 1 iff rank[i] < rank[j]
    Y = (rank[:, None] < rank[None, :]).astype(np.uint8)
    np.fill_diagonal(Y, 0)

    return sigma, rank, Y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_teacher.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/teacher_labels.py \
        block_lo_arm_order_network/tests/test_neural_readout_teacher.py
git commit -m "feat(neural-readout): CDL-source-start teacher with rank-0=earliest convention"
```

---

## Task 4: `dataset` — build, save, load (B, σ_T, rank) pair set

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/dataset.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_neural_readout_dataset.py
import sys, pathlib, numpy as np, tempfile, os
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_dataset_roundtrip(tmp_path):
    from neural_readout.dataset import save_dataset, load_dataset
    M, N = 8, 64
    rng = np.random.default_rng(0)
    B = rng.random((M, N, N)).astype(np.float32)
    for i in range(M):
        np.fill_diagonal(B[i], 0.0)
    sigma = np.stack([rng.permutation(N) for _ in range(M)]).astype(np.int64)
    rank = np.empty_like(sigma)
    for i in range(M):
        rank[i, sigma[i]] = np.arange(N)

    path = tmp_path / "tiny.npz"
    save_dataset(str(path), B, sigma, rank)
    B2, sigma2, rank2 = load_dataset(str(path))

    np.testing.assert_array_equal(B, B2)
    np.testing.assert_array_equal(sigma, sigma2)
    np.testing.assert_array_equal(rank, rank2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_dataset.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/dataset.py`:

```python
"""(B, sigma, rank) dataset on disk (NR-1 §2.4).

Stored as a single .npz with three arrays. No metadata; reproducibility comes
from the (ckpt, M, seed, alpha_dep) used to construct it.
"""
import sys, pathlib, numpy as np
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))


def save_dataset(path: str, B: np.ndarray, sigma: np.ndarray, rank: np.ndarray) -> None:
    assert B.ndim == 3 and B.shape[1] == B.shape[2] == 64, f"unexpected B shape {B.shape}"
    assert sigma.shape == (B.shape[0], 64)
    assert rank.shape == (B.shape[0], 64)
    np.savez_compressed(path,
                        B=B.astype(np.float32),
                        sigma=sigma.astype(np.int64),
                        rank=rank.astype(np.int64))


def load_dataset(path: str):
    data = np.load(path)
    return data["B"], data["sigma"], data["rank"]


def build_dataset_from_ckpt(ckpt_path: str, M: int, seed: int, alpha_dep: float,
                            out_path: str, device: str = "cuda:0") -> str:
    """Forward M chunks through ckpt, build per-sample B, label with teacher, save."""
    from neural_readout.extract_b import extract_per_sample_B
    from neural_readout.teacher_labels import generate_teacher_label

    B = extract_per_sample_B(ckpt_path, M=M, seed=seed, device=device)
    sigmas = np.zeros((M, 64), dtype=np.int64)
    ranks = np.zeros((M, 64), dtype=np.int64)
    for i in range(M):
        sigma, rank, _Y = generate_teacher_label(B[i], alpha_dep=alpha_dep)
        sigmas[i] = sigma
        ranks[i] = rank
    save_dataset(out_path, B, sigmas, ranks)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/dataset.py \
        block_lo_arm_order_network/tests/test_neural_readout_dataset.py
git commit -m "feat(neural-readout): dataset builder + npz round-trip"
```

---

## Task 5: `diversity_stats` — teacher diversity diagnostic

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/diversity_stats.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_diversity.py`

- [ ] **Step 1: Write the failing tests**

```python
# block_lo_arm_order_network/tests/test_neural_readout_diversity.py
import sys, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_degenerate_dataset():
    """All-identical sigmas -> unique=1, first-node entropy=0, mean pairwise tau=1."""
    from neural_readout.diversity_stats import teacher_diversity
    sigma = np.tile(np.arange(64), (8, 1))  # 8 identical permutations
    stats = teacher_diversity(sigma)
    assert stats["unique_sigma_count"] == 1
    assert abs(stats["first_node_entropy"]) < 1e-9
    assert stats["distinct_first3_prefix_count"] == 1
    assert abs(stats["mean_pairwise_tau"] - 1.0) < 1e-9


def test_diverse_dataset():
    """All-random distinct sigmas -> unique=M, first-node entropy near log2(M)."""
    from neural_readout.diversity_stats import teacher_diversity
    rng = np.random.default_rng(0)
    sigma = np.stack([rng.permutation(64) for _ in range(64)]).astype(np.int64)
    stats = teacher_diversity(sigma)
    assert stats["unique_sigma_count"] >= 60   # almost all distinct
    assert stats["first_node_entropy"] > 3.0   # high entropy
    assert stats["mean_pairwise_tau"] < 0.1    # uncorrelated random perms
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_diversity.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/diversity_stats.py`:

```python
"""Teacher order diversity diagnostic (NR-1 §5.1b).

These stats accompany every student tau report so reviewers can distinguish
'student learned a per-sample mapping' from 'student learned a global prior'.
"""
import numpy as np
from scipy.stats import kendalltau


def _first_node_entropy(sigma: np.ndarray) -> float:
    """Entropy (nats) of the empirical first-node distribution over the dataset."""
    first = sigma[:, 0]
    _, counts = np.unique(first, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def teacher_diversity(sigma: np.ndarray) -> dict:
    """sigma: (M, N) int64. Returns 4 diagnostic stats."""
    assert sigma.ndim == 2
    M, N = sigma.shape

    unique_sigma = np.unique(sigma, axis=0)
    first3 = np.unique(sigma[:, :3], axis=0)

    # Mean pairwise Kendall tau over up to 100 random pairs (subsample for cost)
    rng = np.random.default_rng(0)
    n_pairs = min(100, M * (M - 1) // 2) if M > 1 else 0
    taus = []
    for _ in range(n_pairs):
        i, j = rng.choice(M, size=2, replace=False)
        tau, _ = kendalltau(sigma[i], sigma[j])
        taus.append(tau)
    mean_tau = float(np.mean(taus)) if taus else 1.0

    return {
        "unique_sigma_count": int(unique_sigma.shape[0]),
        "first_node_entropy": _first_node_entropy(sigma),
        "distinct_first3_prefix_count": int(first3.shape[0]),
        "mean_pairwise_tau": mean_tau,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_diversity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/diversity_stats.py \
        block_lo_arm_order_network/tests/test_neural_readout_diversity.py
git commit -m "feat(neural-readout): teacher diversity diagnostic (unique/entropy/prefix/tau)"
```

---

## Task 6: `graph_transformer_readout` — g_β model

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/graph_transformer_readout.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# block_lo_arm_order_network/tests/test_neural_readout_model.py
import sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_model_shape():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    g = GraphTransformerReadout(N=64, d_model=64, n_heads=4, n_layers=2)
    B = torch.randn(7, 64, 64)
    s = g(B)
    assert s.shape == (7, 64), s.shape


def test_no_positional_embedding_registered():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    g = GraphTransformerReadout(N=64, d_model=64, n_heads=4, n_layers=2)
    # No buffer or parameter should have 'pos' in its name
    for name, _ in g.named_parameters():
        assert "pos" not in name.lower(), f"unexpected positional param: {name}"
    for name, _ in g.named_buffers():
        assert "pos" not in name.lower(), f"unexpected positional buffer: {name}"


def test_param_count_in_range():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    g = GraphTransformerReadout(N=64, d_model=64, n_heads=4, n_layers=2)
    n = sum(p.numel() for p in g.parameters())
    assert 30_000 <= n <= 200_000, f"param count out of expected range: {n}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_model.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/graph_transformer_readout.py`:

```python
"""g_beta — Graph Transformer over per-node attention features (NR-1 §3).

Per-node token x_v = concat(B[v,:], B[:,v]) in R^{2N}, projected to d_model,
processed by a standard TransformerEncoder with NO positional embedding, then
read off to a scalar score s_v per node.

Honest framing (§3 of the spec): row/column coordinates of B encode node identity
through the fixed graph coordinate system. This is a low-extra-position-bias
readout, NOT a permutation-invariant architecture.
"""
import torch
import torch.nn as nn


class GraphTransformerReadout(nn.Module):
    def __init__(self, N: int = 64, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.N = N
        self.in_proj = nn.Linear(2 * N, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_mult * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        """B: (batch, N, N). Returns s: (batch, N)."""
        assert B.dim() == 3 and B.shape[-1] == B.shape[-2] == self.N, f"unexpected B shape {B.shape}"
        rows = B                    # (b, N, N) — rows: outgoing from v
        cols = B.transpose(-1, -2)  # (b, N, N) — cols: incoming to v
        x = torch.cat([rows, cols], dim=-1)  # (b, N, 2N)
        h = self.in_proj(x)                  # (b, N, d_model)
        h = self.encoder(h)                  # (b, N, d_model)
        s = self.head(h).squeeze(-1)         # (b, N)
        return s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_model.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/graph_transformer_readout.py \
        block_lo_arm_order_network/tests/test_neural_readout_model.py
git commit -m "feat(neural-readout): graph-transformer readout module (no positional embedding)"
```

---

## Task 7: `loss` — pairwise logistic with rank convention asserted

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/loss.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_loss.py`

- [ ] **Step 1: Write the failing tests**

```python
# block_lo_arm_order_network/tests/test_neural_readout_loss.py
import sys, pathlib, torch, math
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_loss_zero_when_scores_perfectly_anti_rank():
    """If s = -rank (so earliest has highest score), all pairs satisfy s_i > s_j
    when rank_i < rank_j. Loss should be near 0 with large score magnitudes."""
    from neural_readout.loss import pairwise_logistic_loss
    N = 64
    rank = torch.arange(N).unsqueeze(0).float()   # (1, N), rank 0..N-1
    s = -10.0 * rank                              # large negative gap
    loss = pairwise_logistic_loss(s, rank.long())
    assert loss.item() < 0.001, loss.item()


def test_loss_max_when_scores_inverted():
    """If s = +rank (so earliest has LOWEST score), every pair is wrong.
    Loss should be much larger than the random-init baseline log(2)."""
    from neural_readout.loss import pairwise_logistic_loss
    N = 64
    rank = torch.arange(N).unsqueeze(0).float()
    s = 10.0 * rank
    loss = pairwise_logistic_loss(s, rank.long())
    assert loss.item() > 5.0, loss.item()


def test_loss_random_baseline_near_log2():
    """Zero scores: every pair contributes -log sigma(0) = log 2 ≈ 0.693."""
    from neural_readout.loss import pairwise_logistic_loss
    N = 64
    rank = torch.arange(N).unsqueeze(0).long()
    s = torch.zeros(1, N)
    loss = pairwise_logistic_loss(s, rank)
    assert abs(loss.item() - math.log(2)) < 1e-4


def test_loss_gradient_pushes_correct_direction():
    """If rank_i < rank_j (i earlier), gradient should push s_i up and s_j down."""
    from neural_readout.loss import pairwise_logistic_loss
    s = torch.zeros(1, 4, requires_grad=True)
    rank = torch.tensor([[0, 1, 2, 3]])  # node 0 earliest -> should get highest score
    loss = pairwise_logistic_loss(s, rank)
    loss.backward()
    grad = s.grad[0]
    # Negative grad on node 0 (push s up), positive on node 3 (push s down)
    assert grad[0] < grad[1] < grad[2] < grad[3], grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_loss.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/loss.py`:

```python
"""Pairwise logistic loss with rank-0=earliest convention asserted (NR-1 §4.1).

For all pairs (i, j) with rank[i] < rank[j]  (i revealed earlier than j),
the model is rewarded for predicting s[i] > s[j]. Concretely:

    L = - (1 / |P|) * sum_{(i,j) in P} log sigma( s[i] - s[j] )
    P = { (i, j) : rank[i] < rank[j] }

All N*(N-1)/2 = 2016 directed pairs are used per graph.
"""
import torch
import torch.nn.functional as F


def pairwise_logistic_loss(scores: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """scores: (batch, N) float. rank: (batch, N) long. Returns scalar loss.

    Convention:
        rank[v] = position of v in reveal order; rank 0 = earliest.
        Earliest should get the HIGHEST score.
    """
    assert scores.dim() == 2 and rank.dim() == 2 and scores.shape == rank.shape, \
        f"shape mismatch: scores {scores.shape}, rank {rank.shape}"
    rank = rank.long()
    # mask[b, i, j] = True iff rank[b, i] < rank[b, j]  (i.e. i should beat j)
    mask = rank.unsqueeze(-1) < rank.unsqueeze(-2)        # (b, N, N)
    # logits[b, i, j] = s[b, i] - s[b, j]
    logits = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # (b, N, N)
    # -log sigma(logits) = softplus(-logits)
    per_pair = F.softplus(-logits)                        # (b, N, N)
    n_pairs = mask.sum(dim=(-1, -2)).clamp_min(1).float()
    loss = (per_pair * mask).sum(dim=(-1, -2)) / n_pairs  # (b,)
    return loss.mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_loss.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/loss.py \
        block_lo_arm_order_network/tests/test_neural_readout_loss.py
git commit -m "feat(neural-readout): pairwise logistic loss with rank-0=earliest convention"
```

---

## Task 8: `eval_metrics` — Kendall τ / pairwise acc / Spearman ρ

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/eval_metrics.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# block_lo_arm_order_network/tests/test_neural_readout_metrics.py
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_identity_metrics():
    from neural_readout.eval_metrics import compute_matching_metrics
    N = 64
    rank = np.arange(N)[None, :]
    s = -rank.astype(np.float32)  # earliest gets highest score
    out = compute_matching_metrics(scores=s, rank=rank)
    assert out["kendall_tau"] > 0.999
    assert out["pairwise_precedence_acc"] > 0.999
    assert out["spearman_rho"] > 0.999


def test_reversed_metrics():
    from neural_readout.eval_metrics import compute_matching_metrics
    N = 64
    rank = np.arange(N)[None, :]
    s = rank.astype(np.float32)  # earliest gets LOWEST -- fully inverted
    out = compute_matching_metrics(scores=s, rank=rank)
    assert out["kendall_tau"] < -0.999
    assert out["pairwise_precedence_acc"] < 0.001
    assert out["spearman_rho"] < -0.999


def test_random_metrics_near_zero():
    from neural_readout.eval_metrics import compute_matching_metrics
    rng = np.random.default_rng(0)
    M, N = 100, 64
    rank = np.stack([rng.permutation(N) for _ in range(M)])
    s = rng.standard_normal((M, N)).astype(np.float32)
    out = compute_matching_metrics(scores=s, rank=rank)
    assert abs(out["kendall_tau"]) < 0.05
    assert abs(out["pairwise_precedence_acc"] - 0.5) < 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_metrics.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/eval_metrics.py`:

```python
"""Attention-order matching metrics (NR-1 §5.1).

Pairwise precedence accuracy ≈ (1 + tau) / 2 numerically, but we report both
because they are the canonical names a reviewer expects.
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr


def _pred_rank(scores: np.ndarray) -> np.ndarray:
    """Convert per-node scores to predicted rank: highest score -> rank 0."""
    order = np.argsort(-scores, axis=-1)             # descending by score
    pred_rank = np.empty_like(order)
    M, N = order.shape
    for i in range(M):
        pred_rank[i, order[i]] = np.arange(N)
    return pred_rank


def compute_matching_metrics(scores: np.ndarray, rank: np.ndarray) -> dict:
    """scores: (M, N) float. rank: (M, N) int (rank 0 = earliest)."""
    assert scores.shape == rank.shape
    M, N = scores.shape
    pred_rank = _pred_rank(np.asarray(scores))

    taus, rhos, accs = [], [], []
    for i in range(M):
        # Compare predicted rank to teacher rank -- both rank-0 = earliest
        tau, _ = kendalltau(pred_rank[i], rank[i])
        rho, _ = spearmanr(pred_rank[i], rank[i])
        # Pairwise precedence accuracy
        true_lt = (rank[i][:, None] < rank[i][None, :])          # (N, N)
        pred_lt = (pred_rank[i][:, None] < pred_rank[i][None, :])
        mask = true_lt                                            # ordered pairs only
        acc = (pred_lt[mask] == True).mean() if mask.any() else 0.5
        taus.append(tau)
        rhos.append(rho)
        accs.append(acc)
    return {
        "kendall_tau": float(np.mean(taus)),
        "spearman_rho": float(np.mean(rhos)),
        "pairwise_precedence_acc": float(np.mean(accs)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_metrics.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/eval_metrics.py \
        block_lo_arm_order_network/tests/test_neural_readout_metrics.py
git commit -m "feat(neural-readout): kendall/pairwise/spearman matching metrics"
```

---

## Task 9: Overfit-tiny-batch sanity test

**Files:**
- Test: `block_lo_arm_order_network/tests/test_neural_readout_overfit.py`

This is a sanity test: with 4 graphs and an over-parameterized run, pairwise loss should drive to near zero and τ should hit ≈ 1. Confirms the model + loss + rank convention are wired correctly end-to-end. No new module — the test itself is the deliverable.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_neural_readout_overfit.py
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_overfit_tiny_batch():
    from neural_readout.graph_transformer_readout import GraphTransformerReadout
    from neural_readout.loss import pairwise_logistic_loss
    from neural_readout.eval_metrics import compute_matching_metrics

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    M, N = 4, 64

    # Synthetic chain B + matching rank, so teacher and student should agree.
    B_np = np.zeros((M, N, N), dtype=np.float32)
    rank_np = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        perm = rng.permutation(N)
        # Place chain so perm[k] is k-th revealed; out edge perm[k] -> perm[k+1].
        for k in range(N - 1):
            B_np[i, perm[k], perm[k+1]] = 1.0
        # rank[v] = k where v == perm[k]
        rank_np[i, perm] = np.arange(N)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = GraphTransformerReadout().to(device)
    opt = torch.optim.AdamW(g.parameters(), lr=3e-3)

    B = torch.from_numpy(B_np).to(device)
    rank = torch.from_numpy(rank_np).to(device)

    for step in range(300):
        s = g(B)
        loss = pairwise_logistic_loss(s, rank)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"final loss {loss.item():.4f} should overfit to near 0"

    with torch.no_grad():
        s = g(B).cpu().numpy()
    out = compute_matching_metrics(scores=s, rank=rank.cpu().numpy())
    assert out["kendall_tau"] > 0.95, out
```

- [ ] **Step 2: Run test to verify it fails (initially) — then implement**

Because the test depends only on already-implemented modules (Tasks 6, 7, 8), this should PASS without new code. If it does not, fix the module that breaks it before continuing.

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_overfit.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add block_lo_arm_order_network/tests/test_neural_readout_overfit.py
git commit -m "test(neural-readout): overfit-tiny-batch end-to-end sanity"
```

---

## Task 10: `train_nr1` — training script with no-NLL-selection assertion

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/train_nr1.py`
- Test: `block_lo_arm_order_network/tests/test_neural_readout_selection_policy.py`

**Selection policy (§8 of spec):** the training script must never import `eval_frozen_nll` during training. Test enforces this with a sentinel.

- [ ] **Step 1: Write the failing test**

```python
# block_lo_arm_order_network/tests/test_neural_readout_selection_policy.py
import sys, pathlib, importlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_train_nr1_does_not_import_frozen_nll():
    """Selection policy assertion: train_nr1 must not import eval_frozen_nll."""
    # Force fresh import
    for mod in list(sys.modules):
        if mod.startswith("neural_readout"):
            del sys.modules[mod]
    mod = importlib.import_module("neural_readout.train_nr1")

    leak = [k for k in sys.modules
            if k.startswith("neural_readout.eval_frozen_nll")]
    assert not leak, (
        "train_nr1 must NOT import eval_frozen_nll; selection by NLL is forbidden "
        "by NR-1 spec §8."
    )

    # The training entrypoint must exist and accept the expected args
    assert hasattr(mod, "train_nr1"), "train_nr1.train_nr1 entrypoint missing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_selection_policy.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/train_nr1.py`:

```python
"""NR-1 training loop (§8 of the spec).

Model selection policy: checkpoints and hyperparameters are selected ONLY by
attention-order matching metrics on same-ckpt val (Kendall tau, pairwise
precedence acc, Spearman rho). Frozen-theta NLL is NEVER imported here.
Any developer that adds an `import neural_readout.eval_frozen_nll` to this
file breaks `test_neural_readout_selection_policy.py`.
"""
import sys, pathlib, json, argparse, time
import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from neural_readout.dataset import load_dataset
from neural_readout.graph_transformer_readout import GraphTransformerReadout
from neural_readout.loss import pairwise_logistic_loss
from neural_readout.eval_metrics import compute_matching_metrics


def _split(B, sigma, rank, train_n, val_n):
    """Sequential split; dataset is already in a deterministic order from build_dataset."""
    return (
        (B[:train_n], sigma[:train_n], rank[:train_n]),
        (B[train_n:train_n + val_n], sigma[train_n:train_n + val_n], rank[train_n:train_n + val_n]),
        (B[train_n + val_n:], sigma[train_n + val_n:], rank[train_n + val_n:]),
    )


@torch.no_grad()
def evaluate(model, B, rank, device, batch=64):
    model.eval()
    M = B.shape[0]
    all_s = np.zeros((M, B.shape[1]), dtype=np.float32)
    for i in range(0, M, batch):
        Bt = torch.from_numpy(B[i:i+batch]).to(device)
        all_s[i:i+batch] = model(Bt).cpu().numpy()
    return compute_matching_metrics(scores=all_s, rank=rank)


def train_nr1(dataset_path: str, out_dir: str, train_n: int, val_n: int,
              epochs: int, batch: int, lr: float, device: str = "cuda:0",
              d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
              eval_every_epochs: int = 1):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    B, sigma, rank = load_dataset(dataset_path)
    (B_tr, _, r_tr), (B_va, _, r_va), _ = _split(B, sigma, rank, train_n, val_n)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GraphTransformerReadout(N=B.shape[1], d_model=d_model,
                                    n_heads=n_heads, n_layers=n_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    M_tr = B_tr.shape[0]
    log = []
    best_tau = -1.0
    best_path = out / "g_beta_best.pt"

    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(ep)
        perm = rng.permutation(M_tr)
        ep_loss = 0.0
        n_steps = 0
        for s_i in range(0, M_tr, batch):
            idx = perm[s_i:s_i + batch]
            Bt = torch.from_numpy(B_tr[idx]).to(dev)
            rt = torch.from_numpy(r_tr[idx]).to(dev).long()
            scores = model(Bt)
            loss = pairwise_logistic_loss(scores, rt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_steps += 1
        ep_loss /= max(n_steps, 1)

        if (ep + 1) % eval_every_epochs == 0 or ep == epochs - 1:
            metrics = evaluate(model, B_va, r_va, dev)
            log.append({"epoch": ep, "train_loss": ep_loss, **metrics})
            print(f"[ep {ep:3d}] loss={ep_loss:.4f}  tau={metrics['kendall_tau']:.4f}  "
                  f"pairwise={metrics['pairwise_precedence_acc']:.4f}  "
                  f"spearman={metrics['spearman_rho']:.4f}")
            # Selection by tau ONLY (NR-1 §8)
            if metrics["kendall_tau"] > best_tau:
                best_tau = metrics["kendall_tau"]
                torch.save({"model": model.state_dict(),
                            "metrics": metrics,
                            "epoch": ep}, best_path)

    with open(out / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return str(best_path), log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-n", type=int, required=True)
    p.add_argument("--val-n", type=int, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    best, _ = train_nr1(args.dataset, args.out_dir, args.train_n, args.val_n,
                        args.epochs, args.batch, args.lr, args.device)
    print(f"best ckpt: {best}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_selection_policy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/train_nr1.py \
        block_lo_arm_order_network/tests/test_neural_readout_selection_policy.py
git commit -m "feat(neural-readout): training loop with selection-by-tau-only policy"
```

---

## Task 11: `eval_frozen_nll` — diagnostic-only frozen-θ NLL

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/eval_frozen_nll.py`

This is reported AFTER `g_beta_best.pt` is selected by Task 10. The module asserts the checkpoint exists and refuses to run otherwise. No test needs to be written before implementation because the script's correctness is operational, not algorithmic — the algorithmic correctness lives in §11's definition ("NLL(σ̂) − NLL(σ_T) under frozen θ"). A smoke test verifies the script runs and produces a finite number.

- [ ] **Step 1: Write the failing smoke test**

```python
# block_lo_arm_order_network/tests/test_neural_readout_frozen_nll.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_frozen_nll_module_imports_and_exposes_entrypoint():
    """Smoke: module imports, has compute_frozen_nll_gap entrypoint."""
    import importlib
    mod = importlib.import_module("neural_readout.eval_frozen_nll")
    assert hasattr(mod, "compute_frozen_nll_gap"), \
        "eval_frozen_nll.compute_frozen_nll_gap entrypoint missing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_frozen_nll.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `block_lo_arm_order_network/neural_readout/eval_frozen_nll.py`:

```python
"""Frozen-theta NLL diagnostic (NR-1 §5.2 / §8 step 5).

NEVER imported by neural_readout/train_nr1.py — that is enforced by
tests/test_neural_readout_selection_policy.py. This script is run AFTER
the g_beta checkpoint is selected by attention-order matching metrics.

Reports NLL(sigma_hat) - NLL(sigma_T) over the held-out val batch. The number
is observational, not selective. Do not use it to choose epochs, learning
rates, dropout, or any other hyperparameter.
"""
import sys, pathlib, argparse, json
import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from train_clean_aogpt import (
    build_model, CleanPermutation, phys_to_model_idx_clean,
    expand_model_blocks_to_token_order,
)
from training_utils import load_train_chunks, SEQ_LEN, N as N_BLOCKS, BLOCK_LEN
from neural_readout.dataset import load_dataset
from neural_readout.graph_transformer_readout import GraphTransformerReadout


@torch.no_grad()
def _nll_under_orders(model, idx_model, orders_phys: np.ndarray, clean_perm, device,
                      block_len: int = BLOCK_LEN) -> float:
    """Mean per-token NLL on idx_model when revealing in the given physical orders.

    orders_phys: (B, N) physical-block reveal order per sample. We map to model
    blocks via clean_perm, expand to token order, and compute the model's loss.
    """
    block_phys_to_model = clean_perm.block_perm_phys_to_model.cpu().numpy()
    M = orders_phys.shape[0]
    losses = []
    for i in range(M):
        order_phys = orders_phys[i]
        order_model = block_phys_to_model[order_phys]   # (N,)
        order_model_t = torch.from_numpy(order_model.astype(np.int64)).unsqueeze(0)
        token_order = expand_model_blocks_to_token_order(order_model_t, block_len).to(device)
        tokens = idx_model[i:i+1].to(device)
        _, loss, _ = model.forward_fn(tokens, token_order, return_attentions=False)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def compute_frozen_nll_gap(ckpt_path: str, g_beta_path: str, dataset_path: str,
                           device: str = "cuda:0") -> dict:
    """NLL(sigma_hat) - NLL(sigma_T) under frozen theta. Diagnostic only."""
    assert pathlib.Path(g_beta_path).exists(), \
        f"g_beta checkpoint not found at {g_beta_path}; this is a post-selection " \
        f"diagnostic, run training first."

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN

    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_model(model_args).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load val portion: use the same dataset that produced sigma_T
    B, sigma, rank = load_dataset(dataset_path)  # sigma is sigma_T

    # Compute sigma_hat from g_beta
    g = GraphTransformerReadout(N=B.shape[1]).to(dev)
    state = torch.load(g_beta_path, map_location=dev, weights_only=False)
    g.load_state_dict(state["model"])
    g.eval()
    Bt = torch.from_numpy(B).to(dev)
    scores = g(Bt).cpu().numpy()
    sigma_hat = np.argsort(-scores, axis=-1)

    # We need the same wikitext chunks used to build the dataset; rebuild deterministically
    from neural_readout.extract_b import extract_per_sample_B  # noqa: F401
    idx_phys = load_train_chunks(n_chunks=None)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    eval_indices = np.asarray(protocol["eval_indices"], dtype=np.int64)
    idx_eval = idx_model[eval_indices]
    # The dataset was built with the same seed in build_dataset_from_ckpt:
    # we need to repeat that choice. Caller must pass dataset_path that was
    # built with extract_per_sample_B(seed=DATASET_SEED). Here we rely on the
    # convention used by Task 12/13: dataset M chunks correspond to the first
    # M elements of idx_eval after rng.choice(seed=DATASET_SEED).
    raise NotImplementedError(
        "Caller must supply chunk selection or refactor build_dataset_from_ckpt "
        "to save the chunk indices alongside the .npz. See Task 11 step 4."
    )
```

- [ ] **Step 4: Refactor dataset to persist chunk indices (so frozen NLL can reuse them)**

Update `block_lo_arm_order_network/neural_readout/dataset.py`:

```python
# Replace save_dataset / load_dataset / build_dataset_from_ckpt with versions
# that also store the `chunk_index` array (indices into idx_eval).
def save_dataset(path, B, sigma, rank, chunk_index):
    np.savez_compressed(path,
                        B=B.astype(np.float32),
                        sigma=sigma.astype(np.int64),
                        rank=rank.astype(np.int64),
                        chunk_index=chunk_index.astype(np.int64))

def load_dataset(path):
    data = np.load(path)
    return data["B"], data["sigma"], data["rank"], data["chunk_index"]

def build_dataset_from_ckpt(ckpt_path, M, seed, alpha_dep, out_path, device="cuda:0"):
    from neural_readout.extract_b import extract_per_sample_B_with_chunks
    from neural_readout.teacher_labels import generate_teacher_label
    B, chunk_index = extract_per_sample_B_with_chunks(ckpt_path, M=M, seed=seed, device=device)
    sigmas = np.zeros((M, 64), dtype=np.int64)
    ranks = np.zeros((M, 64), dtype=np.int64)
    for i in range(M):
        sigma, rank, _ = generate_teacher_label(B[i], alpha_dep=alpha_dep)
        sigmas[i], ranks[i] = sigma, rank
    save_dataset(out_path, B, sigmas, ranks, chunk_index)
    return out_path
```

Update `block_lo_arm_order_network/neural_readout/extract_b.py` to expose `extract_per_sample_B_with_chunks` returning `(B, chunk_index)`. Update the Task 2 test to use the new 2-tuple.

Update `compute_frozen_nll_gap` in `eval_frozen_nll.py` to:

```python
    B, sigma_T, rank, chunk_index = load_dataset(dataset_path)
    # ... compute sigma_hat as before ...
    idx_eval = idx_model[np.asarray(protocol["eval_indices"], dtype=np.int64)]
    chunks = idx_eval[chunk_index]

    # NLL under teacher and predicted orders. Reveal order in this codebase is
    # specified in PHYSICAL block frame; sigma_T and sigma_hat are physical
    # block sequences by construction (extract_per_sample_B remaps).
    nll_T   = _nll_under_orders(model, chunks, sigma_T, clean_perm, dev)
    nll_hat = _nll_under_orders(model, chunks, sigma_hat, clean_perm, dev)
    return {"nll_teacher": nll_T, "nll_student": nll_hat,
            "nll_gap": nll_hat - nll_T}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--g-beta", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    res = compute_frozen_nll_gap(args.ckpt, args.g_beta, args.dataset, args.device)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all neural_readout tests to verify nothing broke**

Run: `pytest block_lo_arm_order_network/tests/test_neural_readout_*.py -v`
Expected: every test PASSES. If `test_neural_readout_extract_b.py` or `test_neural_readout_dataset.py` fails because of the new chunk-index argument, update those tests to the new 2-tuple / 4-tuple signatures.

- [ ] **Step 6: Commit**

```bash
git add block_lo_arm_order_network/neural_readout/eval_frozen_nll.py \
        block_lo_arm_order_network/neural_readout/extract_b.py \
        block_lo_arm_order_network/neural_readout/dataset.py \
        block_lo_arm_order_network/tests/test_neural_readout_frozen_nll.py \
        block_lo_arm_order_network/tests/test_neural_readout_extract_b.py \
        block_lo_arm_order_network/tests/test_neural_readout_dataset.py
git commit -m "feat(neural-readout): frozen-theta NLL diagnostic (post-selection only)"
```

---

## Task 12: Smoke 1k dataset + smoke training run

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/scripts/build_smoke_1k.py`
- Output: `block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.npz`
- Output: `block_lo_arm_order_network/neural_readout/checkpoints/smoke_1k/g_beta_best.pt`

This task verifies the full pipeline works on small data before spending compute on full 10k.

- [ ] **Step 1: Build the smoke dataset**

```bash
mkdir -p block_lo_arm_order_network/neural_readout/data \
         block_lo_arm_order_network/neural_readout/checkpoints/smoke_1k
python -c "
import sys; sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import build_dataset_from_ckpt
out = build_dataset_from_ckpt(
    ckpt_path='block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt',
    M=1000, seed=42, alpha_dep=0.5,
    out_path='block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.npz',
    device='cuda:0',
)
print('saved', out)
"
```

Expected: prints `saved .../text_5k_smoke_1k.npz`.

- [ ] **Step 2: Compute and record teacher diversity**

```bash
python -c "
import sys; sys.path.insert(0, 'block_lo_arm_order_network')
import json
from neural_readout.dataset import load_dataset
from neural_readout.diversity_stats import teacher_diversity
_, sigma, _, _ = load_dataset('block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.npz')
stats = teacher_diversity(sigma)
print(json.dumps(stats, indent=2))
"
```

Expected: a dict with `unique_sigma_count`, `first_node_entropy`, `distinct_first3_prefix_count`, `mean_pairwise_tau`. Record these in a notes file `block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.diversity.json`.

- [ ] **Step 3: Train smoke**

```bash
python -m neural_readout.train_nr1 \
  --dataset block_lo_arm_order_network/neural_readout/data/text_5k_smoke_1k.npz \
  --out-dir block_lo_arm_order_network/neural_readout/checkpoints/smoke_1k \
  --train-n 800 --val-n 100 --epochs 30 --batch 32 --lr 3e-4 \
  --device cuda:0
```

Expected: training prints decreasing loss; the final `tau` printed on val is > 0.5 (smoke threshold — far below the 0.80 hard gate, just confirming the model can move).

- [ ] **Step 4: Sanity gate**

If smoke val τ ≤ 0.5 after 30 epochs:
  - First check the diversity diagnostic — if `mean_pairwise_tau ≥ 0.9`, the teacher is degenerate and even a high student τ would not validate anything. Stop and re-examine teacher diversity in §5.1b.
  - Otherwise, lower lr to 1e-4 or raise epochs to 60 and rerun.
  - Do NOT loosen the hard gates in the spec.

If smoke passes (val τ > 0.5), proceed.

- [ ] **Step 5: Commit smoke artifacts (the runs themselves, not the .npz)**

`.npz` and `.pt` files should be in `.gitignore` already (data + checkpoints are not committed). If not, do **not** add them. Commit only the script and a markdown notes file.

```bash
git add block_lo_arm_order_network/neural_readout/scripts/build_smoke_1k.py  # if you extracted Step 1 into a script
git commit -m "run(neural-readout): smoke 1k dataset built; baseline val tau recorded"
```

---

## Task 13: Full 10k dataset + full training run + report against §5.1 gates

**Files:**
- Output: `block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz`
- Output: `block_lo_arm_order_network/neural_readout/checkpoints/full_10k/g_beta_best.pt`

- [ ] **Step 1: Build the full dataset**

```bash
python -c "
import sys; sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import build_dataset_from_ckpt
build_dataset_from_ckpt(
    ckpt_path='block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt',
    M=10000, seed=42, alpha_dep=0.5,
    out_path='block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz',
    device='cuda:0',
)
"
```

- [ ] **Step 2: Compute and persist full diversity stats**

```bash
python -c "
import sys; sys.path.insert(0, 'block_lo_arm_order_network')
import json
from neural_readout.dataset import load_dataset
from neural_readout.diversity_stats import teacher_diversity
_, sigma, _, _ = load_dataset('block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz')
print(json.dumps(teacher_diversity(sigma), indent=2))
" > block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.diversity.json
```

Expected: JSON file present, content matches the §5.1b schema.

- [ ] **Step 3: Train full**

```bash
python -m neural_readout.train_nr1 \
  --dataset block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz \
  --out-dir block_lo_arm_order_network/neural_readout/checkpoints/full_10k \
  --train-n 8000 --val-n 1000 --epochs 40 --batch 64 --lr 3e-4 \
  --device cuda:0
```

Expected: best val τ printed at end of training. Read `block_lo_arm_order_network/neural_readout/checkpoints/full_10k/g_beta_best.pt` -- its embedded `metrics` is the selected snapshot.

- [ ] **Step 4: Apply §5.1 hard gates**

Check three metrics on the same-ckpt held-out val:

| Metric | Hard gate |
|---|---|
| Kendall τ | ≥ 0.80 |
| Pairwise precedence acc | ≥ 0.90 |
| Spearman ρ | ≥ 0.85 |

All three must pass. If any one fails, follow §10 of the spec ("If gates fail: reduce scope ... do not redefine the gates"). Do not retrain to chase NLL.

- [ ] **Step 5: Run frozen-θ NLL diagnostic (post-selection, not a gate)**

```bash
python -m neural_readout.eval_frozen_nll \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt \
  --g-beta block_lo_arm_order_network/neural_readout/checkpoints/full_10k/g_beta_best.pt \
  --dataset block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz \
  --out block_lo_arm_order_network/neural_readout/checkpoints/full_10k/frozen_nll_diag.json
```

Expected: JSON with `nll_teacher`, `nll_student`, `nll_gap`. Record but do not select on it.

- [ ] **Step 6: Commit notes (not data, not ckpts)**

```bash
git add block_lo_arm_order_network/neural_readout/checkpoints/full_10k/.gitkeep \
        block_lo_arm_order_network/neural_readout/data/.gitkeep
# create the .gitkeeps if needed; do not commit .npz or .pt
git commit -m "run(neural-readout): full 10k run; hard gates reported in next task"
```

---

## Task 14: Four MVP ablations on the full 10k dataset

**Files:**
- Create: `block_lo_arm_order_network/neural_readout/ablation_inputs.py`
- Output: `block_lo_arm_order_network/neural_readout/checkpoints/ablation_<name>/g_beta_best.pt` × 4

Ablation inputs (§6 of the spec):

| Name | Transform on B at training time |
|---|---|
| reverse | `B.transpose(-1, -2)` |
| sym | `0.5 * (B + B.T)` per sample |
| row_shuffle | for each sample i, apply a fixed random permutation to `B[i, :, :]` rows (B[i, π, :]) |
| b_global | replace per-sample `B[i]` with the global mean `B.mean(axis=0)` for ALL i |

For each, the **teacher labels** (σ_T, rank) come from the original (unmodified) `B[i]`. Only the **model's input** changes — that is what defines the ablation.

- [ ] **Step 1: Create `ablation_inputs.py`**

```python
"""Four input transforms for the NR-1 MVP ablations (§6 of the spec).

Each transform is applied to B at training time. Teacher labels (sigma_T,
rank) are always generated from the ORIGINAL B and are NOT changed by the
ablation: the ablation studies whether the student can still recover the
teacher's order when its input is degraded in a specific way.
"""
import numpy as np


def identity(B: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    return B


def reverse(B: np.ndarray, rng=None) -> np.ndarray:
    return np.transpose(B, (0, 2, 1)).copy()


def sym(B: np.ndarray, rng=None) -> np.ndarray:
    return 0.5 * (B + np.transpose(B, (0, 2, 1)))


def row_shuffle(B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(B)
    M, N, _ = B.shape
    perm = rng.permutation(N)        # fixed permutation across all samples
    out = B[:, perm, :]
    return out


def b_global(B: np.ndarray, rng=None) -> np.ndarray:
    g = B.mean(axis=0, keepdims=True)
    return np.broadcast_to(g, B.shape).copy()


TRANSFORMS = {
    "reverse": reverse,
    "sym": sym,
    "row_shuffle": row_shuffle,
    "b_global": b_global,
}
```

- [ ] **Step 2: Add a CLI flag to `train_nr1.py` for the input transform**

In `train_nr1.py`, add to `argparse`:

```python
    p.add_argument("--ablation", default="identity",
                   choices=["identity", "reverse", "sym", "row_shuffle", "b_global"])
```

In `train_nr1`, after `B, sigma, rank = load_dataset(...)`, apply the transform:

```python
    from neural_readout.ablation_inputs import TRANSFORMS, identity
    rng_ab = np.random.default_rng(0)
    fn = TRANSFORMS.get(ablation, identity)
    B = fn(B, rng_ab)
```

(Pass `ablation=args.ablation` from `main()`.)

- [ ] **Step 3: Run the four ablations**

```bash
for AB in reverse sym row_shuffle b_global; do
  python -m neural_readout.train_nr1 \
    --dataset block_lo_arm_order_network/neural_readout/data/text_5k_full_10k.npz \
    --out-dir block_lo_arm_order_network/neural_readout/checkpoints/ablation_$AB \
    --train-n 8000 --val-n 1000 --epochs 40 --batch 64 --lr 3e-4 \
    --device cuda:0 \
    --ablation $AB
done
```

- [ ] **Step 4: Record each ablation's final val (τ, pairwise acc, ρ)**

Read the embedded `metrics` from each `ablation_<name>/g_beta_best.pt`. Save as:

```
block_lo_arm_order_network/neural_readout/checkpoints/ablation_summary.json
```

Shape:
```json
{
  "identity":    {"kendall_tau": 0.xx, "pairwise_precedence_acc": 0.xx, "spearman_rho": 0.xx},
  "reverse":     {...},
  "sym":         {...},
  "row_shuffle": {...},
  "b_global":    {...}
}
```

- [ ] **Step 5: Commit script + summary**

```bash
git add block_lo_arm_order_network/neural_readout/ablation_inputs.py \
        block_lo_arm_order_network/neural_readout/train_nr1.py
git commit -m "feat(neural-readout): four MVP ablations on input transforms"
```

---

## Task 15: Cross-ckpt diagnostic (NR-4 generalization, NOT NR-1 gate)

- [ ] **Step 1: Build 1k datasets for each cross-ckpt**

```bash
for STEP in 10000 20000 30000 40000 50000 60000; do
  python -c "
import sys; sys.path.insert(0, 'block_lo_arm_order_network')
from neural_readout.dataset import build_dataset_from_ckpt
build_dataset_from_ckpt(
    ckpt_path='block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step${STEP}.pt',
    M=1000, seed=42, alpha_dep=0.5,
    out_path='block_lo_arm_order_network/neural_readout/data/text_cross_${STEP}_1k.npz',
    device='cuda:0',
)
"
done
```

- [ ] **Step 2: Evaluate frozen g_β on each cross-ckpt dataset**

```python
# block_lo_arm_order_network/neural_readout/scripts/cross_ckpt_eval.py
import sys, pathlib, json, numpy as np, torch
sys.path.insert(0, "block_lo_arm_order_network")
from neural_readout.dataset import load_dataset
from neural_readout.graph_transformer_readout import GraphTransformerReadout
from neural_readout.eval_metrics import compute_matching_metrics
from neural_readout.diversity_stats import teacher_diversity

g = GraphTransformerReadout(N=64).cuda()
state = torch.load("block_lo_arm_order_network/neural_readout/checkpoints/full_10k/g_beta_best.pt",
                   map_location="cuda", weights_only=False)
g.load_state_dict(state["model"]); g.eval()

results = {}
for step in [10000, 20000, 30000, 40000, 50000, 60000]:
    path = f"block_lo_arm_order_network/neural_readout/data/text_cross_{step}_1k.npz"
    B, sigma, rank, _chunks = load_dataset(path)
    with torch.no_grad():
        s = g(torch.from_numpy(B).cuda()).cpu().numpy()
    results[str(step)] = {
        "metrics": compute_matching_metrics(scores=s, rank=rank),
        "teacher_diversity": teacher_diversity(sigma),
    }
out = "block_lo_arm_order_network/neural_readout/checkpoints/full_10k/cross_ckpt_eval.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", out)
```

Run it:

```bash
python block_lo_arm_order_network/neural_readout/scripts/cross_ckpt_eval.py
```

- [ ] **Step 3: Commit script (not data)**

```bash
git add block_lo_arm_order_network/neural_readout/scripts/cross_ckpt_eval.py
git commit -m "run(neural-readout): cross-ckpt diagnostic (NR-4, not NR-1 gate)"
```

---

## Task 16: Final report

**Files:**
- Create: `analyses/neural_readout_nr1_2026-05-28/REPORT.md`
- Create: `block_lo_arm_order_network/neural_readout/report.py` (helper that aggregates)

- [ ] **Step 1: Aggregate everything into one report file**

`analyses/neural_readout_nr1_2026-05-28/REPORT.md` MUST contain, in this order:

1. **Hard gates (§5.1)** — single table with τ / pairwise acc / Spearman ρ on the full-10k same-ckpt held-out val. Each cell labeled PASS / FAIL against the gate threshold.
2. **Teacher diversity (§5.1b)** — table with `unique_sigma_count`, `first_node_entropy`, `distinct_first3_prefix_count`, `mean_pairwise_tau` for the training dataset. Two-row interpretation block (low-diversity vs high-diversity readings from §5.1b) selecting whichever applies.
3. **Diagnostic metrics (§5.2)** — top-1 first-node match, first-3 set match, frozen-θ NLL gap. Each explicitly tagged `(diagnostic, not a gate)`.
4. **Cross-ckpt diagnostic (§5.2 / NR-4)** — table for steps 10k/20k/30k/40k/50k/60k. Tagged `(diagnostic, not NR-1 gate)`.
5. **MVP ablations (§6)** — 4-row table (reverse / sym / row_shuffle / b_global) with each ablation's val τ; brief 1-line interpretation per row using the "Expected (text)" column of §6 as the rubric.
6. **Interpretation (§9 / §10)** — apply the failure-mode table; pick the single sentence from §10 ("If gates pass" / "If gates pass but NLL gap large" / "If gates fail") that matches the outcome. Quote it verbatim.

- [ ] **Step 2: Commit the report**

```bash
git add analyses/neural_readout_nr1_2026-05-28/REPORT.md \
        block_lo_arm_order_network/neural_readout/report.py
git commit -m "report(neural-readout): NR-1 final report — gates, diagnostics, ablations, cross-ckpt"
```

---

## Self-Review

**Spec coverage check** (against
`docs/superpowers/specs/2026-05-28-neural-attention-to-order-readout-nr1-design.md`):

| Spec section | Task(s) | Covered? |
|---|---|---|
| §1 supervision boundary | 10 (no-NLL-import test), 11 (post-selection only) | yes |
| §2.1 substrate path | 12, 13 (use the exact ckpt path) | yes |
| §2.2 per-sample B + seeding bugfix | 1 (seeding), 2 (B = A^T) | yes |
| §2.3 CDL-source-start teacher with α=0.5 | 3 | yes |
| §2.4 smoke 1k → full 10k two-tier | 12, 13 | yes |
| §3 architecture, no PE | 6 | yes |
| §4 pairwise logistic with rank convention | 7 | yes |
| §5.1 three hard gates | 8, 10, 13 step 4 | yes |
| §5.1b teacher diversity | 5, 12 step 2, 13 step 2, 16 step 1 | yes |
| §5.2 diagnostic-only metrics | 8 (top-1/first-3 implied via metrics), 11, 13 step 5, 15 | partial — top-1 / first-3 not in `eval_metrics.py`; add as follow-up |
| §6 four MVP ablations | 14 | yes |
| §7 out-of-scope discipline | enforced by 10 (selection policy test) | yes |
| §8 model selection policy | 10 | yes |
| §9 failure modes | 16 step 1 §6 (interpretation block) | yes |
| §10 honest claims | 16 step 1 §6 (verbatim quote) | yes |

**Gap fix:** `eval_metrics.py` (Task 8) only computes τ / pairwise acc / Spearman. Top-1 first-node match and first-3 set match are §5.2 diagnostics. Add to Task 8's `compute_matching_metrics` two extra keys:

```python
top1 = float(np.mean([(np.argmax(-scores[i]) == np.argmin(rank[i])) for i in range(M)]))
first3 = float(np.mean([
    len(set(np.argsort(-scores[i])[:3].tolist()) & set(np.argsort(rank[i])[:3].tolist())) / 3
    for i in range(M)
]))
return {..., "top1_first_node_match": top1, "first3_set_match": first3}
```

Add the corresponding test in `test_neural_readout_metrics.py`:

```python
def test_top1_first3_extremes():
    from neural_readout.eval_metrics import compute_matching_metrics
    N = 64
    rank = np.arange(N)[None, :]
    s = -rank.astype(np.float32)
    out = compute_matching_metrics(scores=s, rank=rank)
    assert out["top1_first_node_match"] == 1.0
    assert out["first3_set_match"] == 1.0
```

**Placeholder scan:**
- No "TBD" / "implement later" remain.
- One "see Task 11 step 4" reference inside Task 11 itself is resolved by step 4 directly below it; not a forward reference.

**Type consistency:**
- `load_dataset` returns 4-tuple `(B, sigma, rank, chunk_index)` consistently after Task 11. Task 12-16 callers all unpack 4-tuple.
- `compute_matching_metrics` keys are `kendall_tau`, `pairwise_precedence_acc`, `spearman_rho`, `top1_first_node_match`, `first3_set_match` (added in self-review fix above).
- `GraphTransformerReadout` constructor signature is consistent across tasks 6, 9, 10, 11, 15.

---

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-05-28-neural-attention-to-order-readout-nr1.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
