# V3 Frozen-gβ MLP Order-Policy Port into chenhe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run our frozen, CDL-pretrained MLP (gβ) as a first-class order policy inside chenhe's mainline trainer (`chenhe_rerun/train.py`, seq256/permute/block64), same-table comparable with chenhe AR/Random.

**Architecture:** Port the validated V3 math (strict65 B-extraction + `L0DynamicGBeta` + CDL teacher) as a self-contained chenhe-local package `orderhead_v3/`; an offline Stage-1 producer CDL-pretrains gβ from a chenhe parent ckpt; a batch-mean frozen provider plugs into chenhe's order dispatch. No PG (Stage 3 deferred).

**Tech Stack:** Python 3.8, PyTorch 2.4 (chenhe `X1` env: `/data/users/chenhe/conda_envs/X1/bin/python`), numpy, chenhe `AOGPT` (`AOGPT_block.py`), wandb.

## Global Constraints

- **Env:** run everything with `/data/users/chenhe/conda_envs/X1/bin/python`; CWD `chenhe_rerun/` (relative `data/`, `config/` paths).
- **Layout (hard-fail off it):** `seq256/permute/block64` → `SEQ_LEN=256, N=64, BLOCK_LEN=4, HEADS=8, STRICT_NODES=65`, `permute_seed=42`.
- **gβ frozen throughout:** `requires_grad=False`, never in optimizer, forward under `@torch.no_grad()`.
- **Batch-mean B only (train AND eval):** B = mean over (batch samples × `batch_mean_probes`) → ONE σ per batch/refresh, broadcast to all samples. NEVER per-sample gβ.
- **Frame:** model-frame block indices `[0,63]` throughout; NO `inverse_block_perm`/physical remap in training path or gβ supervision target. chenhe `inverse_block_perm` only inside `val_origin_l2r_loss`.
- **Mode parity:** Stage-1 producer and Stage-2 provider use the same backbone mode for probes; probes always `@torch.no_grad()`. (`dropout=0` ⇒ jitter-free.)
- **Metric:** headline = `val_origin_l2r_loss`; `val` = diagnostic only.
- **Branch:** `feat/v3-frozen-gbeta-chenhe-port`. Artifacts (`out/ wandb/ *.log *.pt`) are already git-ignored by `chenhe_rerun/.gitignore`.
- **Spec:** `docs/superpowers/specs/2026-07-02-v3-frozen-mlp-port-into-chenhe-design.md`.

**Porting rule (all Part-A modules):** *function-level extraction*, not file copy. Copy only the named functions/classes, keep pure (numpy/torch) helpers they need, and **drop** every admin import not required by those functions (esp. `neural_readout.*`, `_load_model_and_chunks`, `training_utils`, `clean_training_protocol`, `batch_readout.diversity_batch`). Rewrite remaining intra-package imports to `orderhead_v3`-local. Every public entry point calls `assert_layout(...)` first.

---

## File Structure

```
chenhe_rerun/
├── orderhead_v3/
│   ├── __init__.py               # exports; nothing heavy at import time
│   ├── constants.py              # SEQ_LEN=256,N=64,BLOCK_LEN=4,HEADS=8,STRICT_NODES=65,PERMUTE_SEED=42 + assert_layout()
│   ├── attn_order_teacher.py     # port: MODES, teacher_scores (+ pure helpers)
│   ├── none_separated_block_graph.py  # port: build_none_separated_B
│   ├── per_head_order_scan.py    # port: _attn_to_A_block_loss_aligned_with_none_model_vec ONLY
│   ├── l0_strict65.py            # port: build_model_frame_strict65 (+ batch_mean_heads)
│   ├── l0_dynamic_gbeta.py       # port: L0DynamicGBeta, normalize_strict65, feature builders
│   ├── soft_pairwise.py          # port: soft_pairwise_bce_loss (+ used losses)
│   ├── cdl_teacher.py            # port: build_dynamic_teacher (model-frame target)
│   └── gbeta_provider.py         # NEW: batch-mean frozen provider (Task 4)
├── gbeta_cdl_pretrain.py         # NEW: Stage-1 offline producer (Task 3)
├── config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py  # NEW (Task 5)
├── train.py                      # MODIFY: dispatch branch + flags (Task 4)
└── tests/
    ├── test_orderhead_v3_strict65.py   # Task 1
    ├── test_orderhead_v3_gbeta.py      # Task 2
    ├── test_gbeta_cdl_pretrain.py      # Task 3
    ├── test_gbeta_provider.py          # Task 4
    └── test_gbeta_frozen_e2e.py        # Task 5
```

Admin source roots (read-only reference): `block_lo_arm_order_network/` and `.../batch_readout/`.

---

### Task 1: strict65 B-extraction chain (Part A, half 1)

**Files:**
- Create: `chenhe_rerun/orderhead_v3/__init__.py`, `constants.py`, `attn_order_teacher.py`, `none_separated_block_graph.py`, `per_head_order_scan.py`, `l0_strict65.py`
- Source: `block_lo_arm_order_network/{attn_order_teacher,none_separated_block_graph,per_head_order_scan}.py`, `.../batch_readout/l0_strict65.py`
- Test: `chenhe_rerun/tests/test_orderhead_v3_strict65.py`

**Interfaces:**
- Produces:
  - `constants.assert_layout(num_blocks:int, block_len:int, n_head:int) -> None` (raises `ValueError` off block64)
  - `l0_strict65.build_model_frame_strict65(attn_l0: np.ndarray[S,H,257,257], probe_orders: np.ndarray[S,256]) -> np.ndarray[S,H,65,65]`

- [ ] **Step 1: Write `constants.py`**

```python
SEQ_LEN = 256
N = 64
BLOCK_LEN = 4
HEADS = 8
STRICT_NODES = 65  # 64 content + 1 None
PERMUTE_SEED = 42


def assert_layout(num_blocks: int, block_len: int, n_head: int) -> None:
    if (int(num_blocks), int(block_len), int(n_head)) != (N, BLOCK_LEN, HEADS):
        raise ValueError(
            f"orderhead_v3 supports only seq256/block64 "
            f"(N={N}, block_len={BLOCK_LEN}, heads={HEADS}); "
            f"got num_blocks={num_blocks}, block_len={block_len}, n_head={n_head}."
        )
```

- [ ] **Step 2: Port `attn_order_teacher.py` and `none_separated_block_graph.py`**

Copy `MODES`, `teacher_scores` (and any pure helpers they call) from `block_lo_arm_order_network/attn_order_teacher.py` into `orderhead_v3/attn_order_teacher.py`. Copy `build_none_separated_B` from `block_lo_arm_order_network/none_separated_block_graph.py` into `orderhead_v3/none_separated_block_graph.py`; rewrite its imports `from attn_order_teacher import MODES, teacher_scores` → `from orderhead_v3.attn_order_teacher import MODES, teacher_scores`. Drop any import not used by the copied functions.

- [ ] **Step 3: Port `per_head_order_scan.py` (function-level)**

Into `orderhead_v3/per_head_order_scan.py`, copy ONLY `_attn_to_A_block_loss_aligned_with_none_model_vec` and the pure numpy helpers it calls. **Drop** all of: `from training_utils import ...`, `from clean_training_protocol import ...`, `from neural_readout.extract_b import _load_model_and_chunks`, `from neural_readout.teacher_labels import ...`, `from batch_readout.diversity_batch import ...`, and the `sys/pathlib` path hacks. If the function needs `SEQ_LEN/N/BLOCK_LEN`, import them `from orderhead_v3.constants import SEQ_LEN, N, BLOCK_LEN`.

- [ ] **Step 4: Port `l0_strict65.py`**

Copy `build_model_frame_strict65` and `batch_mean_heads` from `.../batch_readout/l0_strict65.py` into `orderhead_v3/l0_strict65.py`. Rewrite imports:
`from none_separated_block_graph import build_none_separated_B` → `from orderhead_v3.none_separated_block_graph import build_none_separated_B`;
`from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec` → `from orderhead_v3.per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec`.
Add at the top of `build_model_frame_strict65`, after shape checks: `from orderhead_v3.constants import assert_layout; assert_layout(64, 4, n_heads)`.

- [ ] **Step 5: Write `__init__.py`**

```python
from orderhead_v3.l0_strict65 import build_model_frame_strict65, batch_mean_heads  # noqa: F401
```

- [ ] **Step 6: Write the failing tests**

`chenhe_rerun/tests/test_orderhead_v3_strict65.py`:

```python
import sys, os
import numpy as np
import pytest

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from orderhead_v3.l0_strict65 import build_model_frame_strict65
from orderhead_v3.constants import assert_layout


def _fake_attn(S=2, H=8, T=256, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((S, H, T + 1, T + 1)).astype(np.float32)
    a /= a.sum(-1, keepdims=True)              # row-stochastic like softmax
    probe = np.stack([rng.permutation(T) for _ in range(S)]).astype(np.int64)
    return a, probe


def test_shape_and_wellformed():
    a, probe = _fake_attn()
    B = build_model_frame_strict65(a, probe)
    assert B.shape == (2, 8, 65, 65)
    assert np.isfinite(B).all()
    assert np.allclose(B[:, :, :, 0], 0.0)                 # no edges INTO None
    diag = B[:, :, np.arange(65), np.arange(65)]
    assert np.allclose(diag, 0.0)                          # zero diagonal


def test_frame_index_semantics():
    # Build attention where, in reveal order, block j attends only to block j-1.
    # After un-shuffling by the probe, strict65 content edge must land on the
    # SAME model-frame block index (i -> i), proving no inv_perm mismatch.
    S, H, T, BL, NB = 1, 8, 256, 4, 64
    probe = np.arange(T)[None, :].astype(np.int64)         # identity reveal
    a = np.zeros((S, H, T + 1, T + 1), np.float32)
    for t in range(T):
        a[:, :, t + 1, t] = 1.0                            # token t attends token t-1 (+None offset)
    a[:, :, 0, 0] = 1.0
    B = build_model_frame_strict65(a, probe)
    # content block b (node b+1) should have its dominant incoming edge from block b-1 (node b)
    for b in range(1, NB):
        row = B[0, 0, b + 1, 1:]                            # content->content incoming for block b
        assert row.argmax() == (b - 1), (b, row.argmax())


def test_layout_guard():
    with pytest.raises(ValueError):
        assert_layout(96, 4, 8)                             # block96 must fail
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd chenhe_rerun && /data/users/chenhe/conda_envs/X1/bin/python -m pytest tests/test_orderhead_v3_strict65.py -v`
Expected: `test_shape_and_wellformed` PASS, `test_layout_guard` PASS. If `test_frame_index_semantics` fails, the un-shuffle convention differs — fix `per_head_order_scan` alignment before proceeding (this is the port-correctness gate; do NOT skip).

- [ ] **Step 8: Commit**

```bash
git add chenhe_rerun/orderhead_v3/{__init__.py,constants.py,attn_order_teacher.py,none_separated_block_graph.py,per_head_order_scan.py,l0_strict65.py} chenhe_rerun/tests/test_orderhead_v3_strict65.py
git commit -m "feat(orderhead_v3): port strict65 B-extraction chain + frame-index test"
```

---

### Task 2: gβ model + losses + CDL teacher (Part A, half 2)

**Files:**
- Create: `chenhe_rerun/orderhead_v3/l0_dynamic_gbeta.py`, `soft_pairwise.py`, `cdl_teacher.py`
- Source: `.../batch_readout/{l0_dynamic_gbeta,soft_pairwise,label_free_cdl_teacher}.py`
- Test: `chenhe_rerun/tests/test_orderhead_v3_gbeta.py`

**Interfaces:**
- Consumes: `orderhead_v3.l0_strict65.build_model_frame_strict65`
- Produces:
  - `l0_dynamic_gbeta.L0DynamicGBeta(heads=8, nodes=65, ...)`; `model(B: Tensor[Bsz,8,65,65], apply_head_dropout=False) -> (scores: Tensor[Bsz,64], aux)`
  - `cdl_teacher.build_dynamic_teacher(...) -> dict` with keys `teacher_pairwise`, `teacher_consensus_order`, `teacher_ranks`, `teacher_weights` (model-frame `[0,63]`)
  - `soft_pairwise.soft_pairwise_bce_loss(scores, teacher_pairwise, weights=None) -> Tensor`

- [ ] **Step 1: Port `l0_dynamic_gbeta.py`**

Copy `L0DynamicGBeta`, `normalize_strict65`, and the feature-builder functions/classes (`SharedBlockScorer`, `SharedDynamicGate`, the content-feature builder) from `.../batch_readout/l0_dynamic_gbeta.py` verbatim (pure torch). No admin imports expected; if any, rewrite to `orderhead_v3`-local. Keep the `nodes=65` / `520`-dim feature assumptions.

- [ ] **Step 2: Port `soft_pairwise.py` and `cdl_teacher.py`**

Copy `soft_pairwise_bce_loss` (and any sibling losses referenced by training) into `orderhead_v3/soft_pairwise.py`. Copy `build_dynamic_teacher` into `orderhead_v3/cdl_teacher.py`; rewrite `from attn_order_teacher import teacher_scores` → `from orderhead_v3.attn_order_teacher import teacher_scores`. Keep `from scipy.stats import kendalltau`.

- [ ] **Step 3: Extend `__init__.py`**

```python
from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta, normalize_strict65  # noqa: F401
from orderhead_v3.cdl_teacher import build_dynamic_teacher  # noqa: F401
from orderhead_v3.soft_pairwise import soft_pairwise_bce_loss  # noqa: F401
```

- [ ] **Step 4: Write the failing tests**

`chenhe_rerun/tests/test_orderhead_v3_gbeta.py`:

```python
import sys, os
import numpy as np
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.cdl_teacher import build_dynamic_teacher


def test_gbeta_forward_shape():
    B = torch.rand(3, 8, 65, 65)
    gb = L0DynamicGBeta(heads=8, nodes=65).eval()
    with torch.no_grad():
        scores, _aux = gb(B, apply_head_dropout=False)
    assert scores.shape == (3, 64)
    assert torch.isfinite(scores).all()
    sigma = scores.argsort(dim=1, descending=True)
    assert all(sorted(sigma[i].tolist()) == list(range(64)) for i in range(3))


def test_teacher_is_model_frame():
    # build_dynamic_teacher must emit orders/ranks over model-frame [0,63];
    # no inverse_block_perm applied. Feed a small batch of strict65 B and assert
    # the consensus order is a permutation of range(64) (indices, not remapped).
    rng = np.random.default_rng(0)
    B = rng.random((16, 8, 65, 65)).astype(np.float32)
    teacher = build_dynamic_teacher(B)  # adapt kwargs to the ported signature
    order = np.asarray(teacher["teacher_consensus_order"]).reshape(-1)
    assert sorted(order.tolist()) == list(range(64))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd chenhe_rerun && /data/users/chenhe/conda_envs/X1/bin/python -m pytest tests/test_orderhead_v3_gbeta.py -v`
Expected: both PASS. (Adjust `build_dynamic_teacher` kwargs in the test to the ported signature; the assertion — model-frame `[0,63]` permutation — must hold.)

- [ ] **Step 6: Commit**

```bash
git add chenhe_rerun/orderhead_v3/{l0_dynamic_gbeta.py,soft_pairwise.py,cdl_teacher.py,__init__.py} chenhe_rerun/tests/test_orderhead_v3_gbeta.py
git commit -m "feat(orderhead_v3): port L0DynamicGBeta + CDL teacher (model-frame) + losses"
```

---

### Task 3: Stage-1 CDL producer (Part B)

**Files:**
- Create: `chenhe_rerun/gbeta_cdl_pretrain.py`
- Modify: none
- Test: `chenhe_rerun/tests/test_gbeta_cdl_pretrain.py`

**Interfaces:**
- Consumes: `orderhead_v3.{build_model_frame_strict65, build_dynamic_teacher, L0DynamicGBeta, soft_pairwise_bce_loss}`; chenhe `AOGPT_block.{AOGPT, AOGPTConfig}`.
- Produces: `pretrain_gbeta_cdl(parent_ckpt:str, out_dir:str, *, M:int, batch_mean_probes:int=4, heads:int=8, epochs:int, seed:int=0, device:str, probe_mode:str="eval") -> str` → path to `g_beta_best.pt`; writes `gbeta_provenance.json`.
- Produces: `load_chenhe_backbone(ckpt_path, device) -> (model, model_args)` — chenhe-native loader.

- [ ] **Step 1: Write the chenhe-native loader + extraction (failing test first)**

`chenhe_rerun/tests/test_gbeta_cdl_pretrain.py`:

```python
import sys, os
import numpy as np
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from AOGPT_block import AOGPT, AOGPTConfig
from gbeta_cdl_pretrain import extract_strict65_batch_mean, load_chenhe_backbone


def _tmp_parent_ckpt(tmp_path):
    cfg = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=8, n_embd=384,
               dropout=0.0, bias=True, block_order_block_len=4,
               block_order_layout="contiguous", position_encoding_mode="absolute")
    m = AOGPT(AOGPTConfig(**cfg))
    p = os.path.join(tmp_path, "ckpt.pt")
    torch.save({"model": m.state_dict(), "optimizer": {}, "model_args": cfg,
                "iter_num": 10000}, p)
    return p


def test_load_backbone_and_extract(tmp_path):
    p = _tmp_parent_ckpt(tmp_path)
    model, margs = load_chenhe_backbone(p, device="cpu")
    assert margs["block_order_block_len"] == 4
    idx = torch.randint(0, 50304, (2, 256))
    B = extract_strict65_batch_mean(model, idx, global_step=0, seed=0,
                                    batch_mean_probes=4, device="cpu",
                                    probe_mode="eval")
    assert B.shape == (2, 8, 65, 65)
    assert torch.isfinite(B).all()
```

- [ ] **Step 2: Implement loader + extraction in `gbeta_cdl_pretrain.py`**

```python
import os, json, hashlib
import numpy as np
import torch

from AOGPT_block import AOGPT, AOGPTConfig
from orderhead_v3.l0_strict65 import build_model_frame_strict65
from orderhead_v3.constants import assert_layout


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_chenhe_backbone(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    margs = dict(ckpt["model_args"])
    model = AOGPT(AOGPTConfig(**margs))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    assert_layout(model.num_blocks, model.block_order_block_len, model.config.n_head)
    return model, margs


@torch.no_grad()
def extract_strict65_batch_mean(model, idx_batch, *, global_step, seed,
                                batch_mean_probes, device, probe_mode="eval"):
    from orderhead_v3.constants import N, BLOCK_LEN
    if probe_mode == "eval":
        model.eval()
    else:
        model.train()
    B_sum = None
    for k in range(max(1, batch_mean_probes)):
        g = torch.Generator(device="cpu").manual_seed(
            int(seed) * 100_000_000 + int(global_step) * 1000 + k)
        blocks = torch.stack([torch.randperm(N, generator=g)
                              for _ in range(idx_batch.shape[0])])
        probe = model._expand_block_orders_to_token_orders(blocks).to(device)
        out = model.forward_fn(idx_batch.to(device), probe, return_attentions=True)
        attn_l0 = out[-1][0].cpu().numpy()
        Bk = build_model_frame_strict65(attn_l0, probe.cpu().numpy())
        B_sum = Bk if B_sum is None else B_sum + Bk
    B_mean = B_sum / max(1, batch_mean_probes)
    return torch.from_numpy(B_mean).float().to(device)
```

- [ ] **Step 3: Run the loader/extraction test**

Run: `cd chenhe_rerun && /data/users/chenhe/conda_envs/X1/bin/python -m pytest tests/test_gbeta_cdl_pretrain.py::test_load_backbone_and_extract -v`
Expected: PASS.

- [ ] **Step 4: Implement `pretrain_gbeta_cdl` orchestrator**

Append to `gbeta_cdl_pretrain.py`:

```python
from orderhead_v3.cdl_teacher import build_dynamic_teacher
from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.soft_pairwise import soft_pairwise_bce_loss
from orderhead_v3.constants import SEQ_LEN, N, BLOCK_LEN, HEADS, PERMUTE_SEED


def _sample_data_windows(bin_path, M, seed, block_size):
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - block_size - 1, size=M)
    return torch.from_numpy(
        np.stack([data[s:s + block_size].astype(np.int64) for s in starts]))


def pretrain_gbeta_cdl(parent_ckpt, out_dir, *, M=2000, batch_mean_probes=4,
                       heads=8, epochs=40, seed=0, device="cpu",
                       probe_mode="eval", data_bin="data/wikitext103/train.bin"):
    os.makedirs(out_dir, exist_ok=True)
    model, margs = load_chenhe_backbone(parent_ckpt, device)
    idx = _sample_data_windows(data_bin, M, seed, SEQ_LEN)
    # batch-mean strict65 over the M windows (chunk to fit memory)
    Bs = []
    for i in range(0, M, 64):
        Bs.append(extract_strict65_batch_mean(
            model, idx[i:i + 64], global_step=i, seed=seed,
            batch_mean_probes=batch_mean_probes, device=device,
            probe_mode=probe_mode).cpu())
    B = torch.cat(Bs, 0).numpy()                    # (M, 8, 65, 65)
    teacher = build_dynamic_teacher(B)              # model-frame [0,63] target
    gb = L0DynamicGBeta(heads=heads, nodes=65).to(device).train()
    opt = torch.optim.Adam(gb.parameters(), lr=1e-3)
    Bt = torch.from_numpy(B).float().to(device)
    yt = torch.from_numpy(np.asarray(teacher["teacher_pairwise"])).float().to(device)
    w = teacher.get("teacher_weights")
    wt = torch.from_numpy(np.asarray(w)).float().to(device) if w is not None else None
    best = None
    for _ in range(epochs):
        opt.zero_grad()
        scores, _ = gb(Bt, apply_head_dropout=False)
        loss = soft_pairwise_bce_loss(scores, yt, weights=wt)
        loss.backward(); opt.step()
        best = float(loss.item())
    ckpt_path = os.path.join(out_dir, "g_beta_best.pt")
    torch.save({"model_state_dict": gb.state_dict(),
                "config": {"model_name": "l0_dynamic", "heads": heads,
                           "nodes": 65}}, ckpt_path)
    prov = {"producer": "gbeta_cdl_pretrain", "parent_ckpt": parent_ckpt,
            "parent_hash": _file_hash(parent_ckpt), "seq_len": SEQ_LEN,
            "num_blocks": N, "block_len": BLOCK_LEN, "heads": heads,
            "permute_seed": PERMUTE_SEED, "none_mode": "model", "strict65": True,
            "probe_mode": probe_mode, "batch_mean_probes": batch_mean_probes,
            "M": M, "loss_type": "pairwise_bce", "seed": seed, "final_loss": best}
    with open(os.path.join(out_dir, "gbeta_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return ckpt_path
```

- [ ] **Step 5: Write producer smoke test (tiny M/epochs)**

Add to `test_gbeta_cdl_pretrain.py`:

```python
def test_producer_smoke(tmp_path):
    p = _tmp_parent_ckpt(tmp_path)
    from gbeta_cdl_pretrain import pretrain_gbeta_cdl
    out = pretrain_gbeta_cdl(p, str(tmp_path / "gb"), M=64, batch_mean_probes=2,
                             epochs=2, device="cpu",
                             data_bin=os.path.join(CHENHE, "data/wikitext103/train.bin"))
    import torch, json
    s = torch.load(out, map_location="cpu", weights_only=False)
    assert s["config"]["heads"] == 8 and s["config"]["nodes"] == 65
    prov = json.load(open(os.path.join(str(tmp_path / "gb"), "gbeta_provenance.json")))
    assert prov["num_blocks"] == 64 and prov["none_mode"] == "model"
    assert "parent_hash" in prov
```

- [ ] **Step 6: Run producer smoke**

Run: `cd chenhe_rerun && /data/users/chenhe/conda_envs/X1/bin/python -m pytest tests/test_gbeta_cdl_pretrain.py -v`
Expected: all PASS. (Requires `data/wikitext103/train.bin` symlink — already present.)

- [ ] **Step 7: Commit**

```bash
git add chenhe_rerun/gbeta_cdl_pretrain.py chenhe_rerun/tests/test_gbeta_cdl_pretrain.py
git commit -m "feat(gbeta): Stage-1 chenhe-native CDL producer + provenance"
```

---

### Task 4: batch-mean frozen provider + train.py dispatch/flags (Part C)

**Files:**
- Create: `chenhe_rerun/orderhead_v3/gbeta_provider.py`
- Modify: `chenhe_rerun/train.py` (dispatch ~L7551; init ~L714; CLI config vars near L85/L103)
- Test: `chenhe_rerun/tests/test_gbeta_provider.py`

**Interfaces:**
- Consumes: `gbeta_cdl_pretrain.extract_strict65_batch_mean`; `orderhead_v3.L0DynamicGBeta`.
- Produces: `GBetaFrozenProvider(gbeta_ckpt, init_from_ckpt, *, batch_mean_probes=4, refresh_every=1, seed=0, device, probe_mode)`; `.block_orders(model, idx_batch, global_step, is_eval:bool) -> LongTensor[B,64]` (one σ broadcast).

- [ ] **Step 1: Write provider test (failing)**

`chenhe_rerun/tests/test_gbeta_provider.py`:

```python
import sys, os, json
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from AOGPT_block import AOGPT, AOGPTConfig
from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.gbeta_provider import GBetaFrozenProvider


def _backbone_and_ckpts(tmp_path):
    cfg = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=8, n_embd=384,
               dropout=0.0, bias=True, block_order_block_len=4,
               block_order_layout="contiguous", position_encoding_mode="absolute")
    m = AOGPT(AOGPTConfig(**cfg))
    parent = os.path.join(tmp_path, "ckpt.pt")
    torch.save({"model": m.state_dict(), "model_args": cfg, "optimizer": {},
                "iter_num": 10000}, parent)
    import hashlib
    h = hashlib.sha256(open(parent, "rb").read()).hexdigest()
    gb = L0DynamicGBeta(heads=8, nodes=65)
    gpath = os.path.join(tmp_path, "g_beta_best.pt")
    torch.save({"model_state_dict": gb.state_dict(),
                "config": {"model_name": "l0_dynamic", "heads": 8, "nodes": 65}}, gpath)
    prov = {"parent_hash": h, "num_blocks": 64, "block_len": 4, "heads": 8,
            "seq_len": 256, "permute_seed": 42, "none_mode": "model",
            "strict65": True, "probe_mode": "eval"}
    json.dump(prov, open(os.path.join(tmp_path, "gbeta_provenance.json"), "w"))
    return m, parent, gpath


def test_batch_mean_single_order_frozen(tmp_path):
    m, parent, gpath = _backbone_and_ckpts(tmp_path)
    prov = GBetaFrozenProvider(gpath, parent, batch_mean_probes=2,
                               refresh_every=1, device="cpu", probe_mode="eval")
    before = [p.detach().clone() for p in prov.gbeta.parameters()]
    idx = torch.randint(0, 50304, (4, 256))
    orders = prov.block_orders(m, idx, global_step=0, is_eval=False)
    assert orders.shape == (4, 64)
    # ONE order broadcast to all samples (batch-mean):
    assert (orders == orders[0:1]).all()
    # gβ frozen: no param requires grad, unchanged:
    assert all(not p.requires_grad for p in prov.gbeta.parameters())
    after = list(prov.gbeta.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_provenance_mismatch_rejected(tmp_path):
    m, parent, gpath = _backbone_and_ckpts(tmp_path)
    other = os.path.join(tmp_path, "other.pt")
    torch.save({"model": m.state_dict(), "model_args": {}, "optimizer": {}}, other)
    import pytest
    with pytest.raises(Exception):
        GBetaFrozenProvider(gpath, other, device="cpu")  # parent hash mismatch


def test_eval_refreshes_per_batch(tmp_path):
    m, parent, gpath = _backbone_and_ckpts(tmp_path)
    prov = GBetaFrozenProvider(gpath, parent, refresh_every=1000,
                               device="cpu", probe_mode="eval")
    idx = torch.randint(0, 50304, (2, 256))
    prov.block_orders(m, idx, global_step=0, is_eval=False)   # train cache set
    train_sig = prov._train_sigma.clone()
    prov.block_orders(m, idx, global_step=5, is_eval=True)     # eval must recompute
    assert prov._eval_sigma is not None
    # train cache unchanged despite eval call:
    assert torch.equal(prov._train_sigma, train_sig)
```

- [ ] **Step 2: Implement `gbeta_provider.py`**

```python
import json, os, hashlib
import torch

from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.constants import N, BLOCK_LEN, HEADS, SEQ_LEN, PERMUTE_SEED
from gbeta_cdl_pretrain import extract_strict65_batch_mean


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class GBetaFrozenProvider:
    def __init__(self, gbeta_ckpt, init_from_ckpt, *, batch_mean_probes=4,
                 refresh_every=1, seed=0, device="cpu", probe_mode="eval"):
        prov_path = os.path.join(os.path.dirname(gbeta_ckpt), "gbeta_provenance.json")
        prov = json.load(open(prov_path))
        # hard config asserts (constraint 6/7/8)
        assert (prov["num_blocks"], prov["block_len"], prov["heads"]) == (N, BLOCK_LEN, HEADS)
        assert prov["seq_len"] == SEQ_LEN and prov["permute_seed"] == PERMUTE_SEED
        assert prov["none_mode"] == "model" and prov["strict65"] is True
        if prov["parent_hash"] != _file_hash(init_from_ckpt):
            raise ValueError("gβ provenance parent_hash != init_from_ckpt hash")
        st = torch.load(gbeta_ckpt, map_location="cpu", weights_only=False)
        self.gbeta = L0DynamicGBeta(heads=st["config"]["heads"],
                                    nodes=st["config"]["nodes"]).to(device)
        self.gbeta.load_state_dict(st["model_state_dict"])
        self.gbeta.eval()
        for p in self.gbeta.parameters():
            p.requires_grad_(False)
        self.batch_mean_probes = int(batch_mean_probes)
        self.refresh_every = max(1, int(refresh_every))
        self.seed, self.device, self.probe_mode = int(seed), device, probe_mode
        self._train_sigma = self._train_step = None
        self._eval_sigma = None

    @torch.no_grad()
    def _compute_sigma(self, model, idx_batch, global_step):
        B = extract_strict65_batch_mean(
            model, idx_batch, global_step=global_step, seed=self.seed,
            batch_mean_probes=self.batch_mean_probes, device=self.device,
            probe_mode=self.probe_mode)                 # (Bsz,8,65,65)
        Bmean = B.mean(dim=0, keepdim=True)             # batch-mean -> (1,8,65,65)
        scores, _ = self.gbeta(Bmean, apply_head_dropout=False)  # (1,64)
        return scores.argsort(dim=1, descending=True)[0]         # (64,)

    @torch.no_grad()
    def block_orders(self, model, idx_batch, global_step, is_eval):
        if is_eval:                                     # per-eval-batch refresh
            self._eval_sigma = self._compute_sigma(model, idx_batch, global_step)
            sigma = self._eval_sigma
        else:
            if (self._train_sigma is None or
                    global_step - self._train_step >= self.refresh_every):
                self._train_sigma = self._compute_sigma(model, idx_batch, global_step)
                self._train_step = int(global_step)
            sigma = self._train_sigma
        return sigma.unsqueeze(0).expand(idx_batch.shape[0], -1).to(idx_batch.device)
```

- [ ] **Step 3: Run provider tests**

Run: `cd chenhe_rerun && /data/users/chenhe/conda_envs/X1/bin/python -m pytest tests/test_gbeta_provider.py -v`
Expected: all 3 PASS.

- [ ] **Step 4: Wire the dispatch branch in `train.py`**

Near the config-var block (~L85–108) add defaults:

```python
gbeta_ckpt = ''            # path to g_beta_best.pt for GBetaFrozenOrder
gbeta_batch_mean_probes = 4
gbeta_refresh_every = 1
gbeta_probe_mode = 'eval'
init_from_ckpt = ''        # arbitrary parent ckpt for continuation
init_from_ckpt_mode = 'full_state'   # 'full_state' | 'weights_only'
```

In the order-dispatch function, before `return None, None, str(aogpt_train_mode)` (~L7602) add:

```python
    if aogpt_train_mode == 'GBetaFrozenOrder':
        bo = _gbeta_provider.block_orders(
            raw_model, idx, iter_num, is_eval=not torch.is_grad_enabled())
        ordered_units = _singleton_units_for_block_orders(bo) if return_units else None
        return bo, ordered_units, "GBetaFrozenOrder"
```

Construct the provider once after the model is built (near optimizer setup, ~L7686), guarded by mode:

```python
_gbeta_provider = None
if aogpt_train_mode == 'GBetaFrozenOrder':
    from orderhead_v3.gbeta_provider import GBetaFrozenProvider
    _gbeta_provider = GBetaFrozenProvider(
        gbeta_ckpt, init_from_ckpt, batch_mean_probes=gbeta_batch_mean_probes,
        refresh_every=gbeta_refresh_every, device=device, probe_mode=gbeta_probe_mode)
```

- [ ] **Step 5: Wire `--init-from-ckpt` in `train.py`**

In the `init_from` handling (~L705–718), add an explicit arbitrary-parent branch:

```python
elif init_from == 'ckpt':
    checkpoint = torch.load(init_from_ckpt, map_location=device)
    model_args = checkpoint['model_args']
    # ... build model like 'resume' ...
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    if init_from_ckpt_mode == 'full_state':
        resume_optimizer_state = True   # optimizer loaded below as in 'resume'
    else:
        resume_optimizer_state = False
    iter_num = 0                         # fresh step count for continuation
```

(Follow the exact model-construction lines used by the existing `init_from == 'resume'` branch; only the ckpt path source and `iter_num`/optimizer policy differ.)

- [ ] **Step 6: Backward-compat + 1-step smoke**

Run (mode absent → unchanged behavior):
`cd chenhe_rerun && CUDA_VISIBLE_DEVICES=0 /data/users/chenhe/conda_envs/X1/bin/python train.py config/WikiText103/seq256/permute/block64/random.py --max_iters=1 --eval_interval=1 --eval_iters=1 --wandb_log=False --compile=False`
Expected: runs, `step 0` prints, checkpoint saves (identical to the baseline smoke).

- [ ] **Step 7: Commit**

```bash
git add chenhe_rerun/orderhead_v3/gbeta_provider.py chenhe_rerun/tests/test_gbeta_provider.py chenhe_rerun/train.py
git commit -m "feat(train): GBetaFrozenOrder dispatch + batch-mean frozen provider + init-from-ckpt"
```

---

### Task 5: parent ckpt + method config + end-to-end smoke

**Files:**
- Create: `chenhe_rerun/config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py`
- Test: `chenhe_rerun/tests/test_gbeta_frozen_e2e.py`

**Interfaces:**
- Consumes: everything above + a chenhe random parent ckpt @ 10k + a produced `g_beta_best.pt`.

- [ ] **Step 1: Produce the 10k parent ckpt**

Run a dedicated random-to-10k so a clean parent is snapshotted (its own out_dir):

```bash
cd chenhe_rerun && CUDA_VISIBLE_DEVICES=0 /data/users/chenhe/conda_envs/X1/bin/python train.py \
  config/WikiText103/seq256/permute/block64/random.py \
  --max_iters=10000 --lr_decay_iters=10000 \
  --out_dir=out/rerun/parent_random_10k --wandb_log=False --compile=False
```

Parent ckpt = `out/rerun/parent_random_10k/ckpt.pt`.

- [ ] **Step 2: CDL-pretrain gβ from the parent**

```bash
cd chenhe_rerun && /data/users/chenhe/conda_envs/X1/bin/python -c "
from gbeta_cdl_pretrain import pretrain_gbeta_cdl
print(pretrain_gbeta_cdl('out/rerun/parent_random_10k/ckpt.pt',
      'out/rerun/gbeta_from_parent10k', M=2000, batch_mean_probes=4,
      epochs=40, device='cuda', probe_mode='eval'))"
```

Expected: prints `out/rerun/gbeta_from_parent10k/g_beta_best.pt`; provenance json written.

- [ ] **Step 3: Write the method config**

`config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py` — method-only diff off `random.py` (all LOCKED fields identical):

```python
# Based on chenhe baseline config: config/WikiText103/seq256/permute/block64/random.py
# Only method/order-policy fields changed.
out_dir = 'out/rerun/gbeta_frozen_warmup'
eval_interval = 250; eval_iters = 200; log_interval = 10
wandb_log = True
wandb_project = 'order-rerun-block64'
wandb_run_name = 'seq256-gbeta-frozen-b64-permute-from10k'

dataset = 'wikitext103'
batch_size = 64; block_size = 256; gradient_accumulation_steps = 2
permute_data = True; permute_seed = 42; permute_mode = 'block'
model_type = 'aogpt'; train_stage = 'standard'
aogpt_train_mode = 'GBetaFrozenOrder'
main_eval_mode = 'AR'; generalization_eval_mode = ''
n_layer = 4; n_head = 8; n_embd = 384; dropout = 0
block_order_block_len = 4
learning_rate = 1e-3; max_iters = 50000; lr_decay_iters = 50000
min_lr = 1e-4; beta2 = 0.99; warmup_iters = 0

# method / continuation fields
init_from = 'ckpt'
init_from_ckpt = 'out/rerun/parent_random_10k/ckpt.pt'
init_from_ckpt_mode = 'full_state'
gbeta_ckpt = 'out/rerun/gbeta_from_parent10k/g_beta_best.pt'
gbeta_batch_mean_probes = 4
gbeta_refresh_every = 1
gbeta_probe_mode = 'eval'
```

- [ ] **Step 4: Write the e2e smoke test**

`chenhe_rerun/tests/test_gbeta_frozen_e2e.py`:

```python
import subprocess, os, sys
CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/data/users/chenhe/conda_envs/X1/bin/python"

def test_gbeta_frozen_smoke():
    cmd = [PY, "train.py",
           "config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py",
           "--max_iters=1", "--eval_interval=1", "--eval_iters=1",
           "--wandb_log=False", "--compile=False"]
    r = subprocess.run(cmd, cwd=CHENHE, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    assert r.returncode == 0, r.stderr[-3000:]
    assert "step 0" in r.stdout
```

- [ ] **Step 5: Run the e2e smoke**

Run: `cd chenhe_rerun && /data/users/chenhe/conda_envs/X1/bin/python -m pytest tests/test_gbeta_frozen_e2e.py -v -s`
Expected: PASS — the frozen-gβ run starts, resumes from the parent, evaluates (`val` = gβ-order loss; `val_origin_l2r_loss` present), saves a checkpoint. Confirm no per-sample gβ path and gβ not in optimizer (inspect logs).

- [ ] **Step 6: Commit**

```bash
git add chenhe_rerun/config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py chenhe_rerun/tests/test_gbeta_frozen_e2e.py
git commit -m "feat(gbeta): method config + parent/producer wiring + e2e frozen smoke"
```

---

## Self-Review

**Spec coverage:**
- Part A (ported math) → Tasks 1–2. Part B (producer) → Task 3. Part C (provider + dispatch + flags) → Task 4. Config + parent + e2e → Task 5. ✓
- Hard constraints: frozen (T4 test), batch-mean-only incl. eval (T4 tests `test_batch_mean_single_order_frozen`, `test_eval_refreshes_per_batch`), model-frame supervision (T2 `test_teacher_is_model_frame`), frame-index semantics (T1 `test_frame_index_semantics`), mode parity (T3 probe_mode plumbed, provider uses same), B-source parity (extraction reuse), layout guard (T1 `test_layout_guard`). ✓
- Metric: headline `val_origin_l2r_loss` unchanged (native); `val` = gβ-order loss via dispatch. Comparison protocol = shared parent (T5 Step 1 parent + `init_from_ckpt_mode=full_state`). ✓
- Provenance binding (T4 `test_provenance_mismatch_rejected`). Overhead logging: chenhe already logs sec/step; probe-forward count is implicit in the provider (note: add tokens/sec to wandb if not present — covered by chenhe's existing logging).

**Placeholder scan:** No TBD/TODO. Port steps specify exact source + import rewrites + guard (function-level extraction rule in Global Constraints). Test code is concrete.

**Type consistency:** `extract_strict65_batch_mean` (T3) reused by `GBetaFrozenProvider` (T4) with matching signature. `build_model_frame_strict65` signature consistent T1→T3. `L0DynamicGBeta(heads, nodes)` + `(scores,aux)` consistent T2→T3→T4. `block_orders(model, idx, step, is_eval)` consistent T4→train.py. `gbeta_provenance.json` keys written in T3 == asserted in T4. ✓

**Open risk flagged for execution:** `build_dynamic_teacher` kwargs may differ from the assumed `build_dynamic_teacher(B)`; the executor adapts the call in T2/T3 to the ported signature while preserving the model-frame `[0,63]` output assertion.
