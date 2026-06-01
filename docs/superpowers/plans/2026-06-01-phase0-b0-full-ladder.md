# Phase 0 — B0 (none→block0) Full Ladder Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the 9-ckpt × 5-seed per-head order-scan ladder with the `none→block0` extraction (B0) instead of the contaminated `0.1-sink` (OLD), then decide the §3.0 gate (is B0 healthy + best+ stable across the full ladder?).

**Architecture:** Add a vectorized `none→block0` block-graph aggregation alongside the existing OLD one in `per_head_order_scan.py`, pinned bit-for-bit against the per-chunk reference `b0_fast.py::agg_b0`. Thread a `none_mode` switch (default `"old"`, preserving all existing behavior) through the extractor → `scan_checkpoint` → CLI. Drive a B0 ladder into a separate output dir, then run an offline OLD-vs-B0 comparison that evaluates the §3.0 pass criteria.

**Tech Stack:** Python, numpy (einsum `optimize=True`), torch (forward only), pytest, bash driver. Spec: `docs/superpowers/specs/2026-06-01-auto-order-head-end-to-end-training-design.md` §3.0/§3.2.

---

### Task 1: Vectorized none→block0 aggregation (`_attn_to_A_block_b0_vec`)

**Files:**
- Modify: `block_lo_arm_order_network/per_head_order_scan.py` (add function after `_attn_to_A_block_vec`, ~line 92)
- Test: `block_lo_arm_order_network/tests/test_per_head_scan_b0.py` (new)

- [ ] **Step 1: Write the failing test** (bit-match vs the per-chunk reference `b0_fast.agg_b0`)

Create `block_lo_arm_order_network/tests/test_per_head_scan_b0.py`:

```python
import importlib.util
import pathlib
import numpy as np
import pytest

from per_head_order_scan import _attn_to_A_block_b0_vec
from training_utils import SEQ_LEN, N, BLOCK_LEN

# Load the per-chunk reference agg_b0 from b0_fast.py without running its main loop.
_B0_FAST = pathlib.Path(__file__).resolve().parents[1] / "batch_readout/logs/per_head_scan/b0_fast.py"


def _load_agg_b0_reference():
    import sys
    src = _B0_FAST.read_text().split("idx_phys = get_idx_phys()")[0]  # cut before the main loop
    ns = {}
    saved = sys.argv
    sys.argv = [str(_B0_FAST)]  # b0_fast.py parses sys.argv[2] at module top; avoid pytest's argv
    try:
        exec(compile(src, str(_B0_FAST), "exec"), ns)
    finally:
        sys.argv = saved
    return ns["agg_b0"]


def _random_inputs(L=4, H=8, seed=0):
    rng = np.random.RandomState(seed)
    attn = rng.rand(L, H, SEQ_LEN + 1, SEQ_LEN + 1).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)  # row-stochastic like real attention
    g = np.random.RandomState(seed + 1)
    reveal_tokens = g.permutation(SEQ_LEN).astype(np.int64)
    inv_perm = g.permutation(N).astype(np.int64)
    return attn, reveal_tokens, inv_perm


def test_b0_vec_matches_per_chunk_reference():
    agg_b0_ref = _load_agg_b0_reference()
    attn, reveal_tokens, inv_perm = _random_inputs()
    got = _attn_to_A_block_b0_vec(attn, reveal_tokens, inv_perm)   # (L,H,N,N)
    ref = agg_b0_ref(attn, reveal_tokens, inv_perm)                # (L,H,N,N)
    assert got.shape == (4, 8, N, N)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_b0_vec_zero_diagonal_and_no_lead_dim():
    attn, reveal_tokens, inv_perm = _random_inputs()
    single = _attn_to_A_block_b0_vec(attn[0, 0], reveal_tokens, inv_perm)  # (N,N), no lead dim
    assert single.shape == (N, N)
    assert np.allclose(np.diag(single), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_per_head_scan_b0.py -q`
Expected: FAIL with `ImportError: cannot import name '_attn_to_A_block_b0_vec'`.

- [ ] **Step 3: Write minimal implementation**

In `block_lo_arm_order_network/per_head_order_scan.py`, add immediately after `_attn_to_A_block_vec` (after its `return`, ~line 92):

```python
def _attn_to_A_block_b0_vec(attn, reveal_tokens, inv_perm,
                            seq_len=SEQ_LEN, num_blocks=N, block_len=BLOCK_LEN):
    """none→block0 physical-frame block graph, vectorized over leading dims.

    Folds the [None] token (index 0) into physical block 0 (both its query row
    and key column) via a normalized segment-selection matrix, then segment-means
    the (T+1,T+1) attention into (N,N). No magic none_weight, no coordinate
    mismatch (everything is mapped to the physical frame before aggregation).
    Diagonal zeroed. Mirrors b0_fast.py::agg_b0 for arbitrary leading dims; the
    per-chunk reference pins this bit-for-bit (test_per_head_scan_b0).
    """
    attn = np.asarray(attn, dtype=np.float64)
    lead = attn.shape[:-2]
    K = int(np.prod(lead)) if lead else 1
    a = attn.reshape(K, seq_len + 1, seq_len + 1)

    reveal_tokens = np.asarray(reveal_tokens, dtype=np.int64)
    inv_perm = np.asarray(inv_perm, dtype=np.int64)
    phys_blocks = inv_perm[reveal_tokens // block_len]          # (T,) physical block per revealed token
    labels = np.empty(seq_len + 1, dtype=np.int64)
    labels[0] = 0                                              # [None] -> physical block 0
    labels[1:] = phys_blocks
    counts = np.bincount(labels, minlength=num_blocks).astype(np.float64)  # never 0: every block_len tokens
    S = np.zeros((num_blocks, seq_len + 1), dtype=np.float64)
    S[labels, np.arange(seq_len + 1)] = 1.0
    S = S / counts[:, None]                                    # segment-mean selection rows
    A = np.einsum("bt,ktu,cu->kbc", S, a, S, optimize=True)    # (K, N, N)
    di = np.arange(num_blocks)
    A[:, di, di] = 0.0
    A = A.astype(np.float32, copy=False)
    return A.reshape(lead + (num_blocks, num_blocks)) if lead else A[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_per_head_scan_b0.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/per_head_order_scan.py block_lo_arm_order_network/tests/test_per_head_scan_b0.py
git commit -m "feat(b0): vectorized none->block0 block-graph aggregation, pinned to b0_fast.agg_b0"
```

---

### Task 2: `none_mode` switch through extractor → scan_checkpoint → CLI

**Files:**
- Modify: `block_lo_arm_order_network/per_head_order_scan.py` (`_per_sample_A` ~229, `extract_per_head_and_heavy_A` ~249, `scan_checkpoint` ~314, `main` ~370)
- Test: `block_lo_arm_order_network/tests/test_per_head_scan_b0.py` (extend)

- [ ] **Step 1: Write the failing test** (dispatch picks the right aggregator; OLD path unchanged)

Append to `tests/test_per_head_scan_b0.py`:

```python
from per_head_order_scan import _per_sample_A, _attn_to_A_block_vec


def test_per_sample_A_none_mode_dispatch():
    attn, reveal_tokens, inv_perm = _random_inputs()           # (L,H,T+1,T+1)
    a_old, _ = _per_sample_A(attn, reveal_tokens, inv_perm, n_top=4, none_mode="old")
    a_b0, _ = _per_sample_A(attn, reveal_tokens, inv_perm, n_top=4, none_mode="b0")
    # OLD path must equal the existing OLD aggregator exactly (no behavior change).
    np.testing.assert_allclose(a_old, _attn_to_A_block_vec(attn, reveal_tokens, inv_perm), rtol=1e-5, atol=1e-6)
    # B0 path must equal the new B0 aggregator and differ from OLD.
    np.testing.assert_allclose(a_b0, _attn_to_A_block_b0_vec(attn, reveal_tokens, inv_perm), rtol=1e-5, atol=1e-6)
    assert not np.allclose(a_old, a_b0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_per_head_scan_b0.py::test_per_sample_A_none_mode_dispatch -q`
Expected: FAIL with `TypeError: _per_sample_A() got an unexpected keyword argument 'none_mode'`.

- [ ] **Step 3: Write minimal implementation** (thread `none_mode`, default `"old"`)

In `per_head_order_scan.py`, change `_per_sample_A` signature + body (line ~229):

```python
def _per_sample_A(attn_stack, reveal_tokens, inv_perm, n_top, none_mode="old"):
    """Per-sample physical-frame A_lh (L,H,N,N) and heavy A (N,N) from one
    sample's attention stack (L,H,T+1,T+1). none_mode in {"old","b0"} selects
    the [None]-handling for BOTH the per-head and heavy block graphs."""
    L, H = attn_stack.shape[:2]
    head_vars = np.zeros(H)
    mask = ~np.eye(SEQ_LEN, dtype=bool)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]
        offdiag = content[:, mask].reshape(L, SEQ_LEN, SEQ_LEN - 1)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-n_top:]
    avg_attn_heavy = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))

    agg = _attn_to_A_block_b0_vec if none_mode == "b0" else _attn_to_A_block_vec
    A_heavy_i = agg(avg_attn_heavy, reveal_tokens, inv_perm)
    A_lh_i = agg(attn_stack, reveal_tokens, inv_perm)
    return A_lh_i, A_heavy_i
```

In `extract_per_head_and_heavy_A` (line ~249): add `none_mode="old"` to the signature, and pass it through at the `_per_sample_A` call (line ~295):

```python
def extract_per_head_and_heavy_A(model, chunks, clean_perm, device, seed,
                                 n_top=4, fwd_batch=64, none_mode="old"):
```
```python
            A_lh_i, A_heavy_i = _per_sample_A(attn_stack, reveal_tokens, inv_perm, n_top, none_mode=none_mode)
```

In `scan_checkpoint` (line ~314): add `none_mode="old"` to the signature, pass to the extractor, and record it in config:

```python
def scan_checkpoint(ckpt_path, M, batch_size, seed, device="cuda:0",
                    split="train", alpha_dep=0.5, none_mode="old"):
```
```python
    A_lh, A_heavy = extract_per_head_and_heavy_A(model, chunks, clean_perm, dev, seed, none_mode=none_mode)
```
```python
        "config": {"M": M, "batch_size": batch_size, "seed": seed,
                   "ckpt": str(ckpt_path), "L": Ln, "H": Hn, "alpha_dep": alpha_dep,
                   "none_mode": none_mode},
```

In `main` (line ~374): add the CLI flag and pass it through:

```python
    p.add_argument("--none-mode", default="old", choices=["old", "b0"])
```
```python
    res = scan_checkpoint(args.ckpt, args.M, args.batch_size, args.seed,
                          device=args.device, split=args.split, alpha_dep=args.alpha_dep,
                          none_mode=args.none_mode)
```

- [ ] **Step 4: Run test to verify it passes** (and the OLD-path regression test still passes)

Run: `cd block_lo_arm_order_network && python -m pytest tests/test_per_head_scan_b0.py tests/test_per_head_scan_batched.py -q`
Expected: PASS (all). The `test_per_head_scan_batched` suite confirms the OLD path is byte-unchanged.

- [ ] **Step 5: Commit**

```bash
git add block_lo_arm_order_network/per_head_order_scan.py block_lo_arm_order_network/tests/test_per_head_scan_b0.py
git commit -m "feat(b0): none_mode switch through extractor/scan/CLI (default old, preserves OLD path)"
```

---

### Task 3: B0 ladder driver script

**Files:**
- Create: `scripts/run_per_head_order_scan_ladder_b0.sh`

- [ ] **Step 1: Write the driver** (copy of the OLD driver, B0 output dir + `--none-mode b0`)

Create `scripts/run_per_head_order_scan_ladder_b0.sh`:

```bash
#!/usr/bin/env bash
# B0 (none->block0) per-head order-scan ladder — §3.0 gate for the
# auto-order-head spec. Same ladder as run_per_head_order_scan_ladder.sh but with
# --none-mode b0, writing to a SEPARATE dir so OLD dumps stay intact for OLD-vs-B0
# comparison. Idempotent: non-empty (step,seed) JSON is skipped.
#
# Usage: scripts/run_per_head_order_scan_ladder_b0.sh [DEVICE] [M] [BATCH_SIZE] [SEEDS...]
# Defaults: DEVICE=cuda:0 M=100 BATCH_SIZE=32 SEEDS="0 1 2 3 4"
set -euo pipefail

REPO="/home/admin/lyuyuhuan/order_lyu"
PKG="$REPO/block_lo_arm_order_network"
CKPT_DIR="$PKG/probe_results/clean_base_random_perm"
OUT_DIR="$PKG/batch_readout/logs/per_head_scan_b0"

DEVICE="${1:-cuda:0}"
M="${2:-100}"
BATCH_SIZE="${3:-32}"
shift $(( $# > 3 ? 3 : $# )) || true
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3 4)

STEPS=(0 1000 5000 10000 20000 30000 40000 50000 60000)

mkdir -p "$OUT_DIR"
cd "$REPO"

total=$(( ${#STEPS[@]} * ${#SEEDS[@]} ))
done=0
echo "[b0-ladder] DEVICE=$DEVICE M=$M BATCH_SIZE=$BATCH_SIZE SEEDS=${SEEDS[*]}"
echo "[b0-ladder] $total jobs -> $OUT_DIR"

for step in "${STEPS[@]}"; do
  ckpt="$CKPT_DIR/ckpt_step${step}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[b0-ladder] MISSING ckpt: $ckpt — skipping step $step"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    done=$(( done + 1 ))
    out="$OUT_DIR/ckpt${step}_seed${seed}.json"
    if [ -s "$out" ]; then
      echo "[b0-ladder] ($done/$total) skip existing $out"
      continue
    fi
    echo "[b0-ladder] ($done/$total) scan step=$step seed=$seed -> $out"
    python "$PKG/per_head_order_scan.py" \
      --ckpt "$ckpt" --M "$M" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --device "$DEVICE" --none-mode b0 --out "$out"
  done
done

echo "[b0-ladder] DONE: $(ls "$OUT_DIR"/ckpt*_seed*.json 2>/dev/null | wc -l) JSON files in $OUT_DIR"
```

- [ ] **Step 2: Make executable + smoke ONE job end-to-end** (validates CLI wiring on a real ckpt before the full 45)

```bash
chmod +x scripts/run_per_head_order_scan_ladder_b0.sh
cd /home/admin/lyuyuhuan/order_lyu
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python block_lo_arm_order_network/per_head_order_scan.py \
  --ckpt block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt \
  --M 100 --batch-size 32 --seed 0 --device cuda:0 --none-mode b0 \
  --out block_lo_arm_order_network/batch_readout/logs/per_head_scan_b0/ckpt5000_seed0.json
```
Expected: prints `top head L?H? tau_vs_l2r=...` and `saved -> .../ckpt5000_seed0.json`; the JSON's `config.none_mode == "b0"`. Sanity-check against the b0_fast 5k result (winner should be in the L0H0/L0H1/L1H1 τ≈1.0 cluster, NOT the OLD L0H6).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_per_head_order_scan_ladder_b0.sh
git commit -m "chore(b0): B0 none->block0 ladder driver (9 ckpt x 5 seed, separate out dir, idempotent)"
```

---

### Task 4: Launch full B0 ladder (background) + monitor

**Files:** none (run only)

- [ ] **Step 1: Launch the 45-job ladder in the background**

```bash
cd /home/admin/lyuyuhuan/order_lyu
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup scripts/run_per_head_order_scan_ladder_b0.sh cuda:0 100 32 0 1 2 3 4 \
  > block_lo_arm_order_network/batch_readout/logs/per_head_scan_b0/ladder.log 2>&1 &
echo "launched pid=$!"
```
(Task 3 Step 2 already produced `ckpt5000_seed0.json`; the idempotent driver skips it.)

- [ ] **Step 2: Monitor to completion** (watch progress + failures, exit when 45 JSONs exist or the driver dies)

Use the Monitor tool with: `tail -n +1 -F .../per_head_scan_b0/ladder.log | grep --line-buffered -E "scan step=|DONE|Error|Traceback|MISSING"`, or poll:
```bash
ls block_lo_arm_order_network/batch_readout/logs/per_head_scan_b0/ckpt*_seed*.json | wc -l   # target 45
```
Expected: 45 JSON files; `ladder.log` ends with `[b0-ladder] DONE: 45 ...`.

---

### Task 5: Offline OLD-vs-B0 §3.0 gate analysis

**Files:**
- Create: `block_lo_arm_order_network/batch_readout/logs/per_head_scan/b0_ladder_gate.py`

- [ ] **Step 1: Write the gate analysis script** (rowconc health + pool precision + best+ stability + late-winner identity, OLD vs B0)

Create `block_lo_arm_order_network/batch_readout/logs/per_head_scan/b0_ladder_gate.py`:

```python
"""§3.0 gate: compare OLD vs B0 ladders on row-concentration health, top-k pool
precision, best+ stability, and late-stage winner identity. Reads both dirs'
ckpt*_seed*.json graph dumps. Pass = B0 rowconc median not collapsed, high pool
precision, best+ stable across seeds."""
import json, glob, re, os
import numpy as np
from collections import defaultdict
import statistics as st

OLD_DIR = os.path.dirname(__file__)
B0_DIR = os.path.join(os.path.dirname(OLD_DIR), "per_head_scan_b0")
EPS = 1e-12
TOPK = 5
TAU_STRONG = 0.8


def c_row(A):  # A:(L,H,N,N) -> (L*H,) row-concentration of B=A^T
    L, H, N, _ = A.shape
    B = A.transpose(0, 1, 3, 2).copy()
    di = np.arange(N); B[:, :, di, di] = 0.0
    rs = B.sum(-1, keepdims=True); P = B / (rs + EPS)
    lP = np.where(P > 0, np.log(P + EPS), 0.0); Hr = -(P * lP).sum(-1)
    Cn = 1.0 - Hr / np.log(N - 1)
    Cn = np.where(rs[..., 0] <= EPS, 0.0, Cn)
    return Cn.mean(-1).reshape(-1)


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "ckpt*_seed*.json"))):
        o = json.load(open(f))
        g = o.get("graphs")
        if not g or "A_mean" not in g:
            continue
        step = int(re.search(r"ckpt(\d+)_seed", f).group(1))
        seed = int(re.search(r"_seed(\d+)", f).group(1))
        A = np.array(g["A_mean"], np.float64); L, H = A.shape[:2]
        C = c_row(A)
        tau = np.zeros(L * H)
        for h in o["per_head_layer_sorted_by_abs_tau_vs_l2r"]:
            tau[h["layer"] * H + h["head"]] = h["tau_vs_l2r"]
        topk = np.argsort(-C)[:TOPK]
        rows.append(dict(step=step, seed=seed, med=float(np.median(C)),
                         winconc=float(C[np.argmax(np.abs(tau))]),
                         pool_prec=float(np.mean(np.abs(tau[topk]) >= 0.9)),
                         bestpos_recall=int(np.any(tau[topk] >= TAU_STRONG)),
                         maxtau_topk=float(np.max(np.abs(tau[topk]))),
                         top1=int(topk[0])))
    return rows


def report(name, rows):
    by = defaultdict(list)
    for r in rows: by[r["step"]].append(r)
    print(f"\n=== {name} ({len(rows)} points) ===")
    print(f"{'step':>6} | med C | poolP@5 | best+rec | maxτ@5 | top1-head mode")
    for s in sorted(by):
        rs = by[s]
        from collections import Counter
        mode = Counter(r["top1"] for r in rs).most_common(1)[0]
        print(f"{s:>6} | {st.median([r['med'] for r in rs]):.3f} | "
              f"{st.mean([r['pool_prec'] for r in rs]):.2f}    | "
              f"{st.mean([r['bestpos_recall'] for r in rs]):.2f}     | "
              f"{st.median([r['maxtau_topk'] for r in rs]):.2f}   | "
              f"L{mode[0]//8}H{mode[0]%8} ({mode[1]}/{len(rs)})")


old, b0 = load(OLD_DIR), load(B0_DIR)
report("OLD (0.1-sink)", old)
report("B0 (none->block0)", b0)

# §3.0 PASS check on B0
conv = [r for r in b0 if r["step"] >= 5000]
med_min = min(r["med"] for r in conv) if conv else 0.0
poolP = st.mean([r["pool_prec"] for r in conv]) if conv else 0.0
recall = st.mean([r["bestpos_recall"] for r in conv]) if conv else 0.0
print(f"\n§3.0 GATE (B0, step>=5000): min median C={med_min:.3f} (want >0.02), "
      f"mean poolP@5={poolP:.2f}, mean best+recall@5={recall:.2f} (want ~1.0)")
print("PASS" if (med_min > 0.02 and recall > 0.9) else "REVIEW")
```

- [ ] **Step 2: Run the gate after the ladder completes**

Run: `cd block_lo_arm_order_network/batch_readout/logs/per_head_scan && python b0_ladder_gate.py`
Expected: a per-step OLD vs B0 table + a final `§3.0 GATE ... PASS/REVIEW` line. Read the `top1-head mode` column to confirm whether B0's best head is stable (e.g. L0H0) across steps and whether OLD's "drift to L1H4" was an artifact.

- [ ] **Step 3: Commit the analysis + record the verdict**

```bash
git add block_lo_arm_order_network/batch_readout/logs/per_head_scan/b0_ladder_gate.py
git commit -m "feat(b0): OLD-vs-B0 ladder §3.0 gate analysis (rowconc health, pool precision, best+ stability)"
```

- [ ] **Step 4: Update the spec §3.2/§3.0 with the full-ladder verdict** (provisional → final, or REVIEW notes), and update memory `quick_head_selector_line.md`.

---

## Out of scope (deferred to later plans, gated on this gate passing)
- Phase 1 selected-head dataset builder + g_β pretrain.
- Phase 1.5 MLP generalization gate.
- Phase 2 FrozenBetaHook training integration + selector observe-only.
- Making B0 the SOLE canonical extraction (removing OLD path) — only after the gate passes.
