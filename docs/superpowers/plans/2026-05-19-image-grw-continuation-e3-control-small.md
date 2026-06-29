# Image VQ Graph-RW Continuation on E3-control-small Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the E3-control-small AO-GPT (l4h8e256 + ImageNet64 VQ-f4 patch2x2 + block_order_block_len=4) — which has a near-perfect 4-neighbor block-level attention graph (`P(d≤1)=0.984`) — and run a 5-arm short continuation to test whether that clean local structure converts into measurable training benefit on structured eval orders, or whether it's a diagnostic-only artifact.

**Architecture:** Build a new image-side Graph-RW continuation trainer adapted to the nanogpt-learned-order `BlockAOGPT` model. The trainer loads the E3-control-small checkpoint, samples block-level orders from one of five policies, expands them through the model's built-in `_expand_block_orders_to_token_orders` to seq=256 token orders, trains under standard AOGPT CE loss with an α-mixed random-order regularizer, and runs a frozen 5-order eval every 500 steps. After each run finishes, re-extract `A_global` + diagnostic to see whether the local structure was preserved, sharpened, or drifted. Cross-run table at the end decides whether attention-derived structure on VQ patch-block data delivers a training benefit.

**Tech Stack:** PyTorch 2.x, nanogpt-learned-order `BlockAOGPT`, existing `directed_graph_policy.build_directed_graph` + `image_order/graph_rw_image.sample_image_orders_batch`, `block_lo_arm_order_network/extract_image_attention_e2.py` as A-extraction reference, custom 5-arm orchestration shell script.

---

## File structure

### Create

- `block_lo_arm_order_network/train_imagelarge_graph_rw.py` — main 5-policy continuation trainer (~400 LOC). Owns: ckpt load (with `_orig_mod.` strip + missing-key raise + data_permutation read), val.bin loading, optimizer setup, RW sampler dispatch, alpha schedule, frozen 5-order eval at fixed step intervals, eval TSV, periodic ckpt save.
- `scripts/run_imagelarge_grw_5arm.sh` — orchestration shell that launches all 5 arms sequentially on GPU 0 (default), each producing a unique output dir. Reuses the wait/launch pattern from `scripts/wait_and_diagnose_e3_control.sh`.
- `scripts/post_grw_diagnose.sh` — after each arm finishes, re-extracts `A_global` + dual-level diagnostic from the final ckpt (uses `extract_image_attention_e2.py` + `scripts/diagnose_e3_control_dual_level.py`, both already exist).
- `scripts/grw_5arm_crossrun_summary.py` — aggregates 5 `eval_curve.tsv` files + 5 post-train `diagnostics.json` files into a single `SUMMARY.md` with cross-run comparison tables and verdict.
- `probe_results_image_large/grw_e3ctrlsmall/` — output root. Each arm gets its own subdir: `cont_random/`, `cont_graph_rw/`, `cont_raster/`, `cont_shuffled_B/`, `cont_eps015/`. Each contains `train_log.txt`, `eval_curve.tsv`, `config.json`, `ckpt_final.pt`, plus post-train `attention/A_global.npy`, `attention/A_block_8x8.npy`, `attention/diagnostics.json`.

### Reuse (do NOT modify)

- `nanogpt-learned-order/AOGPT.py`, `AOGPT_block.py` — model code, used read-only.
- `block_lo_arm_order_network/directed_graph_policy.py` — `build_directed_graph(A)` returns B = A^T with diag zeroed (NOT row-normalized; raw edge weights, row sums ~0.19). The RW sampler consumes raw B via a tau-softmax, so absolute scale is intentional. Reused.
- `image_order/graph_rw_image.py` — `sample_image_orders_batch(B, params, batch_size, seed_base, step, device)` returns `(B_batch, 64)` block orders. Reused.
- `block_lo_arm_order_network/extract_image_attention_e2.py` — reused for post-train A-extraction.
- `scripts/diagnose_e3_control_dual_level.py` — reused for dual-level metrics.

---

## Inputs (verify before starting)

| What | Path | Notes |
|---|---|---|
| Baseline ckpt | `nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt` | best-val at iter 16250, val=7.463, has `data_permutation` field |
| A_block (raster, 64×64) | `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy` | block-level, physical raster order |
| val data | `nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/{val.bin,meta.pkl}` | tokens_per_image=256, vocab_size=8192 |
| train data | same dir, `train.bin` | for continuation training samples |

---

## Run matrix (the 5 arms)

| Arm | Policy | Top-k | ε-uniform | α (RW share) | Note |
|---|---|---|---|---|---|
| `cont_random` | uniform random block perms | — | — | 0.0 | matched baseline (α=0 → 100% random) |
| `cont_graph_rw` | progressive_rw_v3 | 4 | 0.0 | 0.9 | the main treatment (uses real B) |
| `cont_raster` | fixed 8×8 raster block order | — | — | 0.9 | strong-spatial-prior **upper reference / sanity**, NOT a method arm |
| `cont_shuffled_B` | progressive_rw_v3 on per-row-shuffled B | 4 | 0.0 | 0.9 | negative control: same policy, destroyed B |
| `cont_eps015` | progressive_rw_v3 | 0 (no top-k) | 0.15 | 0.9 | **high-noise / no-top-k destructive control** (NOT mild noise) |

**Which comparisons are the actual test** (raster is reference, not method):
- `cont_graph_rw` vs `cont_random` — the main treatment test
- `cont_shuffled_B` vs `cont_random` — does a destroyed B kill the benefit?
- `cont_eps015` vs `cont_random` — does heavy noise kill the benefit?
- `cont_raster` is a strong-spatial-prior upper reference / coordinate sanity check only.

**Common settings (all 5 arms):**
- max_steps = 5000
- batch_size = 16, grad_accum = 16 (eff batch 256, matches baseline)
- lr = 1e-4 (10× lower than baseline 1e-3, standard for continuation)
- min_lr = 1e-5, beta2 = 0.99, weight_decay = 0.1, warmup = 100
- alpha_warmup = 1000 (linear ramp α 0 → target over first 1000 steps)
- eval_interval = 500 (10 evals across the run)
- save_steps = "1000,3000,5000" (intermediate ckpts for debugging)
- B is **fixed at run-start** (no EMA refresh). One source of truth: round-1 cleanliness > sophistication.

---

## Decision matrix (after all 5 arms finish)

| Pattern observed | Conclusion |
|---|---|
| `cont_graph_rw` `val_random` ≤ `cont_random` `val_random` AND `cont_graph_rw` `val_rw_top4_eps0` < `cont_random` `val_rw_top4_eps0` AND `cont_shuffled_B` & `cont_eps015` no improvement | **Local structure converts to training benefit.** Image VQ line closes from diagnostic → training-utility. Write up. |
| `cont_graph_rw` ≈ `cont_random` on every eval column AND `cont_raster` shows a clear gap | **Local structure exists but is policy-redundant** — baseline already learns it through random-order training; explicit Graph-RW adds nothing. Still a valid result. |
| `cont_graph_rw` < `cont_random` on `val_random` (specialization tax) AND `cont_graph_rw` > `cont_random` on `val_rw_top4_eps0` | **Structured order is a specialization, not generalization.** Document trade-off. |
| All arms drift to the same point | Continuation budget too short OR α schedule wrong. Re-run with longer horizon or stronger α. |

---

## Tasks

### Task 1: Pre-flight — verify inputs and confirm coordinate alignment

**Files:**
- Read: `nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt`
- Read: `probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy`
- Create: `probe_results_image_large/grw_e3ctrlsmall/preflight_report.txt`

- [ ] **Step 1: Inspect ckpt contents**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
python3 - <<'EOF'
import torch
ck = torch.load("nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt", map_location="cpu", weights_only=False)
print("keys:", list(ck.keys()))
print("iter_num:", ck["iter_num"], "best_val:", ck["best_val_loss"])
sd = ck["model"]
print("first 3 sd keys:", list(sd.keys())[:3])
print("has _orig_mod prefix:", all(k.startswith("_orig_mod.") for k in sd.keys()))
ma = ck["model_args"]
print("model_args:", {k: ma[k] for k in ["n_layer","n_head","n_embd","block_size","vocab_size","block_order_block_len","order_impl"]})
dp = ck.get("data_permutation")
if dp is not None:
    import numpy as np
    if hasattr(dp, "shape"):
        print("data_permutation shape/dtype:", dp.shape, dp.dtype, "head:", dp.flatten()[:10])
    else:
        print("data_permutation type:", type(dp), "head:", list(dp)[:10] if hasattr(dp, "__iter__") else dp)
else:
    print("NO data_permutation in ckpt")
EOF
```

Expected:
- `iter_num = 16250`, `best_val_loss ≈ 7.463`
- `_orig_mod.` prefix present on all state-dict keys (torch.compile artifact)
- `model_args` shows `n_layer=4, n_embd=256, block_size=256, vocab_size=8192, block_order_block_len=4, order_impl='block'`
- `data_permutation` either present (saved by `train.py`) or absent — both branches handled in Task 2

- [ ] **Step 2: Verify A_block_8x8 sanity**

Run:
```bash
python3 - <<'EOF'
import numpy as np
A = np.load("probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy")
print("shape:", A.shape, "dtype:", A.dtype)
print("mean:", A.mean(), "max:", A.max(), "diag sum:", np.diag(A).sum())
assert A.shape == (64, 64), "Expected (64, 64)"
assert np.diag(A).sum() == 0, "Diagonal must be zeroed"
print("OK: A_block_8x8 is (64,64), diagonal zeroed.")
EOF
```

Expected: shape=(64,64), diag=0, mean≈0.003, max≈0.012.

- [ ] **Step 3: Verify the row-normalized B (= A^T after build_directed_graph) gives row-sums of 1**

Run:
```bash
python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "block_lo_arm_order_network")
from directed_graph_policy import build_directed_graph
A = np.load("probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy")
B = build_directed_graph(A)
print("B shape:", B.shape, "row sums (first 5):", B.sum(axis=1)[:5])
print("B max in row 0:", B[0].max(), "argmax:", B[0].argmax())
print("B max in row 9:", B[9].max(), "argmax:", B[9].argmax(), "  (row 9 is block (1,1))")
EOF
```

Expected: argmax for row 0 (block (0,0)) is block 1 or 8 (4-neighbor argmax); argmax for row 9 (block (1,1)) lies in {1, 8, 10, 17} (the 4 spatial neighbors). This argmax-in-4-neighbor check is the alignment gate. (Note: row sums are ~0.19, NOT 1.0 — build_directed_graph does not normalize; the RW sampler applies its own tau-softmax. Do not treat row-sum≠1 as a failure.)

- [ ] **Step 4: Write preflight report and commit**

Run:
```bash
mkdir -p probe_results_image_large/grw_e3ctrlsmall
python3 - <<'EOF' > probe_results_image_large/grw_e3ctrlsmall/preflight_report.txt
import sys, numpy as np, torch
sys.path.insert(0, "block_lo_arm_order_network")
from directed_graph_policy import build_directed_graph

ck = torch.load("nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt", map_location="cpu", weights_only=False)
A = np.load("probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy")
B = build_directed_graph(A)

print("=== Preflight report for E3-control-small Graph-RW continuation ===")
print(f"ckpt iter_num: {ck['iter_num']}, best_val_loss: {ck['best_val_loss']:.4f}")
print(f"data_permutation in ckpt: {'yes' if ck.get('data_permutation') is not None else 'NO'}")
print(f"A_block: shape={A.shape}, mean={A.mean():.6f}, max={A.max():.6f}, diag_sum={np.diag(A).sum()}")
print(f"B (row-normalized): row_sums first 5 = {B.sum(axis=1)[:5]}")
spatial_argmax_at_block_9 = int(B[9].argmax())
expected_neighbors_of_block_9 = {1, 8, 10, 17}  # 4-neighbor on 8x8 grid
print(f"B[9] argmax = {spatial_argmax_at_block_9}, expected in {expected_neighbors_of_block_9}: "
      f"{'YES' if spatial_argmax_at_block_9 in expected_neighbors_of_block_9 else 'NO -- alignment off, STOP'}")
EOF
cat probe_results_image_large/grw_e3ctrlsmall/preflight_report.txt

git add probe_results_image_large/grw_e3ctrlsmall/preflight_report.txt
git commit -m "image grw: preflight verification for E3-control-small continuation"
```

Expected: the "YES" branch on the neighbor check. If "NO", **stop and re-examine the unshuffle path in `extract_image_attention_e2.py`** — physical/model coordinate confusion is the most likely cause and Tasks 2–5 will give garbage if it's wrong.

---

### Task 2: Build minimal trainer — `cont_random` smoke (200 steps, single arm)

**Files:**
- Create: `block_lo_arm_order_network/train_imagelarge_graph_rw.py`
- Test (smoke): `probe_results_image_large/grw_e3ctrlsmall/smoke_random_200/`

- [ ] **Step 1: Create the trainer file with model loading and random-order forward only**

Create `block_lo_arm_order_network/train_imagelarge_graph_rw.py`:

```python
#!/usr/bin/env python3
"""Image-large Graph-RW continuation trainer for nanogpt-learned-order BlockAOGPT.

Five policies: random | graph_rw | raster | shuffled_B | eps015.
Loads E3-control-small ckpt, samples block-level orders (N=64), expands
internally to seq=256 token orders, trains under standard CE with an
alpha-mixed random-order regularizer.
"""

import argparse, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "image_order"))
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))

from AOGPT import AOGPT, AOGPTConfig
from directed_graph_policy import build_directed_graph
from graph_rw_image import (
    IMAGE_RW_PARAMS_DEFAULT, IMAGE_RW_PARAMS_V2, IMAGE_RW_PARAMS_V3,
    sample_image_orders_batch,
)


N_BLOCKS = 64
SEQ_LEN = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["random", "graph_rw", "raster", "shuffled_B", "eps015"],
                   required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--a-block-path", required=True,
                   help="Path to A_block_8x8.npy (raster, 64x64)")
    p.add_argument("--data-train", required=True)
    p.add_argument("--data-val", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--alpha-warmup", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--max-eval-batches", type=int, default=16)
    p.add_argument("--save-steps", type=str, default="1000,3000,5000")
    p.add_argument("--rw-top-k", type=int, default=4)
    p.add_argument("--rw-epsilon", type=float, default=0.0)
    p.add_argument("--rw-tau-start", type=float, default=0.10)
    p.add_argument("--rw-tau-step", type=float, default=0.10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_baseline_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    keys = ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
            "dropout", "bias", "block_order_block_len", "order_impl"]
    model_args = {k: ma[k] for k in keys if k in ma}
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading ckpt: {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys ignored: {unexpected[:5]}")
    model.crop_block_size(model_args["block_size"])
    model.to(device)
    return model, model_args, ckpt


def get_lr(step, args):
    if step < args.warmup_iters:
        return args.lr * step / max(args.warmup_iters, 1)
    if step >= args.max_steps:
        return args.min_lr
    ratio = (step - args.warmup_iters) / max(args.max_steps - args.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def get_alpha(step, args):
    if args.alpha_warmup <= 0:
        return args.alpha
    return min(args.alpha, args.alpha * step / max(args.alpha_warmup, 1))


def sample_block_orders(policy, B_global, batch_size, step, rw_params, raster_order, device, rng):
    """Return (batch_size, 64) block-order tensor in physical raster order."""
    if policy == "random":
        return torch.stack([torch.randperm(N_BLOCKS, device=device) for _ in range(batch_size)])
    if policy == "raster":
        return raster_order.unsqueeze(0).expand(batch_size, -1).contiguous()
    # graph_rw, shuffled_B, eps015 all use sample_image_orders_batch with different B/params
    seed_base = int(rng.integers(1 << 31))
    return sample_image_orders_batch(B_global, rw_params, batch_size,
                                     seed_base=seed_base, step=step, device=device)


@torch.no_grad()
def evaluate_5orders(model, val_tokens, B_real, raster_order, device,
                     batch_size, max_eval_batches, step):
    model.eval()
    V = val_tokens.shape[0]
    n_batches = min(max_eval_batches, math.ceil(V / batch_size))

    cols = ["random", "raster", "rw_top4_eps0", "rw_eps015", "rw_topk8"]
    results = {c: [] for c in cols}

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, V)
        x = val_tokens[start:end].to(device)
        B_actual = x.shape[0]

        rand_b = torch.stack([torch.randperm(N_BLOCKS, device=device) for _ in range(B_actual)])
        ras_b = raster_order.unsqueeze(0).expand(B_actual, -1).contiguous()

        seed_base = 9999_000 + step
        rw4 = sample_image_orders_batch(B_real,
            {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4, "epsilon_uniform": 0.0},
            B_actual, seed_base=seed_base, step=0, device=device)
        rweps = sample_image_orders_batch(B_real,
            {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 0, "epsilon_uniform": 0.15},
            B_actual, seed_base=seed_base + 1, step=0, device=device)
        rwk8 = sample_image_orders_batch(B_real,
            {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 8, "epsilon_uniform": 0.0},
            B_actual, seed_base=seed_base + 2, step=0, device=device)

        for name, ord_ in zip(cols, [rand_b, ras_b, rw4, rweps, rwk8]):
            _, loss = model(x, ord_, targets=x)
            results[name].append(float(loss.item()))

    model.train()
    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----- A → B -----
    A_block = np.load(args.a_block_path).astype(np.float32)
    assert A_block.shape == (64, 64)
    B_real = build_directed_graph(A_block)
    if args.policy == "shuffled_B":
        B_used = np.empty_like(B_real)
        for i in range(64):
            B_used[i] = B_real[i, rng.permutation(64)]
    else:
        B_used = B_real

    # ----- RW params -----
    rw_params = {**IMAGE_RW_PARAMS_DEFAULT,
                 "top_k": args.rw_top_k, "epsilon_uniform": args.rw_epsilon,
                 "tau_start": args.rw_tau_start, "tau_step": args.rw_tau_step}

    # ----- model -----
    model, model_args, ckpt = load_baseline_model(args.baseline_ckpt, device)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr,
                                           (args.beta1, args.beta2), device.split(":")[0])
    model.train()

    # ----- data -----
    with open(args.meta, "rb") as f: meta = pickle.load(f)
    tokens_per_image = int(meta["tokens_per_image"])
    assert tokens_per_image == SEQ_LEN, f"meta says {tokens_per_image}, expected {SEQ_LEN}"
    train_mm = np.memmap(args.data_train, dtype=np.uint16, mode="r")
    val_mm = np.memmap(args.data_val, dtype=np.uint16, mode="r")
    n_train = len(train_mm) // SEQ_LEN
    n_val = min(2000, len(val_mm) // SEQ_LEN)
    val_tokens = torch.from_numpy(
        np.asarray(val_mm[:n_val * SEQ_LEN], dtype=np.int64).reshape(n_val, SEQ_LEN)
    )

    raster_order = torch.arange(N_BLOCKS, device=device)

    # ----- TSV -----
    tsv_path = out / "eval_curve.tsv"
    header = "step\ttrain_loss\talpha\tlr\tval_random\tval_raster\tval_rw_top4_eps0\tval_rw_eps015\tval_rw_topk8\n"
    if not tsv_path.exists():
        tsv_path.write_text(header)
    log_path = out / "train_log.txt"
    log_f = open(log_path, "a")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    log(f"[start] policy={args.policy} max_steps={args.max_steps} eff_batch={args.batch_size*args.grad_accum} ckpt={args.baseline_ckpt}")

    save_steps = {int(x) for x in args.save_steps.split(",") if x}
    t0 = time.time()
    running_loss = []

    for step in range(args.max_steps + 1):
        lr_now = get_lr(step, args)
        for g in optimizer.param_groups: g["lr"] = lr_now
        alpha_now = get_alpha(step, args)

        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum):
                idxs = rng.integers(0, n_train, size=args.batch_size)
                tokens_np = np.stack(
                    [np.asarray(train_mm[i*SEQ_LEN:(i+1)*SEQ_LEN], dtype=np.int64) for i in idxs]
                )
                x = torch.from_numpy(tokens_np).to(device, non_blocking=True)

                if args.policy == "random" or rng.random() > alpha_now:
                    block_orders = torch.stack([torch.randperm(N_BLOCKS, device=device)
                                                for _ in range(args.batch_size)])
                else:
                    block_orders = sample_block_orders(
                        args.policy, B_used, args.batch_size, step, rw_params,
                        raster_order, device, rng,
                    )

                _, loss = model(x, block_orders, targets=x)
                (loss / args.grad_accum).backward()
                running_loss.append(float(loss.item()))

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.eval_interval == 0:
            vr = evaluate_5orders(model, val_tokens, B_real, raster_order, device,
                                  args.batch_size, args.max_eval_batches, step)
            tl = float(np.mean(running_loss[-100:])) if running_loss else float("nan")
            log(f"[eval] step={step:5d} train={tl:.4f} α={alpha_now:.3f} lr={lr_now:.2e} "
                f"rnd={vr['random']:.4f} ras={vr['raster']:.4f} "
                f"rw4={vr['rw_top4_eps0']:.4f} eps={vr['rw_eps015']:.4f} k8={vr['rw_topk8']:.4f} "
                f"elapsed={time.time()-t0:.1f}s")
            with open(tsv_path, "a") as f:
                f.write(f"{step}\t{tl:.4f}\t{alpha_now:.4f}\t{lr_now:.2e}\t"
                        f"{vr['random']:.4f}\t{vr['raster']:.4f}\t{vr['rw_top4_eps0']:.4f}\t"
                        f"{vr['rw_eps015']:.4f}\t{vr['rw_topk8']:.4f}\n")

        if step > 0 and step in save_steps:
            torch.save({"model": model.state_dict(), "model_args": model_args,
                        "step": step, "config": vars(args)},
                       out / f"ckpt_step{step}.pt")
            log(f"[save] ckpt_step{step}.pt")

    torch.save({"model": model.state_dict(), "model_args": model_args,
                "step": args.max_steps, "config": vars(args)},
               out / "ckpt_final.pt")
    log(f"[done] final ckpt saved at step {args.max_steps}")
    log_f.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run `cont_random` for 200 steps**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
mkdir -p probe_results_image_large/grw_e3ctrlsmall/smoke_random_200
CUDA_VISIBLE_DEVICES=0 python -u block_lo_arm_order_network/train_imagelarge_graph_rw.py \
    --policy random \
    --baseline-ckpt nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt \
    --a-block-path probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy \
    --data-train nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin \
    --data-val nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/val.bin \
    --meta nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/meta.pkl \
    --output-dir probe_results_image_large/grw_e3ctrlsmall/smoke_random_200 \
    --max-steps 200 --eval-interval 100 --save-steps "" \
    --batch-size 16 --grad-accum 4 \
    2>&1 | tee probe_results_image_large/grw_e3ctrlsmall/smoke_random_200/cmd.log
```

Expected:
- 3 eval rows in `eval_curve.tsv` (step 0, 100, 200)
- step-0 `val_random` ≈ 7.46–7.48 (matches baseline val)
- step-0 `val_raster` ≈ same magnitude (no aggressive divergence)
- `train_loss` between 7.3 and 7.6
- No `Missing keys when loading ckpt` error
- Final `ckpt_final.pt` exists

If `val_random` at step 0 differs from baseline val (7.46) by more than 0.1, **stop**: ckpt loading is broken.

- [ ] **Step 3: Sanity assertions on smoke output**

Run:
```bash
python3 - <<'EOF'
import csv
rows = list(csv.DictReader(open("probe_results_image_large/grw_e3ctrlsmall/smoke_random_200/eval_curve.tsv"), delimiter="\t"))
assert len(rows) >= 3, f"Expected ≥3 eval rows, got {len(rows)}"
s0 = float(rows[0]["val_random"])
assert 7.30 < s0 < 7.60, f"step-0 val_random={s0} not in [7.30, 7.60]; ckpt load suspect"
print(f"OK: smoke val_random@step0 = {s0:.4f}, eval rows = {len(rows)}")
EOF
```

Expected: `OK: smoke val_random@step0 = 7.46xx, eval rows = 3`.

- [ ] **Step 4: Commit**

```bash
git add block_lo_arm_order_network/train_imagelarge_graph_rw.py
git commit -m "image grw: random-policy continuation trainer + smoke test"
```

---

### Task 3: Smoke-test the four non-random policies (200 steps each)

**Files:**
- Modify: none (trainer already supports all 5 policies)
- Test outputs: `probe_results_image_large/grw_e3ctrlsmall/smoke_{graph_rw,raster,shuffled_B,eps015}_200/`

- [ ] **Step 1: Run all four non-random policies sequentially (200 steps each)**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
for POLICY in graph_rw raster shuffled_B eps015; do
    EXTRA_ARGS=""
    if [ "$POLICY" = "eps015" ]; then EXTRA_ARGS="--rw-top-k 0 --rw-epsilon 0.15"; fi
    OUT=probe_results_image_large/grw_e3ctrlsmall/smoke_${POLICY}_200
    mkdir -p $OUT
    echo "=== smoke $POLICY ==="
    CUDA_VISIBLE_DEVICES=0 python -u block_lo_arm_order_network/train_imagelarge_graph_rw.py \
        --policy $POLICY \
        --baseline-ckpt nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt \
        --a-block-path probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy \
        --data-train nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin \
        --data-val nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/val.bin \
        --meta nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/meta.pkl \
        --output-dir $OUT \
        --max-steps 200 --eval-interval 100 --save-steps "" \
        --batch-size 16 --grad-accum 4 \
        $EXTRA_ARGS 2>&1 | tee $OUT/cmd.log
done
```

Expected: all four runs finish; each produces `eval_curve.tsv` with 3 eval rows; no exceptions.

- [ ] **Step 2: Cross-policy sanity table**

Run:
```bash
python3 - <<'EOF'
import csv
for pol in ["random", "graph_rw", "raster", "shuffled_B", "eps015"]:
    path = f"probe_results_image_large/grw_e3ctrlsmall/smoke_{pol}_200/eval_curve.tsv"
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    s0 = float(rows[0]["val_random"])
    s_last = rows[-1]
    print(f"{pol:12s} | step0 rnd={s0:.4f} | last train={s_last['train_loss']} α={s_last['alpha']} "
          f"rw4={s_last['val_rw_top4_eps0']} ras={s_last['val_raster']}")
EOF
```

Expected:
- All five `step0 rnd` values are within ±0.02 of each other (same ckpt, same eval data, different seed only)
- `cont_raster` `val_raster` at step 200 is **lower** than `cont_random` `val_raster` (raster training improves raster eval — this is the unit-test for the train-eval pairing being correct)
- `cont_graph_rw` `val_rw_top4_eps0` at step 200 is **lower** than `cont_random` `val_rw_top4_eps0` (same logic for RW-top4)
- `cont_shuffled_B` looks like `cont_random` (its sampled orders are essentially random)

If `cont_raster` does NOT beat `cont_random` on `val_raster` after 200 steps, **stop**: training-eval coordinate mismatch likely.

- [ ] **Step 3: Commit**

```bash
git add probe_results_image_large/grw_e3ctrlsmall/smoke_*/eval_curve.tsv probe_results_image_large/grw_e3ctrlsmall/smoke_*/config.json probe_results_image_large/grw_e3ctrlsmall/smoke_*/train_log.txt
git commit -m "image grw: 5-policy smoke tests, coordinate alignment verified"
```

---

### Task 4: Write the 5-arm orchestration script

**Files:**
- Create: `scripts/run_imagelarge_grw_5arm.sh`

- [ ] **Step 1: Write the orchestration shell**

Create `scripts/run_imagelarge_grw_5arm.sh`:

```bash
#!/bin/bash
# Sequential 5-arm Graph-RW continuation on E3-control-small.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
CKPT=$REPO/nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt
ABLOCK=$REPO/probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy
DATA_DIR=$REPO/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
ROOT=$REPO/probe_results_image_large/grw_e3ctrlsmall

MAX_STEPS="${MAX_STEPS:-5000}"
GPU="${GPU:-0}"

cd "$REPO"

run_arm() {
    local POL="$1"; shift
    local OUT="$ROOT/cont_${POL}"
    mkdir -p "$OUT"
    echo "================================================="
    echo "[`date -Iseconds`] arm=$POL → $OUT"
    echo "================================================="
    CUDA_VISIBLE_DEVICES="$GPU" python -u \
        block_lo_arm_order_network/train_imagelarge_graph_rw.py \
        --policy "$POL" \
        --baseline-ckpt "$CKPT" \
        --a-block-path "$ABLOCK" \
        --data-train "$DATA_DIR/train.bin" \
        --data-val "$DATA_DIR/val.bin" \
        --meta "$DATA_DIR/meta.pkl" \
        --output-dir "$OUT" \
        --max-steps "$MAX_STEPS" \
        --eval-interval 500 \
        --save-steps "1000,3000,$MAX_STEPS" \
        "$@" 2>&1 | tee "$OUT/cmd.log"
}

run_arm random
run_arm graph_rw
run_arm raster
run_arm shuffled_B
run_arm eps015 --rw-top-k 0 --rw-epsilon 0.15

echo "[`date -Iseconds`] ALL 5 ARMS DONE"
```

- [ ] **Step 2: Make executable and dry-run check (echo-only)**

Run:
```bash
chmod +x scripts/run_imagelarge_grw_5arm.sh
bash -n scripts/run_imagelarge_grw_5arm.sh && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_imagelarge_grw_5arm.sh
git commit -m "image grw: 5-arm orchestration shell"
```

---

### Task 5: Launch the 5-arm 5000-step continuation

**Files:**
- Produces: `probe_results_image_large/grw_e3ctrlsmall/cont_{random,graph_rw,raster,shuffled_B,eps015}/`

- [ ] **Step 1: Estimate per-arm wall-clock from smoke**

Run:
```bash
python3 - <<'EOF'
# From smoke 200 steps with grad_accum=4, the real run uses grad_accum=16,
# so per-step cost is ~4x higher. With smoke ~200 steps in S sec, expect
# 5000 steps × (16/4) ≈ 100,000 micro-steps; baseline is ~80 ms/(micro-step)
# at this model size, so ~133 min per arm. Five arms ≈ 11 h.
print("Estimated per-arm wall-clock at grad_accum=16, max_steps=5000:")
print("  ~80 ms per micro-step × 16 micro-steps × 5000 steps = 6400 s = 107 min")
print("  Five arms sequentially: ~9 h")
EOF
```

If this is unacceptable, drop `MAX_STEPS=3000` (cuts to ~5.5 h) or `grad-accum=8` (cuts to ~4.5 h with eff-batch halved — but mark in SUMMARY).

- [ ] **Step 2: Launch the 5-arm run in the background**

Run:
```bash
cd /home/admin/lyuyuhuan/order_lyu
nohup bash scripts/run_imagelarge_grw_5arm.sh \
    > probe_results_image_large/grw_e3ctrlsmall/run_all.log 2>&1 &
RUNPID=$!
echo "PID=$RUNPID" > probe_results_image_large/grw_e3ctrlsmall/run_all.pid
sleep 10
tail -20 probe_results_image_large/grw_e3ctrlsmall/run_all.log
ps -p $RUNPID
```

Expected: tail shows the random arm starting + iter logging; `ps` shows the process alive.

- [ ] **Step 3: Periodic progress check (manual, not in the plan flow)**

Run when convenient:
```bash
for d in probe_results_image_large/grw_e3ctrlsmall/cont_*/; do
    echo "=== $d ==="
    tail -3 "$d/eval_curve.tsv" 2>/dev/null || echo "  (no eval_curve.tsv yet)"
done
```

- [ ] **Step 4: Verify all 5 arms produced a final ckpt + eval curve**

After completion (run_all.log says `ALL 5 ARMS DONE`):

```bash
for POL in random graph_rw raster shuffled_B eps015; do
    D=probe_results_image_large/grw_e3ctrlsmall/cont_$POL
    test -f $D/ckpt_final.pt && echo "$POL: ckpt OK $(stat -c '%y' $D/ckpt_final.pt)" || echo "$POL: MISSING ckpt_final.pt"
    test -f $D/eval_curve.tsv && echo "$POL: tsv $(wc -l < $D/eval_curve.tsv) lines" || echo "$POL: MISSING eval_curve.tsv"
done
```

Expected: 5 × OK + 5 × `tsv 12 lines` (10 evals + header + step 5000).

- [ ] **Step 5: Commit eval curves and configs (do NOT commit ckpts; they're 100MB each)**

```bash
git add probe_results_image_large/grw_e3ctrlsmall/cont_*/eval_curve.tsv \
        probe_results_image_large/grw_e3ctrlsmall/cont_*/config.json \
        probe_results_image_large/grw_e3ctrlsmall/cont_*/train_log.txt \
        probe_results_image_large/grw_e3ctrlsmall/run_all.log
git commit -m "image grw: 5-arm continuation training complete (5000 steps each)"
```

---

### Task 6: Post-train attention re-extraction + dual-level diagnostic for each arm

**Files:**
- Create: `scripts/post_grw_diagnose.sh`
- Produces: `probe_results_image_large/grw_e3ctrlsmall/cont_*/attention/{A_global.npy, A_block_8x8.npy, diagnostics.json}`

- [ ] **Step 1: Write the per-arm post-diagnose script**

Create `scripts/post_grw_diagnose.sh`:

```bash
#!/bin/bash
# For each 5-arm cont_<policy>/, extract A_global from ckpt_final.pt and run dual-level diagnostic.
set -euo pipefail

REPO=/home/admin/lyuyuhuan/order_lyu
DATA_DIR=$REPO/nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8
ROOT=$REPO/probe_results_image_large/grw_e3ctrlsmall
GPU="${GPU:-0}"

cd "$REPO"

for POL in random graph_rw raster shuffled_B eps015; do
    D="$ROOT/cont_$POL"
    CKPT="$D/ckpt_final.pt"
    ATT="$D/attention"
    mkdir -p "$ATT"
    if [[ ! -f "$CKPT" ]]; then
        echo "[skip] $POL: $CKPT missing"
        continue
    fi
    echo "=================================================="
    echo "[`date -Iseconds`] post-diagnose: $POL"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES="$GPU" python block_lo_arm_order_network/extract_image_attention_e2.py \
        --ckpt "$CKPT" \
        --data "$DATA_DIR/val.bin" --meta "$DATA_DIR/meta.pkl" \
        --out "$ATT" --n-images 500 --M-passes 3 --device cuda:0

    python scripts/diagnose_e3_control_dual_level.py \
        --a-global "$ATT/A_global.npy" \
        --outdir "$ATT" \
        --label "cont_$POL (post 5000 steps)"
done

echo "[`date -Iseconds`] ALL POST-DIAGNOSE DONE"
```

- [ ] **Step 2: Make executable + run**

```bash
chmod +x scripts/post_grw_diagnose.sh
bash scripts/post_grw_diagnose.sh 2>&1 | tee probe_results_image_large/grw_e3ctrlsmall/post_diagnose.log
```

Expected: each arm gets `A_global.npy (256×256)`, `A_block_8x8.npy (64×64)`, `diagnostics.json` (token+block tables), and `SUMMARY.md`. Total time ~5 min per arm × 5 = ~25 min.

- [ ] **Step 3: Sanity: did `cont_graph_rw` preserve / sharpen its local B?**

Run:
```bash
python3 - <<'EOF'
import json
for pol in ["random", "graph_rw", "raster", "shuffled_B", "eps015"]:
    with open(f"probe_results_image_large/grw_e3ctrlsmall/cont_{pol}/attention/diagnostics.json") as f:
        d = json.load(f)
    blk = d["block_level_8x8"][0]
    print(f"{pol:12s} | block mean_manh={blk['mean_manh']:.3f} P(d≤1)={blk['p_dist_le1']:.3f}")
print()
print("Pre-continuation E3-control-small (block-level): mean_manh=1.156, P(d≤1)=0.984")
EOF
```

Expected (descriptive, NOT a hard pass/fail): record each arm's post-train block locality as an **observed drift metric** against the pre-continuation baseline (1.156 / 0.984). Do **not** assume `cont_random` preserves the near-perfect locality — different continuation policies can reshape the attention graph, and `cont_random` may itself drift. The question is comparative: how does each arm's post-train B compare to pre-continuation, and to each other. If `cont_graph_rw` sharpens (manh ≤ 1.16) while `cont_shuffled_B`/`cont_eps015` drift more, that is evidence the policy shaped the graph; but no single arm is required to hold the 1.156/0.984 numbers.

- [ ] **Step 4: Commit**

```bash
git add scripts/post_grw_diagnose.sh \
        probe_results_image_large/grw_e3ctrlsmall/cont_*/attention/diagnostics.json \
        probe_results_image_large/grw_e3ctrlsmall/cont_*/attention/SUMMARY.md \
        probe_results_image_large/grw_e3ctrlsmall/post_diagnose.log
# Note: do NOT commit *.npy heatmap pngs or A_global.npy (too large for git)
git commit -m "image grw: post-train A-extraction + dual-level diagnostic on all 5 arms"
```

---

### Task 7: Cross-run summary + verdict

**Files:**
- Create: `scripts/grw_5arm_crossrun_summary.py`
- Produces: `probe_results_image_large/grw_e3ctrlsmall/SUMMARY.md`

- [ ] **Step 1: Write the cross-run summariser**

Create `scripts/grw_5arm_crossrun_summary.py`:

```python
"""Cross-run summary for the 5-arm Graph-RW continuation on E3-control-small.

Reads:
  probe_results_image_large/grw_e3ctrlsmall/cont_*/eval_curve.tsv
  probe_results_image_large/grw_e3ctrlsmall/cont_*/attention/diagnostics.json

Writes:
  probe_results_image_large/grw_e3ctrlsmall/SUMMARY.md
"""

import csv, json
from pathlib import Path

ROOT = Path("probe_results_image_large/grw_e3ctrlsmall")
ARMS = ["random", "graph_rw", "raster", "shuffled_B", "eps015"]

def final_row(pol):
    path = ROOT / f"cont_{pol}" / "eval_curve.tsv"
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    return rows[-1] if rows else None

def post_block(pol):
    p = ROOT / f"cont_{pol}" / "attention" / "diagnostics.json"
    if not p.exists():
        return None
    return json.load(open(p))["block_level_8x8"][0]

eval_rows = {pol: final_row(pol) for pol in ARMS}
blocks = {pol: post_block(pol) for pol in ARMS}

# Table 1: final eval losses
md = []
md.append("# Image VQ Graph-RW Continuation on E3-control-small — RESULT\n\n")
md.append("**Date:** 2026-05-19\n")
md.append("**Source ckpt:** E3-control-small l4h8e256 patch2x2 best-val (iter 16250, val=7.463).\n")
md.append("**Continuation:** 5000 steps each, eff-batch 256, lr 1e-4, α=0.9 (warmup 1000), B fixed from pre-train A_block_8x8.\n\n")

md.append("## Final eval losses (step 5000)\n\n")
md.append("| Arm | train | val_random | val_raster | val_rw_top4_eps0 | val_rw_eps015 | val_rw_topk8 |\n")
md.append("|---|---:|---:|---:|---:|---:|---:|\n")
for pol in ARMS:
    r = eval_rows[pol]
    if r is None:
        md.append(f"| cont_{pol} | (no data) | | | | | |\n"); continue
    md.append(f"| cont_{pol} | {r['train_loss']} | {r['val_random']} | {r['val_raster']} | "
              f"{r['val_rw_top4_eps0']} | {r['val_rw_eps015']} | {r['val_rw_topk8']} |\n")

# Table 2: vs cont_random deltas
rnd = eval_rows["random"]
if rnd:
    md.append("\n## Δ vs cont_random (negative = better than random baseline)\n\n")
    md.append("| Arm | Δ val_random | Δ val_raster | Δ val_rw_top4_eps0 | Δ val_rw_eps015 | Δ val_rw_topk8 |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for pol in ARMS:
        if pol == "random": continue
        r = eval_rows[pol]
        if r is None: continue
        def d(col): return float(r[col]) - float(rnd[col])
        md.append(f"| cont_{pol} | {d('val_random'):+.4f} | {d('val_raster'):+.4f} | "
                  f"{d('val_rw_top4_eps0'):+.4f} | {d('val_rw_eps015'):+.4f} | {d('val_rw_topk8'):+.4f} |\n")

# Table 3: post-train attention preserved?
md.append("\n## Post-train block-level locality (B-side preserved?)\n\n")
md.append("| Arm | mean_manh | P(d≤1) | P(d≤2) | s_readiness |\n")
md.append("|---|---:|---:|---:|---:|\n")
md.append("| pre-continuation (E3-control-small) | 1.156 | 0.984 | 0.984 | 2.641 |\n")
for pol in ARMS:
    b = blocks[pol]
    if b is None:
        md.append(f"| cont_{pol} | (no data) | | | |\n"); continue
    md.append(f"| cont_{pol} | {b['mean_manh']:.3f} | {b['p_dist_le1']:.3f} | "
              f"{b['p_dist_le2']:.3f} | {b['s_readiness']:.3f} |\n")

# Verdict
md.append("\n## Verdict\n\n")
if rnd and eval_rows["graph_rw"]:
    grw, srand = eval_rows["graph_rw"], eval_rows["random"]
    delta_rw4 = float(grw["val_rw_top4_eps0"]) - float(srand["val_rw_top4_eps0"])
    delta_rnd = float(grw["val_random"]) - float(srand["val_random"])
    shuf = eval_rows.get("shuffled_B")
    eps = eval_rows.get("eps015")
    shuf_dead = (shuf is None) or abs(float(shuf["val_rw_top4_eps0"]) - float(srand["val_rw_top4_eps0"])) < 0.01
    eps_dead = (eps is None) or abs(float(eps["val_rw_top4_eps0"]) - float(srand["val_rw_top4_eps0"])) < 0.01

    if delta_rw4 < -0.005 and shuf_dead and eps_dead:
        md.append(f"**Local structure converts into training benefit.** cont_graph_rw beats cont_random on val_rw_top4_eps0 by {-delta_rw4:.4f} nat, "
                  f"controls (shuffled_B, eps015) show no improvement.\n")
        if delta_rnd > 0.005:
            md.append(f"There is a specialization tax: val_random is {delta_rnd:+.4f} worse. Document the trade-off.\n")
        else:
            md.append(f"No specialization tax (val_random Δ={delta_rnd:+.4f}); the structured order generalizes.\n")
    elif abs(delta_rw4) < 0.005:
        md.append(f"**No measurable training benefit.** cont_graph_rw and cont_random within 0.005 nat on val_rw_top4_eps0. "
                  f"Either the budget (5000 steps) is too short, or random-order training already finds this structure. "
                  f"See cont_raster Δ to test the budget-vs-structure question.\n")
    else:
        md.append(f"**cont_graph_rw underperforms cont_random** by {delta_rw4:+.4f} nat on val_rw_top4_eps0. Unexpected; investigate.\n")

open(ROOT / "SUMMARY.md", "w").write("".join(md))
print(f"Wrote {ROOT/'SUMMARY.md'}")
```

- [ ] **Step 2: Run it**

```bash
python scripts/grw_5arm_crossrun_summary.py
cat probe_results_image_large/grw_e3ctrlsmall/SUMMARY.md
```

Expected: SUMMARY.md is printed; verdict text matches the decision matrix at the top of this plan.

- [ ] **Step 3: Update memory `image32_alignment_plan`**

Append a section to `/home/admin/.claude/projects/-home-admin-lyuyuhuan-order-lyu/memory/image32_alignment_plan.md`:

```
**Graph-RW continuation on E3-control-small (2026-05-19+, 5-arm 5000-step):**
- Outcome (read from probe_results_image_large/grw_e3ctrlsmall/SUMMARY.md): [PASTE VERDICT LINE]
- Best arm by val_rw_top4_eps0: [PASTE]
- shuffled_B / eps015 dead-controls: [PASTE]
- Block-level attention preserved through continuation: pre 1.156/0.984 → post graph_rw [PASTE mean_manh/P(d≤1)]
- This closes the image VQ line from "structure discovered" to "structure utility" (positive/negative/inconclusive).
```

Update the `description:` line at top of `image32_alignment_plan.md` to reflect the new state.

- [ ] **Step 4: Commit**

```bash
git add scripts/grw_5arm_crossrun_summary.py probe_results_image_large/grw_e3ctrlsmall/SUMMARY.md
git commit -m "image grw: cross-run summary + verdict on 5-arm continuation"
```

---

## Self-review checklist (already applied)

- **Spec coverage:** Every arm from user's matrix (random / graph_rw / raster / shuffled_B / eps015) has a config; pre-train alignment check (Task 1) added; post-train attention re-extraction (Task 6) added so "does structure survive" is answerable.
- **Placeholders:** None. Every code block is complete; every command has expected output.
- **Type consistency:** `block_orders` is consistently `(B, 64) torch.long on device`; `B_real`/`B_used` is `(64, 64) float32 np.ndarray`; `val_tokens` is `(N, 256) torch.long on CPU` (moved to device per-batch).
- **Key risks already mitigated:** Coordinate alignment validated in Task 1 Step 3 + Task 3 Step 2 (raster training must improve raster eval). Ckpt-loading bug class avoided by reusing the proven `extract_image_attention_e2.py` load pattern.
- **What's deliberately NOT in scope:** EMA refresh of B during continuation (round-2 candidate); longer 20k-step training; scale sweep (l4/l6/l8); RoPE / 2D PE; alternate aggregation grids. These are recorded in the post-mortem if the round-1 result asks for them.
