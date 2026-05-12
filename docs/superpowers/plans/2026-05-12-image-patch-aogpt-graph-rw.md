# Image-Patch AOGPT + Graph-RW v2 Minimum Closed Loop

> **For agentic workers:** This plan covers the minimum closed loop (smoke). Long training runs come after smoke validates.

**Goal:** Extend the existing text-token AOGPT + Graph-RW v2 framework to CIFAR-10 image patches (N=64, patch_dim=48). Validate that attention-derived global order on image patches recovers spatial / structural organization (not L2R), and can be reused as a Graph-RW reveal-order curriculum.

**Architecture:**
- Brand-new `image_order/` package that lives alongside the text pipeline; do **NOT** touch `block_lo_arm_order_network/*` or `AO-GPT-MDM/*`.
- Reuse `block_lo_arm_order_network/directed_graph_policy.py` verbatim — it is pure NumPy and N-agnostic.
- Re-implement an image-flavored AOGPT (patch projection in / MSE patch reconstruction out) but keep the AdaLN-conditioned reveal-order mask machinery identical to the text model.
- Outputs land under `probe_results_image/`.

**Tech Stack:** PyTorch (existing repo deps), NumPy, matplotlib (already present), CIFAR-10 from local cache `/home/admin/bw/data/cifar-10-batches-py/`.

**Key reuse / new split:**

| Concern | Source |
| --- | --- |
| any-order autoregressive shuffle/unshuffle | adapt `model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py` shuffle pattern |
| Graph-RW policy (B, source, progressive_rw, sample_order, sample_orders) | `directed_graph_policy.py` (import, do not copy) |
| A_global → B = A^T | use `build_directed_graph(A_global)` from existing file |
| α-mixed loss, refresh, eval curve TSV | structure copied from `train_aogpt_graph_rw.py` (image version) |
| Order diagnostics: locality / paths / heatmap | new image-only file |

---

## Constants (use throughout)

```python
IMG_SIZE = 32
PATCH_GRID = 8          # 8x8 = 64 patches
PATCH_SIZE = 4          # 4x4 pixels per patch
PATCH_DIM  = PATCH_SIZE * PATCH_SIZE * 3   # 48
N_PATCHES  = PATCH_GRID * PATCH_GRID       # 64

CIFAR_DIR  = "/home/admin/bw/data/cifar-10-batches-py"
OUT_ROOT   = Path(__file__).resolve().parent.parent / "probe_results_image"
```

Default model: `n_embd=256, n_layer=4, n_head=8, dropout=0.0, AdaLN cond_dim=128` (matches the text model's AdaLN signature so the Block class can be reused without modification).

Default smoke training: batch=64, steps=500, lr=3e-4 cosine, warmup=50.

Default baseline (Stage 0): batch=128, steps=10000, lr=3e-4 cosine, warmup=200, eval every 500.

Default Graph-RW continuation (Stage 2): from baseline ckpt, steps=5000, lr=3e-5 cosine, α warmup 0→0.9 over 1500 steps, τ_start=0.1 τ_step=0.1, top_k=4, eps=0, refresh_every=2000 (optional, can be disabled with `--no-refresh` for the smoke).

---

## File Structure

```
image_order/
  __init__.py
  data_image_patches.py           # CIFAR-10 loader + patch <-> image utilities
  model_image_aogpt.py            # Image AOGPT (continuous I/O, MSE loss)
  graph_rw_image.py               # thin wrapper around directed_graph_policy
  extract_image_attention.py      # A_global / B_global extraction
  train_image_random.py           # random-order baseline pretraining
  train_image_graph_rw.py         # α-mixed continuation from baseline
  eval_image_orders.py            # frozen ckpt eval under 5 orders
  diagnose_image_orders.py        # locality + path + heatmaps
  run_image_smoke.sh              # minimal closed-loop runner
  run_image_experiments.sh        # full baseline + Graph-RW + ablations

probe_results_image/              # output root (gitignored)
  baseline/<run>/                 # ckpt.pt, train_log.txt, eval_curve.tsv
  attention/<run>/A_global.npy, B_global.npy, A_heatmap.png
  graph_rw/<run>/                 # ckpt, eval_curve, refresh diag
  diagnostics/<run>/              # order_path_*.png, avg_step_heatmap.png, ...
```

`__init__.py` exists so files can `from image_order.<x> import ...` after `sys.path.insert(0, repo_root)`.

---

## Task 1: data_image_patches.py

**Files:** Create `image_order/data_image_patches.py`.

- [ ] **Step 1: load CIFAR-10 batches offline** — `load_cifar10(split: 'train'|'test') -> (images: uint8 (N,3,32,32), labels)` reading `data_batch_1..5` + `test_batch` via `pickle.load(open(p,'rb'), encoding='latin1')`. No `torchvision.datasets.CIFAR10` to avoid any network attempts.
- [ ] **Step 2: image ↔ patch utilities**
  - `image_to_patches(x: (B,3,32,32) float) -> (B,64,48)`: reshape via `unfold(2,4,4).unfold(3,4,4).permute(...).reshape(B,64,48)`. Row-major raster: patch index `r*8+c`.
  - `patches_to_image(p: (B,64,48)) -> (B,3,32,32)`: inverse.
  - `grid_pos(idx: int) -> (row, col)`, `manhattan(idx_a, idx_b) -> int`, `is_raster_order(perm) -> bool`.
  - `RASTER_ORDER = np.arange(64, dtype=np.int64)`.
- [ ] **Step 3: torch Dataset wrapping precomputed patches** — `CIFAR10Patches(split, normalize=True)`; images → float32 / 127.5 - 1.0; precompute patches once and store on CPU; `__getitem__` returns `(patches, label)` patches shape `(64,48)`.
- [ ] **Step 4: smoke `if __name__ == "__main__":`** — instantiate train/test datasets, print shapes, assert round-trip `patches_to_image(image_to_patches(x)) == x` within 1e-5.
- [ ] **Step 5: commit** — `git add image_order/__init__.py image_order/data_image_patches.py`.

```python
# Round-trip check shown in module __main__:
ds = CIFAR10Patches("train")
x  = torch.stack([ds[i][0] for i in range(4)])         # (4,64,48)
img = patches_to_image(x)                              # (4,3,32,32)
x2  = image_to_patches(img)
assert torch.allclose(x, x2, atol=1e-5)
print("OK", ds.patches.shape, ds.images_minmax)
```

---

## Task 2: model_image_aogpt.py

**Files:** Create `image_order/model_image_aogpt.py`. Reuse `Block`, `FinalLayer`, `RMSNorm`, `modulate` from the text model by importing them.

- [ ] **Step 1: ImageAOGPTConfig dataclass** — `n_patches=64, patch_dim=48, n_embd=256, n_layer=4, n_head=8, cond_dim=128, dropout=0.0, bias=True`.
- [ ] **Step 2: ImageAOGPT.__init__** — token embedding replaced by `patch_proj = nn.Linear(patch_dim, n_embd)`; keep `wpe = nn.Embedding(n_patches+1, n_embd)` (position) and `wtpe = nn.Embedding(n_patches, cond_dim)` (target position cond); keep `wnonee = nn.Embedding(1, n_embd)`; reuse imported `Block(cfg_like)` — pass a shim config exposing `.n_embd, .n_head, .dropout, .bias, .block_size=n_patches`; reuse imported `FinalLayer`; replace `lm_head` with `patch_head = nn.Linear(n_embd, patch_dim)`.
- [ ] **Step 3: shuffle/unshuffle** — copy the 2 helper methods that operate on `(B, T, D)` (D=patch_dim) instead of `(B, T)` ids. Build new variants that take a 3-d tensor.
- [ ] **Step 4: forward_fn(patches, orders, return_attentions=False)** — same flow as text model: shuffle patches, project, prepend learned [None] token, add `wpe` & shuffled target-pos cond, run blocks with AdaLN c=target_pos_emb, FinalLayer, `patch_head`. Loss: MSE between `pred[:, :-1, :]` (predictions at each prefix-step) and `targets = shuffled_patches` aligned the same way the text model aligns `shift_logits` vs `shift_targets`. Use `F.mse_loss(reduction='mean')`.
- [ ] **Step 5: helper methods** — `sample_random_orders`, `set_ascending_orders` (raster), `forward(mode='Random'|'AR'|None, orders=None)` like text model.
- [ ] **Step 6: configure_optimizers** — adapt fused AdamW logic verbatim.
- [ ] **Step 7: __main__ smoke** — instantiate with defaults, forward random batch `(4,64,48)`, assert loss is scalar, `loss.backward()` works, prints param count (target ≈3–5 M).
- [ ] **Step 8: commit**.

The key invariant: `forward_fn` must accept a `(B, 64)` long-tensor `orders` and produce per-step prediction targets identical in semantics to the text model — i.e. at position t the model has seen patches `orders[:, :t]` (plus the [None] prefix) and predicts patch `orders[:, t]`. This guarantees the Graph-RW order machinery is plug-compatible.

---

## Task 3: graph_rw_image.py

**Files:** Create `image_order/graph_rw_image.py`.

- [ ] **Step 1: import policy** — `from directed_graph_policy import build_directed_graph, sample_order, sample_orders, compute_source` (set `sys.path` to repo root + `block_lo_arm_order_network`).
- [ ] **Step 2: default params**
  ```python
  IMAGE_RW_PARAMS = dict(
      tau_start=0.10, tau_step=0.10,
      alpha_dep=0.5, alpha_pr=0.85,
      beta_sup=1.0, beta_fut=0.5, beta_src=0.2, beta_loc=0.5,
      top_k=4, epsilon_uniform=0.0,
  )
  POLICY = "progressive_rw"
  ```
- [ ] **Step 3: helpers**
  - `make_B_from_attention(A_global: np.ndarray) -> np.ndarray`: zero diagonal, then `build_directed_graph`.
  - `sample_image_orders_batch(B, params, batch_size, seed_base, step) -> LongTensor (B,64)`: loop `sample_order` with unique seed per batch element, same pattern as `sample_rw_batch_orders` in text trainer.
  - `sample_orders_for_eval(B, params, K, seed_base) -> dict` — passes through to `sample_orders` and additionally exposes `mean_manhattan_step` for image diagnostics (computed here so `diagnose_image_orders.py` can reuse).
- [ ] **Step 4: __main__ smoke** — build fake B from random `A_global`, sample 10 orders, assert each is a valid permutation of `range(64)`, print `tau_vs_l2r_mean` (here L2R = raster scan; should be low).
- [ ] **Step 5: commit**.

---

## Task 4: train_image_random.py (+ smoke)

**Files:** Create `image_order/train_image_random.py`.

- [ ] **Step 1: argparse** — `--output-dir`, `--max-steps` (default 500 smoke / 10000 baseline), `--batch-size` (64 / 128), `--lr` (3e-4), `--warmup-iters` (50/200), `--eval-interval` (100/500), `--seed`, `--device`.
- [ ] **Step 2: train/val split** — train = CIFAR-10 train (50000); val = first 1000 of test.
- [ ] **Step 3: training loop** — random patch order per sample per step, cosine LR with warmup, grad clip 1.0, log loss every 25 steps.
- [ ] **Step 4: eval_image_random(model, val_loader, K=3 seeds)** — log `val_random_loss` and `val_raster_loss` to eval_curve.tsv.
- [ ] **Step 5: smoke acceptance** — at step 0 random-loss ≈ uniform-noise baseline; after 500 steps random-loss should drop by ≥ 25% from step-0 value. If not, halt and report.
- [ ] **Step 6: save** — `ckpt_step{N}.pt` containing `{'model_state_dict','args','train_losses','best_val_loss','best_step','config': asdict(cfg)}`.
- [ ] **Step 7: commit**.

---

## Task 5: extract_image_attention.py

**Files:** Create `image_order/extract_image_attention.py`.

- [ ] **Step 1: load model from ckpt + frozen eval mode**.
- [ ] **Step 2: forward over `num_seqs` random val patches with `return_attentions=True`** — collect raw attention `(L,H,T,T)` where T=65 (with [None]).
- [ ] **Step 3: drop [None] row/col** → (L,H,64,64); average over heads/layers; M=3 random-order passes per image; zero diagonal.
- [ ] **Step 4: aggregate to A_global** — mean over images → (64,64); save `A_global.npy`; `B_global = A_global.T` (zero diag), save `B_global.npy`.
- [ ] **Step 5: heatmap PNG** — matplotlib imshow of A_global with grid lines at 8-block boundaries; save `A_global_heatmap.png`.
- [ ] **Step 6: optional `--per-head` head selection** — keep simple for smoke: average all heads/layers.
- [ ] **Step 7: __main__** — `--ckpt`, `--num-seqs 200`, `--output-dir probe_results_image/attention/<run>`.
- [ ] **Step 8: commit**.

---

## Task 6: diagnose_image_orders.py

**Files:** Create `image_order/diagnose_image_orders.py`. This is the most important deliverable for the smoke — it answers the research question.

- [ ] **Step 1: locality metric** — `def manhattan_locality(orders: (K,64)) -> dict`: compute consecutive grid distance per step, return `{mean, median, per_step}`.
- [ ] **Step 2: center distance** — `def center_distance_curve(orders) -> (64,)` using `grid_pos` and `((3.5, 3.5))` center; mean across K.
- [ ] **Step 3: order path PNG** — `def plot_order_path(order, save_path, title)`: 8x8 grid with cell colored by reveal step (viridis) + arrow overlay between consecutive patches.
- [ ] **Step 4: avg step heatmap** — `def plot_avg_step_heatmap(orders, save_path)`: per patch, mean step at which it is revealed; render as 8x8 imshow.
- [ ] **Step 5: transition heatmap** — `def plot_transition_heatmap(orders, save_path)`: 64x64 transition counts.
- [ ] **Step 6: compare methods** — `if __name__ == "__main__":` driver takes `--B`, `--output-dir`, configs `[("graph_rw_top4", top_k=4), ("graph_rw_top8", top_k=8), ("graph_rw_eps015", epsilon=0.15), ("random", random_baseline), ("raster", raster_only)]`; for each, sample 200 orders, write `summary.tsv` with `mean_manhattan / median_manhattan / mean_center_distance / tau_vs_raster`, plus path PNG (1 example) + avg_step heatmap PNG + transition PNG.
- [ ] **Step 7: success criterion (for smoke report)** —
  - random baseline mean Manhattan ≈ uniform expected ≈ 5.0
  - Graph-RW top_k=4 mean Manhattan should be **noticeably lower** than random (target ≤ 3.0). If equal, A_global lacks spatial structure → flag for investigation.
- [ ] **Step 8: commit**.

---

## Task 7: train_image_graph_rw.py (skeleton only for smoke)

**Files:** Create `image_order/train_image_graph_rw.py`. For the **first deliverable** we build the skeleton but the user's "先做什么" list says we don't need to run it — only run baseline + extract + diagnostics. Keep this thin and verify forward/backward path works.

- [ ] **Step 1: argparse** — `--baseline-ckpt`, `--a-path`, `--output-dir`, `--max-steps` (smoke 500 / pilot 5000), `--alpha-target` (0.9), `--alpha-warmup` (1500), `--rw-top-k` (4), `--no-refresh` (flag for smoke).
- [ ] **Step 2: loop** — clone the hard-α-mixing branch from `train_aogpt_graph_rw.py:692-709`, simplified (no checkpoint permutation, no token-vs-block coordinate translation; patch indices ARE the order directly).
- [ ] **Step 3: eval — 4 order modes** — random / raster / graph_rw / model_order (raster≡model_order here since no permutation training, so collapse to 3 distinct metrics: random, raster, graph_rw).
- [ ] **Step 4: __main__ smoke** — 100-step run to prove it converges (do not gate the deliverable on this — only forward/backward smoke).
- [ ] **Step 5: commit**.

---

## Task 8: eval_image_orders.py + run scripts

**Files:** Create `image_order/eval_image_orders.py`, `image_order/run_image_smoke.sh`, `image_order/run_image_experiments.sh`.

- [ ] **Step 1: eval_image_orders.py** — frozen-ckpt MSE under random / raster / Graph-RW top4 / Graph-RW top8 / Graph-RW eps015; output `eval_summary.tsv`.
- [ ] **Step 2: run_image_smoke.sh** —
  ```bash
  set -e
  cd "$(dirname "$0")/.."
  python -u image_order/data_image_patches.py
  python -u image_order/model_image_aogpt.py
  python -u image_order/train_image_random.py --max-steps 500 \
      --output-dir probe_results_image/baseline/smoke500
  python -u image_order/extract_image_attention.py \
      --ckpt probe_results_image/baseline/smoke500/ckpt_step500.pt \
      --output-dir probe_results_image/attention/smoke500 --num-seqs 100
  python -u image_order/diagnose_image_orders.py \
      --B probe_results_image/attention/smoke500/B_global.npy \
      --output-dir probe_results_image/diagnostics/smoke500
  ```
- [ ] **Step 3: run_image_experiments.sh** — full Stage 0 baseline + extract + Graph-RW α=0.9 + ablations.
- [ ] **Step 4: commit**.

---

## Task 9: Run smoke and report

- [ ] **Step 1:** `bash image_order/run_image_smoke.sh` end-to-end.
- [ ] **Step 2:** verify each acceptance criterion below.
- [ ] **Step 3:** report file paths (PNG, NPY, TSV) and a 5-line summary to the user.

### Smoke acceptance criteria

1. `data_image_patches.py` round-trip passes (atol 1e-5).
2. `model_image_aogpt.py` forward + backward produces finite MSE; printed param count is sane.
3. `train_image_random.py` 500-step run shows ≥ 25% loss drop from step 0.
4. `extract_image_attention.py` saves `A_global.npy` shape (64,64), no NaN, off-diag mass > 0.
5. `graph_rw_image.py` produces valid permutations (sorted == range(64)) for 200 sampled orders.
6. `diagnose_image_orders.py` produces:
   - PNG: random path, raster path, top_k4 path, top_k8 path
   - PNG: avg_step heatmap (top_k4)
   - PNG: transition heatmap (top_k4)
   - TSV row: `top_k4 mean_manhattan` < `random mean_manhattan` (≤ 3.0 vs ~5.0).

If criterion 6 fails (no spatial locality), this is a meaningful negative result — report it but do not consider it a code defect; it would mean the baseline ckpt's attention is too noisy at 500 steps and we should retry after the 10k-step real baseline.

---

## Research README (added inline to image_order/__init__.py as a docstring)

```
This package extends the text AOGPT + Graph-RW v2 framework to CIFAR-10 image
patches (N=64, patch_dim=48). The research question is NOT image-generation
quality — it is: does AOGPT's internal attention on image-patch data encode
dataset-level shared structure (spatial locality, region grouping, center vs
edge tendency)? If yes, the same Graph-RW order policy that recovers L2R-like
order from WikiText attention should produce spatially-coherent reveal orders
here, AND using those orders as an any-order curriculum should match or beat
random-order training.

This file deliberately does not duplicate any text-pipeline assumption:
- No block_perm / inv_perm (no checkpoint permutation).
- No L2R oracle (raster is reported only for diagnostics, never as a label).
- A_global is averaged over heads/layers and images.
```

---

## Self-review

**Spec coverage:**
- Section I (dataset / model spec): Tasks 1+2.
- Section II (Graph-RW on image): Task 3.
- Section III (experiment stages 0/1/2): Tasks 4+5+7.
- Section IV (diagnostics): Task 6.
- Section V (output files): all tasks write to `probe_results_image/`.
- Section VI (do not break text pipeline): enforced by `image_order/` isolation.
- Section VII (README): handled in Task 0 / `__init__.py` docstring.
- Section VIII (先做什么): Task 9 is the deliverable gate (smoke + report). Task 7 train_image_graph_rw is skeleton-only for the smoke.

**No placeholders.** All steps name file paths, default constants, and acceptance values.

**Type consistency:** patch tensors are `(B, 64, 48)` float32 throughout; orders are `(B, 64)` int64; A_global / B_global are `(64,64) float32`. Single coherent vocabulary.
