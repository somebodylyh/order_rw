"""Per-(layer, head) order-signal scan — reconstruction of the lost L0H5 scanner.

For one AOGPT checkpoint, forward M*batch_size random-order chunks, build the
per-(layer, head) batch-mean attention graph B and a top-4-head "heavy" baseline
graph, run the CDL-source-start readout on each batch-mean graph, then summarise
the reveal orders with:

  - tau_vs_l2r       : mean Kendall tau between sigma_m and the L2R order.
  - mean_pairwise_tau: teacher diversity across the M batch orders.
  - first_step_entropy: entropy of sigma[:, 0] across the M batch orders.
  - tau_vs_heavy     : mean Kendall tau between a head's sigma_m and the heavy sigma_m.

Coordinate handling (reveal -> physical remap, none-token source term, NxN block
aggregation, diagonal zeroing) mirrors train_clean_aogpt.extract_A_matrices
exactly so the per-head and heavy graphs are directly comparable; the only change
is which attention slice feeds the remap (single head vs. top-4-head average).

Spec: docs/superpowers/specs/2026-05-29-l0h5-cross-ckpt-seed-stability-design.md
Red lines: attention-only; no NLL / L2R-raster oracle in the readout or selection.
"""
import sys
import pathlib

import numpy as np
import torch
from scipy.stats import kendalltau

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from training_utils import SEQ_LEN, N, BLOCK_LEN
from clean_training_protocol import expand_model_blocks_to_token_order
from neural_readout.extract_b import _load_model_and_chunks
from neural_readout.teacher_labels import generate_teacher_label
from batch_readout.diversity_batch import teacher_diversity_stats


# ---------------------------------------------------------------------------
# Pure coordinate / metric helpers (unit-tested without a model)
# ---------------------------------------------------------------------------
def _attn_to_A_block_vec(attn, reveal_tokens, inv_perm,
                         seq_len=SEQ_LEN, num_blocks=N, block_len=BLOCK_LEN,
                         none_weight=0.1):
    """Vectorized attn257_to_A_block over arbitrary leading dims.

    Same physical-frame remap / block-mean / [None]-source / zero-diagonal math
    as the scalar path, but applied to a whole stack of attention matrices at
    once. `reveal_tokens` is a permutation of [0, seq_len) (it is a token_order
    row), so the reveal->physical token map is a bijection with no collisions —
    the np.add.at scatter therefore reduces to a plain fancy-index assignment,
    and the NxN block aggregate becomes a single reshape+mean (no Python loop).

    Args:
        attn: (..., T+1, T+1) float array; row/col 0 is the [None] sink token.
        reveal_tokens: (T,) int model-coordinate token positions.
        inv_perm: (num_blocks,) int model-block -> physical-block map.

    Returns:
        A: (..., num_blocks, num_blocks) float32, physical frame, zero diagonal.
    """
    attn = np.asarray(attn, dtype=np.float32)
    lead = attn.shape[:-2]
    K = int(np.prod(lead)) if lead else 1
    a = attn.reshape(K, seq_len + 1, seq_len + 1)

    reveal_tokens = np.asarray(reveal_tokens, dtype=np.int64)
    inv_perm = np.asarray(inv_perm, dtype=np.int64)
    model_blocks = reveal_tokens // block_len
    phys_blocks = inv_perm[model_blocks]
    phys_tokens = phys_blocks * block_len + (reveal_tokens % block_len)

    # Bijective remap (phys_tokens is a permutation) => assignment == scatter-add.
    attn_phys = np.zeros((K, seq_len, seq_len), dtype=np.float32)
    attn_phys[:, phys_tokens[:, None], phys_tokens[None, :]] = a[:, 1:, 1:]

    # block-mean aggregate: (K, N, bl, N, bl) -> (K, N, N)
    A = attn_phys.reshape(
        K, num_blocks, block_len, num_blocks, block_len
    ).mean(axis=(2, 4))

    # [None] source term — replicated EXACTLY from extract_A_matrices (lines
    # 146-151): none_block is indexed in model/reveal block order and added to
    # the physical-frame A columns *without* remap. Kept identical (not "fixed")
    # so the heavy graph here matches the canonical B used across NR-1/BR-1.
    none_block = a[:, 1:, 0].reshape(K, num_blocks, block_len).mean(axis=2)  # (K, N)
    A = A + none_block[:, None, :] * none_weight

    di = np.arange(num_blocks)
    A[:, di, di] = 0.0
    A = A.astype(np.float32, copy=False)
    return A.reshape(lead + (num_blocks, num_blocks)) if lead else A[0]


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
    counts = np.bincount(labels, minlength=num_blocks).astype(np.float64)  # every physical block gets block_len revealed tokens (block 0 also gets [None]) -> never 0
    S = np.zeros((num_blocks, seq_len + 1), dtype=np.float64)
    S[labels, np.arange(seq_len + 1)] = 1.0
    S = S / counts[:, None]                                    # segment-mean selection rows
    A = np.einsum("bt,ktu,cu->kbc", S, a, S, optimize=True)    # (K, N, N)
    di = np.arange(num_blocks)
    A[:, di, di] = 0.0
    A = A.astype(np.float32, copy=False)
    return A.reshape(lead + (num_blocks, num_blocks)) if lead else A[0]


def attn257_to_A_block(avg_attn, reveal_tokens, inv_perm,
                       seq_len=SEQ_LEN, num_blocks=N, block_len=BLOCK_LEN,
                       none_weight=0.1):
    """Map one (T+1, T+1) attention matrix to a physical-frame NxN block graph.

    Replicates train_clean_aogpt.extract_A_matrices lines 121-152:
      reveal-frame -> physical-frame token remap, block-mean aggregation,
      none-token ([None]) source term added to columns, diagonal zeroed.
    Thin scalar wrapper over `_attn_to_A_block_vec` so the per-head loop
    reference and the batched extractor share one numerically-identical core.

    Args:
        avg_attn: (T+1, T+1) array; row/col 0 is the [None] sink token.
        reveal_tokens: (T,) int, model-coordinate token positions (token_order row).
        inv_perm: (num_blocks,) int, model-block -> physical-block map.
        none_weight: scale on the [None] source term (0.1 in extract_A_matrices).

    Returns:
        A: (num_blocks, num_blocks) float32, physical frame, zero diagonal.
    """
    return _attn_to_A_block_vec(avg_attn, reveal_tokens, inv_perm,
                                seq_len, num_blocks, block_len, none_weight)


def _mean_tau_vs(sigmas, ref):
    """Mean Kendall tau between each row of sigmas (M, Nn) and ref (Nn,)."""
    ref = np.asarray(ref)
    taus = []
    for s in sigmas:
        t, _ = kendalltau(s, ref)
        if not np.isnan(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


def _mean_tau_pairwise_vs(sigmas_a, sigmas_b):
    """Mean Kendall tau between paired rows of two (M, Nn) order arrays."""
    taus = []
    for sa, sb in zip(sigmas_a, sigmas_b):
        t, _ = kendalltau(sa, sb)
        if not np.isnan(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


def head_order_metrics(sigmas, sigmas_heavy=None):
    """Summarise a head's M batch reveal orders.

    Args:
        sigmas: (M, Nn) int reveal orders for this head.
        sigmas_heavy: optional (M, Nn) int heavy-baseline orders for tau_vs_heavy.

    Returns dict: tau_vs_l2r, mean_pairwise_tau, first_step_entropy[, tau_vs_heavy].
    """
    sigmas = np.asarray(sigmas)
    M, Nn = sigmas.shape
    l2r = np.arange(Nn)
    div = teacher_diversity_stats(sigmas)
    out = {
        "tau_vs_l2r": _mean_tau_vs(sigmas, l2r),
        "mean_pairwise_tau": div["mean_pairwise_tau"],
        "first_step_entropy": div["first_step_entropy"],
    }
    if sigmas_heavy is not None:
        out["tau_vs_heavy"] = _mean_tau_pairwise_vs(sigmas, np.asarray(sigmas_heavy))
    return out


def _orders_from_graphs(B_batch, alpha_dep=0.5):
    """Run CDL-source-start readout on each (Nn, Nn) batch-mean graph."""
    sigmas = np.empty((B_batch.shape[0], B_batch.shape[1]), dtype=np.int64)
    for m in range(B_batch.shape[0]):
        sigma, _rank, _Y = generate_teacher_label(B_batch[m], alpha_dep=alpha_dep)
        sigmas[m] = sigma
    return sigmas


# ---------------------------------------------------------------------------
# Per-head + heavy attention extraction (requires model)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _extract_per_head_and_heavy_A_loop(model, chunks, clean_perm, device, seed, n_top=4):
    """Reference (batch=1) extractor — kept verbatim as the golden behaviour the
    batched `extract_per_head_and_heavy_A` is pinned to (test_per_head_scan_batched).

    Returns:
        A_lh:    (n_chunks, L, H, N, N) float32
        A_heavy: (n_chunks, N, N) float32
    """
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    n_chunks = len(chunks)
    model.eval()

    A_lh = None
    A_heavy = np.zeros((n_chunks, N, N), dtype=np.float32)

    for i in range(n_chunks):
        tokens = chunks[i:i + 1].to(device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_order = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        ).to(device)

        _, _, attn_list = model.forward_fn(tokens, token_order, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, T+1, T+1)
        L, H = attn_stack.shape[:2]
        if A_lh is None:
            A_lh = np.zeros((n_chunks, L, H, N, N), dtype=np.float32)

        reveal_tokens = token_order[0].cpu().numpy()

        # heavy: top-n_top heads by off-diagonal variance, averaged over L + heads
        head_vars = np.zeros(H)
        mask = ~np.eye(SEQ_LEN, dtype=bool)
        for h in range(H):
            content = attn_stack[:, h, 1:, 1:]
            offdiag = content[:, mask].reshape(L, SEQ_LEN, SEQ_LEN - 1)
            head_vars[h] = float(np.var(offdiag))
        top_heads = np.argsort(head_vars)[-n_top:]
        avg_attn_heavy = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))
        A_heavy[i] = attn257_to_A_block(avg_attn_heavy, reveal_tokens, inv_perm)

        for l in range(L):
            for h in range(H):
                A_lh[i, l, h] = attn257_to_A_block(
                    attn_stack[l, h], reveal_tokens, inv_perm
                )

    return A_lh, A_heavy


def _per_sample_A(attn_stack, reveal_tokens, inv_perm, n_top, none_mode="old"):
    """Per-sample physical-frame A_lh (L,H,N,N) and heavy A (N,N) from one
    sample's attention stack (L,H,T+1,T+1). Identical math to the loop body.

    none_mode in {"old","b0"} selects the [None]-handling for BOTH the per-head
    and heavy block graphs ("old" = canonical 0.1-weighted [None] source term,
    "b0" = none->physical-block-0 fold)."""
    L, H = attn_stack.shape[:2]
    head_vars = np.zeros(H)
    mask = ~np.eye(SEQ_LEN, dtype=bool)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]
        offdiag = content[:, mask].reshape(L, SEQ_LEN, SEQ_LEN - 1)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-n_top:]
    avg_attn_heavy = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))

    # "old" agg == attn257_to_A_block (thin wrapper over _attn_to_A_block_vec
    # with default none_weight), so the OLD path is byte-unchanged.
    _AGG_BY_NONE_MODE = {"old": _attn_to_A_block_vec, "b0": _attn_to_A_block_b0_vec}
    if none_mode not in _AGG_BY_NONE_MODE:
        raise ValueError(f"none_mode must be one of {sorted(_AGG_BY_NONE_MODE)}, got {none_mode!r}")
    agg = _AGG_BY_NONE_MODE[none_mode]
    A_heavy_i = agg(avg_attn_heavy, reveal_tokens, inv_perm)

    # (L, H, T+1, T+1) -> (L, H, N, N) in one vectorized call (was an L*H loop).
    A_lh_i = agg(attn_stack, reveal_tokens, inv_perm)
    return A_lh_i, A_heavy_i


@torch.no_grad()
def extract_per_head_and_heavy_A(model, chunks, clean_perm, device, seed,
                                 n_top=4, fwd_batch=64, none_mode="old"):
    """Batched per-chunk per-(layer,head) A and top-n_top-head heavy A.

    Forwards `fwd_batch` chunks at once instead of one-at-a-time. The per-chunk
    seeded reveal permutation (manual_seed(seed+i)) and all downstream math are
    bit-for-bit the same as `_extract_per_head_and_heavy_A_loop` (pinned by
    test_per_head_scan_batched on a batch-invariant synthetic model); only the
    forward batching changes, so on a real model results match to fp tolerance.

    Returns:
        A_lh:    (n_chunks, L, H, N, N) float32
        A_heavy: (n_chunks, N, N) float32
    """
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    n_chunks = len(chunks)
    model.eval()

    # Pre-build the per-chunk seeded reveal token_orders (CPU, deterministic).
    token_orders = torch.empty((n_chunks, SEQ_LEN), dtype=torch.long)
    for i in range(n_chunks):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_orders[i] = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        )[0]

    A_lh = None
    A_heavy = np.zeros((n_chunks, N, N), dtype=np.float32)

    for start in range(0, n_chunks, max(1, int(fwd_batch))):
        stop = min(start + max(1, int(fwd_batch)), n_chunks)
        tokens = chunks[start:stop].to(device)
        order = token_orders[start:stop].to(device)

        _, _, attn_list = model.forward_fn(tokens, order, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # (L, B, H, T+1, T+1)
        attn_batch = torch.stack(attn_list).cpu().numpy()

        for bi in range(stop - start):
            i = start + bi
            attn_stack = attn_batch[:, bi]  # (L, H, T+1, T+1)
            reveal_tokens = token_orders[i].numpy()
            A_lh_i, A_heavy_i = _per_sample_A(attn_stack, reveal_tokens, inv_perm, n_top, none_mode=none_mode)
            if A_lh is None:
                A_lh = np.zeros((n_chunks,) + A_lh_i.shape, dtype=np.float32)
            A_lh[i] = A_lh_i
            A_heavy[i] = A_heavy_i

    return A_lh, A_heavy


def _batch_mean_B(A, M, batch_size):
    """(n, N, N) per-chunk A -> (M, N, N) batch-mean B = A^T, diagonal zeroed."""
    Nn = A.shape[-1]
    grouped = A.reshape(M, batch_size, Nn, Nn).astype(np.float64).mean(axis=1)
    B = np.transpose(grouped, (0, 2, 1)).astype(np.float32)
    diag = np.arange(Nn)
    B[:, diag, diag] = 0.0
    return B


def scan_checkpoint(ckpt_path, M, batch_size, seed, device="cuda:0",
                    split="train", alpha_dep=0.5, none_mode="old"):
    """Full per-head order scan for one checkpoint and one sampling seed.

    Returns a dict matching the original diag_head_layer_scan JSON schema.
    """
    total = M * batch_size
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed, device, split
    )
    A_lh, A_heavy = extract_per_head_and_heavy_A(model, chunks, clean_perm, dev, seed, none_mode=none_mode)
    Ln, Hn = A_lh.shape[1], A_lh.shape[2]

    sigma_heavy = _orders_from_graphs(_batch_mean_B(A_heavy, M, batch_size), alpha_dep)
    heavy = {
        "tau_vs_l2r": _mean_tau_vs(sigma_heavy, np.arange(N)),
        "diversity": teacher_diversity_stats(sigma_heavy),
    }

    # Cheap C1-C4 scores from the grand-mean per-head graph (attached per head so
    # the quick-head-selector validation driver can read cheap-vs-expensive recall
    # straight from the JSON).
    from quick_head_selector import cheap_head_scores
    cheap = cheap_head_scores(A_lh, alpha_dep=alpha_dep)  # {C1..C4: (L,H)}

    per_head = []
    for l in range(Ln):
        for h in range(Hn):
            B = _batch_mean_B(A_lh[:, l, h], M, batch_size)
            sig = _orders_from_graphs(B, alpha_dep)
            m = head_order_metrics(sig, sigmas_heavy=sigma_heavy)
            cheap_lh = {k: float(cheap[k][l, h]) for k in ("C1", "C2", "C3", "C4")}
            per_head.append({"layer": l, "head": h, **m, "cheap": cheap_lh})
    per_head.sort(key=lambda d: abs(d["tau_vs_l2r"]), reverse=True)

    # Persist the mean per-head physical graph + split-half means (L,H,N,N) so
    # ANY cheap score / cross-modal axis (row-concentration, split-half
    # reliability, ...) can be recomputed OFFLINE from the JSON, no re-run.
    n_chunks = A_lh.shape[0]
    half = n_chunks // 2
    graphs = {
        "shape": "L,H,N,N",
        "A_mean": A_lh.mean(axis=0).astype(np.float32).tolist(),
        "A_half1": A_lh[:half].mean(axis=0).astype(np.float32).tolist(),
        "A_half2": A_lh[half:].mean(axis=0).astype(np.float32).tolist(),
    }

    return {
        "config": {"M": M, "batch_size": batch_size, "seed": seed,
                   "ckpt": str(ckpt_path), "L": Ln, "H": Hn, "alpha_dep": alpha_dep,
                   "none_mode": none_mode},
        "heavy_baseline": heavy,
        "per_head_layer_sorted_by_abs_tau_vs_l2r": per_head,
        "graphs": graphs,
    }


def main():
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="train")
    p.add_argument("--alpha-dep", type=float, default=0.5)
    p.add_argument("--none-mode", default="old", choices=["old", "b0"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res = scan_checkpoint(args.ckpt, args.M, args.batch_size, args.seed,
                          device=args.device, split=args.split, alpha_dep=args.alpha_dep,
                          none_mode=args.none_mode)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    top = res["per_head_layer_sorted_by_abs_tau_vs_l2r"][0]
    print(f"[{pathlib.Path(args.ckpt).name} seed{args.seed}] "
          f"heavy tau_vs_l2r={res['heavy_baseline']['tau_vs_l2r']:.4f}  "
          f"top head L{top['layer']}H{top['head']} tau_vs_l2r={top['tau_vs_l2r']:.4f}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
