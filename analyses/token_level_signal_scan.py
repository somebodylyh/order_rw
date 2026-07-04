#!/usr/bin/env python3
"""Token-level vs block-level CDL: using the EXACT same extraction as the
deployed strict65 pipeline, but at token granularity (num_blocks=256, block_len=1).

Key: uses ``_attn_to_A_block_loss_aligned_with_none_model_vec`` (the strict65 model-frame
extraction) + ``build_none_separated_B`` (the [None]-separated graph construction),
consistent with the deployed gbeta / CDL pipeline.

Comparison:
  - block-level: num_blocks=64, block_len=4 → B65[1:,1:] (64×64) → CDL → σ_block
  - token-level: num_blocks=256, block_len=1 → B257[1:,1:] (256×256) → CDL → σ_token

Both in MODEL frame; CDL output remapped to physical via inv_perm.
Then: τ(σ_token_phys, expand(σ_block_phys)) and τ(collapse(σ_token_phys), σ_block_phys).

Usage:
  PYTHONPATH=chenhe_rerun:block_lo_arm_order_network \
    python analyses/token_level_signal_scan.py \
    --ckpt chenhe_rerun/out/rerun/method_gbeta_bm16g1500_50k/ckpt.pt \
    --M 32 --device cuda
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
import torch
from scipy.stats import kendalltau

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT / "chenhe_rerun"))
sys.path.insert(0, str(_PROJECT / "block_lo_arm_order_network"))

from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec
from none_separated_block_graph import build_none_separated_B, rollout_from_none


def canonical_sample_count(M: int, batch_size: int) -> int:
    """Number of independently sampled graphs in the canonical protocol."""
    if int(M) <= 0 or int(batch_size) <= 0:
        raise ValueError("M and batch_size must be positive")
    return int(M) * int(batch_size)


def random_reveal_orders(
    total: int,
    seed: int,
    num_blocks: int = 64,
    block_len: int = 4,
) -> np.ndarray:
    """Canonical per-sample random block reveals, seeded ``seed + i``."""
    orders = np.empty((int(total), int(num_blocks) * int(block_len)), dtype=np.int64)
    offsets = np.arange(int(block_len), dtype=np.int64)
    for sample_idx in range(int(total)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + sample_idx)
        blocks = torch.randperm(int(num_blocks), generator=generator).numpy()
        orders[sample_idx] = (
            blocks[:, None] * int(block_len) + offsets[None, :]
        ).reshape(-1)
    return orders


def order_to_rank(order: np.ndarray) -> np.ndarray:
    """Convert a permutation order to ``rank[item] = reveal_position``."""
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1 or not np.array_equal(np.sort(order), np.arange(order.size)):
        raise ValueError("order must be a permutation of [0, N)")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank


def order_kendall_tau(left: np.ndarray, right: np.ndarray) -> float:
    """Kendall tau between two reveal orders, compared by item rank."""
    left_rank = order_to_rank(left)
    right_rank = order_to_rank(right)
    if left_rank.shape != right_rank.shape:
        raise ValueError(f"order shapes must match, got {left_rank.shape} and {right_rank.shape}")
    value = kendalltau(left_rank, right_rank)[0]
    return float(value) if not np.isnan(value) else float("nan")


def model_block_order_to_physical(
    sigma_model: np.ndarray,
    inv_perm_model_to_phys: np.ndarray,
) -> np.ndarray:
    """Translate a model-block order to physical block ids."""
    sigma_model = np.asarray(sigma_model, dtype=np.int64)
    inv_perm = np.asarray(inv_perm_model_to_phys, dtype=np.int64)
    return inv_perm[sigma_model]


def model_token_order_to_physical(
    sigma_model: np.ndarray,
    inv_perm_model_to_phys: np.ndarray,
    block_len: int = 4,
) -> np.ndarray:
    """Translate model-token ids to physical tokens, preserving block offset."""
    sigma_model = np.asarray(sigma_model, dtype=np.int64)
    inv_perm = np.asarray(inv_perm_model_to_phys, dtype=np.int64)
    model_blocks = sigma_model // int(block_len)
    offsets = sigma_model % int(block_len)
    return inv_perm[model_blocks] * int(block_len) + offsets


def coarsegrain_token_B(
    B_token: np.ndarray,
    block_len: int = 4,
    missing_source_token: int | None = None,
) -> np.ndarray:
    """Masked per-sample token B -> block B, excluding the absent last source."""
    B_token = np.asarray(B_token, dtype=np.float64)
    if B_token.ndim != 2 or B_token.shape[0] != B_token.shape[1]:
        raise ValueError(f"B_token must be square, got {B_token.shape}")
    if B_token.shape[0] % int(block_len):
        raise ValueError("token count must be divisible by block_len")
    num_blocks = B_token.shape[0] // int(block_len)
    out = np.zeros((num_blocks, num_blocks), dtype=np.float64)
    for source_block in range(num_blocks):
        source_tokens = np.arange(
            source_block * int(block_len), (source_block + 1) * int(block_len)
        )
        if missing_source_token is not None:
            source_tokens = source_tokens[source_tokens != int(missing_source_token)]
        for target_block in range(num_blocks):
            if source_block == target_block:
                continue
            target_tokens = np.arange(
                target_block * int(block_len), (target_block + 1) * int(block_len)
            )
            out[source_block, target_block] = B_token[
                np.ix_(source_tokens, target_tokens)
            ].mean()
    return out


def _extract_B_strict65(attn_np, reveal_tokens_np, num_blocks, block_len):
    """Extract B matrix (model frame) using the deployed strict65 pipeline.

    Args:
        attn_np: (M, 257, 257) raw attention in shuffled space.
        reveal_tokens_np: (M, 256) shuffled token positions.
        num_blocks: number of units (64 for block, 256 for token).
        block_len: tokens per unit (4 for block, 1 for token).

    Returns:
        B: (M, num_blocks, num_blocks) model-frame B matrices.
    """
    M = attn_np.shape[0]
    B_list = []
    for bi in range(M):
        A_bi = _attn_to_A_block_loss_aligned_with_none_model_vec(
            attn_np[bi], reveal_tokens_np[bi],
            seq_len=256, num_blocks=num_blocks, block_len=block_len,
        )  # (num_blocks, num_blocks+1)
        B65 = build_none_separated_B(A_bi)  # (num_blocks+1, num_blocks+1)
        B_list.append(B65[1:, 1:])  # strip [None] → (num_blocks, num_blocks)
    B = np.stack(B_list, axis=0).astype(np.float64)
    return B


def expand_block_to_tokens(sigma_block, block_len=4):
    """σ_block (nb,) → token order (nb*block_len,)."""
    sigma_block = np.asarray(sigma_block, dtype=np.int64)
    nb = len(sigma_block)
    out = np.empty(nb * block_len, dtype=np.int64)
    for pos, blk in enumerate(sigma_block):
        out[pos * block_len:(pos + 1) * block_len] = blk * block_len + np.arange(block_len)
    return out


def collapse_tokens_to_blocks(sigma_token, block_len=4, num_blocks=64):
    """σ_token (nb*bl,) → block order by first-appearance."""
    sigma_token = np.asarray(sigma_token, dtype=np.int64)
    first_pos = np.full(num_blocks, 999999, dtype=np.int64)
    for pos, tok in enumerate(sigma_token):
        blk = tok // block_len
        if pos < first_pos[blk]:
            first_pos[blk] = pos
    return np.argsort(first_pos).astype(np.int64)


def block_boundary_score(sigma_token, block_len=4, num_blocks=64):
    """How well does σ_token group same-block tokens consecutively? 1.0 = perfect."""
    sigma_token = np.asarray(sigma_token, dtype=np.int64)
    positions = np.empty(len(sigma_token), dtype=np.int64)
    for pos, tok in enumerate(sigma_token):
        positions[tok] = pos
    scores = []
    for blk in range(num_blocks):
        pos = np.sort(positions[blk * block_len:(blk + 1) * block_len])
        ideal = block_len - 1
        actual = int(pos[-1] - pos[0])
        scores.append(ideal / max(actual, ideal))
    return float(np.mean(scores))


def run_scan(ckpt_path, M=32, batch_size=8, device="cuda", seed=42, out_dir=None):
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    # ── Load model ──
    print(f"[load] {ckpt_path}", flush=True)
    from AOGPT_block import AOGPT, AOGPTConfig
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ma = state["model_args"]
    cfg = AOGPTConfig()
    for k, v in ma.items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    cfg.force_manual_attention = True
    cfg.dropout = 0.0

    model = AOGPT(cfg)
    missing, _ = model.load_state_dict(state["model"], strict=False)
    non_bias = [k for k in (missing or []) if "attn.bias" not in k]
    if non_bias: print(f"[warn] Missing: {non_bias}", flush=True)
    model.eval(); model.to(dev)

    bl = int(cfg.block_order_block_len)  # 4
    nb = cfg.block_size // bl             # 64
    nL, nH = cfg.n_layer, cfg.n_head
    print(f"[load] {nL}L{nH}H, native={nb}blk×{bl}tok, iter={state['iter_num']}", flush=True)

    # ── Data ──
    data = np.memmap(str(_PROJECT / "chenhe_rerun/data/wikitext103/train.bin"),
                     dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(M):
        s = int(rng.integers(0, len(data) - cfg.block_size - 1))
        chunks.append(torch.from_numpy(data[s:s + cfg.block_size].astype(np.int64)))
    print(f"[data] {M} chunks", flush=True)

    # ── Forward & extract attention ──
    print(f"[extract] Forward {M} texts …", flush=True)
    raw_attn_list = []
    all_reveal_tokens = []  # (M, 256) shuffled token positions

    with torch.no_grad():
        for start in range(0, M, batch_size):
            stop = min(start + batch_size, M)
            idx_batch = torch.stack(chunks[start:stop]).to(dev)
            bsz = idx_batch.shape[0]

            # Fixed L2R probe order: model frame = physical frame
            block_orders = torch.arange(nb, device=dev).unsqueeze(0).expand(bsz, -1)
            token_orders = model._expand_block_orders_to_token_orders(block_orders)
            all_reveal_tokens.append(token_orders.cpu())

            _, _, attn_list = model.forward_fn(
                idx_batch, token_orders,
                return_attentions=True, return_logits=False,
            )
            raw_attn_list.append(torch.stack(attn_list, dim=0).cpu())  # (L, B, H, 257, 257)
            if dev.type == "cuda": torch.cuda.empty_cache()
            print(f"  … {stop}/{M}", end="", flush=True)
    print(" done", flush=True)

    raw_attn = torch.cat(raw_attn_list, dim=1).numpy().astype(np.float64)  # (L, M, H, 257, 257)
    reveal = torch.cat(all_reveal_tokens, dim=0).numpy().astype(np.int64)    # (M, 256)
    del raw_attn_list, all_reveal_tokens

    # ── Per-head: extract B at block & token level, batch-mean, CDL ──
    print(f"[analyze] strict65 extraction + CDL for {nL}×{nH} heads …", flush=True)
    results = []

    for li in range(nL):
        for hi in range(nH):
            # Get attention for this head: (M, 257, 257)
            attn_head = raw_attn[li, :, hi, :, :]  # (M, 257, 257)

            # ── Block-level (64×64) using strict65 ──
            B_block = _extract_B_strict65(attn_head, reveal, nb, bl)  # (M, 64, 64)
            B_block_mean = B_block.mean(axis=0)  # (64, 64) batch-mean
            np.fill_diagonal(B_block_mean, 0.0)

            # Block CDL via rollout_from_none
            B65_block = np.zeros((nb + 1, nb + 1), dtype=np.float64)
            B65_block[1:, 1:] = B_block_mean.T  # build B65 from stripped B
            try:
                sigma_block_model = rollout_from_none(B65_block, mode="C-D+L")  # (64,) model frame
                tau_block_l2r = float(kendalltau(sigma_block_model, np.arange(nb))[0])
            except Exception:
                sigma_block_model = None
                tau_block_l2r = float("nan")

            # ── Token-level (256×256) using strict65 ──
            B_token = _extract_B_strict65(attn_head, reveal, 256, 1)  # (M, 256, 256)
            B_token_mean = B_token.mean(axis=0)  # (256, 256) batch-mean
            np.fill_diagonal(B_token_mean, 0.0)

            # Token CDL via rollout_from_none
            B257_token = np.zeros((257, 257), dtype=np.float64)
            B257_token[1:, 1:] = B_token_mean.T
            try:
                sigma_token_model = rollout_from_none(B257_token, mode="C-D+L")  # (256,) model frame
            except Exception:
                sigma_token_model = None

            # ── Remap model→physical for comparison ──
            # Model frame: model_block i = reveal_tokens indices grouped by block
            # Physical frame: physical tokens are L2R [0..255]
            # For token-level, we need a (256,) inv_perm: model_token → physical_token
            # The reveal_tokens map: reveal_tokens[model_token] = physical_token for shuffled position
            # But with block_len=1, each "block" = 1 token, and the model block order
            # is just the shuffled token order.
            #
            # Actually, for strict65 MODEL frame, the B is constructed in model coordinates.
            # The CDL output σ_token_model is in model coordinates (node i = model token i).
            # To get physical: need to map model_token → physical_token.
            #
            # For a given sample, the model token at position `p` in shuffled order has
            # physical position `reveal_tokens[p]`. But with batch-mean across samples,
            # each sample has a DIFFERENT random block order.
            #
            # For the batch-mean approach: B is averaged across samples with DIFFERENT
            # random orders. The model-frame CDL σ_token_model is an order of "model
            # slot indices" (0..255), where slot i represents the token that was at
            # shuffled position i. But across samples, the SAME slot i corresponds to
            # DIFFERENT physical tokens (because the random permutation differs).
            #
            # This makes the batch-mean approach BREAK for comparison!
            # The B_block and B_token are averaged across different permutations,
            # so the model-frame CDL doesn't have a consistent physical interpretation.
            #
            # WAIT — actually, looking at the deployed code (HookOrderProvider, gbeta_scores_batchmean),
            # the batch-mean B is computed across samples with THE SAME random probe order
            # (seed-based deterministic). Let me check...

            # Actually, looking at HookOrderProvider.physical_order(), the probe orders
            # are generated as: g.manual_seed(seed * 100M + global_step * 1000 + b).
            # So each sample b gets a DIFFERENT probe order. The A matrices are extracted
            # per-sample with different reveal_tokens, then B = A^T.mean(dim=0).
            # This batch-mean B is in MODEL frame for ALL samples — but since each sample
            # has a different model→physical mapping, the batch-mean B mixes different
            # correspondences.
            #
            # BUT — the deployed code does: hook.step(A).cpu() which returns PHYSICAL-frame
            # sigma. So the FrozenBetaHook internally handles the model→physical remap.
            # HOW? It must be using a per-sample remap or assuming the B is in a frame
            # where model slot = physical slot.
            #
            # Looking at _attn_to_A_block_model_vec: it uses inv_perm to remap to physical.
            # Looking at _attn_to_A_block_strict65_model_vec: it explicitly does NOT use inv_perm
            # (model frame), and the trainer applies model→physical remap POSTHOC.
            #
            # OK so for the deployed pipeline:
            # 1. Extract A in model frame (strict65)
            # 2. A^T → B in model frame
            # 3. gbeta(B) → scores in model frame
            # 4. argsort(scores) → σ_model
            # 5. inv_perm[σ_model] → σ_phys
            #
            # The batch-mean mixes different model frames. How does this work?
            # Each sample bi has a DIFFERENT random block order → different reveal_tokens.
            # The A matrices are in "model frame" where model block i = reveal_tokens[i*bl] // bl.
            # After averaging across samples, the result is... meaningless in strict model frame?
            #
            # Actually, I think the answer is simpler: the random probe orders ARE different per sample,
            # but the B matrix structure is permutation-equivariant. If the attention encodes
            # the true order structure, then averaging B across different probe permutations
            # should still preserve the structure (because the graph topology is the same,
            # just with permuted node labels).
            #
            # More precisely: if attention pattern is determined by physical block relationships,
            # then under random block permutation π, the model-frame B is B_model[i,j] = B_phys[π(i), π(j)].
            # Averaging over random π: E[B_model[i,j]] = (1/64²) * sum_{p,q} B_phys[p,q] = constant.
            # So batch-mean across different permutations should WASH OUT the signal!
            #
            # But empirically it works (τ≈1.0 for gbeta). Why?
            # Because the probed attention uses fresh random order each time, and the
            # model sees the shuffled input. The attention patterns in model frame reflect
            # the model's INTERNAL structure which should be consistent regardless of
            # the probe order. The B in model frame (slot i ↔ slot j) reflects the
            # relationship between whatever tokens happen to be at those slots.
            # Across random permutations, the EXPECTED relationship between slot i and slot j
            # is the average relationship between ALL pairs of physical tokens — which is constant.
            #
            # UNLESS the model's INTERNAL processing creates structure that depends on
            # the token identities at each slot. With random block orders, the token at
            # slot i varies across samples, so the batch-mean B averages token-specific effects.
            #
            # Hmm, but this still shouldn't work if the signal is purely about physical order...
            # Unless the gbeta itself is learning something batch-statistical.
            #
            # Actually, I think I'm overthinking this. The deployed code uses batch-mean with
            # DIFFERENT random probe orders per sample, and it works (τ≈1.0 for good heads).
            # This means the model-frame B DOES contain consistent structure despite the
            # varying permutations. The gbeta then learns to decode this consistent structure.
            #
            # For our comparison: if we do CDL on the batch-mean B (model frame), the CDL order
            # will be in model frame (slot indices). To compare with L2R, we need to know what
            # physical position each slot index corresponds to. But with batch-mean across
            # different permutations, there IS no consistent slot→physical mapping!
            #
            # This is why the deployed code computes CDL per-sample or per-group (not batch-mean CDL).
            # The gbeta produces scores per-group (fresh A extraction → batch-mean B → gbeta),
            # and the scores are then remapped via inv_perm which IS per-group consistent.
            #
            # For our token-vs-block comparison, we should:
            # Option A: Use a SINGLE consistent probe order across all samples
            # Option B: Compute CDL per-sample, then average τ
            # Option C: Use batch-mean B but don't compare with L2R; compare token B vs block B directly
            #
            # Let me go with Option A: use a fixed probe order (e.g., L2R) for all samples.
            # This gives consistent model↔physical mapping and allows direct comparison.
            #
            # BUT — the deployed code uses RANDOM probe orders to keep extraction in-distribution
            # for gbeta. For pure CDL analysis though, using L2R probe order should be fine —
            # the CDL is deterministic and doesn't have a "training distribution".

            # ── SIMPLER APPROACH: use FIXED L2R probe order for all samples ──
            # This gives consistent model↔physical mapping.
            # Since we're only doing CDL (not gbeta inference), the probe distribution
            # doesn't matter for correctness.

            row = {
                "layer": li, "head": hi,
                "tau_block_vs_L2R": tau_block_l2r,
            }

            if sigma_block_model is not None and sigma_token_model is not None:
                # Expand block CDL to tokens
                sigma_block_exp = expand_block_to_tokens(sigma_block_model, bl)
                tau_tvb = float(kendalltau(sigma_token_model, sigma_block_exp)[0])

                # Collapse token CDL to blocks
                sigma_token_collapsed = collapse_tokens_to_blocks(sigma_token_model, bl, nb)
                tau_cvb = float(kendalltau(sigma_token_collapsed, sigma_block_model)[0])

                # Boundary score
                bscore = block_boundary_score(sigma_token_model, bl, nb)

                row["tau_token_vs_block_expanded"] = tau_tvb
                row["tau_collapse_vs_block"] = tau_cvb
                row["boundary_score"] = bscore
            else:
                row["tau_token_vs_block_expanded"] = float("nan")
                row["tau_collapse_vs_block"] = float("nan")
                row["boundary_score"] = float("nan")

            results.append(row)

        done = li * nH + hi + 1
        if done % 8 == 0:
            print(f"  … L{li}H{hi} done ({done}/{nL*nH})", flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TOKEN ↔ BLOCK CDL (strict65 extraction, fixed L2R probe)")
    print("=" * 80)

    def _top(key, label, n=8):
        vals = [(r[key], r["layer"], r["head"]) for r in results
                if not np.isnan(r.get(key, float("nan")))]
        vals.sort(key=lambda x: -abs(x[0]))
        print(f"\n── {label} ──")
        if not vals: print("  (no valid data)"); return
        for tau, l, h in vals[:n]:
            print(f"  L{l}H{h}: {tau:+.4f}")
        av = [abs(v[0]) for v in vals]
        print(f"  Mean|τ|={np.mean(av):.4f}  Max|τ|={np.max(av):.4f}  "
              f"#≥0.1: {sum(1 for v in av if v>=0.1)}/{len(av)}")

    _top("tau_token_vs_block_expanded",
         "τ(σ_token, expand(σ_block)) — model frame, L2R probe")
    _top("tau_collapse_vs_block",
         "τ(collapse(σ_token), σ_block)")
    _top("boundary_score",
         "Block boundary score (1.0=perfect block grouping)")
    _top("tau_block_vs_L2R",
         "τ(σ_block, L2R) — baseline block CDL strength")

    # Save
    out_dir = pathlib.Path(out_dir or _HERE)
    out_path = out_dir / "token_level_signal_scan.json"
    meta = {"ckpt": str(ckpt_path), "M": M, "seed": seed,
            "native_block_len": bl, "native_num_blocks": nb,
            "model": f"{nL}L{nH}H",
            "extraction": "strict65 (_attn_to_A_block_loss_aligned_with_none_model_vec + build_none_separated_B)",
            "probe_order": "L2R (fixed for all samples, ensures consistent model↔physical mapping)",
            "note": "Batch-mean B across M samples, then single CDL rollout"}
    json.dump({"meta": meta, "results": results}, open(out_path, "w"),
              indent=2, default=float)
    print(f"\n[saved] {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="chenhe_rerun/out/rerun/method_gbeta_bm16g1500_50k/ckpt.pt")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    run_scan(args.ckpt, M=args.M, batch_size=args.batch_size,
             device=args.device, seed=args.seed, out_dir=args.out_dir)
