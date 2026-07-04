#!/usr/bin/env python3
"""Canonical strict65 block CDL vs strict257 token CDL scan.

This matches ``scripts/search_strict_label_free_65.py``:

- ``total = M * batch_size`` model-frame text chunks;
- one independent random block reveal per sample, seeded ``seed + i``;
- global mean A over all samples/reveals;
- strict None-separated B construction and C-D+L rollout;
- model-to-physical translation only after rollout.

Permutation comparisons use inverse-rank Kendall tau.

Usage:
  PYTHONPATH=chenhe_rerun:block_lo_arm_order_network \
    python analyses/token_level_signal_scan.py \
    --ckpt chenhe_rerun/out/rerun/method_gbeta_bm16g1500_50k/ckpt.pt \
    --M 40 --batch-size 4 --forward-batch 4 --device cuda
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
import torch
from scipy.stats import kendalltau

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "chenhe_rerun"))
sys.path.insert(0, str(_PROJECT / "block_lo_arm_order_network"))

from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec
from none_separated_block_graph import build_none_separated_B, rollout_from_none


def canonical_sample_count(M: int, batch_size: int) -> int:
    """Number of independently sampled graphs in the canonical protocol."""
    if int(M) <= 0 or int(batch_size) <= 0:
        raise ValueError("M and batch_size must be positive")
    return int(M) * int(batch_size)


def forward_slices(total: int, forward_batch: int):
    """Yield execution-only microbatch slices covering ``total`` samples."""
    if int(total) <= 0 or int(forward_batch) <= 0:
        raise ValueError("total and forward_batch must be positive")
    for start in range(0, int(total), int(forward_batch)):
        yield start, min(start + int(forward_batch), int(total))


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


def run_scan(
    ckpt_path,
    M=40,
    batch_size=4,
    forward_batch=4,
    device="cuda",
    seed=42,
    out_dir=None,
    perm_orientation="model_to_phys",
):
    from scripts.scan_collaborator_ckpt_b1 import (
        _clean_perm_from_ckpt,
        _load_model,
        _physical_chunks_to_model,
    )
    from training_utils import load_train_chunks

    total = canonical_sample_count(M, batch_size)
    print(f"[load] {ckpt_path}", flush=True)
    state, model, dev = _load_model(str(ckpt_path), str(device))
    clean_perm = _clean_perm_from_ckpt(state, perm_orientation)
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    model_args = state["model_args"]
    block_len = int(model_args["block_order_block_len"])
    seq_len = int(model_args["block_size"])
    num_blocks = seq_len // block_len
    n_layers = int(model_args["n_layer"])
    n_heads = int(model_args["n_head"])
    if seq_len != 256 or num_blocks != 64 or block_len != 4:
        raise ValueError(
            "canonical scan expects seq_len=256, num_blocks=64, block_len=4; "
            f"got {seq_len}, {num_blocks}, {block_len}"
        )
    print(
        f"[load] {n_layers}L{n_heads}H, total={total}=M{M}×B{batch_size}, "
        f"iter={state.get('iter_num')}",
        flush=True,
    )

    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = _physical_chunks_to_model(chunks_phys, clean_perm)
    reveal_orders = random_reveal_orders(total, seed, num_blocks, block_len)

    A_block_sum = np.zeros(
        (n_layers, n_heads, num_blocks, num_blocks + 1), dtype=np.float64
    )
    A_token_sum = np.zeros(
        (n_layers, n_heads, seq_len, seq_len + 1), dtype=np.float64
    )
    coarsegrain_max_abs_error = 0.0
    coarsegrain_checks = 0

    print("[extract] canonical independent random reveals", flush=True)
    with torch.no_grad():
        for start, stop in forward_slices(total, forward_batch):
            orders_batch = torch.from_numpy(reveal_orders[start:stop]).to(dev)
            _, _, attn_list = model.forward_fn(
                chunks_model[start:stop].to(dev),
                orders_batch,
                return_attentions=True,
            )
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            attn_batch = torch.stack(attn_list, dim=0).cpu().numpy()
            del attn_list

            for local_idx in range(stop - start):
                sample_idx = start + local_idx
                sample_attn = attn_batch[:, local_idx]
                A_block = _attn_to_A_block_loss_aligned_with_none_model_vec(
                    sample_attn,
                    reveal_orders[sample_idx],
                    seq_len=seq_len,
                    num_blocks=num_blocks,
                    block_len=block_len,
                )
                A_token = _attn_to_A_block_loss_aligned_with_none_model_vec(
                    sample_attn,
                    reveal_orders[sample_idx],
                    seq_len=seq_len,
                    num_blocks=seq_len,
                    block_len=1,
                )
                A_block_sum += A_block
                A_token_sum += A_token

                if coarsegrain_checks == 0:
                    for layer in range(n_layers):
                        for head in range(n_heads):
                            B_block = build_none_separated_B(A_block[layer, head])[1:, 1:]
                            B_token = build_none_separated_B(A_token[layer, head])[1:, 1:]
                            coarse = coarsegrain_token_B(
                                B_token,
                                block_len=block_len,
                                missing_source_token=int(reveal_orders[sample_idx, -1]),
                            )
                            mask = ~np.eye(num_blocks, dtype=bool)
                            coarsegrain_max_abs_error = max(
                                coarsegrain_max_abs_error,
                                float(np.max(np.abs(coarse[mask] - B_block[mask]))),
                            )
                    coarsegrain_checks = 1

            del attn_batch
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  … {stop}/{total}", end="", flush=True)
    print(" done", flush=True)

    A_block_mean = A_block_sum / total
    A_token_mean = A_token_sum / total
    l2r_blocks = np.arange(num_blocks, dtype=np.int64)
    results = []

    print(f"[analyze] strict65/strict257 CDL for {n_layers}×{n_heads} heads", flush=True)
    for layer in range(n_layers):
        for head in range(n_heads):
            B65 = build_none_separated_B(A_block_mean[layer, head])
            B257 = build_none_separated_B(A_token_mean[layer, head])
            sigma_block_model = rollout_from_none(B65, mode="C-D+L")
            sigma_token_model = rollout_from_none(B257, mode="C-D+L")

            sigma_block_phys = model_block_order_to_physical(
                sigma_block_model, inv_perm
            )
            sigma_token_phys = model_token_order_to_physical(
                sigma_token_model, inv_perm, block_len
            )
            sigma_block_expanded = expand_block_to_tokens(
                sigma_block_phys, block_len
            )
            sigma_token_collapsed = collapse_tokens_to_blocks(
                sigma_token_phys, block_len, num_blocks
            )

            results.append(
                {
                    "layer": layer,
                    "head": head,
                    "tau_block_vs_L2R": order_kendall_tau(
                        sigma_block_phys, l2r_blocks
                    ),
                    "tau_token_vs_block_expanded": order_kendall_tau(
                        sigma_token_phys, sigma_block_expanded
                    ),
                    "tau_collapse_vs_block": order_kendall_tau(
                        sigma_token_collapsed, sigma_block_phys
                    ),
                    "boundary_score": block_boundary_score(
                        sigma_token_phys, block_len, num_blocks
                    ),
                    "sigma_block_phys": sigma_block_phys.tolist(),
                    "sigma_token_collapsed_phys": sigma_token_collapsed.tolist(),
                }
            )

    print("\n" + "=" * 80)
    print("TOKEN ↔ BLOCK CDL (canonical random-reveal global mean)")
    print("=" * 80)

    def print_summary(key, label, n=8):
        values = [
            (float(row[key]), row["layer"], row["head"])
            for row in results
            if np.isfinite(row[key])
        ]
        values.sort(key=lambda item: item[0], reverse=True)
        signed = np.asarray([item[0] for item in values], dtype=np.float64)
        print(f"\n── {label} ──")
        for value, layer, head in values[:n]:
            print(f"  L{layer}H{head}: {value:+.4f}")
        print(
            f"  signed_mean={signed.mean():+.4f}  max={signed.max():+.4f}  "
            f"#>=0.1={int((signed >= 0.1).sum())}/{signed.size}"
        )

    print_summary("tau_token_vs_block_expanded", "tau(token, expand(block))")
    print_summary("tau_collapse_vs_block", "tau(collapse(token), block)")
    print_summary("boundary_score", "block boundary score")
    print_summary("tau_block_vs_L2R", "tau(block, physical L2R)")
    print(f"\ncoarsegrain_max_abs_error={coarsegrain_max_abs_error:.3e}")

    out_dir = pathlib.Path(out_dir or _HERE)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "token_level_signal_scan.json"
    meta = {
        "ckpt": str(ckpt_path),
        "M": int(M),
        "batch_size": int(batch_size),
        "total": int(total),
        "forward_batch": int(forward_batch),
        "seed": int(seed),
        "native_block_len": block_len,
        "native_num_blocks": num_blocks,
        "model": f"{n_layers}L{n_heads}H",
        "iter_num": state.get("iter_num"),
        "protocol": "canonical_random_reveal_global_mean",
        "reveal_seed": "seed + sample_index",
        "extraction": "model-frame loss-aligned strict65/strict257 with None separate",
        "posthoc_mapping": "inv_perm_model_to_phys after CDL rollout",
        "perm_orientation": perm_orientation,
        "kendall_tau": "inverse-rank vectors",
        "coarsegrain_checks": coarsegrain_checks,
        "coarsegrain_max_abs_error": coarsegrain_max_abs_error,
    }
    with open(out_path, "w") as handle:
        json.dump({"meta": meta, "results": results}, handle, indent=2, default=float)
    print(f"[saved] {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="chenhe_rerun/out/rerun/method_gbeta_bm16g1500_50k/ckpt.pt")
    parser.add_argument("--M", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--forward-batch", "--fwd-batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--perm-orientation",
        choices=["phys_to_model", "model_to_phys"],
        default="model_to_phys",
    )
    args = parser.parse_args()
    run_scan(args.ckpt, M=args.M, batch_size=args.batch_size,
             forward_batch=args.forward_batch, device=args.device, seed=args.seed,
             out_dir=args.out_dir, perm_orientation=args.perm_orientation)
