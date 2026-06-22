#!/usr/bin/env python3
"""Build model-frame multi-head distillation dataset for head-gated g_beta training.

For a given checkpoint and layer, extracts batch-mean B matrices for ALL heads
in model-frame coordinates, generates teacher labels using multiple teacher
methods, and saves as .npz.

Teacher methods:
  - manual_l0h2_cd:        CDL-source-start on L0H2's B (current canonical head)
  - per_head_cdl_rollout:   CDL-source-start per head, pick most-consensus head
  - minus_d_only:           -D only greedy rollout (load-bearing ablation)
  - full_cdl:               C-D+L with source-start anchor (operational teacher)
  - semantic_model_path_oracle: ground-truth model-frame order (upper-bound only)

Convention:
  - All B matrices and orders are in MODEL frame.
  - No inv_perm / physical remapping applied.
  - B = A^T, diagonal zeroed.
  - teacher_order_model[t] = earliest revealed block at step t.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
from scipy.stats import kendalltau

_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = _ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from training_utils import SEQ_LEN, N, BLOCK_LEN  # noqa: E402
from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from per_head_order_scan import _attn_to_A_block_model_vec  # noqa: E402
from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from neural_readout.teacher_labels import generate_teacher_label  # noqa: E402
from attn_order_teacher import rollout_order  # noqa: E402

TEACHER_METHODS = (
    "manual_l0h2_cd",
    "per_head_cdl_rollout",
    "minus_d_only",
    "full_cdl",
    "semantic_model_path_oracle",
)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _random_probe_token_orders(batch_size: int, seed: int, step: int, device: str):
    """Random model-coordinate token orders (batch, SEQ_LEN)."""
    rows = []
    for b in range(batch_size):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) * 100_000_000 + int(step) * 1000 + b)
        blocks = torch.randperm(N, generator=g)
        rows.append(expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0])
    return torch.stack(rows).to(device)


@torch.no_grad()
def extract_all_heads_model_frame(
    model, idx_batch, probe_orders, layer, device
) -> np.ndarray:
    """Extract model-frame B for ALL heads in a layer.

    Args:
        model: AOGPT model with forward_fn(..., return_attentions=True).
        idx_batch: (batch, SEQ_LEN) long tensor, model-coordinate token indices.
        probe_orders: (batch, SEQ_LEN) long tensor, model-coordinate token orders.
        layer: int, which transformer layer to extract from.
        device: torch.device or str.

    Returns:
        B_heads: (batch, num_heads, N, N) float32, B = A^T with zero diagonal.
    """
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    _, _, attn_list = model.forward_fn(idx_batch, probe_orders, return_attentions=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    attn_layer = attn_list[layer]  # (B, H, T+1, T+1)
    B_sz, H, Tp1, _ = attn_layer.shape
    attn_np = attn_layer.cpu().numpy()  # (B, H, T+1, T+1)
    probe_np = probe_orders.cpu().numpy()  # (B, T)

    # Model frame: identity inv_perm (no physical remapping).
    inv_perm = np.arange(N, dtype=np.int64)

    B_heads = np.zeros((B_sz, H, N, N), dtype=np.float32)
    # Per-sample loop — matches existing pattern in hook_order_provider.py line 62-63
    for bi in range(B_sz):
        for h in range(H):
            A_h = _attn_to_A_block_model_vec(
                attn_np[bi, h, :, :],  # (T+1, T+1)
                probe_np[bi],          # (T,)
                inv_perm,
            )  # (N, N)
            B_heads[bi, h] = A_h.T  # B = A^T

    # Zero diagonals
    diag = np.arange(N)
    B_heads[:, :, diag, diag] = 0.0

    return B_heads


# ---------------------------------------------------------------------------
# Teacher label generation
# ---------------------------------------------------------------------------

def _tau(a: np.ndarray, b: np.ndarray) -> float:
    val, _ = kendalltau(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    return float(val) if not np.isnan(val) else 0.0


def _make_teacher_order(B_head: np.ndarray, method: str, alpha_dep: float = 0.5,
                        block_perm: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate a single teacher order from one head's batch-mean B.

    Args:
        B_head: (N, N) float64, batch-mean block graph in model frame.
        method: teacher method name.
        alpha_dep: readiness weight for CDL-source-start.
        block_perm: (N,) int64 physical→model mapping (only for oracle).

    Returns:
        sigma_model: (N,) int64 reveal order in model frame.
    """
    B64 = np.asarray(B_head, dtype=np.float64)

    if method == "semantic_model_path_oracle":
        if block_perm is None:
            raise ValueError("block_perm required for semantic_model_path_oracle")
        return np.asarray(block_perm, dtype=np.int64)

    if method == "minus_d_only":
        return np.asarray(rollout_order(B64, mode="-D", greedy=True), dtype=np.int64)

    if method in ("full_cdl", "manual_l0h2_cd", "per_head_cdl_rollout"):
        sigma, _, _ = generate_teacher_label(B64, alpha_dep=alpha_dep)
        return sigma

    raise ValueError(f"unknown teacher method: {method}")


def _consensus_head_order(B_heads: np.ndarray, alpha_dep: float = 0.5) -> tuple[np.ndarray, int, dict]:
    """Pick the head whose CDL order has highest mean tau vs all other heads.

    Args:
        B_heads: (H, N, N) float32, all heads' batch-mean B for one sample.

    Returns:
        teacher_order: (N,) int64 — the order from the most-consensus head.
        best_head: int — index of the selected head.
        info: dict with per_head_orders, tau_matrix, mean_taus.
    """
    H = B_heads.shape[0]
    orders = []
    for h in range(H):
        sigma, _, _ = generate_teacher_label(
            np.asarray(B_heads[h], dtype=np.float64), alpha_dep=alpha_dep
        )
        orders.append(sigma)

    # Pairwise tau matrix
    tau_mat = np.zeros((H, H), dtype=np.float64)
    for i in range(H):
        for j in range(i + 1, H):
            t = _tau(orders[i], orders[j])
            tau_mat[i, j] = t
            tau_mat[j, i] = t

    mean_taus = tau_mat.mean(axis=1)
    best_head = int(np.argmax(mean_taus))

    return orders[best_head], best_head, {
        "per_head_orders": [o.tolist() for o in orders],
        "tau_matrix": tau_mat.tolist(),
        "mean_taus": mean_taus.tolist(),
        "best_head": best_head,
    }


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_headset_dataset(
    ckpt_path: str,
    layer: int,
    teacher: str,
    M: int,
    batch_size: int,
    seed: int = 0,
    device: str = "cuda:0",
    split: str = "train",
    out_path: Optional[str] = None,
):
    """Build and save a multi-head distillation dataset.

    Args:
        ckpt_path: path to AOGPT checkpoint.
        layer: which transformer layer to extract heads from.
        teacher: teacher method name.
        M: number of batch-mean graphs.
        batch_size: per-sample graphs averaged into each batch-mean.
        seed: random seed for sampling and probe orders.
        device: torch device.
        split: "train" or "eval".
        out_path: output .npz path (auto-generated if None).

    Returns:
        dict with dataset summary.
    """
    if teacher not in TEACHER_METHODS:
        raise ValueError(f"unknown teacher {teacher!r}; choose from {TEACHER_METHODS}")

    print(f"Loading checkpoint: {ckpt_path}")
    model, chunks, clean_perm, dev, chunk_indices = _load_model_and_chunks(
        ckpt_path, M * batch_size, seed, device, split
    )
    block_perm = clean_perm.block_perm_phys_to_model.cpu().numpy()  # (N,) physical→model

    total = M * batch_size
    print(f"Extracting {total} per-sample graphs (M={M}, batch_size={batch_size}) "
          f"from layer {layer}...")

    # Extract in mini-batches to manage GPU memory
    all_B_heads = []
    mini_bs = min(batch_size, 4)  # forward batch size for extraction
    for start in range(0, total, mini_bs):
        end = min(start + mini_bs, total)
        idx_b = chunks[start:end].to(dev)
        probe = _random_probe_token_orders(end - start, seed, start // mini_bs, str(dev))
        B_h = extract_all_heads_model_frame(model, idx_b, probe, layer, dev)
        all_B_heads.append(B_h)
        if (end) % 20 == 0 or end == total:
            print(f"  ... {end}/{total} per-sample graphs extracted")

    all_B_heads = np.concatenate(all_B_heads, axis=0)  # (total, H, N, N)
    H = all_B_heads.shape[1]
    print(f"Extracted {total} graphs, {H} heads, shape={all_B_heads.shape}")

    # Group into batch-mean graphs
    B_heads_mean = all_B_heads.reshape(M, batch_size, H, N, N).mean(axis=1).astype(np.float32)
    print(f"Batch-mean B_heads: {B_heads_mean.shape}")

    # Generate teacher labels per batch-mean sample
    print(f"Generating teacher labels (method={teacher})...")
    teacher_orders = np.zeros((M, N), dtype=np.int64)
    per_sample_info = []

    for m in range(M):
        if teacher in ("manual_l0h2_cd", "full_cdl"):
            if teacher == "manual_l0h2_cd":
                # Use head (0, 2) specifically — but we might not have layer 0.
                # For now, use head 2 of the target layer (closest analogue).
                # If layer != 0, fall back to head 2.
                head_idx = 2 if H > 2 else 0
                B_target = np.asarray(B_heads_mean[m, head_idx], dtype=np.float64)
            else:
                # full_cdl: use the most-consensus head
                B_target = np.asarray(B_heads_mean[m], dtype=np.float64)
                teacher_orders[m], _, _ = _consensus_head_order(B_target)
                continue

            teacher_orders[m] = _make_teacher_order(B_target, teacher, block_perm=block_perm)
            per_sample_info.append(None)

        elif teacher == "per_head_cdl_rollout":
            teacher_orders[m], best_h, info = _consensus_head_order(B_heads_mean[m])
            per_sample_info.append(info)

        elif teacher == "minus_d_only":
            # Use the head with strongest -D signal (most variable B)
            head_vars = [np.std(B_heads_mean[m, h]) for h in range(H)]
            best_h = int(np.argmax(head_vars))
            teacher_orders[m] = _make_teacher_order(
                B_heads_mean[m, best_h], teacher, block_perm=block_perm
            )
            per_sample_info.append({"best_head": best_h})

        elif teacher == "semantic_model_path_oracle":
            teacher_orders[m] = _make_teacher_order(
                B_heads_mean[m, 0], teacher, block_perm=block_perm
            )
            per_sample_info.append(None)

        if (m + 1) % 20 == 0:
            print(f"  ... {m + 1}/{M} teacher labels generated")

    print(f"Teacher orders shape: {teacher_orders.shape}")

    # Save
    if out_path is None:
        ckpt_name = pathlib.Path(ckpt_path).stem
        out_path = f"headset_l{layer}_{teacher}_{ckpt_name}.npz"

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        B_heads=B_heads_mean,
        teacher_order_model=teacher_orders,
        block_perm_phys_to_model=block_perm,
        inv_perm_model_to_phys=clean_perm.inv_perm_model_to_phys.cpu().numpy(),
        layer=layer,
        source_ckpt=str(ckpt_path),
        teacher_method=teacher,
        coordinate_frame="model",
        N=N,
        num_heads=H,
        M=M,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"Saved: {out}")
    print(f"  B_heads shape: {B_heads_mean.shape}")
    print(f"  coordinate_frame: model")
    print(f"  teacher: {teacher}")

    return {
        "out_path": str(out),
        "B_heads_shape": list(B_heads_mean.shape),
        "num_heads": H,
        "layer": layer,
        "teacher": teacher,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build model-frame multi-head distillation dataset for head-gated g_beta"
    )
    p.add_argument("--ckpt", required=True, help="Path to AOGPT checkpoint .pt file")
    p.add_argument("--layer", type=int, default=0, help="Transformer layer to extract (default 0)")
    p.add_argument("--teacher", default="manual_l0h2_cd", choices=TEACHER_METHODS,
                   help="Teacher label method")
    p.add_argument("--M", type=int, default=100, help="Number of batch-mean graphs")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-sample graphs per batch-mean")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--device", default="cuda:0", help="Torch device")
    p.add_argument("--split", default="train", choices=("train", "eval"),
                   help="Data split to sample from")
    p.add_argument("--out", default=None, help="Output .npz path (auto-generated if omitted)")
    args = p.parse_args()

    build_headset_dataset(
        ckpt_path=args.ckpt,
        layer=args.layer,
        teacher=args.teacher,
        M=args.M,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        split=args.split,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
