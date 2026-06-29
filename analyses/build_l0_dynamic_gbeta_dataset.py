#!/usr/bin/env python3
"""Build label-free L0 dynamic g_beta pretrain dataset.

Workflow:
  1. Load the random-baseline checkpoint via ``_load_model_and_chunks``.
  2. Forward in bounded batches under random model-frame probe orders.
  3. Extract L0 strict-65 B_raw using ``build_model_frame_strict65`` (Task 1).
  4. Batch-mean group per-sample graphs.
  5. Generate dynamic CDL consensus teacher via ``build_dynamic_teacher`` (Task 2).
  6. Deterministic 80/10/10 train/val/test split.
  7. Save as .npz with ONLY label-free fields.

Physical coordinates (clean_perm, inv_perm, block_perm, L2R order) are
intentionally NOT saved and NOT passed to extraction or teacher.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = _ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from training_utils import SEQ_LEN, N, BLOCK_LEN  # noqa: E402
from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from batch_readout.l0_strict65 import (  # noqa: E402
    batch_mean_heads,
    build_model_frame_strict65,
)
from batch_readout.label_free_cdl_teacher import build_dynamic_teacher  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------

def deterministic_split(M: int, seed: int = 7) -> dict:
    """80/10/10 train/val/test split, deterministic in (M, seed).

    Returns:
        dict with keys "train", "val", "test" → (N,) int64 sorted arrays.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    train_n = int(round(M * 0.8))
    val_n = int(round(M * 0.1))
    # test gets the rest
    return {
        "train": np.sort(perm[:train_n].astype(np.int64)),
        "val": np.sort(perm[train_n:train_n + val_n].astype(np.int64)),
        "test": np.sort(perm[train_n + val_n:].astype(np.int64)),
    }


def weighted_consensus_order(
    ranks: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Sort blocks by their teacher-weighted mean CDL rank."""
    ranks = np.asarray(ranks, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must be [H, N], got {ranks.shape}")
    if weights.shape != (ranks.shape[0],):
        raise ValueError(
            f"weights must be [{ranks.shape[0]}], got {weights.shape}"
        )
    mean_rank = (ranks.astype(np.float64) * weights[:, None]).sum(axis=0)
    return np.argsort(mean_rank, kind="stable").astype(np.int64)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_dataset(
    out_path: str,
    B_raw: np.ndarray,
    teacher: dict,
    split: dict,
    meta: dict,
):
    """Save the label-free dataset as .npz.

    Deliberately does NOT save block_perm, inv_perm, clean_perm,
    physical_order, or l2r_order.
    """
    save = {
        "B_raw": B_raw,
    }
    # Teacher fields — each is [M, ...]
    for key in ("orders", "ranks", "margin", "destroyed_gap",
                "agreement", "quality", "weights", "pairwise",
                "consensus_order"):
        if key in teacher:
            save[f"teacher_{key}"] = teacher[key]
    # Split indices
    for key in ("train", "val", "test"):
        save[f"{key}_idx"] = split[key]
    # Metadata
    save["meta_json"] = json.dumps(meta)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save)
    print(f"Saved: {out_path}")


def load_dataset_fields(path: str) -> list[str]:
    """Return the list of field names in a saved dataset (no data loaded)."""
    with np.load(path, allow_pickle=True) as z:
        return sorted(z.files)


# ---------------------------------------------------------------------------
# Probe orders
# ---------------------------------------------------------------------------

def _random_probe_token_orders(
    batch_size: int, seed: int, step: int, device: str,
) -> torch.Tensor:
    """Random model-coordinate token orders (batch, SEQ_LEN).

    Deterministic in (seed, step, batch_index).
    """
    rows = []
    for b in range(batch_size):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) * 100_000_000 + int(step) * 1000 + b)
        blocks = torch.randperm(N, generator=g)
        rows.append(
            expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0]
        )
    return torch.stack(rows).to(device)


def select_layer_attention(attn_list, layers):
    """Select and stack attention from one or more transformer layers.

    The reveal-order signal can relocate across layers between training
    trajectories (L0 vs L1), so extraction must be layer-aware rather than
    hard-coded to ``attn_list[0]``.

    Args:
        attn_list: sequence of per-layer attention arrays/tensors, each
            shaped ``(B, H, T+1, T+1)``.  List index == transformer layer.
        layers: list of layer indices to read.  A single element returns that
            layer's attention unchanged; multiple indices are concatenated
            along the head axis in the given order, yielding
            ``(B, len(layers)*H, T+1, T+1)``.  Concatenating on the head axis
            lets a head-permutation-equivariant g_beta treat each
            ``(layer, head)`` pair as one element of an expanded head set.

    Returns:
        ``np.ndarray`` of shape ``(B, len(layers)*H, T+1, T+1)``, float32.

    Raises:
        ValueError: if ``layers`` is empty.
        IndexError: if any layer index is out of range.
    """
    if len(layers) == 0:
        raise ValueError("layers must be non-empty")
    n_layers = len(attn_list)
    selected = []
    for l in layers:
        if l < 0 or l >= n_layers:
            raise IndexError(
                f"layer {l} out of range for {n_layers}-layer attn_list"
            )
        a = attn_list[l]
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        selected.append(a.astype(np.float32))
    if len(selected) == 1:
        return selected[0]
    return np.concatenate(selected, axis=1)


def filter_heads(attn: np.ndarray, heads: tuple, layers: tuple) -> np.ndarray:
    """Keep only selected head indices within the given layer stack.

    When a single layer is read (e.g. ``layers=(1,)``), each head index in
    ``heads`` maps directly to that layer's head axis.  Multi-layer filtering
    is not yet supported — pass the layer explicitly and use per-layer head
    indices.

    Args:
        attn: ``(B, len(layers)*8, 257, 257)`` float32.
        heads: tuple of head indices to keep, e.g. ``(0, 7, 1, 5)``.
        layers: tuple of layer indices (must be length 1).

    Returns:
        ``np.ndarray`` of shape ``(B, len(heads), 257, 257)``, float32.
    """
    if len(layers) != 1:
        raise ValueError(
            f"--heads filtering requires a single --readout-layers, "
            f"got {layers}"
        )
    H_per_layer = 8
    return attn[:, list(heads), :, :]


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_l0_dynamic_gbeta_dataset(
    ckpt_path: str,
    M: int = 2000,
    batch_mean_size: int = 16,
    seed: int = 2,
    device: str = "cuda:0",
    split: str = "train",
    forward_batch: int = 8,
    destroy_replicas: int = 1,
    teacher_temperature: float = 1.0,
    teacher_smoothing: float = 0.05,
    out_path: Optional[str] = None,
    layers: tuple = (0,),
    heads: tuple = (),
) -> str:
    """Build and save the label-free L0 dynamic g_beta pretrain dataset.

    Args:
        ckpt_path: path to AOGPT checkpoint.
        M: number of batch-mean graphs (default 2000).
        batch_mean_size: per-sample graphs per batch-mean (default 16).
        seed: random seed for sampling and probe orders.
        device: torch device.
        split: data split to sample from ("train" or "eval").
        forward_batch: max sequences per model forward pass.
        destroy_replicas: destroyed replicas per head for gap computation.
        teacher_temperature: softmax temperature for teacher weights.
        teacher_smoothing: uniform mixture weight.
        out_path: output .npz path (auto-generated if None).

    Returns:
        str: path to the saved .npz file.
    """
    total_samples = M * batch_mean_size
    print(f"Building L0 dynamic g_beta dataset:")
    print(f"  ckpt={ckpt_path}")
    print(f"  M={M}, batch_mean_size={batch_mean_size}, total_samples={total_samples}")
    print(f"  seed={seed}, split={split}, device={device}")

    # ── Load model and chunks ──
    model, chunks, clean_perm, dev, chunk_indices = _load_model_and_chunks(
        ckpt_path, total_samples, seed, device, split,
    )
    # ══════════════════════════════════════════════════════════════════════
    # LABEL-FREE: clean_perm is intentionally NOT passed to extraction or
    # teacher.  It is used ONLY for chunk loading (via _load_model_and_chunks
    # internals).  Any accidental usage downstream is a bug.
    # ══════════════════════════════════════════════════════════════════════
    del clean_perm  # enforce: cannot be used past this point

    model.eval()
    # Infer n_head from the first forward pass's attention shape.
    # model.n_head is also available but attention shape is the ground truth.

    # ── Extract per-sample B_raw in forward batches ──
    all_B_samples = []
    n_batches = (total_samples + forward_batch - 1) // forward_batch

    for bi in range(n_batches):
        start = bi * forward_batch
        end = min(start + forward_batch, total_samples)
        batch_size = end - start

        idx_batch = chunks[start:end].to(dev)
        probe = _random_probe_token_orders(
            batch_size, seed, bi, str(dev),
        )

        _, _, attn_list = model.forward_fn(
            idx_batch, probe, return_attentions=True,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        # Layer-aware attention stack: (B, len(layers)*H, 257, 257).
        # Single layer → that layer; multiple → concatenated on head axis.
        attn_sel = select_layer_attention(attn_list, list(layers))

        # Optional per-head filtering
        if heads:
            attn_sel = filter_heads(attn_sel, heads, layers)

        probe_np = probe.cpu().numpy()

        B_samples = build_model_frame_strict65(attn_sel, probe_np)
        all_B_samples.append(B_samples)

        if (bi + 1) % max(1, n_batches // 10) == 0 or end >= total_samples:
            print(f"  [extract] {end}/{total_samples} per-sample graphs")

    all_B = np.concatenate(all_B_samples, axis=0)  # (total_samples, H, 65, 65)
    print(f"  per-sample B shape: {all_B.shape}")

    H = all_B.shape[1]  # infer from extracted data
    print(f"  n_heads (from attention): {H}")

    # ── Batch-mean ──
    B_raw = batch_mean_heads(all_B, batch_mean_size)  # (M, H, 65, 65)
    print(f"  batch-mean B shape: {B_raw.shape}")

    # ── Generate teacher labels per batch-mean ──
    teacher_orders = np.zeros((M, H, 64), dtype=np.int64)
    teacher_ranks = np.zeros((M, H, 64), dtype=np.int64)
    teacher_margin = np.zeros((M, H), dtype=np.float32)
    teacher_destroyed_gap = np.zeros((M, H), dtype=np.float32)
    teacher_agreement = np.zeros((M, H), dtype=np.float32)
    teacher_quality = np.zeros((M, H), dtype=np.float32)
    teacher_weights = np.zeros((M, H), dtype=np.float32)
    teacher_pairwise = np.zeros((M, 64, 64), dtype=np.float32)
    teacher_consensus_order = np.zeros((M, 64), dtype=np.int64)

    for m in range(M):
        result = build_dynamic_teacher(
            B_raw[m],
            destroy_seed=seed * 1000 + m,
            n_destroy_replicas=destroy_replicas,
            teacher_temperature=teacher_temperature,
            teacher_smoothing=teacher_smoothing,
            mode="C-D+L",
        )
        teacher_orders[m] = result["orders"]
        teacher_ranks[m] = result["ranks"]
        teacher_margin[m] = result["margin"]
        teacher_destroyed_gap[m] = result["destroyed_gap"]
        teacher_agreement[m] = result["agreement"]
        teacher_quality[m] = result["quality"]
        teacher_weights[m] = result["weights"]
        teacher_pairwise[m] = result["pairwise"]
        teacher_consensus_order[m] = weighted_consensus_order(
            result["ranks"], result["weights"],
        )

        if (m + 1) % max(1, M // 5) == 0 or (m + 1) == M:
            print(f"  [teacher] {m + 1}/{M} batch-mean graphs")

    teacher = {
        "orders": teacher_orders,
        "ranks": teacher_ranks,
        "margin": teacher_margin,
        "destroyed_gap": teacher_destroyed_gap,
        "agreement": teacher_agreement,
        "quality": teacher_quality,
        "weights": teacher_weights,
        "pairwise": teacher_pairwise,
        "consensus_order": teacher_consensus_order,
    }

    # ── Split ──
    split_idx = deterministic_split(M, seed=seed)
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(split_idx[k])} samples")

    # ── Save ──
    if out_path is None:
        ckpt_name = pathlib.Path(ckpt_path).stem
        out_path = (
            f"block_lo_arm_order_network/batch_readout/logs/"
            f"l0_dynamic_gbeta_{ckpt_name}_M{M}_s{seed}.npz"
        )

    meta = {
        "source_ckpt": str(ckpt_path),
        "M": M,
        "batch_mean_size": batch_mean_size,
        "seed": seed,
        "split_source": split,
        "layers": list(layers),
        "layer": list(layers)[0] if len(layers) == 1 else -1,
        "n_heads": H,
        "heads": list(heads) if heads else list(range(H)),
        "coordinate_frame": "model",
        "teacher_mode": "C-D+L",
        "destroy_replicas": destroy_replicas,
        "teacher_temperature": teacher_temperature,
        "teacher_smoothing": teacher_smoothing,
        "label_free": True,
    }

    save_dataset(out_path, B_raw, teacher, split_idx, meta)

    # ── Final audit ──
    fields = load_dataset_fields(out_path)
    forbidden_in_save = {"block_perm", "inv_perm", "clean_perm",
                         "physical_order", "l2r_order"}
    leaked = forbidden_in_save & set(fields)
    if leaked:
        raise RuntimeError(
            f"Dataset contains forbidden physical fields: {sorted(leaked)}"
        )
    print(f"  fields: {fields}")
    print(f"  label-free audit: PASS (no physical fields)")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build label-free L0 dynamic g_beta pretrain dataset"
    )
    p.add_argument("--ckpt", required=True,
                   help="Path to AOGPT checkpoint .pt file")
    p.add_argument("--M", type=int, default=2000,
                   help="Number of batch-mean graphs (default 2000)")
    p.add_argument("--batch-mean-size", type=int, default=16,
                   help="Per-sample graphs per batch-mean (default 16)")
    p.add_argument("--seed", type=int, default=2,
                   help="Random seed")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="train",
                   choices=("train", "eval"))
    p.add_argument("--forward-batch", type=int, default=8,
                   help="Max sequences per forward pass")
    p.add_argument("--destroy-replicas", type=int, default=1,
                   help="Destroyed replicas per head (default 1)")
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--teacher-smoothing", type=float, default=0.05)
    p.add_argument("--out", default=None,
                   help="Output .npz path (auto-generated if omitted)")
    p.add_argument("--readout-layers", type=int, nargs="+", default=[0],
                   help="Transformer layer index/indices to read attention "
                        "from (default [0]). Multiple indices are fused on the "
                        "head axis as a layer-head set, e.g. --readout-layers "
                        "0 1 2 3 for all-layer g_beta.")
    p.add_argument("--heads", type=int, nargs="+", default=[],
                   help="Head indices to keep within the selected layer(s). "
                        "Requires a single --readout-layers value. "
                        "Example: --readout-layers 1 --heads 0 7 1 5")
    args = p.parse_args()

    build_l0_dynamic_gbeta_dataset(
        ckpt_path=args.ckpt,
        M=args.M,
        batch_mean_size=args.batch_mean_size,
        seed=args.seed,
        device=args.device,
        split=args.split,
        forward_batch=args.forward_batch,
        destroy_replicas=args.destroy_replicas,
        teacher_temperature=args.teacher_temperature,
        teacher_smoothing=args.teacher_smoothing,
        out_path=args.out,
        layers=tuple(args.readout_layers),
        heads=tuple(args.heads),
    )


if __name__ == "__main__":
    main()
