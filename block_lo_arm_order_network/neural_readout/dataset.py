"""NR-1 Task 4: (B, sigma, rank, chunk_index) dataset on disk (spec §2.4).

Stores everything needed to:
  - train g_beta with attention-order matching (B, sigma, rank)
  - re-locate the original wikitext chunks for the frozen-theta NLL diagnostic
    in Task 11 (chunk_index + split tells us which idx_model[split_indices][k]
    each sample came from)

The npz also carries `meta` (ckpt_path, M, seed, alpha_dep, split) for
provenance — these never participate in training.
"""
import sys
import pathlib

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))


_REQUIRED_KEYS = ("B", "sigma", "rank", "chunk_index", "split")


def save_dataset(path, B, sigma, rank, chunk_index, split, meta=None):
    """Persist a (B, sigma, rank, chunk_index, split, meta) dataset to .npz.

    Args:
        B: (M, N, N) float32, per-sample attention graphs.
        sigma: (M, N) int64, teacher reveal orders.
        rank:  (M, N) int64, teacher ranks (0 = earliest).
        chunk_index: (M,) int64, indices into protocol[f"{split}_indices"].
        split: str, "train" or "eval".
        meta: optional dict of additional provenance keys (ckpt_path, seed, etc).
    """
    B = np.asarray(B, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.int64)
    rank = np.asarray(rank, dtype=np.int64)
    chunk_index = np.asarray(chunk_index, dtype=np.int64)

    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError(f"B must be (M, N, N), got shape {B.shape}")
    M, N = B.shape[0], B.shape[1]
    if sigma.shape != (M, N) or rank.shape != (M, N):
        raise ValueError(f"sigma/rank must be (M, N)=({M}, {N}); got {sigma.shape}, {rank.shape}")
    if chunk_index.shape != (M,):
        raise ValueError(f"chunk_index must be (M,)=({M},), got {chunk_index.shape}")

    payload = {
        "B": B,
        "sigma": sigma,
        "rank": rank,
        "chunk_index": chunk_index,
        "split": np.asarray(split),
    }
    if meta:
        for k, v in meta.items():
            payload[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(path, **payload)


def load_dataset(path):
    """Return (B, sigma, rank, chunk_index, split)."""
    data = np.load(path, allow_pickle=False)
    missing = [k for k in _REQUIRED_KEYS if k not in data.files]
    if missing:
        raise KeyError(f"dataset {path} missing required keys: {missing}")
    return (
        data["B"],
        data["sigma"],
        data["rank"],
        data["chunk_index"],
        str(data["split"].item()) if data["split"].ndim == 0 else str(data["split"]),
    )


def build_dataset_from_ckpt(
    ckpt_path,
    M,
    seed,
    alpha_dep,
    out_path,
    device="cuda:0",
    split="train",
):
    """End-to-end: forward M chunks -> B -> teacher labels -> save.

    Args:
        ckpt_path: AOGPT checkpoint path.
        M: number of graphs.
        seed: int, controls chunk selection + per-sample randperm in extract_A_matrices.
        alpha_dep: source-start readiness weight (default 0.5 in callers).
        out_path: where to save the .npz.
        device: torch device string.
        split: "train" (default) or "eval".

    Returns:
        out_path (str).
    """
    from neural_readout.extract_b import extract_per_sample_B_with_chunks
    from neural_readout.teacher_labels import generate_teacher_label

    B, chunk_index, split_out = extract_per_sample_B_with_chunks(
        ckpt_path, M=M, seed=seed, device=device, split=split,
    )
    M_actual, N, _ = B.shape
    sigmas = np.zeros((M_actual, N), dtype=np.int64)
    ranks = np.zeros((M_actual, N), dtype=np.int64)
    for i in range(M_actual):
        sigma, rank, _Y = generate_teacher_label(B[i], alpha_dep=alpha_dep)
        sigmas[i] = sigma
        ranks[i] = rank

    meta = {
        "ckpt_path": str(ckpt_path),
        "M": int(M_actual),
        "seed": int(seed),
        "alpha_dep": float(alpha_dep),
    }
    save_dataset(out_path, B, sigmas, ranks, chunk_index, split_out, meta=meta)
    return out_path
