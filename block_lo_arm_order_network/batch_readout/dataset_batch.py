"""BR-1 Task 3: build / save / load (B_batch, sigma_T, rank, pairwise_Y) dataset.

Teacher labels come from neural_readout.teacher_labels.generate_teacher_label
applied per batch-mean graph (CDL source-start, default alpha_dep=0.5).
Split is deterministic in `seed` so re-running with the same seed reproduces
exactly the same train/val/test partition.

Two entry points:

  - build_dataset(ckpt_path, M, batch_size, seed, ...): hits the wikitext +
    model pipeline via extract_batch_mean_B. Heavy; covered by the Phase-0
    smoke (BR-1 Task 10).
  - _build_from_B_array(B, chunks, seed, ...): pure-numpy core, testable with
    a synthetic B array. All the split / labeling / IO logic lives here.
"""
from __future__ import annotations

import numpy as np

from batch_readout.extract_b_batch import extract_batch_mean_B
from neural_readout.teacher_labels import generate_teacher_label


def _build_from_B_array(
    B: np.ndarray,
    chunks: np.ndarray,
    seed: int,
    alpha_dep: float = 0.5,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    out_path: str | None = None,
    extra_meta: dict | None = None,
):
    """Build (sigma_T, rank, pairwise_Y) for each graph in B, split, optionally save.

    Args:
        B: (M, N, N) float32 — batch-mean attention graphs with zero diagonal.
        chunks: (M, batch_size) int64 — global chunk indices per graph (for
                downstream NLL eval that needs to reload the same tokens).
        seed: seeds the permutation that defines the train/val/test split.
        alpha_dep: forwarded to generate_teacher_label.
        train_frac, val_frac: split fractions; test takes the remainder.
        out_path: if given, saves an npz with all per-split arrays.
        extra_meta: merged into the on-disk meta dict.

    Returns dict {"M_train", "M_val", "M_test", "sigma_T", "rank", "pairwise_Y"}.
    The full label arrays are returned alongside the split sizes so callers
    can do in-memory work without round-tripping through disk.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(
            f"invalid split fractions train={train_frac} val={val_frac} (must be in (0,1) "
            "with train+val<1)"
        )
    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError(f"B must be (M, N, N); got {B.shape}")
    M, N, _ = B.shape
    if chunks.shape[0] != M:
        raise ValueError(f"chunks first dim must equal M={M}; got {chunks.shape}")

    sigma = np.zeros((M, N), dtype=np.int64)
    rank = np.zeros((M, N), dtype=np.int64)
    Y = np.zeros((M, N, N), dtype=np.uint8)
    for m in range(M):
        s_m, r_m, y_m = generate_teacher_label(B[m], alpha_dep=alpha_dep)
        sigma[m] = s_m
        rank[m] = r_m
        Y[m] = y_m

    # Deterministic split. Distinct seed offset from extractor so callers can
    # reuse the same `seed` argument without aliasing the two RNGs.
    rng = np.random.default_rng(seed + 1_000_003)
    idx = rng.permutation(M)
    n_train = int(round(M * train_frac))
    n_val = int(round(M * val_frac))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    if out_path is not None:
        meta = {"M": M, "N": N, "seed": seed, "alpha_dep": alpha_dep,
                "train_frac": train_frac, "val_frac": val_frac}
        if extra_meta is not None:
            meta.update(extra_meta)
        np.savez(
            out_path,
            train_B_batch=B[tr], train_sigma_T=sigma[tr], train_rank=rank[tr],
            train_pairwise_Y=Y[tr], train_chunks=chunks[tr],
            val_B_batch=B[va], val_sigma_T=sigma[va], val_rank=rank[va],
            val_pairwise_Y=Y[va], val_chunks=chunks[va],
            test_B_batch=B[te], test_sigma_T=sigma[te], test_rank=rank[te],
            test_pairwise_Y=Y[te], test_chunks=chunks[te],
            meta=np.array([meta], dtype=object),
        )

    return {
        "M_train": int(len(tr)), "M_val": int(len(va)), "M_test": int(len(te)),
        "sigma_T": sigma, "rank": rank, "pairwise_Y": Y,
        "train_idx": tr, "val_idx": va, "test_idx": te,
    }


def build_dataset(
    ckpt_path,
    M: int,
    batch_size: int,
    seed: int,
    alpha_dep: float = 0.5,
    out_path: str | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    device: str = "cuda:0",
    split: str = "train",
):
    """End-to-end: extract batch-mean B from ckpt, label, split, save."""
    ext = extract_batch_mean_B(
        ckpt_path=ckpt_path, M=M, batch_size=batch_size, seed=seed,
        device=device, split=split, return_per_sample=False,
    )
    info = _build_from_B_array(
        B=ext["B_batch"], chunks=ext["chunks"], seed=seed,
        alpha_dep=alpha_dep, train_frac=train_frac, val_frac=val_frac,
        out_path=out_path,
        extra_meta={"ckpt": ckpt_path, "batch_size": batch_size, "split": ext["split"]},
    )
    return {k: info[k] for k in ("M_train", "M_val", "M_test")}


def load_dataset(path: str) -> dict:
    """Load an npz produced by build_dataset / _build_from_B_array."""
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
