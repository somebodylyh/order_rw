"""§3.3 selected-head g_β pretrain dataset builder.

Mirrors `batch_readout.dataset_batch.build_dataset`, but swaps the B-extraction
front-end: instead of the top-4-variance *head-mean* heavy graph under the OLD
none-token handling, it extracts ONE selected head's block graph under the **B1
canonical extraction** (`none_mode="b1"` = 65-node), which the §3.0 full-ladder
gate sealed as canonical (winner L0H0, 45/45 stable).

Pipeline (single ckpt, single head):
    load model + M*batch_size chunks
    -> extract_per_head_and_heavy_A(none_mode)        # (n, L, H, N, N)
    -> select head (l*, h*)                            # (n, N, N)
    -> canonical batch-mean B = mean_b (A_b^T), diag 0 # (M, N, N)  [_batch_mean_B]
    -> _build_from_B_array: CDL teacher σ_T + train/val/test split + save

Everything downstream (CDL teacher labels, split, training, eval) is reused
verbatim from NR-1 / BR-1 so the only new logic is head selection + B1 wiring.
This keeps `offline scan == g_β pretrain dataset == hook input` on one extraction
path (the mismatch guard from spec §3.2).
"""
from __future__ import annotations

import numpy as np

from neural_readout.extract_b import _load_model_and_chunks
from per_head_order_scan import _batch_mean_B, extract_per_head_and_heavy_A
from batch_readout.dataset_batch import _build_from_B_array


def selected_head_B(A_lh: np.ndarray, head, batch_size: int) -> np.ndarray:
    """Select head (l, h) from A_lh (n, L, H, N, N) -> canonical batch-mean B.

    Returns (M, N, N) float32 with M = n // batch_size, applying the project's
    canonical B = A^T batch-mean (delegated to `per_head_order_scan._batch_mean_B`,
    diagonal zeroed). The trailing n % batch_size samples are dropped.
    """
    l, h = head
    A_sel = A_lh[:, l, h]  # (n, N, N)
    n = A_sel.shape[0]
    M = n // batch_size
    if M == 0:
        raise ValueError(
            f"need at least batch_size={batch_size} samples; got n={n}"
        )
    return _batch_mean_B(A_sel[: M * batch_size], M, batch_size)


def extract_selected_head_batch_mean_B(
    ckpt_path,
    head,
    M: int,
    batch_size: int,
    seed: int,
    none_mode: str = "b1",
    device: str = "cuda:0",
    split: str = "train",
    fwd_batch: int = 64,
):
    """Extract M batch-mean B graphs for one selected head under `none_mode`.

    Returns dict {B_batch (M,N,N) float32, chunks (M,batch_size) int64, split,
    meta}. Mirrors `extract_b_batch.extract_batch_mean_B`'s output contract so it
    is a drop-in front-end for `_build_from_B_array`.
    """
    if M <= 0 or batch_size <= 0:
        raise ValueError(
            f"M and batch_size must be positive; got M={M}, batch_size={batch_size}"
        )
    total = M * batch_size
    model, chunks, clean_perm, dev, chunk_index = _load_model_and_chunks(
        ckpt_path, total, seed, device, split
    )
    A_lh, _A_heavy = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed, fwd_batch=fwd_batch, none_mode=none_mode,
        head=head,
    )
    B_batch = selected_head_B(A_lh, head, batch_size)
    chunks_grouped = np.asarray(chunk_index, dtype=np.int64).reshape(M, batch_size)
    return {
        "B_batch": B_batch,
        "chunks": chunks_grouped,
        "split": split,
        "meta": {"M": M, "batch_size": batch_size, "seed": seed,
                 "ckpt": str(ckpt_path), "head": [int(head[0]), int(head[1])],
                 "none_mode": none_mode},
    }


def build_selected_head_dataset(
    ckpt_path,
    head,
    M: int,
    batch_size: int,
    seed: int,
    none_mode: str = "b1",
    alpha_dep: float = 0.5,
    out_path: str | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    device: str = "cuda:0",
    split: str = "train",
    fwd_batch: int = 64,
):
    """End-to-end: extract selected-head B1 (65-node) batch-mean B, CDL-label, split, save.

    `head` is (layer, head_index). With `train_frac`/`val_frac` < 1 the remainder
    becomes the test split. Returns the `_build_from_B_array` info dict.
    """
    ext = extract_selected_head_batch_mean_B(
        ckpt_path=ckpt_path, head=head, M=M, batch_size=batch_size, seed=seed,
        none_mode=none_mode, device=device, split=split, fwd_batch=fwd_batch,
    )
    return _build_from_B_array(
        B=ext["B_batch"], chunks=ext["chunks"], seed=seed,
        alpha_dep=alpha_dep, train_frac=train_frac, val_frac=val_frac,
        out_path=out_path,
        extra_meta={"ckpt": str(ckpt_path), "head": ext["meta"]["head"],
                    "none_mode": none_mode, "batch_size": batch_size,
                    "split": ext["split"]},
    )
