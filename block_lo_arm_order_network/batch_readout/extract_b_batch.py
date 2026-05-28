"""BR-1 Task 2: batch-mean attention extractor.

Groups (M * batch_size) per-sample graphs B_i = A_theta(x_i)^T into M
batch-mean graphs B_batch[m] = mean_b B_{m*batch_size + b}, then zeros the
diagonal. Wraps neural_readout.extract_b.extract_per_sample_B_with_chunks
for the per-sample forward pass and chunk-index bookkeeping.

Output dtype is float32 (matches neural_readout convention); the mean is
computed in float64 internally so accumulation noise stays sub-1e-7.
"""
import numpy as np

from neural_readout.extract_b import extract_per_sample_B_with_chunks


def extract_batch_mean_B(
    ckpt_path,
    M: int,
    batch_size: int,
    seed: int,
    device: str = "cuda:0",
    split: str = "train",
    return_per_sample: bool = False,
):
    """Extract M batch-mean attention graphs.

    Args:
        ckpt_path: AOGPT checkpoint with clean_protocol + model_args.
        M: number of batch-mean graphs to produce.
        batch_size: number of per-sample graphs averaged into each B_batch[m].
        seed: seeds chunk sampling AND extract_A_matrices' randperm.
        device: forwarded to extract_per_sample_B_with_chunks.
        split: "train" or "eval" — the per-sample extractor enforces this.
        return_per_sample: if True, include the (M, batch_size, N, N) tensor.

    Returns dict with keys:
        B_batch: (M, N, N) float32, diagonal zeroed.
        chunks:  (M, batch_size) int64 — indices into protocol[f"{split}_indices"].
        split:   str echoed back.
        meta:    {M, batch_size, seed, ckpt}.
        B_per_sample (optional): (M, batch_size, N, N) float32.
    """
    if M <= 0 or batch_size <= 0:
        raise ValueError(
            f"M and batch_size must be positive; got M={M}, batch_size={batch_size}"
        )
    total = M * batch_size
    B_per, chunk_index, split_str = extract_per_sample_B_with_chunks(
        ckpt_path=ckpt_path, M=total, seed=seed, device=device, split=split,
    )
    # B_per is float32 (M*B, N, N). Reshape into (M, B, N, N) and mean over B.
    N = B_per.shape[-1]
    B_per_grouped = B_per.reshape(M, batch_size, N, N)
    # Accumulate in float64 to keep the per-sample noise floor below 1e-7,
    # then cast back to float32 for storage parity with neural_readout.dataset.
    B_batch = B_per_grouped.astype(np.float64).mean(axis=1).astype(np.float32)
    # Re-zero diagonal: averaging zeros stays zero in exact arithmetic, but
    # the float64->float32 round-trip can leave sub-eps non-zeros.
    diag = np.arange(N)
    B_batch[:, diag, diag] = 0.0
    chunks_grouped = np.asarray(chunk_index, dtype=np.int64).reshape(M, batch_size)
    out = {
        "B_batch": B_batch,
        "chunks": chunks_grouped,
        "split": split_str,
        "meta": {"M": M, "batch_size": batch_size, "seed": seed, "ckpt": ckpt_path},
    }
    if return_per_sample:
        out["B_per_sample"] = B_per_grouped
    return out
