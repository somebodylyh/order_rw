"""NR-1 Task 2: per-sample B = A^T extraction with reproducible seeding.

Wraps train_clean_aogpt.extract_A_matrices (with the seed bugfix from Task 1),
applies B = A^T and fills the diagonal with 0. Two entry points:

  - extract_per_sample_B(ckpt, M, seed)              -> B            (shape (M, 64, 64))
  - extract_per_sample_B_with_chunks(ckpt, M, seed)  -> (B, chunks, split)

The "with_chunks" variant is used by neural_readout.dataset to persist the
chunk indices alongside (B, sigma, rank) so that downstream frozen-theta NLL
evaluation (Task 11) can reload the exact same chunks the dataset was built
from.

Default split is "train" because NR-1 needs ~10k graphs and clean_base_random_perm's
eval set has only 200 chunks. Both "train" and "eval" are accepted; "val" is
not exposed since it is reserved for AOGPT main-model validation.
"""
import sys
import pathlib

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from train_clean_aogpt import (
    extract_A_matrices,
    build_model,
    CleanPermutation,
    phys_to_model_idx_clean,
    build_phys_to_model_token_gather,
    BLOCK_LEN as _BLOCK_LEN,
)
from training_utils import load_train_chunks, load_token_stream, sample_stream_batch, SEQ_LEN


_ALLOWED_SPLITS = ("train", "eval")


def _chunks_from_continuous_stream(ckpt, clean_perm, M, seed, device):
    """Sample M random 256-token windows from the continuous train stream.

    For models trained with ``--data-source continuous``, the train split is a
    single memmap of ~119M uint16 tokens rather than pre-chunked arrow data.
    This function draws M contiguous windows at reproducibly seeded offsets,
    maps them from physical→model token space, and returns them in the same
    (M, SEQ_LEN) long-tensor format as the chunk-based path.
    """
    args = ckpt.get("args", {})
    train_bin = args.get("train_bin")
    if not train_bin:
        raise ValueError(
            "continuous model ckpt is missing args.train_bin; "
            "cannot sample from the train stream"
        )
    stream = load_token_stream(train_bin)
    g_gather = build_phys_to_model_token_gather(clean_perm, _BLOCK_LEN)

    # Each call to sample_stream_batch draws batch_size windows at one
    # (seed, step, micro) position.  We fan out over micro steps so large
    # M fits without exceeding max contiguous-token limits.
    MICRO_MAX = 256
    chunks_list = []
    for i in range(M):
        micro = i % MICRO_MAX
        step = i // MICRO_MAX
        batch = sample_stream_batch(
            stream, 1, SEQ_LEN, seed=seed, global_step=step, micro_step=micro,
        )  # (1, SEQ_LEN) physical-token long tensor
        chunks_list.append(batch[0, g_gather])  # (SEQ_LEN,) model-token
    chunks = torch.stack(chunks_list)  # (M, SEQ_LEN)
    chunk_index = np.arange(M, dtype=np.int64)
    return chunks, chunk_index


def _load_model_and_chunks(ckpt_path, M, seed, device, split):
    """Common setup: validate args (cheap), load ckpt protocol, then load model + chunks (expensive).

    Supports both classic chunk-based models and continuous-stream models
    (``--data-source continuous``).  For continuous models the train split
    samples from the memmap stream; the eval split uses the fixed seeded
    windows stored in the checkpoint protocol.
    """
    # Cheap validations first so misuse fails instantly.
    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {_ALLOWED_SPLITS!r}, got {split!r}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    args = ckpt.get("args", {})
    is_continuous = (args.get("data_source") == "continuous")

    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN

    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(
            protocol["block_perm_phys_to_model"], dtype=torch.long
        ),
        inv_perm_model_to_phys=torch.tensor(
            protocol["inv_perm_model_to_phys"], dtype=torch.long
        ),
    )

    if device.startswith("cuda") and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    # Expensive: build model and load state_dict.
    model = build_model(model_args, dev, compile_model=False)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError("ckpt has neither 'model' nor 'model_state_dict'")
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(dev)
    model.eval()

    if is_continuous and split == "train":
        # Sample arbitrary windows from the continuous memmap stream.
        chunks, chunk_index = _chunks_from_continuous_stream(
            ckpt, clean_perm, M, seed, device
        )
        return model, chunks, clean_perm, dev, chunk_index

    # --- classic chunk-based or continuous-eval path ---
    split_indices = np.asarray(protocol[f"{split}_indices"], dtype=np.int64)
    if is_continuous:
        # Eval split for continuous models: the protocol already stores
        # pre-computed model-space token windows.
        if M > len(split_indices):
            raise ValueError(
                f"requested M={M} chunks but continuous eval split only has "
                f"{len(split_indices)} windows"
            )
        rng = np.random.RandomState(seed)
        chunk_index = rng.choice(len(split_indices), size=M, replace=False).astype(np.int64)
        chunk_index.sort()
        idx_split = torch.from_numpy(split_indices[chunk_index]).long()  # (M, SEQ_LEN)
        chunks = idx_split
        return model, chunks, clean_perm, dev, chunk_index

    # Classic chunk-based path (arrow tokenization).
    if M > len(split_indices):
        raise ValueError(
            f"requested M={M} chunks but split={split!r} only has "
            f"{len(split_indices)} chunks; choose split='train' for large M"
        )

    idx_phys = load_train_chunks(n_chunks=None)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    idx_split = idx_model[split_indices]

    rng = np.random.RandomState(seed)
    chunk_index = rng.choice(len(idx_split), size=M, replace=False).astype(np.int64)
    chunk_index.sort()  # sort so chunks pass through model in a stable order
    chunks = idx_split[chunk_index]
    return model, chunks, clean_perm, dev, chunk_index


def extract_per_sample_B_with_chunks(
    ckpt_path,
    M,
    seed,
    device="cuda:0",
    split="train",
):
    """Forward M chunks from `split` through ckpt, return (B, chunk_index, split).

    Returns:
        B: (M, 64, 64) float32, B_i = A_i^T with fill_diagonal(0).
        chunk_index: (M,) int64 indices into protocol[f"{split}_indices"], sorted.
        split: str ("train" or "eval"), echoed back for dataset persistence.

    Bit-reproducible given (ckpt_path, M, seed, split).
    """
    model, chunks, clean_perm, dev, chunk_index = _load_model_and_chunks(
        ckpt_path, M, seed, device, split
    )
    A = extract_A_matrices(model, chunks, clean_perm, dev, n_chunks=M, seed=seed)
    B = np.transpose(A, (0, 2, 1)).copy()
    for i in range(B.shape[0]):
        np.fill_diagonal(B[i], 0.0)
    return B.astype(np.float32), chunk_index, split


def extract_per_sample_B(ckpt_path, M, seed, device="cuda:0", split="train"):
    """Thin wrapper that returns only B, dropping the chunk_index/split."""
    B, _ci, _split = extract_per_sample_B_with_chunks(
        ckpt_path, M, seed, device=device, split=split
    )
    return B
