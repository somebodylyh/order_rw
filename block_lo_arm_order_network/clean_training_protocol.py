"""Shared protocol helpers for clean AO-GPT training runs.

This module uses the new explicit convention for clean experiments:
    block_perm_phys_to_model[physical_block] = model_block
    inv_perm_model_to_phys[model_block] = physical_block
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CleanPermutation:
    block_perm_phys_to_model: torch.Tensor
    inv_perm_model_to_phys: torch.Tensor

    @classmethod
    def from_model_to_phys(cls, model_to_phys):
        inv_perm_model_to_phys = torch.as_tensor(model_to_phys, dtype=torch.long).cpu()
        block_perm_phys_to_model = torch.empty_like(inv_perm_model_to_phys)
        block_perm_phys_to_model[inv_perm_model_to_phys] = torch.arange(
            inv_perm_model_to_phys.numel(), dtype=torch.long
        )
        return cls(
            block_perm_phys_to_model=block_perm_phys_to_model,
            inv_perm_model_to_phys=inv_perm_model_to_phys,
        )


@dataclass(frozen=True)
class FixedDataProtocol:
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_shuffle_order: np.ndarray
    eval_indices: np.ndarray


def build_clean_block_permutation(num_blocks, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    model_to_phys = torch.randperm(int(num_blocks), generator=generator, device="cpu")
    return CleanPermutation.from_model_to_phys(model_to_phys)


def verify_clean_coordinate_round_trip(clean_perm, block_len):
    block_perm = clean_perm.block_perm_phys_to_model.long().cpu()
    inv_perm = clean_perm.inv_perm_model_to_phys.long().cpu()
    num_blocks = block_perm.numel()
    if inv_perm.numel() != num_blocks:
        return False

    expected = torch.arange(num_blocks, dtype=torch.long)
    if not torch.equal(inv_perm[block_perm], expected):
        return False
    if not torch.equal(block_perm[inv_perm], expected):
        return False

    seq_len = num_blocks * int(block_len)
    idx_phys = torch.arange(seq_len, dtype=torch.long).view(1, -1)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    recovered = model_to_phys_idx_clean(idx_model, clean_perm)
    return torch.equal(idx_phys, recovered)


def phys_to_model_idx_clean(idx_phys, clean_perm):
    block_perm = clean_perm.block_perm_phys_to_model.to(idx_phys.device).long()
    batch_size, seq_len = idx_phys.shape
    block_len = seq_len // block_perm.numel()
    idx_model = torch.empty_like(idx_phys)
    for pos in range(seq_len):
        phys_block = pos // block_len
        offset = pos % block_len
        model_block = int(block_perm[phys_block].item())
        idx_model[:, model_block * block_len + offset] = idx_phys[:, pos]
    return idx_model


def model_to_phys_idx_clean(idx_model, clean_perm):
    inv_perm = clean_perm.inv_perm_model_to_phys.to(idx_model.device).long()
    batch_size, seq_len = idx_model.shape
    block_len = seq_len // inv_perm.numel()
    idx_phys = torch.empty_like(idx_model)
    for pos in range(seq_len):
        model_block = pos // block_len
        offset = pos % block_len
        phys_block = int(inv_perm[model_block].item())
        idx_phys[:, phys_block * block_len + offset] = idx_model[:, pos]
    return idx_phys


def physical_blocks_to_model_blocks(physical_orders, clean_perm):
    block_perm = clean_perm.block_perm_phys_to_model.to(physical_orders.device).long()
    return block_perm[physical_orders.long()]


def model_blocks_to_physical_blocks(model_orders, clean_perm):
    inv_perm = clean_perm.inv_perm_model_to_phys.to(model_orders.device).long()
    return inv_perm[model_orders.long()]


def expand_model_blocks_to_token_order(model_block_orders, block_len):
    model_block_orders = model_block_orders.long()
    offsets = torch.arange(int(block_len), device=model_block_orders.device).view(1, 1, -1)
    token_order = model_block_orders.unsqueeze(-1) * int(block_len) + offsets
    return token_order.reshape(model_block_orders.size(0), -1)


def physical_blocks_to_model_token_order(physical_block_orders, clean_perm, block_len):
    model_blocks = physical_blocks_to_model_blocks(physical_block_orders, clean_perm)
    return expand_model_blocks_to_token_order(model_blocks, block_len)


def build_fixed_split_and_shuffle(total_chunks, seed, val_fraction, max_eval_seqs=None):
    rng = np.random.RandomState(int(seed))
    all_perm = rng.permutation(int(total_chunks))
    val_size = max(1, int(round(int(total_chunks) * float(val_fraction))))
    val_indices = np.array(sorted(int(x) for x in all_perm[:val_size]), dtype=np.int64)
    train_indices = np.array(sorted(int(x) for x in all_perm[val_size:]), dtype=np.int64)
    train_shuffle_order = rng.permutation(train_indices).astype(np.int64)
    if max_eval_seqs is None:
        eval_indices = val_indices.copy()
    else:
        eval_indices = val_indices[: min(int(max_eval_seqs), len(val_indices))].copy()
    return FixedDataProtocol(
        train_indices=train_indices,
        val_indices=val_indices,
        train_shuffle_order=train_shuffle_order,
        eval_indices=eval_indices,
    )


def batch_indices_for_step(train_shuffle_order, global_step, micro_step, batch_size, grad_accum):
    order = np.asarray(train_shuffle_order, dtype=np.int64)
    if order.size == 0:
        raise ValueError("train_shuffle_order is empty")
    start = (int(global_step) * int(grad_accum) + int(micro_step)) * int(batch_size)
    positions = (np.arange(int(batch_size), dtype=np.int64) + start) % order.size
    return order[positions]


def train_cursor_for_next_step(train_shuffle_order, next_global_step, batch_size, grad_accum):
    order = np.asarray(train_shuffle_order, dtype=np.int64)
    if order.size == 0:
        raise ValueError("train_shuffle_order is empty")
    return int((int(next_global_step) * int(batch_size) * int(grad_accum)) % order.size)


def sha256_int_array(values):
    arr = np.asarray(values, dtype=np.int64)
    import hashlib

    return hashlib.sha256(arr.tobytes()).hexdigest()
