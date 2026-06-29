#!/usr/bin/env python3
"""Shared helpers for eval-loss mismatch debugging."""

from __future__ import annotations

import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


WORKSPACE = Path(__file__).resolve().parents[1]
NANOGPT_ROOT = Path("/home/admin/ych/nanogpt-learned-order")
if str(NANOGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOGPT_ROOT))
if str(WORKSPACE / "block_lo_arm_order_network") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "block_lo_arm_order_network"))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from order_utils import expand_block_orders_to_token_orders  # noqa: E402


DEFAULT_CKPT = (
    "/home/admin/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
DEFAULT_TOKENS = "block_lo_arm_order_network/probe_results/A32_from_N64_10k.tokens.npy"
DEFAULT_A64 = "block_lo_arm_order_network/probe_results/A64_from_N64_50k_10k.npy"
DEFAULT_ON_CKPT = "block_lo_arm_order_network/probe_results/cotrain_aogpt_on_n64/phase1_on_bc_best.pt"
DEFAULT_OUT_DIR = "block_lo_arm_order_network/probe_results/eval_mismatch_debug"

SEQ_LEN = 256
NUM_BLOCKS = 64
TOKENS_PER_BLOCK = 4


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16 if dtype_name == "float16" else torch.float32
    if dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def load_checkpoint(path: str, map_location="cpu") -> Dict:
    return torch.load(os.path.expanduser(path), map_location=map_location, weights_only=False)


def load_aogpt_from_ckpt(ckpt_path: str, device: torch.device, freeze: bool = True):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model_args = dict(ckpt["model_args"])
    model = AOGPT(AOGPTConfig(**model_args))
    model.load_state_dict(clean_state_dict(ckpt["model"]))
    model.crop_block_size(int(model_args["block_size"]))
    model.to(device).eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model, ckpt


def get_permutation_state(ckpt: Dict, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    perm_state = ckpt["data_permutation"]
    block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long, device=device)
    inverse_block_perm = torch.tensor(perm_state["inverse_block_perm"], dtype=torch.long, device=device)
    token_perm = expand_block_orders_to_token_orders(
        block_perm.view(1, -1), block_len=TOKENS_PER_BLOCK
    ).squeeze(0)
    inverse_token_perm = expand_block_orders_to_token_orders(
        inverse_block_perm.view(1, -1), block_len=TOKENS_PER_BLOCK
    ).squeeze(0)
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
        "inverse_token_perm": inverse_token_perm,
    }


def permute_input_to_model_frame(idx_phys: torch.Tensor, perm_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    token_perm = perm_state["token_perm"].to(idx_phys.device)
    return idx_phys[:, token_perm]


def load_memmap_batch(
    data_path: str,
    batch_size: int,
    block_size: int,
    rng: torch.Generator,
    device: torch.device,
    perm_state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=rng)
    batch = torch.stack([
        torch.from_numpy(data[int(start): int(start) + block_size].astype(np.int64))
        for start in starts
    ])
    batch = batch.to(device)
    if perm_state is not None:
        batch = permute_input_to_model_frame(batch, perm_state)
    return batch


def load_token_chunks(tokens_path: str, start: int, count: int, device: torch.device) -> torch.Tensor:
    arr = np.load(tokens_path, mmap_mode="r")
    end = min(start + count, len(arr))
    return torch.from_numpy(np.array(arr[start:end], dtype=np.int64)).to(device)


def build_model_token_order_from_phys_blocks(
    phys_block_orders: torch.Tensor,
    perm_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Convert physical block order to model-coordinate token order."""
    inverse_block_perm = perm_state["inverse_block_perm"].to(phys_block_orders.device)
    model_blocks = inverse_block_perm[phys_block_orders.long()]
    offsets = torch.arange(TOKENS_PER_BLOCK, device=phys_block_orders.device)
    return (model_blocks.unsqueeze(-1) * TOKENS_PER_BLOCK + offsets).reshape(phys_block_orders.size(0), -1)


def build_block_orders(mode: str, batch_size: int, device: torch.device, seed: int = 123) -> torch.Tensor:
    if mode == "ar":
        return torch.arange(NUM_BLOCKS, device=device).view(1, -1).expand(batch_size, -1)
    if mode == "l2r_phys":
        return torch.arange(NUM_BLOCKS, device=device).view(1, -1).expand(batch_size, -1)
    if mode == "random_fixed":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        base = torch.randperm(NUM_BLOCKS, generator=generator, device=device)
        return base.view(1, -1).expand(batch_size, -1)
    raise ValueError(f"Unsupported block-order mode: {mode}")


@torch.no_grad()
def model_loss(
    model,
    idx: torch.Tensor,
    *,
    mode: Optional[str] = None,
    token_orders: Optional[torch.Tensor] = None,
    ctx=nullcontext(),
    return_token_loss: bool = False,
) -> Tuple[float, Optional[torch.Tensor]]:
    with ctx:
        if mode is not None:
            outputs = model(idx, mode=mode, return_token_loss=return_token_loss)
        else:
            outputs = model.forward_fn(idx, token_orders, return_token_loss=return_token_loss)
    loss = float(outputs[1].detach().float().item())
    token_losses = outputs[2].detach() if return_token_loss and len(outputs) > 2 else None
    return loss, token_losses


@torch.no_grad()
def reference_loss_from_logits(model, idx: torch.Tensor, token_orders: torch.Tensor, ctx=nullcontext()) -> float:
    with ctx:
        logits, _ = model.forward_fn(idx, token_orders)
    targets = model.shuffle(idx, token_orders)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-1,
    )
    return float(loss.detach().float().item())


@torch.no_grad()
def eval_batches(
    model,
    batches: Iterable[torch.Tensor],
    *,
    mode: Optional[str] = None,
    token_order_builder=None,
    ctx=nullcontext(),
    return_counts: bool = False,
) -> Dict:
    losses = []
    predicted_tokens = 0
    for idx in batches:
        if token_order_builder is None:
            loss, _ = model_loss(model, idx, mode=mode, ctx=ctx)
        else:
            token_orders = token_order_builder(idx)
            loss, _ = model_loss(model, idx, mode=None, token_orders=token_orders, ctx=ctx)
        losses.append(loss)
        predicted_tokens += int(idx.numel())
    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "num_batches": len(losses),
    }
    if return_counts:
        out["predicted_tokens"] = predicted_tokens
    return out


def load_on_model(path: str, device: torch.device, d_edge: int = 128, d_model: int = 128):
    from order_network import CrossAttentionOrderNetwork

    ckpt = load_checkpoint(path, map_location=device)
    model = CrossAttentionOrderNetwork(num_blocks=NUM_BLOCKS, d_edge=d_edge, d_model=d_model).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def sample_on_order(on_model, A: torch.Tensor, temperature: float = 0.0, top_k: int = 0) -> torch.Tensor:
    from cotrain_aogpt_on import sample_on_order_n64

    return sample_on_order_n64(on_model, A, temperature=temperature, top_k=top_k)


def order_checksum(order: torch.Tensor) -> str:
    row = order.detach().cpu().reshape(-1).numpy().astype(np.int64)
    # Small deterministic checksum without importing hashlib in every caller.
    value = int(np.sum((np.arange(row.size, dtype=np.int64) + 1) * (row + 17)) % 1_000_000_007)
    return str(value)


def toy_order_sanity(n_blocks: int = 4, block_len: int = 4) -> Dict:
    order = np.array([2, 0, 3, 1], dtype=np.int64)[:n_blocks]
    token_ranges = []
    visible = []
    for step, block in enumerate(order):
        token_ranges.append([int(block * block_len), int((block + 1) * block_len - 1)])
        visible.append(order[:step].astype(int).tolist())
    return {
        "n_blocks": n_blocks,
        "block_len": block_len,
        "order": order.astype(int).tolist(),
        "visible_before_step": visible,
        "predicted_token_ranges": token_ranges,
        "num_predicted_tokens": int(n_blocks * block_len),
        "unique_predicted_tokens": int(n_blocks * block_len),
    }


def write_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
