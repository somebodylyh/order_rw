"""
Evaluate AO-GPT language checkpoints with PPL reported in the original
0..T-1 token frame.

The model still scores tokens under a chosen reveal/order policy. This script
then unshuffles token losses back from reveal order to current token positions,
maps current positions back to true original positions for permuted-data
checkpoints, and reports CE/PPL over that original-position frame.

Example:
CUDA_VISIBLE_DEVICES=0 /data/users/chenhe/conda_envs/X1/bin/python \
  scripts/eval/eval_lm_original_order_ppl.py \
  --ckpt_path out/curriculum/nonpermute/seq256/block64/out-wikitext103-seq256-random-b64-attn-spectral-crossaxis-4-stage/ckpt.pt \
  --eval_mode SegmentGuided \
  --order_json Report/curriculum/nonpermute/seq256/block64/attention_spectral_crossaxis-4-stage/stage_04/results.json \
  --out_dir Report/eval/wikitext103_attention_spectral_crossaxis_stage04_original_order_ppl/segmentguided \
  --split val \
  --batch_size 64 \
  --num_batches 200 \
  --device cuda \
  --dtype bfloat16
"""

import argparse
import json
import math
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    invert_permutation,
    token_losses_to_block_losses,
)


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def load_model(checkpoint: Dict, device: str):
    model_args = checkpoint["model_args"]
    model = AOGPT(AOGPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def resolve_data_dir(args, checkpoint: Dict) -> Path:
    if args.data_dir is not None:
        return Path(args.data_dir)
    dataset = args.dataset or checkpoint.get("config", {}).get("dataset")
    if not dataset:
        raise ValueError("Could not determine dataset. Pass --dataset or --data_dir.")
    return REPO_ROOT / "data" / str(dataset)


def infer_data_record_mode(data_dir: Path, checkpoint: Dict) -> str:
    checkpoint_mode = checkpoint.get("config", {}).get("data_record_mode")
    if checkpoint_mode is not None:
        return str(checkpoint_mode)
    meta_path = data_dir / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        return str(meta.get("data_record_mode", "stream"))
    return "stream"


def load_tokens(data_dir: Path, split: str) -> np.memmap:
    path = data_dir / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Could not find split file: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def resolve_permutation_state(checkpoint: Dict, model, block_size: int):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    perm_state = checkpoint.get("data_permutation") or {}
    block_len = int(model.block_order_block_len)
    num_blocks = int(block_size) // block_len
    if perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    layout = str(getattr(model, "block_order_layout", "contiguous"))
    token_perm = block_permutation_to_token_permutation(
        block_perm,
        block_len=block_len,
        block_order_layout=layout,
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )
    inverse_token_perm = block_permutation_to_token_permutation(
        inverse_block_perm,
        block_len=block_len,
        block_order_layout=layout,
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
        "inverse_token_perm": inverse_token_perm,
    }


def extract_segment(row) -> List[int]:
    if isinstance(row, dict):
        if "blocks" in row:
            return [int(v) for v in row["blocks"]]
        if "segment" in row:
            return [int(v) for v in row["segment"]]
        if "order" in row:
            return [int(v) for v in row["order"]]
    if isinstance(row, list):
        return [int(v) for v in row]
    return []


def extract_order_from_json(path: Path, num_blocks: int) -> Dict:
    payload = json.loads(path.read_text())
    order = None
    source = "unknown"
    if isinstance(payload.get("best_candidate"), dict) and payload["best_candidate"].get("order") is not None:
        order = [int(v) for v in payload["best_candidate"]["order"]]
        source = "best_candidate.order"
    elif payload.get("final_units"):
        units = [extract_segment(row) for row in payload["final_units"]]
        order = [int(v) for unit in units for v in unit]
        source = "concat(final_units[].blocks)"
    elif payload.get("levels"):
        levels = payload.get("levels") or []
        last = levels[-1] if levels else {}
        raw_units = last.get("next_units") or last.get("aggregated_segments") or []
        units = [extract_segment(row) for row in raw_units]
        order = [int(v) for unit in units for v in unit]
        source = "concat(levels[-1] units)"
    else:
        for key in ("order", "block_order", "blocks", "segment"):
            if payload.get(key) is not None:
                order = [int(v) for v in payload[key]]
                source = key
                break
    if order is None:
        raise ValueError(f"Could not find an order in {path}")
    if len(order) != int(num_blocks) or sorted(order) != list(range(int(num_blocks))):
        raise ValueError(
            f"Order from {path} is not a valid permutation of 0..{int(num_blocks)-1}: "
            f"len={len(order)}, unique={len(set(order))}"
        )
    return {
        "order": order,
        "source": source,
        "name": payload.get("name") or payload.get("best_candidate", {}).get("name") or path.stem,
    }


def _extract_checkpoint_attn_mlp_order(checkpoint: Dict, num_blocks: int) -> List[int]:
    state = checkpoint.get("attn_mlp_policy_state") or {}
    raw_order = state.get("cached_order")
    source = "attn_mlp_policy_state.cached_order"
    if raw_order is None:
        raw_order = state.get("cached_order_current")
        source = "attn_mlp_policy_state.cached_order_current"
    if raw_order is None:
        raise ValueError("Checkpoint does not contain attn_mlp_policy_state.cached_order.")
    if torch.is_tensor(raw_order):
        order = [int(v) for v in raw_order.detach().cpu().tolist()]
    else:
        order = [int(v) for v in raw_order]
    if len(order) != int(num_blocks) or sorted(order) != list(range(int(num_blocks))):
        raise ValueError(
            f"Checkpoint Attn-MLP order from {source} is not a valid current-frame permutation: "
            f"len={len(order)}, unique={len(set(order))}"
        )
    return order


def _extract_checkpoint_online_spectral_order(checkpoint: Dict, num_blocks: int) -> List[int]:
    state = checkpoint.get("online_spectral_policy_state") or {}
    raw_order = state.get("cached_order")
    source = "online_spectral_policy_state.cached_order"
    if raw_order is None:
        raw_order = state.get("cached_order_current")
        source = "online_spectral_policy_state.cached_order_current"
    if raw_order is None:
        raise ValueError("Checkpoint does not contain online_spectral_policy_state.cached_order.")
    if torch.is_tensor(raw_order):
        order = [int(v) for v in raw_order.detach().cpu().tolist()]
    else:
        order = [int(v) for v in raw_order]
    if len(order) != int(num_blocks) or sorted(order) != list(range(int(num_blocks))):
        raise ValueError(
            f"Checkpoint online spectral order from {source} is not a valid current-frame permutation: "
            f"len={len(order)}, unique={len(set(order))}"
        )
    return order


def _extract_checkpoint_online_spectral_priority(checkpoint: Dict, num_blocks: int) -> torch.Tensor:
    state = checkpoint.get("online_spectral_policy_state") or {}
    raw_priority = state.get("priority_ema")
    if raw_priority is None:
        raise ValueError("Checkpoint does not contain online_spectral_policy_state.priority_ema.")
    priority = torch.as_tensor(raw_priority, dtype=torch.float32, device="cpu").view(-1)
    if priority.numel() != int(num_blocks):
        raise ValueError(
            "Checkpoint online spectral priority_ema has wrong length: "
            f"got {priority.numel()}, expected {int(num_blocks)}"
        )
    if not torch.isfinite(priority).all():
        raise ValueError("Checkpoint online spectral priority_ema contains non-finite values.")
    return priority


def _checkpoint_distribution_default(checkpoint: Dict, key: str, default):
    state = checkpoint.get("online_spectral_policy_state") or {}
    return state.get(key, checkpoint.get("config", {}).get(f"online_spectral_policy_{key}", default))


def _sample_order_from_priority(priority: torch.Tensor, temperature: float, rng: torch.Generator) -> torch.Tensor:
    eps = 1e-6
    logits = priority.float()
    u = torch.rand(logits.shape, generator=rng, device=logits.device, dtype=logits.dtype).clamp(eps, 1.0 - eps)
    gumbel = -torch.log(-torch.log(u))
    return torch.argsort(logits / max(float(temperature), 1e-6) + gumbel, descending=True)


def build_block_orders(args, model, batch_size: int, permutation_state, rng: torch.Generator, checkpoint=None):
    device = next(model.parameters()).device
    mode = str(args.eval_mode)
    if mode == "AR":
        order = torch.arange(model.num_blocks, device=device, dtype=torch.long)
        return order.unsqueeze(0).expand(batch_size, -1), {"source": "current_frame_l2r"}
    if mode == "OriginalL2R":
        if permutation_state is None:
            order = torch.arange(model.num_blocks, device=device, dtype=torch.long)
        else:
            order = permutation_state["inverse_block_perm"].to(device=device, dtype=torch.long)
        return order.unsqueeze(0).expand(batch_size, -1), {"source": "original_l2r_mapped_to_current_frame"}
    if mode == "Random":
        orders = [torch.randperm(model.num_blocks, generator=rng, device=device) for _ in range(batch_size)]
        return torch.stack(orders, dim=0), {"source": "sampled_random_orders"}
    if mode in {"BlockOrder", "SegmentGuided"}:
        if args.order_json is None:
            raise ValueError(f"--eval_mode {mode} requires --order_json")
        info = extract_order_from_json(Path(args.order_json), model.num_blocks)
        order = torch.tensor(info["order"], device=device, dtype=torch.long)
        return order.unsqueeze(0).expand(batch_size, -1), info
    if mode == "CheckpointAttnMLPOrder":
        if checkpoint is None:
            raise ValueError("CheckpointAttnMLPOrder requires checkpoint metadata.")
        order_values = _extract_checkpoint_attn_mlp_order(checkpoint, model.num_blocks)
        order = torch.tensor(order_values, device=device, dtype=torch.long)
        return order.unsqueeze(0).expand(batch_size, -1), {
            "source": "checkpoint_attn_mlp_cached_order_current",
            "name": "CheckpointAttnMLPOrder",
        }
    if mode == "CheckpointOnlineSpectralOrder":
        if checkpoint is None:
            raise ValueError("CheckpointOnlineSpectralOrder requires checkpoint metadata.")
        order_values = _extract_checkpoint_online_spectral_order(checkpoint, model.num_blocks)
        order = torch.tensor(order_values, device=device, dtype=torch.long)
        return order.unsqueeze(0).expand(batch_size, -1), {
            "source": "checkpoint_online_spectral_cached_order_current",
            "name": "CheckpointOnlineSpectralOrder",
        }
    if mode == "CheckpointOnlineSpectralDistributionMAP":
        if checkpoint is None:
            raise ValueError("CheckpointOnlineSpectralDistributionMAP requires checkpoint metadata.")
        priority = _extract_checkpoint_online_spectral_priority(checkpoint, model.num_blocks).to(device=device)
        order = torch.argsort(priority, descending=True)
        return order.unsqueeze(0).expand(batch_size, -1), {
            "source": "checkpoint_online_spectral_priority_ema_map_current",
            "name": "CheckpointOnlineSpectralDistributionMAP",
        }
    if mode in {"CheckpointOnlineSpectralDistributionSampled", "CheckpointOnlineSpectralDistributionMixSampled"}:
        if checkpoint is None:
            raise ValueError(f"{mode} requires checkpoint metadata.")
        if int(getattr(args, "distribution_num_order_samples", 1)) != 1:
            raise NotImplementedError("distribution_num_order_samples > 1 is not implemented in this first eval path.")
        priority = _extract_checkpoint_online_spectral_priority(checkpoint, model.num_blocks).to(device=device)
        default_temp = float(_checkpoint_distribution_default(checkpoint, "sample_temperature", 0.7))
        sample_temperature = (
            default_temp
            if getattr(args, "distribution_sample_temperature", None) is None
            else float(args.distribution_sample_temperature)
        )
        if mode == "CheckpointOnlineSpectralDistributionSampled":
            random_mix_prob = 0.0
        else:
            default_mix = float(_checkpoint_distribution_default(checkpoint, "random_mix_prob", 0.2))
            random_mix_prob = (
                default_mix
                if getattr(args, "distribution_random_mix_prob", None) is None
                else float(args.distribution_random_mix_prob)
            )
        random_mix_prob = max(0.0, min(1.0, float(random_mix_prob)))
        orders = []
        random_mix_count = 0
        for _ in range(batch_size):
            if random_mix_prob > 0.0:
                draw = torch.rand((), generator=rng, device=device).item()
            else:
                draw = 1.0
            if draw < random_mix_prob:
                orders.append(torch.randperm(model.num_blocks, generator=rng, device=device))
                random_mix_count += 1
            else:
                orders.append(_sample_order_from_priority(priority, sample_temperature, rng).to(dtype=torch.long))
        return torch.stack(orders, dim=0), {
            "source": "checkpoint_online_spectral_priority_ema_gumbel_current",
            "name": mode,
            "sample_temperature": float(sample_temperature),
            "random_mix_prob": float(random_mix_prob),
            "random_mix_count_first_batch": int(random_mix_count),
            "distribution_num_order_samples": int(getattr(args, "distribution_num_order_samples", 1)),
        }
    raise ValueError(f"Unsupported eval_mode={mode!r}")


def expand_orders(model, block_orders: torch.Tensor):
    return expand_block_orders_to_token_orders(
        block_orders,
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )


def iter_starts(tokens_len: int, block_size: int, data_record_mode: str, sample_mode: str, num_samples: int, seed: int):
    rng = np.random.default_rng(int(seed))
    if data_record_mode == "fixed":
        num_records = tokens_len // block_size
        if num_records <= 0:
            raise ValueError("Dataset split is shorter than one fixed record.")
        if sample_mode == "sequential":
            starts = [idx * block_size for idx in range(num_records)]
            return starts[:num_samples] if num_samples > 0 else starts
        record_ids = rng.integers(0, num_records, size=num_samples)
        return [int(v) * block_size for v in record_ids]
    if data_record_mode == "stream":
        max_start = tokens_len - block_size
        if max_start <= 0:
            raise ValueError("Dataset split is shorter than block_size.")
        if sample_mode == "sequential":
            starts = list(range(0, max_start + 1, block_size))
            return starts[:num_samples] if num_samples > 0 else starts
        return [int(v) for v in rng.integers(0, max_start, size=num_samples)]
    raise ValueError(f"Unsupported data_record_mode={data_record_mode!r}")


def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def apply_sequential_tail_coverage(
    starts: List[int],
    tokens_len: int,
    block_size: int,
    data_record_mode: str,
    sample_mode: str,
    num_batches: int,
    enabled: bool,
):
    if not enabled:
        return starts, None, {
            "enabled": False,
            "total_tokens": int(tokens_len),
            "covered_tokens": int(len(starts) * block_size),
            "tail_tokens": int(max(0, tokens_len - len(starts) * block_size)),
        }
    if data_record_mode != "stream" or sample_mode != "sequential" or int(num_batches) != 0:
        raise ValueError("--sequential_cover_tail requires stream data, --sample_mode sequential, and --num_batches 0.")
    if int(tokens_len) < int(block_size):
        raise ValueError("Cannot cover tail when the split is shorter than block_size.")

    masks = [torch.ones(int(block_size), dtype=torch.bool) for _ in starts]
    covered_tokens = len(starts) * int(block_size)
    tail_tokens = int(tokens_len) - int(covered_tokens)
    if tail_tokens > 0:
        tail_start = int(tokens_len) - int(block_size)
        mask = torch.zeros(int(block_size), dtype=torch.bool)
        mask[-tail_tokens:] = True
        starts = list(starts) + [tail_start]
        masks.append(mask)
        covered_tokens = int(tokens_len)
    return starts, masks, {
        "enabled": True,
        "total_tokens": int(tokens_len),
        "covered_tokens": int(covered_tokens),
        "tail_tokens": int(max(0, tail_tokens)),
        "tail_window_added": bool(tail_tokens > 0),
    }


def unshuffle_to_current(values_reveal: torch.Tensor, token_orders: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = values_reveal.shape[:2]
    batch_idx = torch.arange(batch_size, device=values_reveal.device).unsqueeze(1).expand(batch_size, seq_len)
    out = torch.empty_like(values_reveal)
    out[batch_idx, token_orders.long()] = values_reveal
    return out


def maybe_to_original_frame(values_current: torch.Tensor, permutation_state) -> torch.Tensor:
    if permutation_state is None:
        return values_current
    inverse_token_perm = permutation_state["inverse_token_perm"].to(values_current.device, dtype=torch.long)
    return values_current.index_select(1, inverse_token_perm)


def get_autocast(device: str, dtype: str):
    if "cuda" not in str(device) or str(dtype) == "float32":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype])


@torch.no_grad()
def evaluate(args):
    ckpt_path = Path(args.ckpt_path)
    checkpoint = load_checkpoint(ckpt_path)
    model = load_model(checkpoint, args.device)
    block_size = int(args.block_size or checkpoint["model_args"]["block_size"])
    data_dir = resolve_data_dir(args, checkpoint)
    data_record_mode = infer_data_record_mode(data_dir, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    permutation_state = resolve_permutation_state(checkpoint, model, block_size)
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    token_perm_device = None if token_perm is None else token_perm.to(args.device, dtype=torch.long)

    if int(args.num_batches) < 0:
        raise ValueError("--num_batches must be >= 0; use 0 with --sample_mode sequential to evaluate all chunks.")
    if args.sample_mode == "random" and int(args.num_batches) == 0:
        raise ValueError("--sample_mode random requires --num_batches > 0")
    num_samples = int(args.num_batches) * int(args.batch_size)
    starts = iter_starts(
        len(tokens),
        block_size,
        data_record_mode,
        args.sample_mode,
        num_samples,
        int(args.seed),
    )
    starts, score_masks, coverage_info = apply_sequential_tail_coverage(
        starts,
        len(tokens),
        block_size,
        data_record_mode,
        args.sample_mode,
        int(args.num_batches),
        bool(getattr(args, "sequential_cover_tail", False)),
    )
    if not starts:
        raise ValueError("No evaluation samples were selected.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = get_autocast(args.device, args.dtype)
    rng = torch.Generator(device="cuda" if "cuda" in str(args.device) else "cpu")
    rng.manual_seed(int(args.seed))

    total_nll = 0.0
    total_tokens = 0
    total_reveal_nll = 0.0
    total_samples = 0
    token_loss_sum = torch.zeros(block_size, dtype=torch.float64)
    token_loss_count = torch.zeros(block_size, dtype=torch.float64)
    first_order_info = None

    for batch_start in range(0, len(starts), int(args.batch_size)):
        start_batch = starts[batch_start : batch_start + int(args.batch_size)]
        mask_batch = None
        if score_masks is not None:
            mask_batch = score_masks[batch_start : batch_start + int(args.batch_size)]
        batch_np = np.stack([np.asarray(tokens[s : s + block_size], dtype=np.int64) for s in start_batch], axis=0)
        batch_original = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
        if token_perm_device is not None:
            batch_current = batch_original.index_select(1, token_perm_device)
        else:
            batch_current = batch_original

        block_orders, order_info = build_block_orders(
            args,
            model,
            batch_current.size(0),
            permutation_state,
            rng,
            checkpoint=checkpoint,
        )
        if first_order_info is None:
            first_order_info = dict(order_info)
            first_order_info["block_order_first64"] = [int(v) for v in block_orders[0].detach().cpu().tolist()]
        token_orders = expand_orders(model, block_orders)
        with ctx:
            _, loss, token_losses_reveal = model(
                batch_current,
                mode=None,
                orders=token_orders,
                return_token_loss=True,
                return_logits=False,
            )
        token_losses_current = unshuffle_to_current(token_losses_reveal, token_orders)
        token_losses_original = maybe_to_original_frame(token_losses_current, permutation_state)
        mask_original = None
        mask_reveal = None
        if mask_batch is not None:
            mask_original = torch.stack(mask_batch, dim=0).to(args.device)
            if token_perm_device is not None:
                mask_current = mask_original.index_select(1, token_perm_device)
            else:
                mask_current = mask_original
            mask_reveal = torch.gather(mask_current, 1, token_orders.long())
        if bool(args.ignore_first_token):
            token_losses_original = token_losses_original[:, 1:]
            token_losses_reveal_for_mean = token_losses_reveal[:, 1:]
            if mask_original is not None:
                mask_original = mask_original[:, 1:]
                mask_reveal = mask_reveal[:, 1:]
            position_offset = 1
        else:
            token_losses_reveal_for_mean = token_losses_reveal
            position_offset = 0

        if mask_original is None:
            token_losses_original_for_mean = token_losses_original
            token_losses_reveal_selected = token_losses_reveal_for_mean
            per_pos_sum = token_losses_original.double().sum(dim=0).detach().cpu()
            per_pos_count = torch.full_like(per_pos_sum, fill_value=float(batch_current.size(0)), dtype=torch.float64)
        else:
            token_losses_original_for_mean = token_losses_original[mask_original]
            token_losses_reveal_selected = token_losses_reveal_for_mean[mask_reveal]
            per_pos_sum = (token_losses_original.double() * mask_original.double()).sum(dim=0).detach().cpu()
            per_pos_count = mask_original.double().sum(dim=0).detach().cpu()

        total_nll += float(token_losses_original_for_mean.double().sum().item())
        total_reveal_nll += float(token_losses_reveal_selected.double().sum().item())
        total_tokens += int(token_losses_original_for_mean.numel())
        total_samples += int(batch_current.size(0))

        token_loss_sum[position_offset : position_offset + per_pos_sum.numel()] += per_pos_sum
        token_loss_count[position_offset : position_offset + per_pos_count.numel()] += per_pos_count

    mean_nll = total_nll / max(1, total_tokens)
    reveal_mean_nll = total_reveal_nll / max(1, total_tokens)
    token_loss_mean = token_loss_sum / token_loss_count.clamp_min(1.0)
    if bool(args.ignore_first_token):
        token_loss_mean[0] = float("nan")
    block_loss_mean = token_losses_to_block_losses(
        token_loss_mean.view(1, -1).float(),
        block_len=int(model.block_order_block_len),
    ).view(-1)

    summary = {
        "ckpt_path": str(ckpt_path),
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "dataset": str(args.dataset or checkpoint.get("config", {}).get("dataset")),
        "data_dir": str(data_dir),
        "split": str(args.split),
        "data_record_mode": str(data_record_mode),
        "sample_mode": str(args.sample_mode),
        "num_batches_arg": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "num_samples": int(total_samples),
        "num_scored_tokens": int(total_tokens),
        "block_size": int(block_size),
        "block_order_block_len": int(model.block_order_block_len),
        "num_blocks": int(model.num_blocks),
        "eval_mode": str(args.eval_mode),
        "order_json": None if args.order_json is None else str(args.order_json),
        "distribution_sample_temperature": getattr(args, "distribution_sample_temperature", None),
        "distribution_random_mix_prob": getattr(args, "distribution_random_mix_prob", None),
        "distribution_num_order_samples": int(getattr(args, "distribution_num_order_samples", 1)),
        "order_info": first_order_info,
        "device": str(args.device),
        "dtype": str(args.dtype),
        "ignore_first_token": bool(args.ignore_first_token),
        "coverage_info": coverage_info,
        "mean_nll_original_frame": float(mean_nll),
        "ppl_original_frame": float(math.exp(mean_nll)),
        "mean_nll_reveal_frame": float(reveal_mean_nll),
        "ppl_reveal_frame": float(math.exp(reveal_mean_nll)),
        "frame_mapping": {
            "input_batch_frame": "current_permuted" if permutation_state is not None else "original",
            "loss_report_frame": "true_original_0_to_T_minus_1",
            "unshuffled_from_reveal_order": True,
            "data_permutation_applied": bool(permutation_state is not None),
        },
        "position_loss_original_frame": [None if not math.isfinite(float(v)) else float(v) for v in token_loss_mean.tolist()],
        "block_loss_original_frame": [float(v) for v in block_loss_mean.tolist()],
    }
    if permutation_state is not None:
        summary["data_permutation"] = {
            "block_perm_first16": [int(v) for v in permutation_state["block_perm"][:16].tolist()],
            "inverse_block_perm_first16": [int(v) for v in permutation_state["inverse_block_perm"][:16].tolist()],
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "mean_nll_original_frame": summary["mean_nll_original_frame"],
        "ppl_original_frame": summary["ppl_original_frame"],
        "mean_nll_reveal_frame": summary["mean_nll_reveal_frame"],
        "ppl_reveal_frame": summary["ppl_reveal_frame"],
        "num_samples": summary["num_samples"],
        "num_scored_tokens": summary["num_scored_tokens"],
        "out_dir": str(out_dir),
    }, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate language PPL in original 0..T-1 token frame.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--sample_mode", choices=("random", "sequential"), default="random")
    parser.add_argument(
        "--sequential_cover_tail",
        action="store_true",
        help="With --sample_mode sequential --num_batches 0 on stream data, add one tail window and score only previously uncovered tokens.",
    )
    parser.add_argument(
        "--eval_mode",
        choices=(
            "AR",
            "OriginalL2R",
            "Random",
            "BlockOrder",
            "SegmentGuided",
            "CheckpointAttnMLPOrder",
            "CheckpointOnlineSpectralOrder",
            "CheckpointOnlineSpectralDistributionMAP",
            "CheckpointOnlineSpectralDistributionSampled",
            "CheckpointOnlineSpectralDistributionMixSampled",
        ),
        default="OriginalL2R",
    )
    parser.add_argument("--order_json", type=Path, default=None)
    parser.add_argument("--distribution_sample_temperature", type=float, default=None)
    parser.add_argument("--distribution_random_mix_prob", type=float, default=None)
    parser.add_argument("--distribution_num_order_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--ignore_first_token", action="store_true")
    return parser.parse_args()


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
