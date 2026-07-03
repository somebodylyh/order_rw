from __future__ import annotations

import csv
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from order_utils import (  # noqa: E402
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    invert_permutation,
)


FORBIDDEN_SIGNALS = [
    "original_l2r",
    "original_tau",
    "OriginalL2R_ppl",
    "human_scan",
    "validation_ppl_for_policy_selection",
]


def parse_int_list(text: str | None, default: list[int] | None = None) -> list[int]:
    if text is None or str(text).strip() == "":
        return list(default or [])
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=json_default) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_torch_dtype(name: str):
    name = str(name or "float32").lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype={name!r}")


def autocast_context(device: torch.device | str, dtype_name: str):
    device = torch.device(device)
    dtype = resolve_torch_dtype(dtype_name)
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def strip_orig_mod_prefix(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        key = str(key)
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        out[key] = value
    return out


def load_frozen_aogpt(ckpt_path: Path | str, device: torch.device | str):
    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_args = dict(checkpoint["model_args"])
    model_args.setdefault("force_manual_attention", False)
    config = AOGPTConfig(**model_args)
    model = AOGPT(config)
    model.load_state_dict(strip_orig_mod_prefix(checkpoint["model"]), strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint


def infer_data_dir(dataset: str | None, data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    if dataset is None:
        raise ValueError("Provide --dataset or --data_dir.")
    return REPO_ROOT / "data" / str(dataset)


def infer_data_record_mode(checkpoint: dict) -> str:
    config = checkpoint.get("config") or {}
    return str(config.get("data_record_mode", "stream"))


def load_tokens(data_dir: Path, split: str):
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_permutation_state(checkpoint: dict, num_blocks: int, block_len: int):
    config = checkpoint.get("config") or {}
    if not bool(config.get("permute_data", False)):
        return None
    model_args = checkpoint.get("model_args") or {}
    block_order_layout = str(model_args.get("block_order_layout", config.get("block_order_layout", "contiguous")))
    image_size = int(model_args.get("image_size", config.get("image_size", 0)))
    image_block_size = int(model_args.get("image_block_size", config.get("image_block_size", 0)))
    image_block_height = int(model_args.get("image_block_height", config.get("image_block_height", 0)))
    image_block_width = int(model_args.get("image_block_width", config.get("image_block_width", 0)))
    perm_state = checkpoint.get("data_permutation") or {}
    if perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
        inverse_block_perm = torch.tensor(
            perm_state.get("inverse_block_perm", invert_permutation(block_perm).tolist()),
            dtype=torch.long,
        )
        block_order_layout = str(perm_state.get("block_order_layout", block_order_layout))
        image_size = int(perm_state.get("image_size", image_size))
        image_block_size = int(perm_state.get("image_block_size", image_block_size))
        image_block_height = int(perm_state.get("image_block_height", image_block_height))
        image_block_width = int(perm_state.get("image_block_width", image_block_width))
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
        inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(
        block_perm,
        block_len=block_len,
        block_order_layout=block_order_layout,
        image_size=image_size,
        image_block_size=image_block_size,
        image_block_height=image_block_height,
        image_block_width=image_block_width,
    )
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
        "block_order_layout": block_order_layout,
        "image_size": image_size,
        "image_block_size": image_block_size,
        "image_block_height": image_block_height,
        "image_block_width": image_block_width,
    }


def sample_batch(
    tokens,
    *,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: torch.device | str,
    token_perm: torch.Tensor | None,
    data_record_mode: str,
):
    data_record_mode = str(data_record_mode or "stream")
    if data_record_mode == "fixed":
        num_records = len(tokens) // int(block_size)
        starts = rng.integers(0, num_records, size=int(batch_size)) * int(block_size)
        batch = np.stack([np.asarray(tokens[start : start + block_size], dtype=np.int64) for start in starts])
    elif data_record_mode == "stream":
        starts = rng.integers(0, len(tokens) - int(block_size), size=int(batch_size))
        batch = np.stack([np.asarray(tokens[start : start + block_size], dtype=np.int64) for start in starts])
    else:
        raise ValueError(f"Unsupported data_record_mode={data_record_mode!r}")
    x = torch.from_numpy(batch).to(device=device, dtype=torch.long)
    if token_perm is not None:
        x = x.index_select(1, token_perm.to(device=device))
    return x


def random_block_orders(batch_size: int, num_blocks: int, device: torch.device | str):
    return torch.stack([torch.randperm(int(num_blocks), device=device) for _ in range(int(batch_size))])


def expand_orders_for_model(model, block_orders: torch.Tensor):
    return expand_block_orders_to_token_orders(
        block_orders,
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )


def aggregate_layerhead_attention_to_current_blocks(
    layer_attn: torch.Tensor,
    block_orders: torch.Tensor,
    *,
    block_len: int,
    export_type: str,
):
    if str(export_type) == "with_none":
        shifted = layer_attn[:, :, :-1, :-1]
    elif str(export_type) == "without_none":
        shifted = layer_attn[:, :, 1:, 1:]
    else:
        raise ValueError(f"Unsupported export_type={export_type!r}")
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % int(block_len) != 0:
        raise ValueError(f"attention shape {tuple(shifted.shape)} is incompatible with block_len={block_len}")
    num_blocks = seq_a // int(block_len)
    block_batch = shifted.float().view(
        batch,
        heads,
        num_blocks,
        int(block_len),
        num_blocks,
        int(block_len),
    ).mean(dim=(3, 5))
    total = torch.zeros((heads, num_blocks, num_blocks), dtype=torch.float64, device=block_batch.device)
    for sample_idx in range(batch):
        inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=block_batch.device)
        total += block_batch[sample_idx][:, inverse, :][:, :, inverse].double()
    return total / float(max(1, batch))


@torch.no_grad()
def collect_attention_matrices(
    *,
    model,
    tokens,
    num_samples: int,
    probe_batch_size: int,
    layer: int,
    head: int,
    export_type: str,
    rng: np.random.Generator,
    device: torch.device | str,
    dtype: str,
    token_perm: torch.Tensor | None,
    data_record_mode: str,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 50,
):
    matrices = []
    block_orders_first = []
    layer = int(layer)
    head = int(head)
    for sample_idx in range(int(num_samples)):
        x = sample_batch(
            tokens,
            batch_size=int(probe_batch_size),
            block_size=int(model.config.block_size),
            rng=rng,
            device=device,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
        )
        block_orders = random_block_orders(int(probe_batch_size), int(model.num_blocks), device)
        token_orders = expand_orders_for_model(model, block_orders)
        with autocast_context(device, dtype):
            outputs = model(
                x,
                mode=None,
                orders=token_orders,
                return_attentions=True,
                return_logits=False,
            )
        attentions = outputs[-1]
        local_layer = layer if layer >= 0 else len(attentions) + layer
        if local_layer < 0 or local_layer >= len(attentions):
            raise ValueError(f"layer={layer} is outside available layers 0..{len(attentions) - 1}")
        layer_heads = aggregate_layerhead_attention_to_current_blocks(
            attentions[local_layer].detach(),
            block_orders,
            block_len=int(model.block_order_block_len),
            export_type=export_type,
        )
        if head < 0:
            matrix = layer_heads.mean(dim=0)
        else:
            if head >= int(layer_heads.size(0)):
                raise ValueError(f"head={head} is outside available heads 0..{int(layer_heads.size(0)) - 1}")
            matrix = layer_heads[head]
        matrix = matrix.detach().float().cpu()
        matrix.fill_diagonal_(0.0)
        matrices.append(matrix)
        if len(block_orders_first) < 4:
            block_orders_first.append(block_orders[0].detach().cpu().tolist())
        if progress is not None and (sample_idx + 1) % max(1, int(progress_interval)) == 0:
            progress(f"collected_attention_samples={sample_idx + 1}/{num_samples}")
    return torch.stack(matrices, dim=0), block_orders_first


def robust_z_offdiag(matrix: torch.Tensor, eps: float = 1e-8):
    values = matrix.detach().float()
    n = int(values.size(0))
    eye = torch.eye(n, dtype=torch.bool, device=values.device)
    finite = torch.isfinite(values) & (~eye)
    out = torch.zeros_like(values, dtype=torch.float32)
    if not bool(finite.any()):
        return out
    vals = values[finite]
    median = torch.median(vals)
    q25 = torch.quantile(vals, 0.25)
    q75 = torch.quantile(vals, 0.75)
    scale = (q75 - q25) / 1.349
    if (not torch.isfinite(scale)) or float(scale.item()) < eps:
        scale = vals.std(unbiased=False)
    if (not torch.isfinite(scale)) or float(scale.item()) < eps:
        scale = vals.new_tensor(1.0)
    out[finite] = (vals - median) / scale
    out.fill_diagonal_(0.0)
    return out


def build_graph_coefficients(
    matrix: torch.Tensor,
    *,
    sym_mode: str,
    dir_mode: str,
    threshold_percentile: float,
    eps: float = 1e-8,
):
    z = robust_z_offdiag(matrix, eps=eps)
    if str(sym_mode) == "max":
        sym = torch.maximum(z, z.t())
    elif str(sym_mode) == "mean":
        sym = 0.5 * (z + z.t())
    else:
        raise ValueError(f"Unsupported sym_mode={sym_mode!r}")
    sym.fill_diagonal_(0.0)
    n = int(sym.size(0))
    offdiag = ~torch.eye(n, dtype=torch.bool, device=sym.device)
    finite = torch.isfinite(sym) & offdiag
    if bool(finite.any()):
        threshold = torch.quantile(sym[finite], float(threshold_percentile) / 100.0)
    else:
        threshold = sym.new_tensor(0.0)
    w = torch.relu(sym - threshold)
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    w.fill_diagonal_(0.0)
    degree = w.sum(dim=1).clamp_min(eps)
    s_norm = w / torch.sqrt(degree[:, None] * degree[None, :])
    d = 0.5 * (z - z.t())
    if str(dir_mode) == "query_key":
        pass
    elif str(dir_mode) == "key_query":
        d = -d
    else:
        raise ValueError(f"Unsupported dir_mode={dir_mode!r}")
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    d.fill_diagonal_(0.0)
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=w.device), diagonal=1)
    edge_density = float((w[offdiag] > 0).float().mean().item()) if bool(offdiag.any()) else 0.0
    d_abs = d[upper].abs()
    stats = {
        "graph_threshold": float(threshold.detach().cpu().item()),
        "graph_degree_mean": float(w.sum(dim=1).mean().detach().cpu().item()),
        "graph_degree_min": float(w.sum(dim=1).min().detach().cpu().item()),
        "graph_degree_max": float(w.sum(dim=1).max().detach().cpu().item()),
        "graph_edge_density": edge_density,
        "dir_abs_mean": float(d_abs.mean().detach().cpu().item()) if d_abs.numel() else 0.0,
        "dir_abs_p90": float(torch.quantile(d_abs, 0.9).detach().cpu().item()) if d_abs.numel() else 0.0,
        "attention_mean": float(matrix.float()[offdiag].mean().detach().cpu().item()) if bool(offdiag.any()) else 0.0,
        "attention_std": float(matrix.float()[offdiag].std(unbiased=False).detach().cpu().item()) if bool(offdiag.any()) else 0.0,
    }
    return {"W": w.cpu(), "S": s_norm.cpu(), "D": d.cpu(), "stats": stats}


def build_coefficients_for_matrices(matrices: torch.Tensor, **kwargs):
    coeffs = []
    for matrix in matrices:
        coeffs.append(build_graph_coefficients(matrix, **kwargs))
    return coeffs


def _entropy_from_logits(logits: torch.Tensor):
    probs = torch.softmax(logits.float(), dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum()


def implicit_axis_loss(
    logits: torch.Tensor,
    coeffs: dict,
    *,
    lambda_dir: float,
    lambda_var: float,
    min_std: float,
    tau_d: float,
    tau_s: float,
    margin_d: float,
    min_conf: float,
    eps: float = 1e-8,
):
    device = logits.device
    s = logits.float()
    w = coeffs["W"].to(device=device, dtype=torch.float32)
    d = coeffs["D"].to(device=device, dtype=torch.float32)
    std = s.std(unbiased=False)
    x = (s - s.mean()) / std.clamp_min(eps)
    diff = x[:, None] - x[None, :]
    w_sum = w.sum().clamp_min(eps)
    loss_smooth = (w * diff.pow(2)).sum() / w_sum
    n = int(s.numel())
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    d_pair = d[ii, jj]
    w_dir = (d_pair.abs() / max(float(margin_d), eps)).clamp(0.0, 1.0)
    mask = w_dir > float(min_conf)
    if bool(mask.any()):
        pair_logits = diff[ii, jj][mask] / max(float(tau_s), eps)
        target = torch.sigmoid(d_pair[mask] / max(float(tau_d), eps))
        weights = w_dir[mask]
        bce = F.binary_cross_entropy_with_logits(pair_logits, target, reduction="none")
        loss_dir = (bce * weights).sum() / weights.sum().clamp_min(eps)
        pred_sign = torch.sign(diff[ii, jj][mask].detach())
        truth_sign = torch.sign(d_pair[mask].detach())
        valid = truth_sign != 0
        if bool(valid.any()):
            pairwise_agreement = ((pred_sign[valid] == truth_sign[valid]).float() * weights[valid]).sum()
            pairwise_agreement = pairwise_agreement / weights[valid].sum().clamp_min(eps)
        else:
            pairwise_agreement = s.new_tensor(float("nan"))
        dir_pair_conf_mean = weights.detach().mean()
    else:
        loss_dir = s.new_tensor(0.0)
        pairwise_agreement = s.new_tensor(float("nan"))
        dir_pair_conf_mean = s.new_tensor(0.0)
    loss_var = torch.relu(s.new_tensor(float(min_std)) - std).pow(2)
    loss_total = loss_smooth + float(lambda_dir) * loss_dir + float(lambda_var) * loss_var
    return {
        "loss_total": loss_total,
        "loss_smooth": loss_smooth,
        "loss_dir": loss_dir,
        "loss_var": loss_var,
        "logit_std": std.detach(),
        "logit_entropy": _entropy_from_logits(s).detach(),
        "pairwise_direction_agreement": pairwise_agreement.detach(),
        "dir_pair_conf_mean": dir_pair_conf_mean.detach(),
        "smoothness_score": (-loss_smooth).detach(),
    }


def implicit_axis_batch_loss(logits: torch.Tensor, coeffs_batch: list[dict], **kwargs):
    rows = [implicit_axis_loss(logits[idx], coeffs_batch[idx], **kwargs) for idx in range(logits.size(0))]
    out = {}
    for key in rows[0].keys():
        values = [row[key] for row in rows]
        if torch.is_tensor(values[0]) and values[0].requires_grad:
            out[key] = torch.stack(values).mean()
        else:
            tensors = [value if torch.is_tensor(value) else torch.tensor(value, device=logits.device) for value in values]
            out[key] = torch.stack([tensor.to(device=logits.device, dtype=torch.float32) for tensor in tensors]).mean()
    return out


def tensor_metrics_to_floats(metrics: dict):
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            out[key] = float(value.item()) if value.ndim == 0 else float(value.mean().item())
        else:
            out[key] = float(value)
    return out


def mean_dicts(rows: list[dict]):
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out = {}
    for key in keys:
        vals = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                vals.append(float(value))
        if vals:
            out[key] = float(np.mean(vals))
    return out


def order_from_logits_desc(logits: torch.Tensor):
    return torch.argsort(logits.detach().float(), descending=True)


def kendall_tau_to_l2r_order(order: torch.Tensor):
    values = order.detach().cpu().tolist()
    n = len(values)
    if n < 2:
        return 1.0
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] > values[j]:
                inversions += 1
    total = n * (n - 1) / 2.0
    return float(1.0 - 2.0 * inversions / total)


def kendall_tau_between_orders(order_a: torch.Tensor, order_b: torch.Tensor):
    a = order_a.detach().cpu().long()
    b = order_b.detach().cpu().long()
    n = int(a.numel())
    if n < 2:
        return 1.0
    pos_a = torch.empty(n, dtype=torch.long)
    pos_b = torch.empty(n, dtype=torch.long)
    pos_a[a] = torch.arange(n, dtype=torch.long)
    pos_b[b] = torch.arange(n, dtype=torch.long)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pos_a[i] - pos_a[j]).item() * (pos_b[i] - pos_b[j]).item() < 0:
                inversions += 1
    total = n * (n - 1) / 2.0
    return float(1.0 - 2.0 * inversions / total)


def adjacent_rate(order: torch.Tensor):
    vals = order.detach().cpu().long()
    if vals.numel() < 2:
        return 1.0
    return float((vals[1:] - vals[:-1]).abs().eq(1).float().mean().item())


def original_order_diagnostics(order: torch.Tensor, permutation_state: dict | None):
    out = {
        "current_l2r_tau": kendall_tau_to_l2r_order(order),
        "within_unit_adjacent_rate": adjacent_rate(order),
        "order_first16_current": [int(v) for v in order.detach().cpu().tolist()[:16]],
    }
    if permutation_state is not None:
        block_perm = permutation_state["block_perm"].to(dtype=torch.long)
        original_ids = block_perm.index_select(0, order.detach().cpu().long())
        original_l2r = permutation_state["inverse_block_perm"].to(dtype=torch.long)
        out.update(
            {
                "original_l2r_tau": kendall_tau_between_orders(order.detach().cpu(), original_l2r),
                "original_id_tau": kendall_tau_to_l2r_order(original_ids),
                "within_unit_adjacent_rate_original": adjacent_rate(original_ids),
                "order_first16_original": [int(v) for v in original_ids.tolist()[:16]],
            }
        )
    return out


@torch.no_grad()
def full_loss_for_block_orders(
    *,
    model,
    x: torch.Tensor,
    block_orders: torch.Tensor,
    device: torch.device | str,
    dtype: str,
):
    if block_orders.ndim == 1:
        block_orders = block_orders.unsqueeze(0).expand(x.size(0), -1)
    block_orders = block_orders.to(device=x.device, dtype=torch.long)
    token_orders = expand_orders_for_model(model, block_orders)
    with autocast_context(device, dtype):
        outputs = model(
            x,
            mode=None,
            orders=token_orders,
            return_token_loss=True,
            return_logits=False,
        )
    return float(outputs[1].detach().float().cpu().item())


@torch.no_grad()
def evaluate_order_source_nll(
    *,
    model,
    tokens,
    matrices: torch.Tensor,
    coeffs: list[dict],
    order_source: Callable[[torch.Tensor, dict], tuple[torch.Tensor, torch.Tensor]],
    rng: np.random.Generator,
    device: torch.device | str,
    dtype: str,
    token_perm: torch.Tensor | None,
    data_record_mode: str,
    eval_batch_size: int,
    random_eval_orders: int,
    permutation_state: dict | None,
    loss_kwargs: dict,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 10,
):
    rows = []
    device = torch.device(device)
    for idx, matrix in enumerate(matrices):
        logits, order = order_source(matrix, coeffs[idx])
        logits = logits.detach().float().cpu()
        order = order.detach().long().cpu()
        reverse_order = torch.flip(order, dims=[0])
        x = sample_batch(
            tokens,
            batch_size=int(eval_batch_size),
            block_size=int(model.config.block_size),
            rng=rng,
            device=device,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
        )
        forward_loss = full_loss_for_block_orders(
            model=model,
            x=x,
            block_orders=order.to(device),
            device=device,
            dtype=dtype,
        )
        reverse_loss = full_loss_for_block_orders(
            model=model,
            x=x,
            block_orders=reverse_order.to(device),
            device=device,
            dtype=dtype,
        )
        random_losses = []
        for _ in range(int(random_eval_orders)):
            random_orders = random_block_orders(x.size(0), int(model.num_blocks), device)
            random_losses.append(
                full_loss_for_block_orders(
                    model=model,
                    x=x,
                    block_orders=random_orders,
                    device=device,
                    dtype=dtype,
                )
            )
        obj = tensor_metrics_to_floats(
            implicit_axis_loss(logits.to(device), coeffs[idx], **loss_kwargs)
        )
        diag = original_order_diagnostics(order, permutation_state)
        row = {
            "sample_id": idx,
            "mlp_map_full_loss": forward_loss,
            "order_full_loss": forward_loss,
            "reverse_mlp_full_loss": reverse_loss,
            "random_mean_full_loss": float(np.mean(random_losses)) if random_losses else float("nan"),
            "random_best_full_loss": float(np.min(random_losses)) if random_losses else float("nan"),
            "mlp_minus_random_mean": forward_loss - float(np.mean(random_losses)) if random_losses else float("nan"),
            "mlp_minus_random_best": forward_loss - float(np.min(random_losses)) if random_losses else float("nan"),
            "forward_minus_reverse_loss": forward_loss - reverse_loss,
            "reverse_order_loss_minus_forward": reverse_loss - forward_loss,
            "logit_std": float(logits.std(unbiased=False).item()),
            "logit_entropy": float(_entropy_from_logits(logits).item()),
            "pairwise_direction_agreement": obj.get("pairwise_direction_agreement", float("nan")),
            "smoothness_objective": obj.get("loss_total", float("nan")),
            "smoothness_score": obj.get("smoothness_score", float("nan")),
            "loss_smooth": obj.get("loss_smooth", float("nan")),
            "loss_dir": obj.get("loss_dir", float("nan")),
            "loss_var": obj.get("loss_var", float("nan")),
            "diagnostic/current_l2r_tau": diag["current_l2r_tau"],
            "diagnostic/within_unit_adjacent_rate": diag["within_unit_adjacent_rate"],
            "diagnostic/order_first16_current": json.dumps(diag["order_first16_current"]),
        }
        if "original_l2r_tau" in diag:
            row["diagnostic/original_l2r_tau"] = diag["original_l2r_tau"]
            row["diagnostic/original_id_tau"] = diag["original_id_tau"]
            row["diagnostic/within_unit_adjacent_rate_original"] = diag["within_unit_adjacent_rate_original"]
            row["diagnostic/order_first16_original"] = json.dumps(diag["order_first16_original"])
        rows.append(row)
        if progress is not None and (idx + 1) % max(1, int(progress_interval)) == 0:
            progress(f"evaluated_nll_samples={idx + 1}/{len(matrices)}")
    summary = mean_dicts(rows)
    summary.update(
        {
            "num_eval_attention_samples": int(len(rows)),
            "random_eval_orders": int(random_eval_orders),
            "eval_batch_size": int(eval_batch_size),
        }
    )
    return rows, summary


def order_stability_summary(orders: list[torch.Tensor]):
    if len(orders) < 2:
        return {"order_change_kendall_current": float("nan")}
    taus = [kendall_tau_between_orders(orders[idx - 1], orders[idx]) for idx in range(1, len(orders))]
    return {"order_change_kendall_current": float(np.mean(taus))}
