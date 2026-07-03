#!/usr/bin/env python3
"""Probe directed-only order recovery from a single attention head.

This diagnostic is intentionally offline: it loads a checkpoint, averages one
layer/head block-attention matrix, derives several directed orders, and scores
them as fixed block orders. Original-frame Kendall tau is diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from online_spectral_order_policy import (  # noqa: E402
    FixedHeadSpectralPolicyConfig,
    recover_fixed_head_spectral_candidates,
    robust_z,
)
from order_utils import (  # noqa: E402
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    invert_permutation,
)


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def load_model(checkpoint: Dict, device: str):
    model_args = dict(checkpoint["model_args"])
    model = AOGPT(AOGPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def load_tokens(checkpoint: Dict, split: str, data_dir: Path | None):
    if data_dir is None:
        dataset = checkpoint.get("config", {}).get("dataset", "wikitext103")
        data_dir = REPO_ROOT / "data" / str(dataset)
    path = data_dir / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r"), data_dir


def infer_record_mode(checkpoint: Dict, data_dir: Path) -> str:
    if checkpoint.get("config", {}).get("data_record_mode") is not None:
        return str(checkpoint["config"]["data_record_mode"])
    meta_path = data_dir / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            return str(pickle.load(handle).get("data_record_mode", "stream"))
    return "stream"


def permutation_state(checkpoint: Dict, model, block_size: int):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    block_len = int(model.block_order_block_len)
    num_blocks = int(block_size) // block_len
    raw = (checkpoint.get("data_permutation") or {}).get("block_perm")
    if raw is not None:
        block_perm = torch.as_tensor(raw, dtype=torch.long)
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
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def iter_starts(tokens_len: int, block_size: int, record_mode: str, count: int, seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    if str(record_mode) == "fixed":
        num_records = tokens_len // block_size
        ids = rng.integers(0, num_records, size=int(count))
        return [int(v) * block_size for v in ids]
    if str(record_mode) == "stream":
        return [int(v) for v in rng.integers(0, tokens_len - block_size, size=int(count))]
    raise ValueError(f"Unsupported data_record_mode={record_mode!r}")


def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def current_batch(tokens, starts: List[int], block_size: int, device: str, perm_state):
    arr = np.stack([np.asarray(tokens[s : s + block_size], dtype=np.int64) for s in starts], axis=0)
    batch = torch.from_numpy(arr).to(device)
    if perm_state is not None:
        token_perm = perm_state["token_perm"].to(device=device, dtype=torch.long)
        batch = batch.index_select(1, token_perm)
    return batch


def expand_block_orders(model, block_orders: torch.Tensor) -> torch.Tensor:
    return expand_block_orders_to_token_orders(
        block_orders.long(),
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )


def extract_attentions(outputs):
    for value in outputs[2:]:
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            return value
    return None


def aggregate_layerhead_attention(layer_attn, block_orders, block_len: int, export_type: str):
    if str(export_type) == "with_none":
        shifted = layer_attn[:, :, :-1, :-1]
    elif str(export_type) == "without_none":
        shifted = layer_attn[:, :, 1:, 1:]
    else:
        raise ValueError(f"Unsupported export_type={export_type!r}")
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % int(block_len) != 0:
        raise ValueError(f"attention shape={tuple(shifted.shape)} incompatible with block_len={block_len}")
    num_blocks = seq_a // int(block_len)
    block_batch = shifted.float().view(batch, heads, num_blocks, block_len, num_blocks, block_len).mean(dim=(3, 5))
    total = torch.zeros((heads, num_blocks, num_blocks), dtype=torch.float64, device=block_batch.device)
    for sample_idx in range(batch):
        inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=block_batch.device)
        total += block_batch[sample_idx][:, inverse, :][:, :, inverse].double()
    return total / float(max(1, batch))


def attention_block_orders(mode: str, batch_size: int, num_blocks: int, device: str, rng: torch.Generator, perm_state):
    mode = str(mode)
    if mode == "current_ar":
        order = torch.arange(num_blocks, device=device, dtype=torch.long)
        return order.unsqueeze(0).expand(batch_size, -1)
    if mode == "random":
        return torch.stack([torch.randperm(num_blocks, generator=rng, device=device) for _ in range(batch_size)], dim=0)
    if mode == "original_l2r":
        if perm_state is None:
            order = torch.arange(num_blocks, device=device, dtype=torch.long)
        else:
            order = perm_state["inverse_block_perm"].to(device=device, dtype=torch.long)
        return order.unsqueeze(0).expand(batch_size, -1)
    raise ValueError(f"Unsupported attention_order_mode={mode!r}")


@torch.no_grad()
def collect_attention_matrix(args, model, checkpoint, tokens, data_record_mode: str, perm_state, ctx):
    block_size = int(model.config.block_size)
    num_blocks = int(model.num_blocks)
    starts = iter_starts(
        len(tokens),
        block_size,
        data_record_mode,
        int(args.attention_samples),
        int(args.attention_seed),
    )
    rng = torch.Generator(device="cuda" if "cuda" in str(args.device) else "cpu")
    rng.manual_seed(int(args.attention_seed) + 17)
    matrix_sum = None
    total = 0
    for batch_starts in batched(starts, int(args.attention_batch_size)):
        batch = current_batch(tokens, batch_starts, block_size, args.device, perm_state)
        block_orders = attention_block_orders(
            args.attention_order_mode,
            int(batch.size(0)),
            num_blocks,
            args.device,
            rng,
            perm_state,
        )
        token_orders = expand_block_orders(model, block_orders)
        with ctx:
            outputs = model(
                batch,
                mode=None,
                orders=token_orders,
                return_attentions=True,
                return_logits=False,
            )
        attentions = extract_attentions(outputs)
        if not attentions:
            raise RuntimeError("Model did not return attentions.")
        layer = int(args.layer)
        head = int(args.head)
        layer_heads = aggregate_layerhead_attention(
            attentions[layer].detach(),
            block_orders,
            int(model.block_order_block_len),
            str(args.export_type),
        )
        matrix = layer_heads[head].detach().cpu().to(dtype=torch.float64)
        matrix.fill_diagonal_(0.0)
        matrix_sum = matrix * float(batch.size(0)) if matrix_sum is None else matrix_sum + matrix * float(batch.size(0))
        total += int(batch.size(0))
    if matrix_sum is None or total <= 0:
        raise RuntimeError("No attention samples were collected.")
    out = (matrix_sum / float(total)).numpy()
    np.fill_diagonal(out, 0.0)
    return out, total


def robust_z_offdiag(matrix: np.ndarray) -> np.ndarray:
    out = robust_z(np.asarray(matrix, dtype=np.float64))
    np.fill_diagonal(out, 0.0)
    return np.where(np.isfinite(out), out, 0.0)


def threshold_positive(z: np.ndarray, percentile: float) -> np.ndarray:
    values = z[np.isfinite(z) & (~np.eye(z.shape[0], dtype=bool))]
    threshold = float(np.percentile(values, float(percentile))) if values.size else 0.0
    out = np.maximum(z - threshold, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def valid_order_from_scores(scores: np.ndarray, descending: bool = True) -> List[int]:
    scores = np.asarray(scores, dtype=np.float64)
    safe = np.where(np.isfinite(scores), scores, np.nanmedian(scores[np.isfinite(scores)]) if np.isfinite(scores).any() else 0.0)
    return [int(v) for v in np.argsort(safe, kind="mergesort")[::-1 if descending else 1]]


def order_from_eigenvector(matrix: np.ndarray, which: str):
    vals, vecs = np.linalg.eig(matrix)
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError("No finite eigenvalues.")
    idxs = np.where(finite)[0]
    if which == "largest_real":
        idx = idxs[int(np.argmax(vals[idxs].real))]
        scores = vecs[:, idx].real
        order = valid_order_from_scores(scores, descending=True)
    elif which == "largest_mag_real":
        idx = idxs[int(np.argmax(np.abs(vals[idxs])))]
        scores = vecs[:, idx].real
        order = valid_order_from_scores(scores, descending=True)
    elif which == "largest_mag_phase":
        idx = idxs[int(np.argmax(np.abs(vals[idxs])))]
        scores = np.angle(vecs[:, idx])
        order = valid_order_from_scores(scores, descending=False)
    else:
        raise ValueError(which)
    return order, {
        "eig_index": int(idx),
        "eig_value_real": float(vals[idx].real),
        "eig_value_imag": float(vals[idx].imag),
        "eig_value_abs": float(abs(vals[idx])),
    }


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.zeros_like(out), where=row_sum > 1e-12)


def stationary_order(matrix: np.ndarray):
    p = row_normalize(matrix)
    if (p.sum(axis=1) <= 1e-12).any():
        n = p.shape[0]
        bad = p.sum(axis=1) <= 1e-12
        p[bad] = 1.0 / float(n)
    vals, vecs = np.linalg.eig(p.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    scores = np.maximum(vecs[:, idx].real, 0.0)
    if scores.sum() <= 1e-12:
        scores = np.abs(vecs[:, idx].real)
    order = valid_order_from_scores(scores, descending=True)
    return order, {
        "eig_index": int(idx),
        "eig_value_real": float(vals[idx].real),
        "eig_value_imag": float(vals[idx].imag),
        "eig_value_abs": float(abs(vals[idx])),
    }


def directed_orders(matrix: np.ndarray, args):
    z = robust_z_offdiag(matrix)
    b = threshold_positive(z, float(args.threshold_percentile))
    d = z - z.T
    q = b - b.T
    orders = {}
    orders["current_ar"] = (list(range(matrix.shape[0])), {"kind": "baseline"})
    orders["current_r2l"] = (list(reversed(range(matrix.shape[0]))), {"kind": "baseline"})
    orders["directed_netflow_z"] = (
        valid_order_from_scores(d.mean(axis=1), descending=True),
        {"kind": "A_minus_AT", "score": "row_mean"},
    )
    orders["directed_netflow_thresh"] = (
        valid_order_from_scores(q.mean(axis=1), descending=True),
        {"kind": "thresholded_A_minus_AT", "score": "row_mean"},
    )
    for which in ("largest_real", "largest_mag_real", "largest_mag_phase"):
        try:
            orders[f"directed_raw_eig_{which}"] = order_from_eigenvector(b, which)
        except Exception as exc:
            orders[f"directed_raw_eig_{which}"] = (list(range(matrix.shape[0])), {"error": str(exc)})
    try:
        orders["directed_rowstoch_stationary"] = stationary_order(b)
    except Exception as exc:
        orders["directed_rowstoch_stationary"] = (list(range(matrix.shape[0])), {"error": str(exc)})

    for weight in (0.25, 1.0, 2.0):
        try:
            cfg = FixedHeadSpectralPolicyConfig(
                directed_score_weight=float(weight),
                direction_lambdas=str(args.sym_direction_lambdas),
                score_adjacency_sym=str(args.sym_score_adjacency),
                threshold_percentile=float(args.threshold_percentile),
            )
            cand = recover_fixed_head_spectral_candidates(matrix, cfg, top_m=1)[0]
            orders[f"sym_spectral_directed_w{weight:g}"] = (
                [int(v) for v in cand["order"]],
                {
                    "kind": "sym_spectral",
                    "directed_score_weight": float(weight),
                    "score": float(cand.get("score", float("nan"))),
                    "meta": cand.get("meta", {}),
                },
            )
        except Exception as exc:
            orders[f"sym_spectral_directed_w{weight:g}"] = (list(range(matrix.shape[0])), {"error": str(exc)})
    return orders


def kendall_tau(order: List[int]) -> float:
    inv = {int(v): i for i, v in enumerate(order)}
    n = len(order)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if inv[i] < inv[j]:
                concordant += 1
            else:
                discordant += 1
    denom = n * (n - 1) / 2
    return float((concordant - discordant) / denom)


def to_original_order(order_current: List[int], perm_state) -> List[int] | None:
    if perm_state is None:
        return [int(v) for v in order_current]
    mapper = perm_state["block_perm"].to(dtype=torch.long, device="cpu")
    current = torch.as_tensor(order_current, dtype=torch.long)
    return [int(v) for v in mapper[current].tolist()]


def unshuffle_to_current(values_reveal: torch.Tensor, token_orders: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = values_reveal.shape
    batch_idx = torch.arange(batch_size, device=values_reveal.device).unsqueeze(1).expand(batch_size, seq_len)
    out = torch.empty_like(values_reveal)
    out[batch_idx, token_orders.long()] = values_reveal
    return out


@torch.no_grad()
def evaluate_order(args, model, tokens, data_record_mode: str, perm_state, order_current: List[int], ctx):
    block_size = int(model.config.block_size)
    starts = iter_starts(
        len(tokens),
        block_size,
        data_record_mode,
        int(args.eval_samples),
        int(args.eval_seed),
    )
    fixed = torch.as_tensor(order_current, dtype=torch.long, device=args.device)
    total_nll = 0.0
    total_tokens = 0
    for batch_starts in batched(starts, int(args.eval_batch_size)):
        batch = current_batch(tokens, batch_starts, block_size, args.device, perm_state)
        block_orders = fixed.unsqueeze(0).expand(int(batch.size(0)), -1)
        token_orders = expand_block_orders(model, block_orders)
        with ctx:
            outputs = model(
                batch,
                mode=None,
                orders=token_orders,
                return_token_loss=True,
                return_logits=False,
            )
        token_losses_reveal = None
        for value in outputs[2:]:
            if torch.is_tensor(value) and value.ndim == 2:
                token_losses_reveal = value
                break
        if token_losses_reveal is None:
            raise RuntimeError("Could not find token losses in model output.")
        token_losses_current = unshuffle_to_current(token_losses_reveal, token_orders)
        if perm_state is not None:
            inverse_token_perm = invert_permutation(perm_state["token_perm"]).to(device=args.device, dtype=torch.long)
            token_losses_original = token_losses_current.index_select(1, inverse_token_perm)
        else:
            token_losses_original = token_losses_current
        total_nll += float(token_losses_original.double().sum().item())
        total_tokens += int(token_losses_original.numel())
    mean_nll = total_nll / max(1, total_tokens)
    return mean_nll, math.exp(mean_nll)


def write_order_json(path: Path, name: str, order: List[int], meta: Dict):
    path.write_text(
        json.dumps(
            {
                "name": str(name),
                "order": [int(v) for v in order],
                "best_candidate": {"name": str(name), "order": [int(v) for v in order], "meta": meta},
            },
            indent=2,
        )
    )


def save_heatmap(path: Path, matrix: np.ndarray, title: str):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("key block")
        ax.set_ylabel("query block")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc))


def parse_args():
    parser = argparse.ArgumentParser(description="Directed single-head order probe.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=7)
    parser.add_argument("--export_type", choices=("with_none", "without_none"), default="without_none")
    parser.add_argument("--attention_order_mode", choices=("current_ar", "random", "original_l2r"), default="current_ar")
    parser.add_argument("--attention_samples", type=int, default=4096)
    parser.add_argument("--attention_batch_size", type=int, default=32)
    parser.add_argument("--attention_seed", type=int, default=12345)
    parser.add_argument("--eval_samples", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_seed", type=int, default=23456)
    parser.add_argument("--threshold_percentile", type=float, default=60.0)
    parser.add_argument("--sym_direction_lambdas", type=str, default="0,0.1,0.25,0.5,1.0")
    parser.add_argument("--sym_score_adjacency", type=str, default="max")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(args.ckpt_path)
    model = load_model(ckpt, args.device)
    tokens, data_dir = load_tokens(ckpt, args.split, args.data_dir)
    record_mode = infer_record_mode(ckpt, data_dir)
    perm_state = permutation_state(ckpt, model, int(model.config.block_size))
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    ctx = nullcontext() if "cuda" not in str(args.device) or args.dtype == "float32" else torch.amp.autocast("cuda", dtype=dtype)

    matrix, total_attention = collect_attention_matrix(args, model, ckpt, tokens, record_mode, perm_state, ctx)
    np.save(args.out_dir / "attention_matrix_current.npy", matrix)
    save_heatmap(
        args.out_dir / "attention_matrix_current.png",
        matrix,
        f"L{args.layer}H{args.head} {args.attention_order_mode} {args.export_type}",
    )

    orders = directed_orders(matrix, args)
    order_dir = args.out_dir / "orders"
    order_dir.mkdir(exist_ok=True)
    rows = []
    seen_orders = {}
    for name, (order, meta) in orders.items():
        order = [int(v) for v in order]
        if len(order) != int(model.num_blocks) or sorted(order) != list(range(int(model.num_blocks))):
            continue
        order_original = to_original_order(order, perm_state)
        tau_current = kendall_tau(order)
        tau_original = kendall_tau(order_original) if order_original is not None else tau_current
        mean_nll, ppl = evaluate_order(args, model, tokens, record_mode, perm_state, order, ctx)
        reverse = list(reversed(order))
        reverse_mean_nll, reverse_ppl = evaluate_order(args, model, tokens, record_mode, perm_state, reverse, ctx)
        write_order_json(order_dir / f"{name}.json", name, order, meta)
        seen_orders[name] = {
            "order_current": order,
            "order_original": order_original,
            "meta": meta,
        }
        rows.append(
            {
                "name": name,
                "mean_nll": mean_nll,
                "ppl": ppl,
                "reverse_mean_nll": reverse_mean_nll,
                "reverse_ppl": reverse_ppl,
                "reverse_minus_forward_nll": reverse_mean_nll - mean_nll,
                "tau_current_l2r": tau_current,
                "tau_original_l2r": tau_original,
                "order_current": " ".join(str(v) for v in order),
                "order_original": "" if order_original is None else " ".join(str(v) for v in order_original),
                "meta_json": json.dumps(meta, sort_keys=True),
            }
        )
    rows.sort(key=lambda row: float(row["mean_nll"]))
    with (args.out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": int(ckpt.get("iter_num", -1)),
        "split": str(args.split),
        "data_dir": str(data_dir),
        "data_record_mode": str(record_mode),
        "permute_data": bool(perm_state is not None),
        "attention": {
            "layer": int(args.layer),
            "head": int(args.head),
            "export_type": str(args.export_type),
            "order_mode": str(args.attention_order_mode),
            "samples": int(total_attention),
            "threshold_percentile": float(args.threshold_percentile),
        },
        "eval": {
            "samples": int(args.eval_samples),
            "batch_size": int(args.eval_batch_size),
            "seed": int(args.eval_seed),
        },
        "best_by_nll": rows[0],
        "orders": seen_orders,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Directed Head Order Probe",
        "",
        f"- checkpoint: `{args.ckpt_path}`",
        f"- attention: L{args.layer}H{args.head}, `{args.attention_order_mode}`, `{args.export_type}`, samples={total_attention}",
        f"- eval: split={args.split}, samples={args.eval_samples}",
        "",
        "| name | NLL | PPL | reverse-forward NLL | tau_original | first16 current |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        first16 = " ".join(row["order_current"].split()[:16])
        lines.append(
            f"| {row['name']} | {float(row['mean_nll']):.6f} | {float(row['ppl']):.3f} | "
            f"{float(row['reverse_minus_forward_nll']):+.6f} | {float(row['tau_original_l2r']):+.4f} | `{first16}` |"
        )
    (args.out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
