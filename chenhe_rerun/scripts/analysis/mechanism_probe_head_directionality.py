#!/usr/bin/env python3
"""Minimal mechanism probes for continuous head directionality results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ANALYSIS = ROOT / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from head_directionality_continuous_runner import (  # noqa: E402
    EXPORT_TYPES,
    MetricAccumulator,
    WikiTextData,
    build_context,
    expand_orders,
    invert_permutation,
    make_token_perm,
    triangle_stats,
)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def parse_heads(text: str) -> List[Tuple[int, int]]:
    heads = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.replace("L", "").replace("H", ":").split(":", 1)
        heads.append((int(left), int(right)))
    return heads


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_args" not in ckpt:
        cfg = ckpt["config"]
        state_for_shape = ckpt["model"]
        vocab_size = int(state_for_shape["transformer.wte.weight"].shape[0])
        ckpt["model_args"] = {
            "n_layer": int(cfg["n_layer"]),
            "n_head": int(cfg["n_head"]),
            "n_embd": int(cfg["n_embd"]),
            "block_size": int(cfg["block_size"]),
            "bias": bool(cfg.get("bias", True)),
            "vocab_size": vocab_size,
            "dropout": float(cfg.get("dropout", 0.0)),
            "block_order_block_len": int(cfg["block_order_block_len"]),
            "block_order_layout": str(cfg.get("block_order_layout", "contiguous")),
            "image_size": int(cfg.get("image_size", 0)),
            "image_block_size": int(cfg.get("image_block_size", 0)),
            "image_block_height": int(cfg.get("image_block_height", 0)),
            "image_block_width": int(cfg.get("image_block_width", 0)),
            "order_impl": str(cfg.get("order_impl", "block")),
            "position_encoding_mode": str(cfg.get("position_encoding_mode", "absolute")),
            "rope_theta": float(cfg.get("rope_theta", 10000.0)),
        }
    model = AOGPT(AOGPTConfig(**ckpt["model_args"]))
    state = ckpt["model"]
    unwanted = "_orig_mod."
    for key in list(state.keys()):
        if key.startswith(unwanted):
            state[key[len(unwanted) :]] = state.pop(key)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt


def permutation_state(cfg: Dict[str, object], num_blocks: int, block_len: int, device: torch.device):
    if bool(cfg.get("permute_data", False)):
        from order_utils import build_fixed_block_permutation

        fixed = build_fixed_block_permutation(num_blocks, int(cfg.get("permute_seed", 42)))
        inv = invert_permutation(fixed)
        token = make_token_perm(fixed, block_len, cfg)
        return fixed.to(device), inv.to(device), token
    return None, None, None


def get_instances(report_dir: Path, count: int) -> List[int]:
    manifest = json.loads((report_dir / "probe_manifests" / "full_probe_manifest.json").read_text())
    return [int(v) for v in manifest["starts"][:count]]


def paired_orders(num_blocks: int, count: int, prefix_size: int, seed: int, device: torch.device):
    rng = np.random.default_rng(seed + prefix_size * 1009)
    forward, reverse, pairs = [], [], []
    for _ in range(count):
        base = rng.permutation(num_blocks).astype(int).tolist()
        p = min(int(prefix_size), num_blocks - 2)
        i, j = base[p], base[p + 1]
        order_f = base[:p] + [i, j] + base[p + 2 :]
        order_r = base[:p] + [j, i] + base[p + 2 :]
        forward.append(order_f)
        reverse.append(order_r)
        pairs.append((i, j))
    return (
        torch.tensor(forward, dtype=torch.long, device=device),
        torch.tensor(reverse, dtype=torch.long, device=device),
        pairs,
    )


class QKCapture:
    def __init__(self, model, layers: Iterable[int]):
        self.model = model
        self.layers = sorted(set(int(v) for v in layers))
        self.handles = []
        self.logits: Dict[int, torch.Tensor] = {}

    def __enter__(self):
        for layer in self.layers:
            module = self.model.transformer.h[layer].attn

            def hook(mod, args, kwargs, layer_idx=layer):
                x = args[0]
                B, T, C = x.size()
                q, k, _v = mod.c_attn(x).split(mod.n_embd, dim=2)
                k = k.view(B, T, mod.n_head, C // mod.n_head).transpose(1, 2)
                q = q.view(B, T, mod.n_head, C // mod.n_head).transpose(1, 2)
                q, k = mod.q_norm(q), mod.k_norm(k)
                if getattr(mod, "use_rope", False):
                    qpos = kwargs.get("query_positions")
                    kpos = kwargs.get("key_positions")
                    q = mod._apply_rope(q, qpos)
                    k = mod._apply_rope(k, kpos)
                logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                if hasattr(mod, "bias"):
                    logits = logits.masked_fill(mod.bias[:, :, :T, :T] == 0, 0.0)
                else:
                    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
                    logits = logits.masked_fill(~mask, 0.0)
                self.logits[layer_idx] = logits.detach()

            self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def selected_metrics(
    tensors: List[torch.Tensor],
    block_orders: torch.Tensor,
    cfg: Dict[str, object],
    block_len: int,
    fixed_perm,
    inv_perm,
    heads: List[Tuple[int, int]],
    kind: str,
) -> List[Dict[str, object]]:
    device = block_orders.device
    acc = MetricAccumulator(int(cfg["n_layer"]), int(cfg["n_head"]), int(cfg["block_size"]) // block_len, device)
    acc.update(tensors, block_orders, block_len, fixed_perm, inv_perm)
    rows = []
    for export in EXPORT_TYPES:
        for frame in ("current", "original"):
            exp = acc.exposure_average(export, frame)
            for layer, head in heads:
                stats = triangle_stats(exp[layer, head])
                rows.append(
                    {
                        "kind": kind,
                        "export_type": export,
                        "frame": frame,
                        "layer": layer,
                        "head": head,
                        "lower_mean": stats["lower"],
                        "upper_mean": stats["upper"],
                        "triangle_index": stats["t"],
                    }
                )
    return rows


@contextmanager
def embedding_condition(model, condition: str):
    wpe = model.transformer.wpe.weight
    wtpe = model.transformer.wtpe.weight
    old_wpe = wpe.detach().clone()
    old_wtpe = wtpe.detach().clone()
    try:
        if condition == "no_wpe":
            wpe.data.zero_()
        elif condition == "no_wtpe":
            wtpe.data.zero_()
        elif condition == "no_wpe_no_wtpe":
            wpe.data.zero_()
            wtpe.data.zero_()
        yield
    finally:
        wpe.data.copy_(old_wpe)
        wtpe.data.copy_(old_wtpe)


@contextmanager
def zero_head_output(model, layer: int, head: int):
    module = model.transformer.h[int(layer)].attn
    original_forward = module.forward

    def patched_forward(x, query_positions=None, key_positions=None, return_attn=False):
        B, T, C = x.size()
        q, k, v = module.c_attn(x).split(module.n_embd, dim=2)
        k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        v = v.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        q, k = module.q_norm(q), module.k_norm(k)
        if getattr(module, "use_rope", False):
            q = module._apply_rope(q, query_positions)
            k = module._apply_rope(k, key_positions)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if hasattr(module, "bias"):
            att = att.masked_fill(module.bias[:, :, :T, :T] == 0, float("-inf"))
        else:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y[:, int(head), :, :] = 0.0
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = module.resid_dropout(module.c_proj(y))
        if return_attn:
            return y, att
        return y

    module.forward = patched_forward
    try:
        yield
    finally:
        module.forward = original_forward


def eval_loss(model, x, block_orders, cfg, block_len, ctx) -> float:
    token_orders = expand_orders(block_orders, block_len, cfg)
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, x.size(0), 32):
            xb = x[start : start + 32]
            ob = token_orders[start : start + 32]
            with ctx:
                out = model(xb, mode=None, orders=ob, return_logits=True)
            total += float(out[1].detach().item()) * xb.size(0)
            count += xb.size(0)
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heads", required=True)
    parser.add_argument("--neutral-heads", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    cfg = ckpt["config"]
    cfg["device"] = str(device)
    cfg["dtype"] = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    block_len = int(ckpt["model_args"]["block_order_block_len"])
    num_blocks = int(ckpt["model_args"]["block_size"]) // block_len
    fixed_perm, inv_perm, token_perm = permutation_state(cfg, num_blocks, block_len, device)
    data = WikiTextData(cfg, device, "cuda" if device.type == "cuda" else "cpu", token_perm)
    starts = get_instances(args.report_dir, int(args.instances))
    x = data.fixed_windows(starts, split="train")
    heads = parse_heads(args.heads)
    neutral = parse_heads(args.neutral_heads) if args.neutral_heads.strip() else []
    all_heads = heads + [h for h in neutral if h not in heads]
    ctx = build_context("cuda" if device.type == "cuda" else "cpu", cfg["dtype"])
    layers = [h[0] for h in all_heads]
    paired_rows: List[Dict[str, object]] = []
    position_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    for prefix in (0, 8, 16, 32):
        orders_f, orders_r, pairs = paired_orders(num_blocks, x.size(0), prefix, args.seed, device)
        for direction, orders in (("forward_pair", orders_f), ("reverse_pair", orders_r)):
            token_orders = expand_orders(orders, block_len, cfg)
            with torch.no_grad(), QKCapture(model, layers) as capture:
                with ctx:
                    out = model(x, mode=None, orders=token_orders, return_attentions=True, return_logits=False)
                attn = list(out[-1])
                qk = [capture.logits.get(layer, torch.zeros_like(attn[layer])) for layer in range(int(cfg["n_layer"]))]
            for row in selected_metrics(attn, orders, cfg, block_len, fixed_perm, inv_perm, all_heads, "attention"):
                row.update({"prefix_size": prefix, "paired_direction": direction})
                paired_rows.append(row)
            for row in selected_metrics(qk, orders, cfg, block_len, fixed_perm, inv_perm, all_heads, "pre_softmax_qk"):
                row.update({"prefix_size": prefix, "paired_direction": direction})
                paired_rows.append(row)
        if prefix == 16:
            for condition in ("full", "no_wpe", "no_wtpe", "no_wpe_no_wtpe"):
                orders = orders_f
                token_orders = expand_orders(orders, block_len, cfg)
                with embedding_condition(model, condition if condition != "full" else "none"):
                    with torch.no_grad(), QKCapture(model, layers) as capture:
                        with ctx:
                            out = model(x, mode=None, orders=token_orders, return_attentions=True, return_logits=False)
                        attn = list(out[-1])
                        qk = [capture.logits.get(layer, torch.zeros_like(attn[layer])) for layer in range(int(cfg["n_layer"]))]
                for row in selected_metrics(attn, orders, cfg, block_len, fixed_perm, inv_perm, all_heads, "attention"):
                    row.update({"condition": condition, "prefix_size": prefix})
                    position_rows.append(row)
                for row in selected_metrics(qk, orders, cfg, block_len, fixed_perm, inv_perm, all_heads, "pre_softmax_qk"):
                    row.update({"condition": condition, "prefix_size": prefix})
                    position_rows.append(row)
            clean_f = eval_loss(model, x, orders_f, cfg, block_len, ctx)
            clean_r = eval_loss(model, x, orders_r, cfg, block_len, ctx)
            rng_orders = torch.stack(
                [torch.randperm(num_blocks, device=device) for _ in range(x.size(0))],
                dim=0,
            )
            clean_rand = eval_loss(model, x, rng_orders, cfg, block_len, ctx)
            for layer, head in all_heads:
                with zero_head_output(model, layer, head):
                    z_f = eval_loss(model, x, orders_f, cfg, block_len, ctx)
                    z_r = eval_loss(model, x, orders_r, cfg, block_len, ctx)
                    z_rand = eval_loss(model, x, rng_orders, cfg, block_len, ctx)
                ablation_rows.extend(
                    [
                        {"layer": layer, "head": head, "order_type": "forward_pair", "clean_loss": clean_f, "zero_head_loss": z_f, "delta_loss": z_f - clean_f},
                        {"layer": layer, "head": head, "order_type": "reverse_pair", "clean_loss": clean_r, "zero_head_loss": z_r, "delta_loss": z_r - clean_r},
                        {"layer": layer, "head": head, "order_type": "random", "clean_loss": clean_rand, "zero_head_loss": z_rand, "delta_loss": z_rand - clean_rand},
                    ]
                )
    write_csv(args.report_dir / "mechanism" / "paired_order" / "paired_order_qk_softmax.csv", paired_rows)
    write_csv(args.report_dir / "mechanism" / "qk_softmax" / "paired_order_qk_softmax.csv", paired_rows)
    write_csv(args.report_dir / "mechanism" / "position_content" / "position_embedding_ablation.csv", position_rows)
    write_csv(args.report_dir / "mechanism" / "head_output_ablation" / "zero_head_loss_delta.csv", ablation_rows)
    write_json(
        args.report_dir / "mechanism" / "summary.json",
        {
            "checkpoint": str(args.checkpoint),
            "instances": int(args.instances),
            "heads": [f"L{l}H{h}" for l, h in all_heads],
            "paired_order_completed": True,
            "qk_softmax_completed": True,
            "position_content_completed": ["no_wpe", "no_wtpe", "no_wpe_no_wtpe"],
            "head_output_ablation_completed": True,
        },
    )


if __name__ == "__main__":
    main()
