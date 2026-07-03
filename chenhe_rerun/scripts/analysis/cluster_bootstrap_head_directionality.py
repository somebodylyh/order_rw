#!/usr/bin/env python3
"""Posthoc cluster bootstrap for full-probe head directionality."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ANALYSIS = ROOT / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from head_directionality_continuous_runner import (  # noqa: E402
    EXPORT_TYPES,
    WikiTextData,
    block_attention,
    build_context,
    expand_orders,
    invert_permutation,
    make_token_perm,
    sample_triangle_values_batch,
    shift_attention,
)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    if "model_args" in ckpt:
        model_args = ckpt["model_args"]
    else:
        state = ckpt["model"]
        model_args = {
            "n_layer": int(cfg["n_layer"]),
            "n_head": int(cfg["n_head"]),
            "n_embd": int(cfg["n_embd"]),
            "block_size": int(cfg["block_size"]),
            "bias": bool(cfg.get("bias", True)),
            "vocab_size": int(state["transformer.wte.weight"].shape[0]),
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
    model = AOGPT(AOGPTConfig(**model_args))
    state = ckpt["model"]
    unwanted = "_orig_mod."
    for key in list(state.keys()):
        if key.startswith(unwanted):
            state[key[len(unwanted) :]] = state.pop(key)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg, model_args


def permutation_state(cfg, model_args, num_blocks, block_len, device):
    if bool(cfg.get("permute_data", False)):
        from order_utils import build_fixed_block_permutation

        fixed = build_fixed_block_permutation(num_blocks, int(cfg.get("permute_seed", 42)))
        inv = invert_permutation(fixed)
        token = make_token_perm(fixed, block_len, cfg)
        return fixed.to(device), inv.to(device), token
    return None, None, None


def original_visible(block_orders: torch.Tensor, inv_perm, num_blocks: int):
    inv_reveal = torch.argsort(block_orders.detach(), dim=1)
    rank = inv_reveal
    visible_current = rank[:, None, :] < rank[:, :, None]
    if inv_perm is None:
        return visible_current
    cur_for_orig = inv_perm.to(block_orders.device)
    return visible_current[:, cur_for_orig, :][:, :, cur_for_orig]


def original_block_batch(layer_attn, block_orders, export, block_len, inv_perm):
    block = block_attention(shift_attention(layer_attn.detach(), export), block_len)
    batch, heads, num_blocks, _ = block.shape
    inv_reveal = torch.argsort(block_orders.detach(), dim=1)
    rows = inv_reveal[:, None, :, None].expand(batch, heads, num_blocks, num_blocks)
    cols = inv_reveal[:, None, None, :].expand(batch, heads, num_blocks, num_blocks)
    current = block.gather(2, rows).gather(3, cols)
    if inv_perm is None:
        return current
    cur_for_orig = inv_perm.to(block.device)
    return current[:, :, cur_for_orig, :][:, :, :, cur_for_orig]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()
    report_dir = args.report_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads((report_dir / "probe_manifests" / "full_probe_manifest.json").read_text())
    starts = [int(v) for v in manifest["starts"]]
    orders = manifest["orders"]
    n_text = int(manifest["num_texts"])
    n_orders = int(manifest["full_orders_per_text"])
    rng = np.random.default_rng(args.seed)
    rows = []
    for step in range(0, 5001, 500):
        ckpt_path = report_dir / "runtime" / "checkpoints" / f"checkpoint_iter{step:06d}.pt"
        model, cfg, model_args = load_model(ckpt_path, device)
        cfg["dtype"] = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        block_len = int(model_args["block_order_block_len"])
        num_blocks = int(model_args["block_size"]) // block_len
        _fixed, inv_perm, token_perm = permutation_state(cfg, model_args, num_blocks, block_len, device)
        data = WikiTextData(cfg, device, "cuda" if device.type == "cuda" else "cpu", token_perm)
        ctx = build_context("cuda" if device.type == "cuda" else "cpu", cfg["dtype"])
        values = {
            export: np.zeros((n_text, n_orders, int(cfg["n_layer"]), int(cfg["n_head"])), dtype=np.float32)
            for export in EXPORT_TYPES
        }
        instances = []
        for text_idx in range(n_text):
            for order_idx in range(n_orders):
                instances.append((text_idx, order_idx, starts[text_idx], [int(v) for v in orders[text_idx][order_idx]]))
        with torch.no_grad():
            for start in range(0, len(instances), int(args.batch_size)):
                chunk = instances[start : start + int(args.batch_size)]
                idxs = [c[2] for c in chunk]
                block_orders = torch.tensor([c[3] for c in chunk], dtype=torch.long, device=device)
                x = data.fixed_windows(idxs, split=str(manifest.get("split", "train")))
                token_orders = expand_orders(block_orders, block_len, cfg)
                with ctx:
                    out = model(x, mode=None, orders=token_orders, return_attentions=True, return_logits=False)
                attentions = list(out[-1])
                visible = original_visible(block_orders, inv_perm, num_blocks)
                for export in EXPORT_TYPES:
                    per_layer = []
                    for layer_attn in attentions:
                        original = original_block_batch(layer_attn, block_orders, export, block_len, inv_perm)
                        per_layer.append(sample_triangle_values_batch(original, visible))
                    arr = np.stack(per_layer, axis=1)  # B,L,H
                    for local_idx, (text_idx, order_idx, _s, _o) in enumerate(chunk):
                        values[export][text_idx, order_idx] = arr[local_idx]
        for export in EXPORT_TYPES:
            cluster = values[export].mean(axis=1)  # text,L,H
            mean = cluster.mean(axis=0)
            for layer in range(mean.shape[0]):
                for head in range(mean.shape[1]):
                    boot = []
                    for _ in range(int(args.bootstraps)):
                        sample_idx = rng.integers(0, n_text, size=n_text)
                        boot.append(float(cluster[sample_idx, layer, head].mean()))
                    lo, hi = np.quantile(np.asarray(boot), [0.025, 0.975])
                    rows.append(
                        {
                            "iter": step,
                            "export_type": export,
                            "frame": "original",
                            "layer": layer,
                            "head": head,
                            "cluster_mean_t": float(mean[layer, head]),
                            "cluster_ci95_low": float(lo),
                            "cluster_ci95_high": float(hi),
                            "num_text_clusters": n_text,
                            "orders_per_text": n_orders,
                            "bootstraps": int(args.bootstraps),
                        }
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(report_dir / "metrics" / "full_probe_4096_cluster_bootstrap.csv", rows)


if __name__ == "__main__":
    main()
