#!/usr/bin/env python3
"""Continuous per-head attention directionality probe for WikiText103 AO-GPT.

This runner is intentionally narrow: it reproduces the seq256/block64 Random
training path and adds read-only attention measurements from the same forwards.
It avoids spectral recovery, candidate orders, MLP policies, and tau-based
selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from order_utils import (  # noqa: E402
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    invert_permutation,
    sample_random_block_orders,
)


DEFAULTS = {
    "out_dir": "out/head_directionality_continuous",
    "eval_interval": 250,
    "eval_iters": 200,
    "log_interval": 10,
    "wandb_log": False,
    "dataset": "wikitext103",
    "batch_size": 64,
    "block_size": 256,
    "gradient_accumulation_steps": 2,
    "permute_data": False,
    "permute_seed": 42,
    "permute_mode": "block",
    "model_type": "aogpt",
    "train_stage": "standard",
    "aogpt_train_mode": "Random",
    "n_layer": 4,
    "n_head": 8,
    "n_embd": 384,
    "dropout": 0.0,
    "bias": True,
    "block_order_block_len": 16,
    "block_order_layout": "contiguous",
    "image_size": 0,
    "image_block_size": 0,
    "image_block_height": 0,
    "image_block_width": 0,
    "order_impl": "block",
    "position_encoding_mode": "absolute",
    "rope_theta": 10000.0,
    "run_seed": 1337,
    "learning_rate": 6e-4,
    "max_iters": 50000,
    "lr_decay_iters": 50000,
    "min_lr": 6e-5,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 1e-1,
    "grad_clip": 1.0,
    "decay_lr": True,
    "warmup_iters": 0,
    "dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
    "device": "cuda",
    "init_from": "scratch",
    "data_record_mode": "stream",
    "compile": False,
}

EXPORT_TYPES = ("with_none", "without_none")
FRAMES = ("reveal", "current", "original")
DISTANCE_BINS = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
)


def load_config(path: Path) -> Dict[str, object]:
    cfg = dict(DEFAULTS)
    namespace: Dict[str, object] = {}
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), namespace)
    for key, value in namespace.items():
        if not key.startswith("__"):
            cfg[key] = value
    return cfg


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()
    except Exception as exc:  # pragma: no cover - environment reporting only
        return f"ERROR: {exc}"


def expand_orders(block_orders: torch.Tensor, block_len: int, cfg: Dict[str, object]) -> torch.Tensor:
    return expand_block_orders_to_token_orders(
        block_orders,
        block_len=block_len,
        block_order_layout=str(cfg.get("block_order_layout", "contiguous")),
        image_size=int(cfg.get("image_size", 0)),
        image_block_size=int(cfg.get("image_block_size", 0)),
        image_block_height=int(cfg.get("image_block_height", 0)),
        image_block_width=int(cfg.get("image_block_width", 0)),
    )


def make_token_perm(block_perm: torch.Tensor, block_len: int, cfg: Dict[str, object]) -> torch.Tensor:
    return block_permutation_to_token_permutation(
        block_perm,
        block_len=block_len,
        block_order_layout=str(cfg.get("block_order_layout", "contiguous")),
        image_size=int(cfg.get("image_size", 0)),
        image_block_size=int(cfg.get("image_block_size", 0)),
        image_block_height=int(cfg.get("image_block_height", 0)),
        image_block_width=int(cfg.get("image_block_width", 0)),
    )


def build_context(device_type: str, dtype_name: str):
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


class WikiTextData:
    def __init__(
        self,
        cfg: Dict[str, object],
        device: torch.device,
        device_type: str,
        fixed_token_perm: Optional[torch.Tensor],
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.device_type = device_type
        self.fixed_token_perm = fixed_token_perm
        self.data_dir = ROOT / "data" / str(cfg["dataset"])
        self.block_size = int(cfg["block_size"])
        self.batch_size = int(cfg["batch_size"])

    def _memmap(self, split: str) -> np.memmap:
        return np.memmap(self.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")

    def batch(self, split: str, batch_size: Optional[int] = None) -> torch.Tensor:
        data = self._memmap(split)
        bs = self.batch_size if batch_size is None else int(batch_size)
        mode = str(self.cfg.get("data_record_mode", "stream"))
        if mode == "fixed":
            num_records = len(data) // self.block_size
            ix = torch.randint(num_records, (bs,))
            starts = (ix * self.block_size).tolist()
        elif mode == "stream":
            ix = torch.randint(len(data) - self.block_size, (bs,))
            starts = ix.tolist()
        else:
            raise ValueError(f"Unsupported data_record_mode={mode!r}")
        x = torch.stack(
            [torch.from_numpy((data[i : i + self.block_size]).astype(np.int64)) for i in starts]
        )
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        if self.fixed_token_perm is not None:
            x = x[:, self.fixed_token_perm.to(self.device)]
        return x

    def fixed_windows(self, starts: Iterable[int], split: str = "train") -> torch.Tensor:
        data = self._memmap(split)
        rows = []
        for start in starts:
            start_i = int(start)
            rows.append(torch.from_numpy((data[start_i : start_i + self.block_size]).astype(np.int64)))
        x = torch.stack(rows)
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        if self.fixed_token_perm is not None:
            x = x[:, self.fixed_token_perm.to(self.device)]
        return x


def make_manifest(
    cfg: Dict[str, object],
    report_dir: Path,
    num_texts: int,
    full_orders_per_text: int,
    core_texts: int,
    core_orders_per_text: int,
    num_blocks: int,
    seed: int,
) -> Dict[str, object]:
    data_path = ROOT / "data" / str(cfg["dataset"]) / "train.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    max_start = len(data) - int(cfg["block_size"]) - 1
    starts = rng.choice(max_start, size=num_texts, replace=False).astype(int).tolist()
    orders = []
    for text_idx in range(num_texts):
        per_text = []
        for order_idx in range(full_orders_per_text):
            order_rng = np.random.default_rng(seed + 1000003 * (text_idx + 1) + 9176 * (order_idx + 1))
            per_text.append(order_rng.permutation(num_blocks).astype(int).tolist())
        orders.append(per_text)
    manifest = {
        "dataset": str(cfg["dataset"]),
        "split": "train",
        "block_size": int(cfg["block_size"]),
        "num_blocks": int(num_blocks),
        "num_texts": int(num_texts),
        "full_orders_per_text": int(full_orders_per_text),
        "core_texts": int(core_texts),
        "core_orders_per_text": int(core_orders_per_text),
        "seed": int(seed),
        "starts": starts,
        "orders": orders,
        "core_subset_rule": "first core_texts and first core_orders_per_text from full manifest",
    }
    write_json(report_dir / "probe_manifests" / "full_probe_manifest.json", manifest)
    core_manifest = dict(manifest)
    core_manifest["num_texts"] = int(core_texts)
    core_manifest["full_orders_per_text"] = int(core_orders_per_text)
    core_manifest["starts"] = starts[:core_texts]
    core_manifest["orders"] = [row[:core_orders_per_text] for row in orders[:core_texts]]
    write_json(report_dir / "probe_manifests" / "core_probe_manifest.json", core_manifest)
    return manifest


def iter_manifest_instances(
    manifest: Dict[str, object],
    core: bool,
) -> Iterable[Tuple[int, int, int, List[int]]]:
    starts: List[int] = manifest["starts"]  # type: ignore[assignment]
    orders: List[List[List[int]]] = manifest["orders"]  # type: ignore[assignment]
    n_texts = int(manifest["core_texts"] if core else manifest["num_texts"])
    n_orders = int(manifest["core_orders_per_text"] if core else manifest["full_orders_per_text"])
    for text_idx in range(n_texts):
        for order_idx in range(n_orders):
            yield text_idx, order_idx, int(starts[text_idx]), [int(v) for v in orders[text_idx][order_idx]]


def shift_attention(layer_attn: torch.Tensor, export_type: str) -> torch.Tensor:
    if export_type == "with_none":
        return layer_attn[:, :, :-1, :-1]
    if export_type == "without_none":
        return layer_attn[:, :, 1:, 1:]
    raise ValueError(f"Unsupported export_type={export_type!r}")


def block_attention(attn: torch.Tensor, block_len: int) -> torch.Tensor:
    batch, heads, seq_a, seq_b = attn.shape
    if seq_a != seq_b or seq_a % block_len != 0:
        raise ValueError(f"attention shape {tuple(attn.shape)} incompatible with block_len={block_len}")
    num_blocks = seq_a // block_len
    return attn.float().view(batch, heads, num_blocks, block_len, num_blocks, block_len).mean(dim=(3, 5))


def align_one_matrix(
    matrix_reveal: torch.Tensor,
    block_order: torch.Tensor,
    frame: str,
    fixed_block_perm: Optional[torch.Tensor],
    inverse_block_perm: Optional[torch.Tensor],
) -> torch.Tensor:
    if frame == "reveal":
        return matrix_reveal
    inv_reveal = invert_permutation(block_order.detach()).to(matrix_reveal.device)
    current = matrix_reveal[:, inv_reveal, :][:, :, inv_reveal]
    if frame == "current":
        return current
    if frame == "original":
        if fixed_block_perm is None or inverse_block_perm is None:
            return current
        cur_for_orig = inverse_block_perm.to(matrix_reveal.device)
        return current[:, cur_for_orig, :][:, :, cur_for_orig]
    raise ValueError(f"Unsupported frame={frame!r}")


def exposure_matrix(
    block_order: torch.Tensor,
    frame: str,
    num_blocks: int,
    fixed_block_perm: Optional[torch.Tensor],
    inverse_block_perm: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    rank = torch.empty(num_blocks, dtype=torch.long, device=device)
    rank[block_order.to(device)] = torch.arange(num_blocks, dtype=torch.long, device=device)
    visible_current = rank.view(1, -1) < rank.view(-1, 1)  # row query i, col key j
    if frame == "reveal":
        idx = torch.arange(num_blocks, device=device)
        return idx.view(1, -1) < idx.view(-1, 1)
    if frame == "current" or fixed_block_perm is None or inverse_block_perm is None:
        return visible_current
    cur_for_orig = inverse_block_perm.to(device)
    return visible_current[cur_for_orig, :][:, cur_for_orig]


class MetricAccumulator:
    def __init__(self, n_layer: int, n_head: int, num_blocks: int, device: torch.device) -> None:
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_blocks = num_blocks
        self.device = device
        shape = (n_layer, n_head, num_blocks, num_blocks)
        self.sum_matrices: Dict[Tuple[str, str], torch.Tensor] = {}
        self.exposure_sums: Dict[Tuple[str, str], torch.Tensor] = {}
        self.exposure_counts: Dict[Tuple[str, str], torch.Tensor] = {}
        self.sample_t: Dict[Tuple[str, str, int], List[np.ndarray]] = {}
        self.sample_count = 0
        for export_type in EXPORT_TYPES:
            for frame in FRAMES:
                key = (export_type, frame)
                self.sum_matrices[key] = torch.zeros(shape, dtype=torch.float64, device=device)
                self.exposure_sums[key] = torch.zeros(shape, dtype=torch.float64, device=device)
                self.exposure_counts[key] = torch.zeros((num_blocks, num_blocks), dtype=torch.float64, device=device)
                for layer_idx in range(n_layer):
                    self.sample_t[(export_type, frame, layer_idx)] = []

    def update(
        self,
        attentions: List[torch.Tensor],
        block_orders: torch.Tensor,
        block_len: int,
        fixed_block_perm: Optional[torch.Tensor],
        inverse_block_perm: Optional[torch.Tensor],
    ) -> None:
        batch = int(block_orders.size(0))
        self.sample_count += batch
        inv_reveal = torch.argsort(block_orders.detach(), dim=1)
        gather_rows = inv_reveal[:, None, :, None].expand(batch, self.n_head, self.num_blocks, self.num_blocks)
        gather_cols = inv_reveal[:, None, None, :].expand(batch, self.n_head, self.num_blocks, self.num_blocks)
        rank_current = inv_reveal.to(self.device)
        visible_current = rank_current[:, None, :] < rank_current[:, :, None]
        idx = torch.arange(self.num_blocks, device=self.device)
        visible_reveal = (idx.view(1, -1) < idx.view(-1, 1)).unsqueeze(0).expand(batch, -1, -1)
        if fixed_block_perm is not None and inverse_block_perm is not None:
            cur_for_orig = inverse_block_perm.to(self.device)
            visible_original = visible_current[:, cur_for_orig, :][:, :, cur_for_orig]
        else:
            cur_for_orig = None
            visible_original = visible_current
        for layer_idx, layer_attn in enumerate(attentions):
            for export_type in EXPORT_TYPES:
                block = block_attention(shift_attention(layer_attn.detach(), export_type), block_len)
                current = block.gather(2, gather_rows).gather(3, gather_cols)
                if cur_for_orig is None:
                    original = current
                else:
                    original = current[:, :, cur_for_orig, :][:, :, :, cur_for_orig]
                frame_payloads = {
                    "reveal": (block, visible_reveal),
                    "current": (current, visible_current),
                    "original": (original, visible_original),
                }
                for frame, (aligned_batch, visible_batch) in frame_payloads.items():
                    key = (export_type, frame)
                    aligned64 = aligned_batch.double()
                    visible64 = visible_batch.to(torch.float64)
                    self.sum_matrices[key][layer_idx] += aligned64.sum(dim=0)
                    self.exposure_sums[key][layer_idx] += (aligned64 * visible64[:, None, :, :]).sum(dim=0)
                    if layer_idx == 0:
                        self.exposure_counts[key] += visible64.sum(dim=0)
                    self.sample_t[(export_type, frame, layer_idx)].append(
                        sample_triangle_values_batch(aligned_batch, visible_batch)
                    )

    def averaged_matrix(self, export_type: str, frame: str) -> torch.Tensor:
        return self.sum_matrices[(export_type, frame)] / float(max(1, self.sample_count))

    def exposure_average(self, export_type: str, frame: str) -> torch.Tensor:
        counts = self.exposure_counts[(export_type, frame)].clamp_min(1.0)
        return self.exposure_sums[(export_type, frame)] / counts.view(1, 1, self.num_blocks, self.num_blocks)

    def rows(
        self,
        step: int,
        stream: str,
        fixed_manifest_count: Optional[int] = None,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        metric_rows: List[Dict[str, object]] = []
        distance_rows: List[Dict[str, object]] = []
        for export_type in EXPORT_TYPES:
            for frame in FRAMES:
                avg = self.averaged_matrix(export_type, frame)
                exp = self.exposure_average(export_type, frame)
                for layer_idx in range(self.n_layer):
                    sample_array = np.concatenate(self.sample_t[(export_type, frame, layer_idx)], axis=0)
                    for head_idx in range(self.n_head):
                        mat = avg[layer_idx, head_idx]
                        exp_mat = exp[layer_idx, head_idx]
                        stats = triangle_stats(mat)
                        row_stats = triangle_stats(row_normalize(mat))
                        exp_stats = triangle_stats(exp_mat)
                        samples = sample_array[:, head_idx]
                        pos_frac = float(np.mean(samples > 0.0)) if samples.size else 0.0
                        neg_frac = float(np.mean(samples < 0.0)) if samples.size else 0.0
                        near_frac = float(np.mean(np.abs(samples) <= 1e-6)) if samples.size else 0.0
                        entropy = attention_entropy(mat)
                        metric_rows.append(
                            {
                                "stream": stream,
                                "iter": int(step),
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "export_type": export_type,
                                "frame": frame,
                                "num_instances": int(fixed_manifest_count or self.sample_count),
                                "lower_triangle_mean": stats["lower"],
                                "upper_triangle_mean": stats["upper"],
                                "triangle_index_raw": stats["t"],
                                "triangle_index_row_normalized": row_stats["t"],
                                "triangle_index_exposure_corrected": exp_stats["t"],
                                "attention_entropy": entropy,
                                "attention_mass": float(mat.sum().item()),
                                "none_related_mass": none_related_mass(mat, export_type),
                                "mean_signed_distance": mean_signed_distance(exp_mat),
                                "mean_absolute_distance": mean_absolute_distance(exp_mat),
                                "sample_t_median": float(np.median(samples)) if samples.size else 0.0,
                                "sample_t_q25": float(np.quantile(samples, 0.25)) if samples.size else 0.0,
                                "sample_t_q75": float(np.quantile(samples, 0.75)) if samples.size else 0.0,
                                "sample_t_std": float(np.std(samples)) if samples.size else 0.0,
                                "sample_positive_fraction": pos_frac,
                                "sample_negative_fraction": neg_frac,
                                "sample_near_zero_fraction": near_frac,
                            }
                        )
                        for label, lo, hi in DISTANCE_BINS:
                            distance_rows.append(
                                {
                                    "stream": stream,
                                    "iter": int(step),
                                    "layer": int(layer_idx),
                                    "head": int(head_idx),
                                    "export_type": export_type,
                                    "frame": frame,
                                    "distance_bin": label,
                                    "triangle_index_exposure_corrected": triangle_stats_by_distance(exp_mat, lo, hi),
                                }
                            )
        return metric_rows, distance_rows


def lower_upper_masks(num_blocks: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(num_blocks, device=device)
    lower = idx.view(-1, 1) > idx.view(1, -1)
    upper = idx.view(-1, 1) < idx.view(1, -1)
    return lower, upper


def triangle_stats(mat: torch.Tensor) -> Dict[str, float]:
    lower, upper = lower_upper_masks(mat.size(0), mat.device)
    lower_mean = mat[lower].mean() if lower.any() else mat.new_tensor(0.0)
    upper_mean = mat[upper].mean() if upper.any() else mat.new_tensor(0.0)
    denom = lower_mean + upper_mean + 1e-12
    return {
        "lower": float(lower_mean.item()),
        "upper": float(upper_mean.item()),
        "t": float(((lower_mean - upper_mean) / denom).item()),
    }


def row_normalize(mat: torch.Tensor) -> torch.Tensor:
    return mat / mat.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def attention_entropy(mat: torch.Tensor) -> float:
    probs = row_normalize(mat.clamp_min(0.0))
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return float(entropy.item())


def none_related_mass(mat: torch.Tensor, export_type: str) -> float:
    if export_type != "with_none":
        return 0.0
    row_col = mat[0, :].sum() + mat[:, 0].sum() - mat[0, 0]
    return float(row_col.item())


def mean_signed_distance(mat: torch.Tensor) -> float:
    idx = torch.arange(mat.size(0), device=mat.device, dtype=mat.dtype)
    dist = idx.view(-1, 1) - idx.view(1, -1)
    denom = mat.sum().clamp_min(1e-12)
    return float(((mat * dist).sum() / denom).item())


def mean_absolute_distance(mat: torch.Tensor) -> float:
    idx = torch.arange(mat.size(0), device=mat.device, dtype=mat.dtype)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).abs()
    denom = mat.sum().clamp_min(1e-12)
    return float(((mat * dist).sum() / denom).item())


def sample_triangle_values(aligned: torch.Tensor, visible: torch.Tensor) -> np.ndarray:
    values = []
    lower, upper = lower_upper_masks(aligned.size(-1), aligned.device)
    lower_visible = lower & visible
    upper_visible = upper & visible
    for head_idx in range(aligned.size(0)):
        h = aligned[head_idx]
        lower_mean = h[lower_visible].mean() if lower_visible.any() else h.new_tensor(0.0)
        upper_mean = h[upper_visible].mean() if upper_visible.any() else h.new_tensor(0.0)
        denom = lower_mean + upper_mean + 1e-12
        values.append(float(((lower_mean - upper_mean) / denom).item()))
    return np.asarray(values, dtype=np.float64)


def sample_triangle_values_batch(aligned: torch.Tensor, visible: torch.Tensor) -> np.ndarray:
    # aligned: [B, H, N, N], visible: [B, N, N]
    n = aligned.size(-1)
    idx = torch.arange(n, device=aligned.device)
    lower = (idx.view(-1, 1) > idx.view(1, -1)).unsqueeze(0) & visible
    upper = (idx.view(-1, 1) < idx.view(1, -1)).unsqueeze(0) & visible
    lower_f = lower.to(aligned.dtype)
    upper_f = upper.to(aligned.dtype)
    lower_count = lower_f.sum(dim=(1, 2)).clamp_min(1.0)
    upper_count = upper_f.sum(dim=(1, 2)).clamp_min(1.0)
    lower_mean = (aligned * lower_f[:, None, :, :]).sum(dim=(2, 3)) / lower_count[:, None]
    upper_mean = (aligned * upper_f[:, None, :, :]).sum(dim=(2, 3)) / upper_count[:, None]
    t = (lower_mean - upper_mean) / (lower_mean + upper_mean + 1e-12)
    return t.detach().cpu().double().numpy()


def triangle_stats_by_distance(mat: torch.Tensor, lo: int, hi: int) -> float:
    n = mat.size(0)
    idx = torch.arange(n, device=mat.device)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).abs()
    band = (dist >= lo) & (dist <= hi)
    lower = band & (idx.view(-1, 1) > idx.view(1, -1))
    upper = band & (idx.view(-1, 1) < idx.view(1, -1))
    if not lower.any() or not upper.any():
        return 0.0
    lower_mean = mat[lower].mean()
    upper_mean = mat[upper].mean()
    return float(((lower_mean - upper_mean) / (lower_mean + upper_mean + 1e-12)).item())


def write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_matrix_snapshot(path: Path, step: int, acc: MetricAccumulator) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for export_type in EXPORT_TYPES:
        for frame in ("current", "original"):
            arrays[f"{export_type}_{frame}"] = acc.averaged_matrix(export_type, frame).detach().cpu().numpy().astype(np.float32)
            arrays[f"{export_type}_{frame}_exposure"] = acc.exposure_average(export_type, frame).detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path / f"step_{step:06d}.npz", **arrays)


def run_probe(
    model,
    data: WikiTextData,
    cfg: Dict[str, object],
    manifest: Dict[str, object],
    core: bool,
    batch_instances: int,
    step: int,
    device: torch.device,
    ctx,
    block_len: int,
    num_blocks: int,
    fixed_block_perm: Optional[torch.Tensor],
    inverse_block_perm: Optional[torch.Tensor],
    report_dir: Path,
    stream: str,
    save_matrix: bool,
) -> None:
    was_training = model.training
    rng_torch = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    rng_np = np.random.get_state()
    rng_py = random.getstate()
    model.eval()
    acc = MetricAccumulator(int(cfg["n_layer"]), int(cfg["n_head"]), num_blocks, device)
    instances = list(iter_manifest_instances(manifest, core=core))
    with torch.no_grad():
        for start_idx in range(0, len(instances), batch_instances):
            chunk = instances[start_idx : start_idx + batch_instances]
            starts = [item[2] for item in chunk]
            block_orders = torch.tensor([item[3] for item in chunk], dtype=torch.long, device=device)
            x = data.fixed_windows(starts, split=str(manifest.get("split", "train")))
            token_orders = expand_orders(block_orders, block_len, cfg)
            with ctx:
                outputs = model(
                    x,
                    mode=None,
                    orders=token_orders,
                    return_attentions=True,
                    return_logits=False,
                )
            attentions = list(outputs[-1])
            acc.update(attentions, block_orders, block_len, fixed_block_perm, inverse_block_perm)
            del outputs, attentions, x, block_orders, token_orders
    metric_rows, distance_rows = acc.rows(step=step, stream=stream, fixed_manifest_count=len(instances))
    metric_csv = report_dir / "metrics" / ("fixed_core_128.csv" if core else "full_probe_4096.csv")
    metric_jsonl = report_dir / "metrics" / ("fixed_core_128.jsonl" if core else "full_probe_4096.jsonl")
    write_csv_rows(metric_csv, metric_rows)
    write_csv_rows(report_dir / "metrics" / "distance_profiles.csv", distance_rows)
    for row in metric_rows:
        append_jsonl(metric_jsonl, row)
    if save_matrix:
        save_matrix_snapshot(report_dir / "matrices" / "all_heads_every100", step, acc)
    if was_training:
        model.train()
    torch.set_rng_state(rng_torch)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    np.random.set_state(rng_np)
    random.setstate(rng_py)


def synthetic_remap_sanity(report_dir: Path, num_blocks: int, device: torch.device) -> bool:
    rows = []
    torch.manual_seed(20260619)
    orders = [torch.randperm(num_blocks, device=device) for _ in range(8)]
    mats = {
        "upper": torch.triu(torch.ones(num_blocks, num_blocks, device=device), diagonal=1),
        "lower": torch.tril(torch.ones(num_blocks, num_blocks, device=device), diagonal=-1),
        "symmetric": torch.ones(num_blocks, num_blocks, device=device) - torch.eye(num_blocks, device=device),
        "random": torch.rand(num_blocks, num_blocks, device=device),
        "offset_plus1": torch.diag(torch.ones(num_blocks - 1, device=device), diagonal=1),
        "offset_minus1": torch.diag(torch.ones(num_blocks - 1, device=device), diagonal=-1),
    }
    ok = True
    for name, mat in mats.items():
        for order in orders:
            reveal = mat[order, :][:, order]
            recovered = align_one_matrix(reveal.unsqueeze(0), order, "current", None, None)[0]
            err = torch.linalg.matrix_norm(recovered - mat) / torch.linalg.matrix_norm(mat).clamp_min(1e-12)
            err_f = float(err.item())
            rows.append({"case": name, "frobenius_relative_error": err_f})
            ok = ok and err_f < 1e-6
    write_csv_rows(report_dir / "sanity" / "synthetic_remap" / "results.csv", rows)
    write_json(report_dir / "sanity" / "synthetic_remap" / "summary.json", {"passed": ok, "max_error": max(r["frobenius_relative_error"] for r in rows)})
    return ok


def with_without_index_audit(report_dir: Path, block_len: int, block_size: int, samples: int = 8) -> None:
    rows = []
    for sample_idx in range(samples):
        for export_type in EXPORT_TYPES:
            if export_type == "with_none":
                kept_positions = list(range(0, block_size))
                semantics = "predictor_aligned: includes [None] at position 0, drops final predictor-less real-token position"
            else:
                kept_positions = list(range(1, block_size + 1))
                semantics = "real_token_only: removes [None], keeps all real-token positions"
            for block_idx in range(block_size // block_len):
                positions = kept_positions[block_idx * block_len : (block_idx + 1) * block_len]
                rows.append(
                    {
                        "sample": sample_idx,
                        "export_type": export_type,
                        "block_idx_reveal_frame": block_idx,
                        "query_positions_kept": " ".join(str(v) for v in positions),
                        "key_positions_kept": " ".join(str(v) for v in positions),
                        "row_semantics": "query",
                        "column_semantics": "key",
                        "semantics": semantics,
                    }
                )
    write_csv_rows(report_dir / "sanity" / "with_without_index_audit" / "index_audit.csv", rows)


def uniform_attention_null(
    report_dir: Path,
    manifest: Dict[str, object],
    num_blocks: int,
    fixed_block_perm: Optional[torch.Tensor],
    inverse_block_perm: Optional[torch.Tensor],
    device: torch.device,
) -> bool:
    rows = []
    for frame in ("current", "original"):
        values = []
        for _, _, _, order_values in iter_manifest_instances(manifest, core=True):
            order = torch.tensor(order_values, dtype=torch.long, device=device)
            visible = exposure_matrix(order, frame, num_blocks, fixed_block_perm, inverse_block_perm, device)
            mat = visible.to(torch.float64)
            exp_stats = triangle_stats(mat)
            values.append(exp_stats["t"])
        mean_abs = float(abs(np.mean(values)))
        rows.append({"frame": frame, "mean_triangle_index": float(np.mean(values)), "mean_abs": mean_abs, "num_instances": len(values)})
    write_csv_rows(report_dir / "sanity" / "uniform_attention_null" / "results.csv", rows)
    passed = all(float(row["mean_abs"]) < 0.05 for row in rows)
    write_json(report_dir / "sanity" / "uniform_attention_null" / "summary.json", {"passed": passed, "rows": rows})
    return passed


def learning_rate_for_iter(cfg: Dict[str, object], it: int) -> float:
    learning_rate = float(cfg["learning_rate"])
    warmup_iters = int(cfg.get("warmup_iters", 0))
    lr_decay_iters = int(cfg.get("lr_decay_iters", cfg.get("max_iters", 0)))
    min_lr = float(cfg.get("min_lr", learning_rate / 10.0))
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def build_model_and_env(cfg: Dict[str, object], report_dir: Path, device_arg: str):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        device_arg = "cpu"
    device = torch.device(device_arg)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype_name = str(cfg["dtype"])
    if device_type == "cpu":
        dtype_name = "float32"
    run_seed = int(cfg.get("run_seed", 1337))
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    np.random.seed(int(cfg.get("permute_seed", 42)))
    random.seed(run_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    block_len = 1 if str(cfg.get("order_impl", "block")) == "token" else int(cfg["block_order_block_len"])
    if int(cfg["block_size"]) % block_len != 0:
        raise ValueError("block_size must be divisible by block_order_block_len")
    num_blocks = int(cfg["block_size"]) // block_len
    if bool(cfg.get("permute_data", False)):
        fixed_block_perm = build_fixed_block_permutation(num_blocks, int(cfg.get("permute_seed", 42)))
        inverse_block_perm = invert_permutation(fixed_block_perm)
        fixed_token_perm = make_token_perm(fixed_block_perm, block_len, cfg)
    else:
        fixed_block_perm = None
        inverse_block_perm = None
        fixed_token_perm = None
    data_dir = ROOT / "data" / str(cfg["dataset"])
    meta_path = data_dir / "meta.pkl"
    meta_vocab_size = None
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            meta_vocab_size = pickle.load(handle)["vocab_size"]
    model_args = dict(
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        block_size=int(cfg["block_size"]),
        bias=bool(cfg.get("bias", True)),
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=float(cfg.get("dropout", 0.0)),
        block_order_block_len=block_len,
        block_order_layout=str(cfg.get("block_order_layout", "contiguous")),
        image_size=int(cfg.get("image_size", 0)),
        image_block_size=int(cfg.get("image_block_size", 0)),
        image_block_height=int(cfg.get("image_block_height", 0)),
        image_block_width=int(cfg.get("image_block_width", 0)),
        order_impl=str(cfg.get("order_impl", "block")),
        position_encoding_mode=str(cfg.get("position_encoding_mode", "absolute")),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
    )
    model = AOGPT(AOGPTConfig(**model_args)).to(device)
    optimizer = model.configure_optimizers(
        float(cfg.get("weight_decay", 0.1)),
        float(cfg["learning_rate"]),
        (float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95))),
        device_type,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype_name == "float16"))
    data = WikiTextData(cfg, device, device_type, fixed_token_perm)
    ctx = build_context(device_type, dtype_name)
    env_payload = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "dtype": dtype_name,
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "config": {k: cfg[k] for k in sorted(cfg.keys()) if isinstance(cfg[k], (str, int, float, bool, type(None)))},
        "model_args": model_args,
        "data_train_sha256": sha256_file(data_dir / "train.bin") if (data_dir / "train.bin").exists() else "",
        "data_val_sha256": sha256_file(data_dir / "val.bin") if (data_dir / "val.bin").exists() else "",
        "fixed_block_perm": fixed_block_perm.tolist() if fixed_block_perm is not None else None,
        "inverse_block_perm": inverse_block_perm.tolist() if inverse_block_perm is not None else None,
    }
    write_json(report_dir / "environment.json", env_payload)
    with (report_dir / "environment.md").open("w", encoding="utf-8") as handle:
        handle.write("# Environment\n\n")
        for key in ("python", "torch", "torch_cuda", "device", "dtype", "git_head", "git_branch"):
            handle.write(f"- `{key}`: `{env_payload[key]}`\n")
    return model, optimizer, scaler, data, ctx, device, device_type, block_len, num_blocks, fixed_block_perm, inverse_block_perm, model_args


def run_training(
    cfg: Dict[str, object],
    report_dir: Path,
    args,
    model,
    optimizer,
    scaler,
    data: WikiTextData,
    ctx,
    device: torch.device,
    block_len: int,
    num_blocks: int,
    fixed_block_perm: Optional[torch.Tensor],
    inverse_block_perm: Optional[torch.Tensor],
    manifest: Dict[str, object],
) -> None:
    model.train()
    grad_acc = int(cfg["gradient_accumulation_steps"])
    batch_size = int(cfg["batch_size"])
    passive_csv = report_dir / "metrics" / "passive_per_step.csv"
    passive_jsonl = report_dir / "metrics" / "passive_per_step.jsonl"
    train_log = report_dir / "logs" / "train_metrics.jsonl"
    t0 = time.time()
    run_probe(
        model,
        data,
        cfg,
        manifest,
        core=True,
        batch_instances=int(args.probe_batch_size),
        step=0,
        device=device,
        ctx=ctx,
        block_len=block_len,
        num_blocks=num_blocks,
        fixed_block_perm=fixed_block_perm,
        inverse_block_perm=inverse_block_perm,
        report_dir=report_dir,
        stream="core_probe",
        save_matrix=True,
    )
    run_probe(
        model,
        data,
        cfg,
        manifest,
        core=False,
        batch_instances=int(args.probe_batch_size),
        step=0,
        device=device,
        ctx=ctx,
        block_len=block_len,
        num_blocks=num_blocks,
        fixed_block_perm=fixed_block_perm,
        inverse_block_perm=inverse_block_perm,
        report_dir=report_dir,
        stream="full_probe",
        save_matrix=True,
    )
    (report_dir / "runtime" / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": 0,
            "config": cfg,
        },
        report_dir / "runtime" / "checkpoints" / "checkpoint_iter000000.pt",
    )
    for iter_num in range(1, int(args.max_iters) + 1):
        lr_iter = iter_num - 1
        lr = learning_rate_for_iter(cfg, lr_iter) if bool(cfg.get("decay_lr", True)) else float(cfg["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        step_acc = MetricAccumulator(int(cfg["n_layer"]), int(cfg["n_head"]), num_blocks, device)
        loss_value = 0.0
        for _micro in range(grad_acc):
            x = data.batch("train", batch_size=batch_size)
            block_orders = sample_random_block_orders(batch_size, num_blocks, device)
            token_orders = expand_orders(block_orders, block_len, cfg)
            with ctx:
                outputs = model(
                    x,
                    mode=None,
                    orders=token_orders,
                    return_attentions=True,
                    return_logits=True,
                )
                logits, loss = outputs[:2]
                attentions = list(outputs[-1])
                step_acc.update(attentions, block_orders, block_len, fixed_block_perm, inverse_block_perm)
                loss_value += float(loss.detach().item())
                loss = loss / grad_acc
            scaler.scale(loss).backward()
            del logits, outputs, attentions, x, block_orders, token_orders, loss
        if float(cfg.get("grad_clip", 1.0)) != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        rows, distance_rows = step_acc.rows(step=iter_num, stream="passive_train")
        write_csv_rows(passive_csv, rows)
        write_csv_rows(report_dir / "metrics" / "distance_profiles.csv", distance_rows)
        for row in rows:
            append_jsonl(passive_jsonl, row)
        if iter_num % int(args.core_interval) == 0:
            run_probe(
                model,
                data,
                cfg,
                manifest,
                core=True,
                batch_instances=int(args.probe_batch_size),
                step=iter_num,
                device=device,
                ctx=ctx,
                block_len=block_len,
                num_blocks=num_blocks,
                fixed_block_perm=fixed_block_perm,
                inverse_block_perm=inverse_block_perm,
                report_dir=report_dir,
                stream="core_probe",
                save_matrix=(iter_num % int(args.matrix_interval) == 0),
            )
        if iter_num % int(args.full_interval) == 0:
            run_probe(
                model,
                data,
                cfg,
                manifest,
                core=False,
                batch_instances=int(args.probe_batch_size),
                step=iter_num,
                device=device,
                ctx=ctx,
                block_len=block_len,
                num_blocks=num_blocks,
                fixed_block_perm=fixed_block_perm,
                inverse_block_perm=inverse_block_perm,
                report_dir=report_dir,
                stream="full_probe",
                save_matrix=True,
            )
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "config": cfg,
            }
            (report_dir / "runtime" / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, report_dir / "runtime" / "checkpoints" / f"checkpoint_iter{iter_num:06d}.pt")
        elapsed = time.time() - t0
        append_jsonl(
            train_log,
            {
                "iter": iter_num,
                "loss": loss_value / grad_acc,
                "learning_rate": lr,
                "elapsed_sec": elapsed,
                "gpu_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
                "gpu_memory_reserved": int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0,
            },
        )
        if iter_num % int(cfg.get("log_interval", 10)) == 0 or iter_num == 1:
            print(
                f"iter {iter_num}: loss {loss_value / grad_acc:.4f}, lr {lr:.6g}, "
                f"elapsed {elapsed / 60.0:.1f} min",
                flush=True,
            )


def diagnostic_parity(
    cfg: Dict[str, object],
    report_dir: Path,
    args,
    device_arg: str,
) -> None:
    # A short deterministic check: metrics collection is read-only relative to
    # the same manual-attention training path. This is intentionally separate
    # from the formal model instance.
    local_cfg = dict(cfg)
    local_cfg["batch_size"] = min(4, int(cfg["batch_size"]))
    local_cfg["gradient_accumulation_steps"] = 1
    model_a, opt_a, scaler_a, data_a, ctx_a, device, _device_type, block_len, num_blocks, fixed_perm, inv_perm, _ = build_model_and_env(local_cfg, report_dir / "sanity" / "diagnostic_parity" / "with_metrics", device_arg)
    model_b, opt_b, scaler_b, data_b, ctx_b, _, _, _, _, _, _, _ = build_model_and_env(local_cfg, report_dir / "sanity" / "diagnostic_parity" / "without_metrics", device_arg)
    model_b.load_state_dict(model_a.state_dict())
    losses_a, losses_b = [], []
    for step in range(int(args.parity_steps)):
        torch.manual_seed(9000 + step)
        x_a = data_a.batch("train", batch_size=int(local_cfg["batch_size"]))
        block_orders_a = sample_random_block_orders(int(local_cfg["batch_size"]), num_blocks, device)
        token_orders_a = expand_orders(block_orders_a, block_len, local_cfg)
        torch.manual_seed(9000 + step)
        x_b = data_b.batch("train", batch_size=int(local_cfg["batch_size"]))
        block_orders_b = block_orders_a.clone()
        token_orders_b = token_orders_a.clone()
        with ctx_a:
            out_a = model_a(x_a, mode=None, orders=token_orders_a, return_attentions=True, return_logits=True)
            loss_a = out_a[1]
            acc = MetricAccumulator(int(local_cfg["n_layer"]), int(local_cfg["n_head"]), num_blocks, device)
            acc.update(list(out_a[-1]), block_orders_a, block_len, fixed_perm, inv_perm)
        scaler_a.scale(loss_a).backward()
        scaler_a.step(opt_a)
        scaler_a.update()
        opt_a.zero_grad(set_to_none=True)
        with ctx_b:
            out_b = model_b(x_b, mode=None, orders=token_orders_b, return_attentions=True, return_logits=True)
            loss_b = out_b[1]
        scaler_b.scale(loss_b).backward()
        scaler_b.step(opt_b)
        scaler_b.update()
        opt_b.zero_grad(set_to_none=True)
        losses_a.append(float(loss_a.detach().item()))
        losses_b.append(float(loss_b.detach().item()))
    max_diff = max(abs(a - b) for a, b in zip(losses_a, losses_b)) if losses_a else 0.0
    write_json(
        report_dir / "sanity" / "diagnostic_parity" / "summary.json",
        {"steps": int(args.parity_steps), "max_loss_abs_diff": max_diff, "losses_with_metrics": losses_a, "losses_without_metrics": losses_b, "passed": max_diff < 1e-6},
    )


def initial_docs(report_dir: Path, cfg_path: Path, cfg: Dict[str, object], args) -> None:
    (report_dir / "repro" / "configs").mkdir(parents=True, exist_ok=True)
    (report_dir / "repro" / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, report_dir / "repro" / "configs" / cfg_path.name)
    shutil.copy2(Path(__file__), report_dir / "repro" / "scripts" / Path(__file__).name)
    with (report_dir / "commands.sh").open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n")
        handle.write(" ".join(map(str, sys.argv)) + "\n")
    files_read = [
        "README.md",
        "base.md",
        "base_cn.md",
        "research-state.yaml",
        "research-log.md",
        "literature/survey.md",
        "docs/prompts/Prompt_language.md",
        "docs/prompts/prompt_online_language_current.md",
        "docs/prompts/prompt_pro.md",
        "docs/findings/findings_language.md",
        "docs/structure/PROJECT_STRUCTURE_language.md",
        "config/WikiText103/README.md",
        "online_spectral_order_policy.py",
        "attn_mlp_order_policy.py",
        "train.py",
        "AOGPT.py",
        "AOGPT_block.py",
        "AOGPT_token.py",
        "order_utils.py",
        "scripts/analysis/head_attention_gradient_orientation_diagnostic.py",
        "scripts/analysis/head_candidate_loss_profile_diagnostic.py",
        "scripts/analysis/head_orientation_loss_flow_diagnostic.py",
        "scripts/analysis/summarize_head_signal_stability.py",
        "scripts/analysis/head_angle_tau_diagnostic.py",
        "scripts/analysis/seq80_random_ckpt_online_order_attn_batch.py",
        "scripts/analysis/seq80_loss_rerank_existing_attention_candidates.py",
    ]
    (report_dir / "files_read.md").write_text("\n".join(f"- `{item}`" for item in files_read) + "\n", encoding="utf-8")
    (report_dir / "experiment_design.md").write_text(
        "\n".join(
            [
                "# Experiment Design",
                "",
                "Research question: whether raw single-head AO-GPT attention, remapped from reveal frame to current/original block coordinates, shows persistent lower-triangle, upper-triangle, or fluctuating direction over a continuous optimizer-step window.",
                "",
                "Phenomenon definition: per-head `T = (lower_mean - upper_mean) / (lower_mean + upper_mean + eps)` with raw, row-normalized, and exposure-corrected variants. Positive is lower-triangle/L2R-like routing; negative is upper-triangle/R2L-like routing.",
                "",
                "Coordinate convention: row = query block, column = key block. `reveal` is the causal reveal order, `current` is the model/policy block coordinate after inverse reveal remap, and `original` is a diagnostic-only mapping via fixed data permutation metadata.",
                "",
                "`with_none` uses `layer_attn[:, :, :-1, :-1]` and is predictor-aligned; `without_none` uses `layer_attn[:, :, 1:, 1:]` and is real-token-only. Both are exported from the same forward.",
                "",
                f"Base config: `{cfg_path}`.",
                f"Observation window: 0..{int(args.max_iters)} optimizer steps.",
                f"Core probe: {int(args.core_texts)} text windows x {int(args.core_orders)} orders every {int(args.core_interval)} steps.",
                f"Full probe: {int(args.full_texts)} text windows x {int(args.full_orders)} orders every {int(args.full_interval)} steps.",
                "",
                "No-prior boundary: no OriginalL2R, original tau, validation PPL, spectral candidates, MLP logits, or cross-head consensus is used for training, head selection, or change-point selection.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def final_summary(report_dir: Path, args, synthetic_passed: bool, uniform_passed: bool) -> None:
    core_csv = report_dir / "metrics" / "fixed_core_128.csv"
    full_csv = report_dir / "metrics" / "full_probe_4096.csv"
    passive_csv = report_dir / "metrics" / "passive_per_step.csv"
    def count_iters(path: Path) -> int:
        if not path.exists():
            return 0
        vals = set()
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                vals.add(int(row["iter"]))
        return len(vals)
    summary = {
        "try_id": report_dir.name,
        "base_head": git_value(["rev-parse", "HEAD"]),
        "base_config": str(args.config),
        "observation_start_iter": 0,
        "observation_end_iter": int(args.max_iters),
        "formal_steps_completed": count_iters(passive_csv),
        "core_probe_points": count_iters(core_csv),
        "full_probe_points": count_iters(full_csv),
        "full_probe_sequences_per_point": int(args.full_texts) * int(args.full_orders),
        "num_heads": 32,
        "with_none_completed": core_csv.exists(),
        "without_none_completed": core_csv.exists(),
        "synthetic_remap_passed": bool(synthetic_passed),
        "uniform_null_passed": bool(uniform_passed),
        "stable_lower_heads": [],
        "stable_upper_heads": [],
        "confirmed_flip_heads": [],
        "amplitude_oscillation_heads": [],
        "neutral_heads": [],
        "export_sensitive_heads": [],
        "mechanism_ablations_completed": [],
        "original_repo_restored": False,
        "only_report_artifacts_remaining": False,
        "success": False,
        "next_direction": "Run offline classification/change-point/bootstrap analysis, then Phase B mechanism probes for confirmed representative heads.",
    }
    write_json(report_dir / "summary.json", summary)
    (report_dir / "experiment_result.md").write_text(
        "\n".join(
            [
                "# Experiment Result",
                "",
                f"Formal steps completed: {summary['formal_steps_completed']}.",
                f"Core probe points: {summary['core_probe_points']}.",
                f"Full probe points: {summary['full_probe_points']}.",
                f"Synthetic remap passed: {summary['synthetic_remap_passed']}.",
                f"Uniform-visible null passed: {summary['uniform_null_passed']}.",
                "",
                "Head classification and Phase B mechanism ablations require the offline analysis stage.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "next_direction.md").write_text(summary["next_direction"] + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--core-interval", type=int, default=10)
    parser.add_argument("--full-interval", type=int, default=500)
    parser.add_argument("--matrix-interval", type=int, default=100)
    parser.add_argument("--full-texts", type=int, default=512)
    parser.add_argument("--full-orders", type=int, default=8)
    parser.add_argument("--core-texts", type=int, default=32)
    parser.add_argument("--core-orders", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--manifest-seed", type=int, default=20260619)
    parser.add_argument("--run-seed", type=int, default=1337)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--parity-steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg["max_iters"] = int(args.max_iters)
    cfg["run_seed"] = int(args.run_seed)
    cfg["out_dir"] = str(report_dir / "runtime" / "out")
    cfg["wandb_log"] = False
    initial_docs(report_dir, args.config, cfg, args)
    model, optimizer, scaler, data, ctx, device, _device_type, block_len, num_blocks, fixed_perm, inv_perm, _model_args = build_model_and_env(cfg, report_dir, args.device)
    manifest = make_manifest(
        cfg,
        report_dir,
        int(args.full_texts),
        int(args.full_orders),
        int(args.core_texts),
        int(args.core_orders),
        num_blocks,
        int(args.manifest_seed),
    )
    synthetic_passed = synthetic_remap_sanity(report_dir, num_blocks, device)
    with_without_index_audit(report_dir, block_len, int(cfg["block_size"]))
    uniform_passed = uniform_attention_null(report_dir, manifest, num_blocks, fixed_perm, inv_perm, device)
    if not synthetic_passed:
        raise RuntimeError("synthetic remap sanity failed")
    if not uniform_passed:
        raise RuntimeError("uniform-visible attention null failed")
    if not args.skip_parity:
        diagnostic_parity(cfg, report_dir, args, args.device)
    run_training(
        cfg,
        report_dir,
        args,
        model,
        optimizer,
        scaler,
        data,
        ctx,
        device,
        block_len,
        num_blocks,
        fixed_perm,
        inv_perm,
        manifest,
    )
    final_summary(report_dir, args, synthetic_passed, uniform_passed)


if __name__ == "__main__":
    main()
