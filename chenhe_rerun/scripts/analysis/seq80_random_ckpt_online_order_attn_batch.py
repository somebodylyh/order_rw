from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from export_block_attention_heatmap import save_heatmap_png, save_distance_curve_png  # noqa: E402
from head_angle_tau_diagnostic import (  # noqa: E402
    TokenArray,
    aggregate_layer_attention_to_block,
    build_probe_orders,
    extract_attentions_from_outputs,
    infer_data_record_mode,
    recover_candidates_for_angles,
    sample_batch,
    tau_to_l2r,
)
from online_spectral_order_policy import FixedHeadSpectralPolicyConfig  # noqa: E402
from order_utils import expand_block_orders_to_token_orders, invert_permutation, token_losses_to_block_losses  # noqa: E402


DEFAULT_CKPT_DIR = (
    "out/base/nonpermute/seq80/block1/"
    "out-wikitext103-seq80-random-b1-nonpermute-save-attn-ckpts-50000-iters"
)
DEFAULT_OUT_DIR = (
    "Report/analysis/attn/nonpermute/seq80/block1/"
    "random_baseline_ckpt_online_order_warmup_diagnostic"
)
DEFAULT_STEPS = "1000,2000,5000,8000,10000,15000,20000,25000,50000"


def parse_steps(text: str) -> list[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in str(device) or str(dtype) == "float32":
        return nullcontext()
    return torch.amp.autocast(
        device_type="cuda",
        dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[str(dtype)],
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu")


def build_model(checkpoint: dict, device: str):
    model_args = dict(checkpoint["model_args"])
    model = AOGPT(AOGPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    if hasattr(model, "set_attention_backend"):
        model.set_attention_backend(True)
    model.to(device)
    model.eval()
    return model


def resolve_data_dir(checkpoint: dict, dataset: str | None, data_dir: Path | None) -> Path:
    if data_dir is not None:
        return data_dir
    name = dataset or checkpoint.get("config", {}).get("dataset")
    if name is None:
        raise ValueError("Could not infer dataset. Pass --dataset or --data_dir.")
    return REPO_ROOT / "data" / str(name)


def load_tokens(data_dir: Path, split: str, checkpoint: dict):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    mode = str(checkpoint.get("config", {}).get("data_record_mode") or infer_data_record_mode(data_dir))
    values = np.memmap(split_path, dtype=np.uint16, mode="r")
    return TokenArray(values, data_record_mode=mode)


def ckpt_path_for_step(ckpt_dir: Path, step: int) -> Path:
    if int(step) == 50000:
        path = ckpt_dir / "checkpoints" / "ckpt_iter0050000.pt"
        if path.exists():
            return path
    return ckpt_dir / "checkpoints" / f"ckpt_iter{int(step):07d}.pt"


def compute_distance_profile(matrix: np.ndarray):
    values = np.asarray(matrix, dtype=np.float32)
    distances = np.arange(values.shape[0], dtype=np.int32)
    means = np.zeros_like(distances, dtype=np.float32)
    for distance in distances:
        diag = np.diag(values, k=-int(distance))
        finite = diag[np.isfinite(diag)]
        means[distance] = float(finite.mean()) if finite.size else float("nan")
    return distances, means


def plot_matrix(matrix: np.ndarray, out_path: Path, title: str, cmap: str = "viridis"):
    values = np.asarray(matrix, dtype=np.float32)
    plot_values = np.nan_to_num(values, nan=0.0)
    save_heatmap_png(plot_values, out_path, title=title, cmap=cmap)
    positive = plot_values[plot_values > 0]
    if positive.size:
        vmax = float(np.percentile(positive, 99.0))
        save_heatmap_png(
            plot_values,
            out_path.with_name(out_path.stem + "_p99.png"),
            title=title + " p99",
            cmap=cmap,
            vmax=vmax if vmax > 0.0 else None,
        )
    log_values = np.log10(np.maximum(plot_values, 0.0) + 1e-6)
    save_heatmap_png(
        log_values,
        out_path.with_name(out_path.stem + "_log10.png"),
        title=title + " log10",
        cmap="magma",
    )
    distances, means = compute_distance_profile(values)
    np.save(out_path.with_name(out_path.stem + "_distance_profile_distances.npy"), distances)
    np.save(out_path.with_name(out_path.stem + "_distance_profile_mean_attention.npy"), means)
    save_distance_curve_png(
        distances,
        means,
        out_path.with_name(out_path.stem + "_distance_profile.png"),
        title=title + " distance profile",
    )


@torch.no_grad()
def mine_attention_matrices_both(
    model,
    tokens,
    *,
    sample_count: int,
    batch_size: int,
    split_seed: int,
    order_seed: int,
    reveal_mode: str,
    device: str,
    ctx,
):
    np_rng = np.random.default_rng(int(split_seed))
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(int(order_seed))
    sums = None
    total_samples = 0
    while total_samples < int(sample_count):
        local_batch = min(int(batch_size), int(sample_count) - int(total_samples))
        idx = sample_batch(
            tokens,
            local_batch,
            int(model.config.block_size),
            np_rng,
            device,
            token_perm=None,
        )
        token_orders, block_orders = build_probe_orders(
            model,
            int(idx.size(0)),
            device,
            reveal_mode,
            torch_gen,
        )
        with ctx:
            outputs = model(
                idx,
                mode=None,
                orders=token_orders,
                return_attentions=True,
                return_logits=False,
            )
        attn_outputs = extract_attentions_from_outputs(outputs)
        if sums is None:
            shape = (len(attn_outputs), int(attn_outputs[0].size(1)), int(model.num_blocks), int(model.num_blocks))
            sums = {
                "with_none": torch.zeros(shape, dtype=torch.float64, device=device),
                "without_none": torch.zeros(shape, dtype=torch.float64, device=device),
            }
        for export_type in ("with_none", "without_none"):
            for layer_idx, layer_attn in enumerate(attn_outputs):
                block_batch = aggregate_layer_attention_to_block(
                    layer_attn.detach().float(),
                    block_len=int(model.block_order_block_len),
                    export_type=export_type,
                )
                for sample_idx in range(block_batch.size(0)):
                    inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=block_batch.device)
                    sums[export_type][layer_idx] += block_batch[sample_idx][:, inverse, :][:, :, inverse].double()
        batch_samples = int(idx.size(0))
        total_samples += batch_samples
    matrices = {}
    for export_type, value in sums.items():
        arr = (value / float(total_samples)).detach().cpu().numpy()
        for layer_idx in range(arr.shape[0]):
            for head_idx in range(arr.shape[1]):
                np.fill_diagonal(arr[layer_idx, head_idx], np.nan)
        matrices[export_type] = arr
    return matrices, None, int(total_samples)


def zscore(values: torch.Tensor):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.zeros_like(values)
    mean = finite.mean()
    std = finite.std(unbiased=False)
    safe = torch.where(torch.isfinite(values), values, mean)
    if float(std.item()) < 1e-8:
        return torch.zeros_like(safe)
    return (safe - mean) / std


@torch.no_grad()
def loss_rerank_candidates(
    model,
    tokens,
    candidates: list[dict],
    *,
    batch_size: int,
    batches: int,
    candidate_batch_size: int,
    prefix_k: int,
    split_seed: int,
    device: str,
    ctx,
    attention_weight: float,
    prefix_weight: float,
    full_weight: float,
):
    if int(batches) <= 0 or not candidates:
        return candidates, []
    np_rng = np.random.default_rng(int(split_seed))
    candidate_orders = torch.tensor(
        [[int(v) for v in candidate["order"]] for candidate in candidates],
        dtype=torch.long,
        device=device,
    )
    num_candidates = int(candidate_orders.size(0))
    num_blocks = int(candidate_orders.size(1))
    prefix_k = max(1, min(int(prefix_k), num_blocks))
    candidate_batch_size = max(1, int(candidate_batch_size))
    prefix_loss_sum = torch.zeros(num_candidates, dtype=torch.float64, device=device)
    full_loss_sum = torch.zeros(num_candidates, dtype=torch.float64, device=device)
    count_sum = torch.zeros(num_candidates, dtype=torch.float64, device=device)

    was_training = model.training
    model.eval()
    try:
        for _ in range(int(batches)):
            x_rerank = sample_batch(
                tokens,
                int(batch_size),
                int(model.config.block_size),
                np_rng,
                device,
                token_perm=None,
            )
            local_batch = int(x_rerank.size(0))
            for start in range(0, num_candidates, candidate_batch_size):
                end = min(num_candidates, start + candidate_batch_size)
                chunk = candidate_orders[start:end]
                chunk_size = int(chunk.size(0))
                block_orders = (
                    chunk[:, None, :]
                    .expand(chunk_size, local_batch, num_blocks)
                    .reshape(chunk_size * local_batch, num_blocks)
                )
                x_expanded = (
                    x_rerank[None, :, :]
                    .expand(chunk_size, local_batch, int(x_rerank.size(1)))
                    .reshape(chunk_size * local_batch, int(x_rerank.size(1)))
                )
                token_orders = expand_block_orders_to_token_orders(
                    block_orders,
                    block_len=int(model.block_order_block_len),
                    block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
                    image_size=int(getattr(model, "image_size", 0)),
                    image_block_size=int(getattr(model, "image_block_size", 0)),
                    image_block_height=int(getattr(model, "image_block_height", 0)),
                    image_block_width=int(getattr(model, "image_block_width", 0)),
                )
                with ctx:
                    outputs = model(
                        x_expanded,
                        mode=None,
                        orders=token_orders,
                        return_token_loss=True,
                        return_logits=False,
                    )
                token_losses = outputs[2]
                block_losses = token_losses_to_block_losses(
                    token_losses.detach(),
                    block_len=int(model.block_order_block_len),
                ).float()
                block_losses = block_losses.view(chunk_size, local_batch, num_blocks)
                prefix_loss_sum[start:end] += block_losses[:, :, :prefix_k].double().mean(dim=2).sum(dim=1)
                full_loss_sum[start:end] += block_losses.double().mean(dim=2).sum(dim=1)
                count_sum[start:end] += float(local_batch)
    finally:
        if was_training:
            model.train()

    count_sum = count_sum.clamp_min(1.0)
    prefix_loss = (prefix_loss_sum / count_sum).float()
    full_loss = (full_loss_sum / count_sum).float()
    attention_scores = torch.tensor(
        [float(candidate.get("score", 0.0)) for candidate in candidates],
        dtype=torch.float32,
        device=device,
    )
    final_scores = (
        float(attention_weight) * zscore(attention_scores)
        - float(prefix_weight) * zscore(prefix_loss)
        - float(full_weight) * zscore(full_loss)
    )
    reranked = []
    rows = []
    for idx, candidate in enumerate(candidates):
        updated = dict(candidate)
        meta = dict(updated.get("meta", {}))
        meta.update(
            {
                "attention_spectral_score": float(attention_scores[idx].detach().cpu().item()),
                "loss_rerank_prefix_loss": float(prefix_loss[idx].detach().cpu().item()),
                "loss_rerank_full_loss": float(full_loss[idx].detach().cpu().item()),
                "loss_rerank_score": float(final_scores[idx].detach().cpu().item()),
                "loss_rerank_prefix_k": int(prefix_k),
                "loss_rerank_batches": int(batches),
                "loss_rerank_batch_size": int(batch_size),
            }
        )
        updated["meta"] = meta
        updated["score"] = float(final_scores[idx].detach().cpu().item())
        reranked.append(updated)
    reranked.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    for rank, candidate in enumerate(reranked, start=1):
        meta = candidate.get("meta", {})
        rows.append(
            {
                "loss_rerank_rank": int(rank),
                "candidate_name": str(candidate.get("name", "")),
                "loss_rerank_score": float(candidate.get("score", float("nan"))),
                "attention_spectral_score": float(meta.get("attention_spectral_score", float("nan"))),
                "prefix_loss": float(meta.get("loss_rerank_prefix_loss", float("nan"))),
                "full_loss": float(meta.get("loss_rerank_full_loss", float("nan"))),
                "order_current": json.dumps([int(v) for v in candidate.get("order", [])], separators=(",", ":")),
            }
        )
    return reranked, rows


def order_stats(order: list[int]) -> dict:
    order = [int(v) for v in order]
    if len(order) < 2:
        return {
            "tau_l2r": 1.0,
            "mean_abs_jump": 0.0,
            "median_abs_jump": 0.0,
            "max_abs_jump": 0,
            "forward_adjacent_pairs": 0,
            "backward_adjacent_pairs": 0,
            "either_adjacent_pairs": 0,
            "either_adjacent_rate": 0.0,
            "longest_abs1_run": 1,
        }
    diffs = [order[i + 1] - order[i] for i in range(len(order) - 1)]
    abs_diffs = [abs(v) for v in diffs]
    longest = 1
    current = 1
    for diff in diffs:
        if abs(diff) == 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    longest = max(longest, current)
    either = sum(1 for diff in diffs if abs(diff) == 1)
    return {
        "tau_l2r": float(tau_to_l2r(order)),
        "mean_abs_jump": float(np.mean(abs_diffs)),
        "median_abs_jump": float(np.median(abs_diffs)),
        "max_abs_jump": int(max(abs_diffs)),
        "forward_adjacent_pairs": int(sum(1 for diff in diffs if diff == 1)),
        "backward_adjacent_pairs": int(sum(1 for diff in diffs if diff == -1)),
        "either_adjacent_pairs": int(either),
        "either_adjacent_rate": float(either / max(1, len(diffs))),
        "longest_abs1_run": int(longest),
    }


def summarize_candidate(candidate: dict, prefix: str) -> dict:
    order = [int(v) for v in candidate.get("order", [])]
    meta = candidate.get("meta", {})
    row = {
        f"{prefix}_name": str(candidate.get("name", "")),
        f"{prefix}_score": float(candidate.get("score", float("nan"))),
        f"{prefix}_order_current": json.dumps(order, separators=(",", ":")),
        f"{prefix}_first16": json.dumps(order[:16], separators=(",", ":")),
        f"{prefix}_selected_k": int(meta.get("k", -1)),
        f"{prefix}_primary_angle_idx": int((meta.get("primary_axis") or {}).get("angle_index", -1)),
        f"{prefix}_secondary_angle_idx": int((meta.get("secondary_axis") or {}).get("angle_index", -1)),
        f"{prefix}_primary_reverse": bool(meta.get("primary_reverse", False)),
        f"{prefix}_secondary_reverse": bool(meta.get("secondary_reverse", False)),
        f"{prefix}_group_order_reverse": bool(meta.get("group_order_reverse", False)),
        f"{prefix}_adjacency_path_score": float(meta.get("adjacency_path_score", float("nan"))),
        f"{prefix}_directed_path_score": float(meta.get("best_directed_path_score", float("nan"))),
        f"{prefix}_band_quality": float(meta.get("band_quality", float("nan"))),
    }
    for key, value in order_stats(order).items():
        row[f"{prefix}_{key}"] = value
    if "loss_rerank_prefix_loss" in meta:
        row[f"{prefix}_attention_spectral_score"] = float(meta.get("attention_spectral_score", float("nan")))
        row[f"{prefix}_prefix_loss"] = float(meta.get("loss_rerank_prefix_loss", float("nan")))
        row[f"{prefix}_full_loss"] = float(meta.get("loss_rerank_full_loss", float("nan")))
    return row


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_summary_plots(rows: list[dict], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[seq80-random-attn] plot import failed: {exc}")
        return
    steps = [int(row["step"]) for row in rows]
    attn_tau = [float(row["attention_top1_tau_l2r"]) for row in rows]
    loss_tau = [
        float(row["loss_rerank_top1_tau_l2r"])
        if row.get("loss_rerank_top1_tau_l2r") not in (None, "")
        else float("nan")
        for row in rows
    ]
    attn_adj = [float(row["attention_top1_either_adjacent_rate"]) for row in rows]
    loss_adj = [
        float(row["loss_rerank_top1_either_adjacent_rate"])
        if row.get("loss_rerank_top1_either_adjacent_rate") not in (None, "")
        else float("nan")
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, attn_tau, marker="o", label="attention top1")
    if any(math.isfinite(v) for v in loss_tau):
        ax.plot(steps, loss_tau, marker="o", label="loss rerank top1")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("Kendall tau vs L2R")
    ax.set_title("Recovered order tau across random checkpoints")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tau_vs_step.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, attn_adj, marker="o", label="attention top1")
    if any(math.isfinite(v) for v in loss_adj):
        ax.plot(steps, loss_adj, marker="o", label="loss rerank top1")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("adjacent |diff|=1 pair rate")
    ax.set_title("Recovered order local-adjacency across random checkpoints")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "adjacent_rate_vs_step.png", dpi=200)
    plt.close(fig)


def analyze_checkpoint(args, step: int, ckpt_path: Path, policy_config: FixedHeadSpectralPolicyConfig):
    print(f"[seq80-random-attn] step={step} loading {ckpt_path}")
    checkpoint = load_checkpoint(ckpt_path)
    model = build_model(checkpoint, str(args.device))
    data_dir = resolve_data_dir(checkpoint, args.dataset, args.data_dir)
    tokens = load_tokens(data_dir, str(args.split), checkpoint)
    ctx = get_autocast_context(str(args.device), str(args.dtype))
    ckpt_out = Path(args.out_dir) / f"ckpt_{int(step):06d}"
    ckpt_out.mkdir(parents=True, exist_ok=True)
    matrices, mean_loss, total_samples = mine_attention_matrices_both(
        model,
        tokens,
        sample_count=int(args.sample_count),
        batch_size=int(args.batch_size),
        split_seed=int(args.seed),
        order_seed=int(args.order_seed),
        reveal_mode=str(args.reveal_mode),
        device=str(args.device),
        ctx=ctx,
    )
    for export_type, arr in matrices.items():
        np.save(ckpt_out / f"{export_type}_all_layers_heads_current_l2r.npy", arr)
        fixed_matrix = arr[int(args.layer), int(args.head)]
        mean_matrix = np.nanmean(arr, axis=(0, 1))
        np.save(ckpt_out / f"l{int(args.layer)}h{int(args.head)}_{export_type}_current_l2r.npy", fixed_matrix)
        np.save(ckpt_out / f"mean_all_{export_type}_current_l2r.npy", mean_matrix)
        plot_matrix(
            fixed_matrix,
            ckpt_out / f"l{int(args.layer)}h{int(args.head)}_{export_type}_current_l2r.png",
            title=f"step {step} L{int(args.layer)}H{int(args.head)} {export_type}",
        )
        plot_matrix(
            mean_matrix,
            ckpt_out / f"mean_all_{export_type}_current_l2r.png",
            title=f"step {step} all-layer/head mean {export_type}",
        )

    recovery_matrix = matrices["with_none"][int(args.layer), int(args.head)]
    candidates, eigvals = recover_candidates_for_angles(
        recovery_matrix,
        policy_config,
        num_angles=int(args.num_angles),
        angle_indices=list(range(int(args.num_angles))),
        top_m=max(int(args.top_m), int(args.loss_rerank_top_k)),
    )
    candidate_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        order = [int(v) for v in candidate["order"]]
        row = {
            "step": int(step),
            "attention_rank": int(rank),
            "candidate_name": str(candidate.get("name", "")),
            "candidate_score": float(candidate.get("score", float("nan"))),
            "order_current": json.dumps(order, separators=(",", ":")),
            "first16": json.dumps(order[:16], separators=(",", ":")),
        }
        for key, value in order_stats(order).items():
            row[key] = value
        candidate_rows.append(row)
    write_csv(ckpt_out / "attention_candidates.csv", candidate_rows)

    top_attention = candidates[0]
    row = {
        "step": int(step),
        "ckpt_path": str(ckpt_path),
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "probe_split": str(args.split),
        "probe_samples": int(total_samples),
        "probe_mean_loss_random_reveal": None if mean_loss is None else float(mean_loss),
        "reveal_mode": str(args.reveal_mode),
        "layer": int(args.layer),
        "head": int(args.head),
        "attn_export_type_for_recovery": "with_none",
        "num_unique_candidates": int((top_attention.get("meta", {}) or {}).get("num_unique_candidates", len(candidates))),
        "eigvals_first8": json.dumps([float(v) for v in np.asarray(eigvals).tolist()[:8]], separators=(",", ":")),
    }
    row.update(summarize_candidate(top_attention, "attention_top1"))

    loss_rows = []
    if bool(args.loss_rerank):
        rerank_candidates = candidates[: int(args.loss_rerank_top_k)]
        reranked, loss_rows = loss_rerank_candidates(
            model,
            tokens,
            rerank_candidates,
            batch_size=int(args.loss_rerank_batch_size),
            batches=int(args.loss_rerank_batches),
            candidate_batch_size=int(args.loss_rerank_candidate_batch_size),
            prefix_k=int(args.loss_rerank_prefix_k),
            split_seed=int(args.loss_rerank_seed),
            device=str(args.device),
            ctx=ctx,
            attention_weight=float(args.loss_rerank_attention_weight),
            prefix_weight=float(args.loss_rerank_prefix_weight),
            full_weight=float(args.loss_rerank_full_weight),
        )
        for loss_row in loss_rows:
            loss_row["step"] = int(step)
        write_csv(ckpt_out / "loss_rerank_candidates.csv", loss_rows)
        row.update(summarize_candidate(reranked[0], "loss_rerank_top1"))

    (ckpt_out / "summary.json").write_text(json.dumps(json_safe(row), indent=2), encoding="utf-8")
    del model
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def write_markdown(rows: list[dict], out_dir: Path):
    lines = [
        "# Seq80 Random Checkpoint Online-Order Attention Diagnostic",
        "",
        "This report analyzes the random non-permute seq80/block1 checkpoint evolution.",
        "Attention is collected with current-frame random reveal orders and recovered with the same fixed-head attention-spectral proposal used by the online policy.",
        "",
        "## Summary",
        "",
        "| step | attn tau | attn adj rate | attn first16 | loss-rerank tau | loss-rerank adj rate | loss-rerank first16 |",
        "| ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        loss_tau = row.get("loss_rerank_top1_tau_l2r")
        loss_adj = row.get("loss_rerank_top1_either_adjacent_rate")
        loss_first16 = row.get("loss_rerank_top1_first16", "")
        lines.append(
            "| {step} | {attn_tau:.4f} | {attn_adj:.3f} | `{attn_first16}` | {loss_tau} | {loss_adj} | `{loss_first16}` |".format(
                step=int(row["step"]),
                attn_tau=float(row["attention_top1_tau_l2r"]),
                attn_adj=float(row["attention_top1_either_adjacent_rate"]),
                attn_first16=row["attention_top1_first16"],
                loss_tau="" if loss_tau in (None, "") else f"{float(loss_tau):.4f}",
                loss_adj="" if loss_adj in (None, "") else f"{float(loss_adj):.3f}",
                loss_first16=loss_first16,
            )
        )
    lines.extend(
        [
            "",
            "## Heatmap Files",
            "",
            "Each `ckpt_<step>/` directory contains:",
            "",
            "- `l0h7_with_none_current_l2r.png`: fixed-head matrix used for recovery.",
            "- `l0h7_without_none_current_l2r.png`: same head without the `[None]` predictor alignment.",
            "- `mean_all_with_none_current_l2r.png`: all-layer/all-head mean view.",
            "- `attention_candidates.csv`: top attention-spectral candidates.",
            "- `loss_rerank_candidates.csv`: present only when loss rerank was enabled.",
            "",
        ]
    )
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch seq80 random checkpoint attention/order diagnostic.")
    parser.add_argument("--ckpt_dir", type=Path, default=Path(DEFAULT_CKPT_DIR))
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--steps", type=str, default=DEFAULT_STEPS)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--sample_count", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--order_seed", type=int, default=22345)
    parser.add_argument("--reveal_mode", type=str, default="Random", choices=("Random", "AR"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=7)
    parser.add_argument("--top_m", type=int, default=96)
    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument("--component_pairs", type=str, default="1-2")
    parser.add_argument("--num_angles", type=int, default=16)
    parser.add_argument("--k_values", type=str, default="4,6,7,8,9,10,12")
    parser.add_argument("--group_methods", type=str, default="gap")
    parser.add_argument("--threshold_percentile", type=float, default=60.0)
    parser.add_argument("--transform", type=str, default="relu", choices=("relu", "exp", "softplus"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--direction_lambdas", type=str, default="0,0.1,0.25")
    parser.add_argument("--directed_score_weight", type=float, default=0.25)
    parser.add_argument("--band_quality_weight", type=float, default=0.05)
    parser.add_argument("--score_adjacency_sym", type=str, default="max", choices=("max", "mean"))
    parser.add_argument("--loss_rerank", action="store_true")
    parser.add_argument("--loss_rerank_top_k", type=int, default=96)
    parser.add_argument("--loss_rerank_batches", type=int, default=4)
    parser.add_argument("--loss_rerank_batch_size", type=int, default=128)
    parser.add_argument("--loss_rerank_candidate_batch_size", type=int, default=8)
    parser.add_argument("--loss_rerank_prefix_k", type=int, default=16)
    parser.add_argument("--loss_rerank_attention_weight", type=float, default=1.0)
    parser.add_argument("--loss_rerank_prefix_weight", type=float, default=0.4)
    parser.add_argument("--loss_rerank_full_weight", type=float, default=0.1)
    parser.add_argument("--loss_rerank_seed", type=int, default=32345)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_config = FixedHeadSpectralPolicyConfig(
        num_components=int(args.num_components),
        component_pairs=str(args.component_pairs),
        num_angles=int(args.num_angles),
        k_values=str(args.k_values),
        group_methods=str(args.group_methods),
        threshold_percentile=float(args.threshold_percentile),
        transform=str(args.transform),
        temperature=float(args.temperature),
        direction_lambdas=str(args.direction_lambdas),
        directed_score_weight=float(args.directed_score_weight),
        band_quality_weight=float(args.band_quality_weight),
        score_adjacency_sym=str(args.score_adjacency_sym),
    )
    config_payload = vars(args).copy()
    config_payload["policy_config"] = vars(policy_config)
    (args.out_dir / "config.json").write_text(json.dumps(json_safe(config_payload), indent=2), encoding="utf-8")

    rows = []
    for step in parse_steps(args.steps):
        ckpt_path = ckpt_path_for_step(args.ckpt_dir, step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for step {step}: {ckpt_path}")
        row = analyze_checkpoint(args, step, ckpt_path, policy_config)
        rows.append(row)
        write_csv(args.out_dir / "summary.csv", rows)
        write_markdown(rows, args.out_dir)
        make_summary_plots(rows, args.out_dir)
    write_csv(args.out_dir / "summary.csv", rows)
    write_markdown(rows, args.out_dir)
    make_summary_plots(rows, args.out_dir)
    print(json.dumps({"out_dir": str(args.out_dir), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
