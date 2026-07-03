"""
Diagnose sign/orientation stability for head-level order signals.

The method is intentionally no-prior. It first recovers an unsigned attention
spectral order axis from the current checkpoint and probe batch, then chooses
between order S and reverse(S) using current-model conditional loss on current
samples. Original L2R tau is reported only as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from head_angle_tau_diagnostic import (  # noqa: E402
    FixedHeadSpectralPolicyConfig,
    build_model,
    get_autocast_context,
    kendall_between_orders,
    load_checkpoint,
    load_permutation_state,
    load_tokens,
    matrix_stats,
    mine_attention_matrices,
    order_to_original,
    parse_heads,
    recover_candidates_for_angles,
    resolve_data_dir,
    sample_batch,
    tau_to_l2r,
)
from online_spectral_order_policy import anti, path_score, robust_z, sym  # noqa: E402
from order_utils import expand_block_orders_to_token_orders, token_losses_to_block_losses  # noqa: E402


def sign_of(value: float, eps: float = 1e-9) -> int:
    value = float(value)
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
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


def mean_std(values):
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def pairwise_tau_summary(orders: list[list[int]]) -> dict:
    taus = []
    abs_taus = []
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            tau = kendall_between_orders(orders[i], orders[j])
            taus.append(float(tau))
            abs_taus.append(abs(float(tau)))
    mean, std, min_v, max_v = mean_std(taus)
    abs_mean, abs_std, abs_min, abs_max = mean_std(abs_taus)
    return {
        "pairwise_order_tau_mean": mean,
        "pairwise_order_tau_std": std,
        "pairwise_order_tau_min": min_v,
        "pairwise_order_tau_max": max_v,
        "pairwise_abs_order_tau_mean": abs_mean,
        "pairwise_abs_order_tau_std": abs_std,
        "pairwise_abs_order_tau_min": abs_min,
        "pairwise_abs_order_tau_max": abs_max,
        "num_order_pairs": int(len(taus)),
    }


def flow_margins(matrix: np.ndarray, order: list[int], direction_lambda: float, score_adjacency_sym: str) -> dict:
    values = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(values, np.nan)
    z = robust_z(values)
    adjacency = robust_z(sym(z, str(score_adjacency_sym)))
    reverse = list(reversed(order))
    attn_query_key = anti(z)
    attn_key_query = anti(z.T)
    qk_forward = path_score(order, adjacency, attn_query_key, float(direction_lambda))
    qk_reverse = path_score(reverse, adjacency, attn_query_key, float(direction_lambda))
    kq_forward = path_score(order, adjacency, attn_key_query, float(direction_lambda))
    kq_reverse = path_score(reverse, adjacency, attn_key_query, float(direction_lambda))
    return {
        "flow_qk_margin_forward_minus_reverse": float(qk_forward - qk_reverse),
        "flow_kq_margin_forward_minus_reverse": float(kq_forward - kq_reverse),
        "flow_qk_forward": float(qk_forward),
        "flow_qk_reverse": float(qk_reverse),
        "flow_kq_forward": float(kq_forward),
        "flow_kq_reverse": float(kq_reverse),
    }


def position_anchor_scores(model, source: str) -> np.ndarray | None:
    source = str(source).strip().lower()
    if not hasattr(model, "transformer"):
        return None
    transformer = model.transformer
    block_size = int(model.config.block_size)
    num_blocks = int(model.num_blocks)
    block_len = int(model.block_order_block_len)
    if block_size != num_blocks * block_len:
        return None
    if source in {"target_index", "causal_index", "causal_context", "target_precedence", "causal_precedence"}:
        scores = torch.arange(num_blocks, dtype=torch.float32)
        scores = scores - scores.mean()
        scale = float(scores.norm().item())
        if not math.isfinite(scale) or scale <= 1e-12:
            return None
        return (scores.to(dtype=torch.float64) / scale).numpy()

    causal_pc1 = source in {"wtpe_causal_pc1", "wpe_causal_pc1"}
    feature_source = source.replace("_causal_pc1", "_pc1") if causal_pc1 else source
    if feature_source.startswith("wtpe"):
        if "wtpe" not in transformer:
            return None
        token_features = transformer.wtpe.weight[:block_size].detach().float()
    elif feature_source.startswith("wpe"):
        if "wpe" not in transformer:
            return None
        token_features = transformer.wpe.weight[1 : block_size + 1].detach().float()
    else:
        raise ValueError(
            f"Unsupported position_anchor_source={source!r}. "
            "Expected wtpe_pc1, wtpe_causal_pc1, wtpe_norm, wpe_pc1, "
            "wpe_causal_pc1, wpe_norm, target_index, or target_precedence."
        )
    block_features = token_features.view(num_blocks, block_len, -1).mean(dim=1)
    if feature_source.endswith("_norm"):
        scores = block_features.norm(dim=-1)
    elif feature_source.endswith("_pc1"):
        centered = block_features - block_features.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
        except Exception:
            return None
        axis = vh[0]
        scores = centered @ axis
        if causal_pc1:
            causal_scores = torch.arange(num_blocks, dtype=scores.dtype, device=scores.device)
            causal_scores = causal_scores - causal_scores.mean()
            if float((scores * causal_scores).sum().item()) < 0.0:
                scores = -scores
        elif scores.numel() > 0:
            max_idx = int(torch.argmax(scores.abs()).item())
            if float(scores[max_idx].item()) < 0.0:
                scores = -scores
    else:
        raise ValueError(
            f"Unsupported position_anchor_source={source!r}. "
            "Expected wtpe_pc1, wtpe_causal_pc1, wtpe_norm, wpe_pc1, "
            "wpe_causal_pc1, wpe_norm, target_index, or target_precedence."
        )
    scores = scores.detach().to(dtype=torch.float64, device="cpu")
    scores = scores - scores.mean()
    scale = float(scores.norm().item())
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    return (scores / scale).numpy()


def orient_order_with_position_anchor(
    raw_order: list[int],
    anchor_scores: np.ndarray | None,
    block_perm,
    source: str,
) -> dict | None:
    if anchor_scores is None:
        return None
    raw_order = [int(value) for value in raw_order]
    if len(raw_order) != len(anchor_scores):
        return None
    if str(source).strip().lower() in {"target_precedence", "causal_precedence"}:
        target_order = list(range(len(raw_order)))
        margin = float((precedence_matrix(raw_order) * precedence_matrix(target_order)).sum())
    else:
        ranks = np.zeros(len(raw_order), dtype=np.float64)
        for rank, block_idx in enumerate(raw_order):
            ranks[int(block_idx)] = float(rank)
        ranks = ranks - float(ranks.mean())
        anchor = np.asarray(anchor_scores, dtype=np.float64)
        anchor = anchor - float(anchor.mean())
        margin = float((ranks * anchor).sum())
    selected_reverse = bool(margin < 0.0)
    selected_order = list(reversed(raw_order)) if selected_reverse else raw_order
    selected_original = order_to_original(selected_order, block_perm)
    selected_tau = tau_to_l2r(selected_original)
    return {
        "position_anchor_source": str(source),
        "position_anchor_alignment_margin": float(margin),
        "position_anchor_selected_reverse": bool(selected_reverse),
        "position_anchor_tau_origin_l2r_diagnostic": float(selected_tau),
        "position_anchor_abs_tau_origin_l2r_diagnostic": abs(float(selected_tau)),
        "position_anchor_sign_diagnostic": int(sign_of(selected_tau)),
        "position_anchor_order_current": json.dumps(selected_order, separators=(",", ":")),
        "position_anchor_order_original": json.dumps(selected_original, separators=(",", ":")),
    }


def precedence_matrix(order: list[int]) -> np.ndarray:
    order = [int(value) for value in order]
    rank = {value: idx for idx, value in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=np.float64)
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            matrix[i, j] = 1.0 if rank[i] < rank[j] else -1.0
    return matrix


def apply_position_consensus(rows: list[dict], block_perm, leave_one_out: bool) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("ckpt_label", ""), row.get("repeat_idx", ""))].append(row)
    for group_rows in groups.values():
        signed = []
        weights = []
        indexed_rows = []
        for row in group_rows:
            if "raw_order_current" not in row or "position_anchor_selected_reverse" not in row:
                continue
            raw_order = json.loads(row["raw_order_current"])
            sign = -1.0 if str(row.get("position_anchor_selected_reverse")).lower() == "true" else 1.0
            margin = abs(float(row.get("position_anchor_alignment_margin", 0.0))) + 1e-4
            indexed_rows.append(row)
            signed.append(precedence_matrix(raw_order) * sign)
            weights.append(float(margin))
        if not signed:
            continue
        for idx, row in enumerate(indexed_rows):
            if bool(leave_one_out) and len(signed) > 1:
                denom = sum(float(weight) for j, weight in enumerate(weights) if j != idx)
                if denom <= 0.0:
                    denom = sum(float(weight) for weight in weights)
                q_matrix = (
                    sum(float(weights[j]) * signed[j] for j in range(len(signed)) if j != idx)
                    / float(max(denom, 1e-8))
                )
            else:
                denom = sum(float(weight) for weight in weights)
                q_matrix = sum(float(weight) * matrix for weight, matrix in zip(weights, signed)) / float(
                    max(denom, 1e-8)
                )
            raw_order = json.loads(row["raw_order_current"])
            align = float((precedence_matrix(raw_order) * q_matrix).sum())
            selected_reverse = bool(align < 0.0)
            selected_order = list(reversed(raw_order)) if selected_reverse else raw_order
            selected_original = order_to_original(selected_order, block_perm)
            selected_tau = tau_to_l2r(selected_original)
            row.update(
                {
                    "position_consensus_alignment_margin": float(align),
                    "position_consensus_selected_reverse": bool(selected_reverse),
                    "position_consensus_tau_origin_l2r_diagnostic": float(selected_tau),
                    "position_consensus_abs_tau_origin_l2r_diagnostic": abs(float(selected_tau)),
                    "position_consensus_sign_diagnostic": int(sign_of(selected_tau)),
                    "position_consensus_order_current": json.dumps(selected_order, separators=(",", ":")),
                    "position_consensus_order_original": json.dumps(selected_original, separators=(",", ":")),
                }
            )


def expand_orders(model, block_orders: torch.Tensor) -> torch.Tensor:
    return expand_block_orders_to_token_orders(
        block_orders,
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )


@torch.no_grad()
def evaluate_order_losses(
    *,
    model,
    tokens,
    order_list: list[list[int]],
    sample_count: int,
    batch_size: int,
    candidate_batch_size: int,
    seed: int,
    token_perm,
    device: str,
    ctx,
    prefix_k: int,
) -> dict[tuple[int, ...], dict]:
    unique_orders = []
    seen = set()
    for order in order_list:
        key = tuple(int(v) for v in order)
        if key not in seen:
            seen.add(key)
            unique_orders.append(list(key))
    if not unique_orders:
        return {}

    sums = {
        tuple(order): {
            "full_loss_sum": 0.0,
            "prefix_loss_sum": 0.0,
            "suffix_loss_sum": 0.0,
            "count": 0,
        }
        for order in unique_orders
    }
    order_tensor_cpu = torch.tensor(unique_orders, dtype=torch.long, device="cpu")
    rng = np.random.default_rng(int(seed))
    total = 0
    while total < int(sample_count):
        local_batch_size = min(int(batch_size), int(sample_count) - int(total))
        idx = sample_batch(
            tokens,
            local_batch_size,
            int(model.config.block_size),
            rng,
            device,
            token_perm=token_perm,
        )
        for start in range(0, len(unique_orders), int(candidate_batch_size)):
            end = min(len(unique_orders), start + int(candidate_batch_size))
            chunk_cpu = order_tensor_cpu[start:end]
            chunk_size = int(chunk_cpu.size(0))
            idx_flat = idx.unsqueeze(1).expand(-1, chunk_size, -1).reshape(local_batch_size * chunk_size, -1)
            block_orders = (
                chunk_cpu.to(device=device)
                .unsqueeze(0)
                .expand(local_batch_size, -1, -1)
                .reshape(local_batch_size * chunk_size, -1)
            )
            token_orders = expand_orders(model, block_orders)
            with ctx:
                _, _, token_losses = model(
                    idx_flat,
                    mode=None,
                    orders=token_orders,
                    return_token_loss=True,
                    return_logits=False,
                )
            block_losses = token_losses_to_block_losses(
                token_losses,
                block_len=int(model.block_order_block_len),
            ).float()
            prefix = max(1, min(int(prefix_k), int(block_losses.size(1))))
            full_loss = token_losses.float().mean(dim=1)
            prefix_loss = block_losses[:, :prefix].mean(dim=1)
            suffix_loss = block_losses[:, -prefix:].mean(dim=1)
            for local_idx, order_idx in enumerate(range(start, end)):
                view = slice(local_idx, local_batch_size * chunk_size, chunk_size)
                key = tuple(unique_orders[order_idx])
                sums[key]["full_loss_sum"] += float(full_loss[view].double().sum().item())
                sums[key]["prefix_loss_sum"] += float(prefix_loss[view].double().sum().item())
                sums[key]["suffix_loss_sum"] += float(suffix_loss[view].double().sum().item())
                sums[key]["count"] += int(local_batch_size)
        total += int(local_batch_size)

    out = {}
    for key, item in sums.items():
        count = max(1, int(item["count"]))
        out[key] = {
            "full_loss": float(item["full_loss_sum"] / count),
            "prefix_loss": float(item["prefix_loss_sum"] / count),
            "suffix_loss": float(item["suffix_loss_sum"] / count),
            "count": int(item["count"]),
        }
    return out


def summarize_method(rows: list[dict], method: str, tau_key: str, order_key: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    summaries = []
    for (layer, head), items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: int(row["repeat_idx"]))
        taus = [float(row[tau_key]) for row in items]
        signs = [sign_of(value) for value in taus]
        nonzero_signs = [value for value in signs if value != 0]
        sign_flips = sum(
            int(nonzero_signs[idx] != nonzero_signs[idx - 1])
            for idx in range(1, len(nonzero_signs))
        )
        positive_rate = float(np.mean([value > 0.0 for value in taus])) if taus else float("nan")
        tau_mean, tau_std, tau_min, tau_max = mean_std(taus)
        abs_mean, abs_std, abs_min, abs_max = mean_std([abs(value) for value in taus])
        orders = [json.loads(row[order_key]) for row in items]
        pairwise = pairwise_tau_summary(orders)
        loss_margins = [float(row.get("loss_selection_margin_reverse_minus_forward", float("nan"))) for row in items]
        loss_mean, loss_std, loss_min, loss_max = mean_std(loss_margins)
        summaries.append(
            {
                "method": str(method),
                "layer": int(layer),
                "head": int(head),
                "num_repeats": int(len(items)),
                "tau_mean": float(tau_mean),
                "tau_std": float(tau_std),
                "tau_min": float(tau_min),
                "tau_max": float(tau_max),
                "abs_tau_mean": float(abs_mean),
                "abs_tau_std": float(abs_std),
                "abs_tau_min": float(abs_min),
                "abs_tau_max": float(abs_max),
                "positive_rate": float(positive_rate),
                "sign_flips": int(sign_flips),
                "stable_sign": bool(len(set(nonzero_signs)) <= 1 and len(nonzero_signs) > 0),
                "loss_margin_mean": float(loss_mean),
                "loss_margin_std": float(loss_std),
                "loss_margin_min": float(loss_min),
                "loss_margin_max": float(loss_max),
                **pairwise,
            }
        )
    return summaries


def write_plots(out_dir: Path, head_summary: list[dict], per_probe_rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[head-orientation] plot import failed: {exc}")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(set(row["method"] for row in head_summary))
    layers = sorted(set(int(row["layer"]) for row in head_summary))
    heads = sorted(set(int(row["head"]) for row in head_summary))
    metrics = [
        ("tau_mean", "Mean tau", -1.0, 1.0, "coolwarm"),
        ("abs_tau_mean", "Mean abs tau", 0.0, 1.0, "viridis"),
        ("sign_flips", "Sign flips", 0.0, None, "magma"),
        ("pairwise_abs_order_tau_mean", "Pairwise abs order tau", 0.0, 1.0, "viridis"),
    ]
    for metric, title, vmin, vmax, cmap in metrics:
        fig, axes = plt.subplots(1, len(methods), figsize=(max(5, 4 * len(methods)), 3.8), squeeze=False)
        for ax, method in zip(axes[0], methods):
            heat = np.full((len(layers), len(heads)), np.nan)
            for row in head_summary:
                if row["method"] != method:
                    continue
                i = layers.index(int(row["layer"]))
                j = heads.index(int(row["head"]))
                heat[i, j] = float(row[metric])
            im = ax.imshow(heat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(method)
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_xticks(range(len(heads)), labels=[str(h) for h in heads])
            ax.set_yticks(range(len(layers)), labels=[str(l) for l in layers])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    selected = [row for row in per_probe_rows if "loss_selection_margin_reverse_minus_forward" in row]
    if selected:
        labels = []
        means = []
        for key, items in sorted(defaultdict(list, {
            (int(row["layer"]), int(row["head"])): [] for row in selected
        }).items()):
            _ = key, items
        grouped = defaultdict(list)
        for row in selected:
            grouped[(int(row["layer"]), int(row["head"]))].append(
                float(row["loss_selection_margin_reverse_minus_forward"])
            )
        for (layer, head), values in sorted(grouped.items()):
            labels.append(f"L{layer}H{head}")
            means.append(float(np.mean(values)))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.32), 4))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.bar(range(len(labels)), means)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
        ax.set_ylabel("reverse loss - forward loss")
        ax.set_title("Loss orientation margin by head")
        fig.tight_layout()
        fig.savefig(plots_dir / "loss_orientation_margin_by_head.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(Path(args.ckpt_path))
    model = build_model(checkpoint, str(args.device))
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, str(args.split), checkpoint)
    permutation_state = load_permutation_state(checkpoint, model)
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    block_perm = None if permutation_state is None else permutation_state["block_perm"]
    ctx = get_autocast_context(str(args.device), str(args.dtype))
    heads = parse_heads(str(args.heads), int(model.config.n_layer), int(model.config.n_head), int(args.max_heads))

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
    anchor_scores = None
    if not bool(args.disable_position_anchor):
        anchor_scores = position_anchor_scores(model, str(args.position_anchor_source))

    config_payload = {
        "ckpt_label": str(args.ckpt_label),
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "out_dir": str(out_dir),
        "dataset": str(args.dataset or checkpoint.get("config", {}).get("dataset")),
        "split": str(args.split),
        "data_dir": str(data_dir),
        "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
        "num_blocks": int(model.num_blocks),
        "block_order_block_len": int(model.block_order_block_len),
        "heads": [{"layer": int(layer), "head": int(head)} for layer, head in heads],
        "probe_budget": int(args.probe_budget),
        "loss_sample_count": int(args.loss_sample_count),
        "repeats": int(args.repeats),
        "fixed_angle_idx": int(args.fixed_angle_idx),
        "top_m": int(args.top_m),
        "batch_size": int(args.batch_size),
        "loss_batch_size": int(args.loss_batch_size),
        "candidate_batch_size": int(args.candidate_batch_size),
        "reveal_mode": str(args.reveal_mode),
        "attn_export_type": str(args.attn_export_type),
        "selection_rule": (
            "Recover an attention-spectral axis from the current head, then choose "
            "between S and reverse(S) by lower current-model conditional loss. "
            "Original L2R tau/PPL are diagnostic-only."
        ),
        "position_anchor": {
            "enabled": bool(anchor_scores is not None),
            "source": str(args.position_anchor_source),
        },
        "position_consensus": {
            "enabled": bool(args.position_consensus_enabled),
            "leave_one_out": bool(args.position_consensus_leave_one_out),
        },
        "loss_selection": {
            "prefix_k": int(args.loss_prefix_k),
            "prefix_weight": float(args.loss_prefix_weight),
            "full_weight": float(args.loss_full_weight),
            "tie_flow_lambda": float(args.flow_direction_lambda),
        },
        "policy_config": vars(policy_config),
    }
    (out_dir / "config.json").write_text(json.dumps(json_safe(config_payload), indent=2), encoding="utf-8")

    per_probe_rows = []
    per_candidate_rows = []
    for repeat_idx in range(int(args.repeats)):
        probe_seed = int(args.seed) + int(args.seed_stride) * int(repeat_idx)
        order_seed = int(args.order_seed) + int(args.seed_stride) * int(repeat_idx)
        loss_seed = int(args.loss_seed) + int(args.seed_stride) * int(repeat_idx)
        print(
            "[head-orientation] mining attention "
            f"repeat={repeat_idx + 1}/{int(args.repeats)} "
            f"probe_seed={probe_seed} order_seed={order_seed}",
            flush=True,
        )
        matrices = mine_attention_matrices(
            model,
            tokens,
            sample_count=int(args.probe_budget),
            batch_size=int(args.batch_size),
            split_seed=probe_seed,
            order_seed=order_seed,
            reveal_mode=str(args.reveal_mode),
            token_perm=token_perm,
            device=str(args.device),
            ctx=ctx,
            export_type=str(args.attn_export_type),
        )

        repeat_items = []
        order_loss_requests = []
        for layer, head in heads:
            matrix = matrices[int(layer), int(head)]
            stats = matrix_stats(matrix)
            try:
                candidates, eigvals = recover_candidates_for_angles(
                    matrix,
                    policy_config,
                    num_angles=int(args.num_angles),
                    angle_indices=[int(args.fixed_angle_idx)],
                    top_m=int(args.top_m),
                )
            except Exception as exc:
                row = {
                    "ckpt_label": str(args.ckpt_label),
                    "ckpt_iter": int(checkpoint.get("iter_num", -1)),
                    "repeat_idx": int(repeat_idx),
                    "probe_seed": int(probe_seed),
                    "order_seed": int(order_seed),
                    "loss_seed": int(loss_seed),
                    "layer": int(layer),
                    "head": int(head),
                    "error": str(exc),
                }
                row.update(stats)
                per_probe_rows.append(row)
                continue
            top = candidates[0]
            raw_order = [int(value) for value in top["order"]]
            reverse_order = list(reversed(raw_order))
            order_loss_requests.extend([raw_order, reverse_order])
            eigvals_list = [float(value) for value in np.asarray(eigvals).tolist()]
            repeat_items.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "matrix": matrix,
                    "stats": stats,
                    "candidates": candidates,
                    "raw_order": raw_order,
                    "reverse_order": reverse_order,
                    "eigvals_first8": eigvals_list[:8],
                }
            )
            for rank, candidate in enumerate(candidates, start=1):
                order_current = [int(value) for value in candidate["order"]]
                order_original = order_to_original(order_current, block_perm)
                per_candidate_rows.append(
                    {
                        "ckpt_label": str(args.ckpt_label),
                        "ckpt_iter": int(checkpoint.get("iter_num", -1)),
                        "repeat_idx": int(repeat_idx),
                        "layer": int(layer),
                        "head": int(head),
                        "candidate_rank": int(rank),
                        "candidate_score": float(candidate.get("score", float("nan"))),
                        "candidate_tau_origin_l2r_diagnostic": float(tau_to_l2r(order_original)),
                        "candidate_abs_tau_origin_l2r_diagnostic": abs(float(tau_to_l2r(order_original))),
                        "candidate_order_current": json.dumps(order_current, separators=(",", ":")),
                        "candidate_order_original": json.dumps(order_original, separators=(",", ":")),
                    }
                )

        print(
            "[head-orientation] evaluating current-model order/reverse losses "
            f"repeat={repeat_idx + 1}/{int(args.repeats)} unique_orders={len({tuple(o) for o in order_loss_requests})}",
            flush=True,
        )
        loss_by_order = evaluate_order_losses(
            model=model,
            tokens=tokens,
            order_list=order_loss_requests,
            sample_count=int(args.loss_sample_count),
            batch_size=int(args.loss_batch_size),
            candidate_batch_size=int(args.candidate_batch_size),
            seed=loss_seed,
            token_perm=token_perm,
            device=str(args.device),
            ctx=ctx,
            prefix_k=int(args.loss_prefix_k),
        )

        for item in repeat_items:
            layer = int(item["layer"])
            head = int(item["head"])
            raw_order = item["raw_order"]
            reverse_order = item["reverse_order"]
            raw_loss = loss_by_order[tuple(raw_order)]
            reverse_loss = loss_by_order[tuple(reverse_order)]
            raw_selection_loss = (
                float(args.loss_prefix_weight) * float(raw_loss["prefix_loss"])
                + float(args.loss_full_weight) * float(raw_loss["full_loss"])
            )
            reverse_selection_loss = (
                float(args.loss_prefix_weight) * float(reverse_loss["prefix_loss"])
                + float(args.loss_full_weight) * float(reverse_loss["full_loss"])
            )
            selection_margin = float(reverse_selection_loss - raw_selection_loss)
            selected_is_reverse = bool(selection_margin < 0.0)
            if abs(selection_margin) <= float(args.loss_tie_epsilon):
                flows = flow_margins(
                    item["matrix"],
                    raw_order,
                    float(args.flow_direction_lambda),
                    str(args.score_adjacency_sym),
                )
                selected_is_reverse = bool(float(flows["flow_kq_margin_forward_minus_reverse"]) < 0.0)
            else:
                flows = flow_margins(
                    item["matrix"],
                    raw_order,
                    float(args.flow_direction_lambda),
                    str(args.score_adjacency_sym),
                )
            selected_order = reverse_order if selected_is_reverse else raw_order
            raw_original = order_to_original(raw_order, block_perm)
            selected_original = order_to_original(selected_order, block_perm)
            raw_tau = tau_to_l2r(raw_original)
            selected_tau = tau_to_l2r(selected_original)
            position_anchor_payload = orient_order_with_position_anchor(
                raw_order,
                anchor_scores,
                block_perm,
                str(args.position_anchor_source),
            )
            top = item["candidates"][0]
            top_meta = top.get("meta", {})
            score_margin = float("nan")
            if len(item["candidates"]) > 1:
                score_margin = float(item["candidates"][0]["score"]) - float(item["candidates"][1]["score"])
            row = {
                "ckpt_label": str(args.ckpt_label),
                "ckpt_iter": int(checkpoint.get("iter_num", -1)),
                "split": str(args.split),
                "repeat_idx": int(repeat_idx),
                "probe_seed": int(probe_seed),
                "order_seed": int(order_seed),
                "loss_seed": int(loss_seed),
                "layer": int(layer),
                "head": int(head),
                "probe_budget": int(args.probe_budget),
                "loss_sample_count": int(args.loss_sample_count),
                "fixed_angle_idx": int(args.fixed_angle_idx),
                "top_m": int(args.top_m),
                "raw_top1_score": float(top.get("score", float("nan"))),
                "raw_score_margin_top1_top2": float(score_margin),
                "raw_tau_origin_l2r_diagnostic": float(raw_tau),
                "raw_abs_tau_origin_l2r_diagnostic": abs(float(raw_tau)),
                "raw_sign_diagnostic": int(sign_of(raw_tau)),
                "loss_oriented_tau_origin_l2r_diagnostic": float(selected_tau),
                "loss_oriented_abs_tau_origin_l2r_diagnostic": abs(float(selected_tau)),
                "loss_oriented_sign_diagnostic": int(sign_of(selected_tau)),
                "loss_selected_reverse": bool(selected_is_reverse),
                "loss_selection_margin_reverse_minus_forward": float(selection_margin),
                "raw_selection_loss": float(raw_selection_loss),
                "reverse_selection_loss": float(reverse_selection_loss),
                "raw_prefix_loss": float(raw_loss["prefix_loss"]),
                "reverse_prefix_loss": float(reverse_loss["prefix_loss"]),
                "raw_full_loss": float(raw_loss["full_loss"]),
                "reverse_full_loss": float(reverse_loss["full_loss"]),
                "raw_suffix_loss": float(raw_loss["suffix_loss"]),
                "reverse_suffix_loss": float(reverse_loss["suffix_loss"]),
                "loss_eval_count": int(raw_loss["count"]),
                "selected_primary_reverse": bool(top_meta.get("primary_reverse", False)),
                "selected_secondary_reverse": bool(top_meta.get("secondary_reverse", False)),
                "selected_group_order_reverse": bool(top_meta.get("group_order_reverse", False)),
                "selected_directed_path_score": float(top_meta.get("best_directed_path_score", float("nan"))),
                "selected_adjacency_path_score": float(top_meta.get("adjacency_path_score", float("nan"))),
                "eigvals_first8": json.dumps(item["eigvals_first8"], separators=(",", ":")),
                "raw_order_current": json.dumps(raw_order, separators=(",", ":")),
                "raw_order_original": json.dumps(raw_original, separators=(",", ":")),
                "loss_oriented_order_current": json.dumps(selected_order, separators=(",", ":")),
                "loss_oriented_order_original": json.dumps(selected_original, separators=(",", ":")),
            }
            if position_anchor_payload is not None:
                row.update(position_anchor_payload)
            row.update(item["stats"])
            row.update(flows)
            per_probe_rows.append(row)
        write_csv(out_dir / "per_probe_results.csv", per_probe_rows)
        write_csv(out_dir / "per_candidate_topk.csv", per_candidate_rows)

    if bool(args.position_consensus_enabled):
        apply_position_consensus(
            per_probe_rows,
            block_perm,
            leave_one_out=bool(args.position_consensus_leave_one_out),
        )

    ok_rows = [row for row in per_probe_rows if "raw_tau_origin_l2r_diagnostic" in row]
    raw_summary = summarize_method(ok_rows, "raw_attention_top1", "raw_tau_origin_l2r_diagnostic", "raw_order_current")
    loss_summary = summarize_method(
        ok_rows,
        "loss_oriented",
        "loss_oriented_tau_origin_l2r_diagnostic",
        "loss_oriented_order_current",
    )
    position_anchor_summary = []
    if any("position_anchor_tau_origin_l2r_diagnostic" in row for row in ok_rows):
        position_anchor_summary = summarize_method(
            ok_rows,
            "position_anchor",
            "position_anchor_tau_origin_l2r_diagnostic",
            "position_anchor_order_current",
        )
    position_consensus_summary = []
    if any("position_consensus_tau_origin_l2r_diagnostic" in row for row in ok_rows):
        position_consensus_summary = summarize_method(
            ok_rows,
            "position_consensus",
            "position_consensus_tau_origin_l2r_diagnostic",
            "position_consensus_order_current",
        )
    head_summary = raw_summary + loss_summary + position_anchor_summary + position_consensus_summary
    write_csv(out_dir / "per_probe_results.csv", per_probe_rows)
    write_csv(out_dir / "per_candidate_topk.csv", per_candidate_rows)
    write_csv(out_dir / "head_summary.csv", head_summary)
    write_plots(out_dir, head_summary, ok_rows)

    summary_payload = {
        "out_dir": str(out_dir),
        "ckpt_label": str(args.ckpt_label),
        "ckpt_iter": int(checkpoint.get("iter_num", -1)),
        "num_probe_rows": int(len(per_probe_rows)),
        "num_ok_rows": int(len(ok_rows)),
        "num_candidate_rows": int(len(per_candidate_rows)),
        "head_summary": str(out_dir / "head_summary.csv"),
        "per_probe_results": str(out_dir / "per_probe_results.csv"),
        "method_totals": {},
    }
    for method in sorted(set(row["method"] for row in head_summary)):
        rows = [row for row in head_summary if row["method"] == method]
        summary_payload["method_totals"][method] = {
            "stable_sign_heads": int(sum(1 for row in rows if bool(row["stable_sign"]))),
            "total_sign_flips": int(sum(int(row["sign_flips"]) for row in rows)),
            "mean_abs_tau": float(np.mean([float(row["abs_tau_mean"]) for row in rows])) if rows else float("nan"),
            "heads_abs_tau_ge_0.4": int(sum(float(row["abs_tau_mean"]) >= 0.4 for row in rows)),
            "heads_abs_tau_ge_0.5": int(sum(float(row["abs_tau_mean"]) >= 0.5 for row in rows)),
            "mean_pairwise_abs_order_tau": (
                float(np.mean([float(row["pairwise_abs_order_tau_mean"]) for row in rows]))
                if rows
                else float("nan")
            ),
        }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary_payload), indent=2), encoding="utf-8")
    print(json.dumps(json_safe(summary_payload), indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--ckpt_label", type=str, default="checkpoint")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--probe_budget", type=int, default=4096)
    parser.add_argument("--loss_sample_count", type=int, default=1024)
    parser.add_argument("--loss_batch_size", type=int, default=16)
    parser.add_argument("--candidate_batch_size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--order_seed", type=int, default=22345)
    parser.add_argument("--loss_seed", type=int, default=32345)
    parser.add_argument("--seed_stride", type=int, default=1009)
    parser.add_argument("--reveal_mode", type=str, default="Random", choices=("Random", "AR"))
    parser.add_argument("--attn_export_type", type=str, default="with_none", choices=("with_none", "without_none"))
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--max_heads", type=int, default=0)
    parser.add_argument("--fixed_angle_idx", type=int, default=4)
    parser.add_argument("--num_angles", type=int, default=16)
    parser.add_argument("--top_m", type=int, default=32)
    parser.add_argument("--loss_prefix_k", type=int, default=16)
    parser.add_argument("--loss_prefix_weight", type=float, default=0.7)
    parser.add_argument("--loss_full_weight", type=float, default=0.3)
    parser.add_argument("--loss_tie_epsilon", type=float, default=1e-6)
    parser.add_argument("--flow_direction_lambda", type=float, default=0.25)
    parser.add_argument("--position_anchor_source", type=str, default="wtpe_pc1")
    parser.add_argument("--disable_position_anchor", action="store_true")
    parser.add_argument("--position_consensus_enabled", action="store_true")
    parser.add_argument("--position_consensus_leave_one_out", action="store_true", default=True)
    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument("--component_pairs", type=str, default="1-2")
    parser.add_argument("--k_values", type=str, default="8,10")
    parser.add_argument("--group_methods", type=str, default="gap")
    parser.add_argument("--threshold_percentile", type=float, default=60.0)
    parser.add_argument("--transform", type=str, default="relu", choices=("relu", "exp", "softplus"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--direction_lambdas", type=str, default="0,0.1,0.25")
    parser.add_argument("--directed_score_weight", type=float, default=0.25)
    parser.add_argument("--band_quality_weight", type=float, default=0.05)
    parser.add_argument("--score_adjacency_sym", type=str, default="max", choices=("max", "mean"))
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
