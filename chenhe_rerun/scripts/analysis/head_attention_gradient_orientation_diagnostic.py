#!/usr/bin/env python3
"""Head-order orientation from current attention-loss gradients.

This is a no-prior diagnostic: it reads raw recovered orders, evaluates each
order and its reverse on current samples, and chooses the orientation whose
selected head has stronger loss-reducing attention saliency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from head_angle_tau_diagnostic import (  # noqa: E402
    build_model,
    extract_attentions_from_outputs,
    get_autocast_context,
    load_checkpoint,
    load_permutation_state,
    load_tokens,
    order_to_original,
    parse_heads,
    resolve_data_dir,
    sample_batch,
    tau_to_l2r,
)
from head_orientation_loss_flow_diagnostic import (  # noqa: E402
    pairwise_tau_summary,
    precedence_matrix,
)
from order_utils import expand_block_orders_to_token_orders, invert_permutation  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _sign(value: float, eps: float = 1e-9) -> int:
    value = float(value)
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _parse_order(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(v) for v in json.loads(str(value))]


def _expand_block_order(model, block_orders: torch.Tensor) -> torch.Tensor:
    return expand_block_orders_to_token_orders(
        block_orders,
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )


def _block_saliency_from_attention(attn: torch.Tensor, grad: torch.Tensor, block_orders: torch.Tensor, block_len: int) -> np.ndarray:
    # Positive entries mean "increasing this attention probability should reduce loss".
    saliency = (-grad.float() * attn.detach().float())[:, :, :-1, :-1]
    batch, heads, seq_a, seq_b = saliency.shape
    if seq_a != seq_b or seq_a % int(block_len) != 0:
        raise ValueError(f"attention shape {tuple(saliency.shape)} incompatible with block_len={block_len}")
    num_blocks = seq_a // int(block_len)
    block_batch = saliency.view(
        batch,
        heads,
        num_blocks,
        int(block_len),
        num_blocks,
        int(block_len),
    ).mean(dim=(3, 5))
    total = torch.zeros((num_blocks, num_blocks), dtype=torch.float64, device=saliency.device)
    for sample_idx in range(batch):
        inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=saliency.device)
        total += block_batch[sample_idx][:, inverse, :][:, :, inverse].mean(dim=0).double()
    out = (total / float(max(1, batch))).detach().cpu().numpy()
    np.fill_diagonal(out, 0.0)
    return out


def _score_order_gradient(
    *,
    model,
    tokens,
    token_perm,
    order: list[int],
    layer: int,
    head: int,
    sample_count: int,
    batch_size: int,
    seed: int,
    device: str,
    ctx,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    order_tensor_single = torch.tensor(order, dtype=torch.long, device=device)
    total_loss = 0.0
    total_score = 0.0
    total_score_grad_only = 0.0
    total_count = 0
    block_len = int(model.block_order_block_len)
    p_matrix = precedence_matrix(order)

    was_training = model.training
    model.eval()
    while total_count < int(sample_count):
        local_batch = min(int(batch_size), int(sample_count) - int(total_count))
        idx = sample_batch(
            tokens,
            local_batch,
            int(model.config.block_size),
            rng,
            device,
            token_perm=token_perm,
        )
        block_orders = order_tensor_single.unsqueeze(0).expand(local_batch, -1)
        token_orders = _expand_block_order(model, block_orders)
        model.zero_grad(set_to_none=True)
        with ctx:
            outputs = model(
                idx,
                mode=None,
                orders=token_orders,
                return_attentions=True,
                return_logits=True,
            )
            loss = outputs[1]
        attentions = extract_attentions_from_outputs(outputs)
        selected_attn = attentions[int(layer)]
        selected_attn.retain_grad()
        loss.backward()
        grad = selected_attn.grad
        if grad is None:
            raise RuntimeError("attention gradient was not retained")
        saliency = _block_saliency_from_attention(
            selected_attn[:, int(head) : int(head) + 1],
            grad[:, int(head) : int(head) + 1],
            block_orders,
            block_len,
        )
        grad_only = _block_saliency_from_attention(
            selected_attn[:, int(head) : int(head) + 1].detach() + 1.0,
            grad[:, int(head) : int(head) + 1],
            block_orders,
            block_len,
        )
        anti_saliency = saliency - saliency.T
        anti_grad = grad_only - grad_only.T
        total_score += float((p_matrix * anti_saliency).sum()) * float(local_batch)
        total_score_grad_only += float((p_matrix * anti_grad).sum()) * float(local_batch)
        total_loss += float(loss.detach().item()) * float(local_batch)
        total_count += int(local_batch)
        model.zero_grad(set_to_none=True)

    if was_training:
        model.train()
    count = max(1, int(total_count))
    return {
        "loss": float(total_loss / count),
        "grad_attn_precedence_score": float(total_score / count),
        "grad_only_precedence_score": float(total_score_grad_only / count),
        "count": int(total_count),
    }


def _summarize(rows: list[dict[str, Any]], method_prefix: str) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["layer"]), int(row["head"]))].append(row)
    out = []
    for (layer, head), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: int(r.get("repeat_idx", 0)))
        taus = [float(row[f"{method_prefix}_tau_origin_l2r_diagnostic"]) for row in items]
        signs = [_sign(v) for v in taus]
        nonzero = [s for s in signs if s != 0]
        flips = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a * b < 0)
        orders = [_parse_order(row[f"{method_prefix}_order_current"]) for row in items]
        pairwise = pairwise_tau_summary(orders)
        out.append(
            {
                "method": method_prefix,
                "layer": int(layer),
                "head": int(head),
                "head_id": f"L{layer}H{head}",
                "num_points": int(len(items)),
                "tau_mean": float(np.mean(taus)),
                "tau_std": float(np.std(taus)),
                "abs_tau_mean": float(np.mean([abs(v) for v in taus])),
                "positive_rate": float(sum(1 for s in signs if s > 0) / max(1, len(signs))),
                "negative_rate": float(sum(1 for s in signs if s < 0) / max(1, len(signs))),
                "sign_flips": int(flips),
                "stable_sign": int(flips == 0 and bool(nonzero)),
                **pairwise,
            }
        )
    return out


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
    selected_heads = set(parse_heads(str(args.heads), int(model.config.n_layer), int(model.config.n_head), 0))

    input_rows = _read_csv(Path(args.input_csv))
    probe_rows = []
    for row in input_rows:
        layer = int(row["layer"])
        head = int(row["head"])
        if (layer, head) not in selected_heads:
            continue
        raw_order = _parse_order(row["raw_order_current"])
        reverse_order = list(reversed(raw_order))
        repeat_idx = int(row.get("repeat_idx", len(probe_rows)))
        seed_base = int(args.seed) + int(args.seed_stride) * repeat_idx + 97 * layer + head
        print(
            f"[grad-orient] repeat={repeat_idx} L{layer}H{head} samples={int(args.sample_count)}",
            flush=True,
        )
        raw_score = _score_order_gradient(
            model=model,
            tokens=tokens,
            token_perm=token_perm,
            order=raw_order,
            layer=layer,
            head=head,
            sample_count=int(args.sample_count),
            batch_size=int(args.batch_size),
            seed=seed_base,
            device=str(args.device),
            ctx=ctx,
        )
        reverse_score = _score_order_gradient(
            model=model,
            tokens=tokens,
            token_perm=token_perm,
            order=reverse_order,
            layer=layer,
            head=head,
            sample_count=int(args.sample_count),
            batch_size=int(args.batch_size),
            seed=seed_base,
            device=str(args.device),
            ctx=ctx,
        )
        margin = float(reverse_score["grad_attn_precedence_score"] - raw_score["grad_attn_precedence_score"])
        selected_reverse = bool(margin > 0.0)
        selected_order = reverse_order if selected_reverse else raw_order
        selected_original = order_to_original(selected_order, block_perm)
        selected_tau = tau_to_l2r(selected_original)

        loss_margin = float(reverse_score["loss"] - raw_score["loss"])
        loss_selected_reverse = bool(loss_margin < 0.0)
        loss_order = reverse_order if loss_selected_reverse else raw_order
        loss_original = order_to_original(loss_order, block_perm)
        loss_tau = tau_to_l2r(loss_original)

        probe_rows.append(
            {
                "ckpt_label": str(args.ckpt_label),
                "ckpt_iter": int(checkpoint.get("iter_num", -1)),
                "split": str(args.split),
                "repeat_idx": int(repeat_idx),
                "layer": int(layer),
                "head": int(head),
                "sample_count": int(args.sample_count),
                "raw_order_current": json.dumps(raw_order, separators=(",", ":")),
                "reverse_order_current": json.dumps(reverse_order, separators=(",", ":")),
                "raw_grad_attn_precedence_score": float(raw_score["grad_attn_precedence_score"]),
                "reverse_grad_attn_precedence_score": float(reverse_score["grad_attn_precedence_score"]),
                "grad_attn_margin_reverse_minus_forward": float(margin),
                "grad_attn_selected_reverse": bool(selected_reverse),
                "grad_attn_tau_origin_l2r_diagnostic": float(selected_tau),
                "grad_attn_abs_tau_origin_l2r_diagnostic": abs(float(selected_tau)),
                "grad_attn_sign_diagnostic": int(_sign(selected_tau)),
                "grad_attn_order_current": json.dumps(selected_order, separators=(",", ":")),
                "grad_attn_order_original": json.dumps(selected_original, separators=(",", ":")),
                "raw_grad_only_precedence_score": float(raw_score["grad_only_precedence_score"]),
                "reverse_grad_only_precedence_score": float(reverse_score["grad_only_precedence_score"]),
                "raw_loss": float(raw_score["loss"]),
                "reverse_loss": float(reverse_score["loss"]),
                "loss_margin_reverse_minus_forward": float(loss_margin),
                "loss_selected_reverse": bool(loss_selected_reverse),
                "loss_tau_origin_l2r_diagnostic": float(loss_tau),
                "loss_abs_tau_origin_l2r_diagnostic": abs(float(loss_tau)),
                "loss_sign_diagnostic": int(_sign(loss_tau)),
                "loss_order_current": json.dumps(loss_order, separators=(",", ":")),
                "loss_order_original": json.dumps(loss_original, separators=(",", ":")),
            }
        )

    summary_rows = _summarize(probe_rows, "grad_attn") + _summarize(probe_rows, "loss")
    _write_csv(out_dir / "per_probe_results.csv", probe_rows)
    _write_csv(out_dir / "head_summary.csv", summary_rows)
    global_summary = {}
    for method in ["grad_attn", "loss"]:
        items = [row for row in summary_rows if row["method"] == method]
        global_summary[method] = {
            "heads": int(len(items)),
            "stable_sign_heads": int(sum(int(row["stable_sign"]) for row in items)),
            "total_sign_flips": int(sum(int(row["sign_flips"]) for row in items)),
            "mean_abs_tau": float(np.mean([float(row["abs_tau_mean"]) for row in items])) if items else float("nan"),
            "mean_pairwise_abs_order_tau": float(np.mean([float(row["pairwise_abs_order_tau_mean"]) for row in items])) if items else float("nan"),
        }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe({"global_summary": global_summary}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(global_summary), indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--ckpt_label", default="ckpt")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=777123)
    parser.add_argument("--seed_stride", type=int, default=1009)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=("float32", "float16", "bfloat16"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
