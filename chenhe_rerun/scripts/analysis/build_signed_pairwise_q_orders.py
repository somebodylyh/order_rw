#!/usr/bin/env python3
"""Build signed multi-head pairwise-Q orders from saved attention matrices.

This script is a diagnostic/order-construction utility. It never uses original
L2R, Kendall tau, or validation loss to choose head signs. Those can be computed
later as diagnostics when the produced fixed orders are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from online_spectral_order_policy import robust_z  # noqa: E402
from order_utils import build_fixed_block_permutation, invert_permutation  # noqa: E402


@dataclass
class HeadMatrix:
    label: str
    layer: int
    head: int
    path: Path
    matrix: np.ndarray
    d_z: np.ndarray
    d_thresh: np.ndarray
    norm_z: float
    norm_thresh: float


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def load_model_meta(checkpoint: Dict):
    model = AOGPT(AOGPTConfig(**dict(checkpoint["model_args"])))
    return model


def permutation_state(checkpoint: Dict, model):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    block_len = int(model.block_order_block_len)
    num_blocks = int(model.config.block_size) // block_len
    raw = (checkpoint.get("data_permutation") or {}).get("block_perm")
    if raw is not None:
        block_perm = torch.as_tensor(raw, dtype=torch.long)
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    return {
        "block_perm": block_perm,
        "inverse_block_perm": invert_permutation(block_perm),
    }


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


def to_original_order(order_current: List[int], perm_state) -> List[int]:
    if perm_state is None:
        return [int(v) for v in order_current]
    mapper = perm_state["block_perm"].to(dtype=torch.long, device="cpu")
    current = torch.as_tensor(order_current, dtype=torch.long)
    return [int(v) for v in mapper[current].tolist()]


def offdiag_values(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[mask]
    return values[np.isfinite(values)]


def threshold_positive(z: np.ndarray, percentile: float) -> np.ndarray:
    values = offdiag_values(z)
    threshold = float(np.percentile(values, float(percentile))) if values.size else 0.0
    out = np.maximum(z - threshold, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def normalize_q(q: np.ndarray) -> Tuple[np.ndarray, float]:
    q = np.asarray(q, dtype=np.float64)
    mask = ~np.eye(q.shape[0], dtype=bool)
    norm = float(np.linalg.norm(q[mask]))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros_like(q), 0.0
    return q / norm, norm


def load_head(label: str, matrix_path: Path, percentile: float) -> HeadMatrix:
    matrix = np.load(matrix_path).astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{matrix_path} is not a square matrix: shape={matrix.shape}")
    z = robust_z(matrix)
    z = np.where(np.isfinite(z), z, 0.0)
    np.fill_diagonal(z, 0.0)
    b = threshold_positive(z, percentile)
    d_z, norm_z = normalize_q(z - z.T)
    d_thresh, norm_thresh = normalize_q(b - b.T)
    lower = label.lower()
    if "l" in lower and "h" in lower:
        try:
            layer = int(lower.split("l", 1)[1].split("h", 1)[0])
            head = int(lower.split("h", 1)[1].split("_", 1)[0])
        except Exception:
            layer = -1
            head = -1
    else:
        layer = -1
        head = -1
    return HeadMatrix(
        label=label,
        layer=layer,
        head=head,
        path=matrix_path,
        matrix=matrix,
        d_z=d_z,
        d_thresh=d_thresh,
        norm_z=norm_z,
        norm_thresh=norm_thresh,
    )


def valid_order_from_priority(priority: np.ndarray, descending: bool = True) -> List[int]:
    priority = np.asarray(priority, dtype=np.float64)
    safe = np.where(np.isfinite(priority), priority, 0.0)
    idx = np.argsort(safe, kind="mergesort")
    if descending:
        idx = idx[::-1]
    return [int(v) for v in idx.tolist()]


def write_order_json(path: Path, name: str, order: List[int], meta: Dict):
    payload = {
        "name": str(name),
        "order": [int(v) for v in order],
        "best_candidate": {
            "name": str(name),
            "order": [int(v) for v in order],
            "meta": meta,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def flatten_q(q: np.ndarray) -> np.ndarray:
    mask = ~np.eye(q.shape[0], dtype=bool)
    return np.asarray(q, dtype=np.float64)[mask]


def dot_q(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(flatten_q(a), flatten_q(b)))


def aggregate_signed(
    heads: List[HeadMatrix],
    q_kind: str,
    signs: Dict[str, int],
    weight_mode: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    qs = []
    weights = []
    for head in heads:
        q = head.d_z if q_kind == "z" else head.d_thresh
        raw_norm = head.norm_z if q_kind == "z" else head.norm_thresh
        sign = int(signs[head.label])
        if weight_mode == "strength":
            weight = max(float(raw_norm), 1e-12)
        elif weight_mode == "uniform":
            weight = 1.0
        else:
            raise ValueError(f"Unsupported weight_mode={weight_mode!r}")
        qs.append(sign * q * weight)
        weights.append(weight)
    denom = float(np.sum(weights)) if weights else 1.0
    q_total = np.sum(qs, axis=0) / max(denom, 1e-12)
    np.fill_diagonal(q_total, 0.0)
    return q_total, {head.label: float(weight) for head, weight in zip(heads, weights)}


def signs_by_anchor(heads: List[HeadMatrix], q_kind: str, anchor_label: str | None) -> Tuple[Dict[str, int], str]:
    if anchor_label is None:
        anchor = max(heads, key=lambda h: h.norm_z if q_kind == "z" else h.norm_thresh)
    else:
        matches = [head for head in heads if head.label == anchor_label]
        if not matches:
            raise ValueError(f"Anchor {anchor_label!r} is not in head set {[h.label for h in heads]}")
        anchor = matches[0]
    anchor_q = anchor.d_z if q_kind == "z" else anchor.d_thresh
    signs = {}
    for head in heads:
        q = head.d_z if q_kind == "z" else head.d_thresh
        signs[head.label] = 1 if dot_q(q, anchor_q) >= 0.0 else -1
    signs[anchor.label] = 1
    return signs, anchor.label


def signs_by_svd(heads: List[HeadMatrix], q_kind: str) -> Tuple[Dict[str, int], str]:
    matrix = np.stack([flatten_q(head.d_z if q_kind == "z" else head.d_thresh) for head in heads], axis=0)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    axis = vt[0]
    anchor = max(heads, key=lambda h: h.norm_z if q_kind == "z" else h.norm_thresh)
    anchor_vec = flatten_q(anchor.d_z if q_kind == "z" else anchor.d_thresh)
    if float(np.dot(anchor_vec, axis)) < 0.0:
        axis = -axis
    signs = {}
    for head, row in zip(heads, matrix):
        signs[head.label] = 1 if float(np.dot(row, axis)) >= 0.0 else -1
    signs[anchor.label] = 1
    return signs, anchor.label


def signs_by_leave_one_out(heads: List[HeadMatrix], q_kind: str, max_iter: int = 20) -> Tuple[Dict[str, int], str]:
    anchor = max(heads, key=lambda h: h.norm_z if q_kind == "z" else h.norm_thresh)
    anchor_q = anchor.d_z if q_kind == "z" else anchor.d_thresh
    signs = {}
    for head in heads:
        q = head.d_z if q_kind == "z" else head.d_thresh
        signs[head.label] = 1 if dot_q(q, anchor_q) >= 0.0 else -1
    signs[anchor.label] = 1
    for _ in range(max(1, int(max_iter))):
        changed = False
        for head in heads:
            if head.label == anchor.label:
                continue
            q = head.d_z if q_kind == "z" else head.d_thresh
            others = []
            for other in heads:
                if other.label == head.label:
                    continue
                other_q = other.d_z if q_kind == "z" else other.d_thresh
                others.append(signs[other.label] * other_q)
            consensus = np.mean(others, axis=0)
            new_sign = 1 if dot_q(q, consensus) >= 0.0 else -1
            if new_sign != signs[head.label]:
                signs[head.label] = new_sign
                changed = True
        if not changed:
            break
    signs[anchor.label] = 1
    return signs, anchor.label


def build_methods(heads: List[HeadMatrix], set_name: str, q_kind: str, weight_mode: str) -> Iterable[Tuple[str, Dict[str, int], str, str]]:
    signs, anchor = signs_by_anchor(heads, q_kind, anchor_label=None)
    yield f"{set_name}_anchor_maxnorm_{q_kind}_{weight_mode}", signs, anchor, "anchor_maxnorm"

    if any(head.label == "L0H7" for head in heads):
        signs, anchor = signs_by_anchor(heads, q_kind, anchor_label="L0H7")
        yield f"{set_name}_anchor_l0h7_{q_kind}_{weight_mode}", signs, anchor, "anchor_l0h7"

    signs, anchor = signs_by_svd(heads, q_kind)
    yield f"{set_name}_svd_consensus_{q_kind}_{weight_mode}", signs, anchor, "svd_consensus"

    signs, anchor = signs_by_leave_one_out(heads, q_kind)
    yield f"{set_name}_loo_consensus_{q_kind}_{weight_mode}", signs, anchor, "leave_one_out_consensus"


def save_heatmap(path: Path, matrix: np.ndarray, title: str):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("block j")
        ax.set_ylabel("block i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc))


def parse_head_arg(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--head_matrix entries must be LABEL=PATH")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build signed pairwise-Q consensus orders from saved head matrices.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--head_matrix", action="append", required=True, help="LABEL=attention_matrix_current.npy")
    parser.add_argument("--set", dest="sets", action="append", required=True, help="SET_NAME:LABEL,LABEL,...")
    parser.add_argument("--threshold_percentile", type=float, default=60.0)
    parser.add_argument("--write_reverse", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    order_dir = args.out_dir / "orders"
    order_dir.mkdir(exist_ok=True)
    plot_dir = args.out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    ckpt = load_checkpoint(args.ckpt_path)
    model = load_model_meta(ckpt)
    perm_state = permutation_state(ckpt, model)
    num_blocks = int(model.num_blocks)

    head_map: Dict[str, HeadMatrix] = {}
    for raw in args.head_matrix:
        label, path = parse_head_arg(raw)
        head = load_head(label, path, float(args.threshold_percentile))
        if head.matrix.shape != (num_blocks, num_blocks):
            raise ValueError(f"{label} has shape {head.matrix.shape}; expected {(num_blocks, num_blocks)}")
        head_map[label] = head

    set_specs: Dict[str, List[str]] = {}
    for raw in args.sets:
        if ":" not in raw:
            raise ValueError("--set entries must be SET_NAME:LABEL,LABEL,...")
        name, labels_raw = raw.split(":", 1)
        labels = [v.strip() for v in labels_raw.split(",") if v.strip()]
        missing = [label for label in labels if label not in head_map]
        if missing:
            raise ValueError(f"Set {name!r} references missing labels: {missing}")
        set_specs[name.strip()] = labels

    head_rows = []
    for head in head_map.values():
        for q_kind, q in (("z", head.d_z), ("thresh", head.d_thresh)):
            order = valid_order_from_priority(q.mean(axis=1), descending=True)
            order_original = to_original_order(order, perm_state)
            method = f"{head.label}_single_{q_kind}_netflow"
            write_order_json(
                order_dir / f"{method}.json",
                method,
                order,
                {
                    "kind": "single_head_netflow",
                    "label": head.label,
                    "q_kind": q_kind,
                    "matrix_path": str(head.path),
                    "uses_original_or_loss_for_selection": False,
                },
            )
            if args.write_reverse:
                write_order_json(
                    order_dir / f"{method}_reverse.json",
                    f"{method}_reverse",
                    list(reversed(order)),
                    {
                        "kind": "single_head_netflow_reverse_diagnostic",
                        "label": head.label,
                        "q_kind": q_kind,
                        "matrix_path": str(head.path),
                        "uses_original_or_loss_for_selection": False,
                    },
                )
            head_rows.append(
                {
                    "label": head.label,
                    "matrix_path": str(head.path),
                    "norm_z": head.norm_z,
                    "norm_thresh": head.norm_thresh,
                    "single_z_tau_original_diagnostic": kendall_tau(order_original) if q_kind == "z" else "",
                }
            )

    method_rows = []
    sign_rows = []
    for set_name, labels in set_specs.items():
        heads = [head_map[label] for label in labels]
        for q_kind in ("z", "thresh"):
            for weight_mode in ("uniform", "strength"):
                for method, signs, anchor, sign_rule in build_methods(heads, set_name, q_kind, weight_mode):
                    q_total, weights = aggregate_signed(heads, q_kind, signs, weight_mode)
                    priority = q_total.mean(axis=1)
                    order = valid_order_from_priority(priority, descending=True)
                    order_original = to_original_order(order, perm_state)
                    tau_original = kendall_tau(order_original)
                    q_path = args.out_dir / f"{method}_pairwise_Q.npy"
                    priority_path = args.out_dir / f"{method}_priority.npy"
                    np.save(q_path, q_total)
                    np.save(priority_path, priority)
                    save_heatmap(plot_dir / f"{method}_pairwise_Q.png", q_total, method)
                    meta = {
                        "kind": "signed_pairwise_Q_consensus",
                        "set_name": set_name,
                        "heads": labels,
                        "q_kind": q_kind,
                        "weight_mode": weight_mode,
                        "sign_rule": sign_rule,
                        "anchor": anchor,
                        "signs": signs,
                        "weights": weights,
                        "pairwise_Q_path": str(q_path),
                        "priority_path": str(priority_path),
                        "uses_original_or_loss_for_selection": False,
                    }
                    write_order_json(order_dir / f"{method}.json", method, order, meta)
                    if args.write_reverse:
                        write_order_json(
                            order_dir / f"{method}_reverse.json",
                            f"{method}_reverse",
                            list(reversed(order)),
                            {**meta, "kind": "signed_pairwise_Q_consensus_reverse_diagnostic"},
                        )
                    method_rows.append(
                        {
                            "method": method,
                            "set_name": set_name,
                            "q_kind": q_kind,
                            "weight_mode": weight_mode,
                            "sign_rule": sign_rule,
                            "anchor": anchor,
                            "num_heads": len(heads),
                            "tau_original_diagnostic": tau_original,
                            "tau_current_diagnostic": kendall_tau(order),
                            "q_mean_abs": float(np.mean(np.abs(q_total))),
                            "priority_std": float(np.std(priority)),
                            "order_current": " ".join(str(v) for v in order),
                            "order_original": " ".join(str(v) for v in order_original),
                        }
                    )
                    for head in heads:
                        sign_rows.append(
                            {
                                "method": method,
                                "set_name": set_name,
                                "label": head.label,
                                "sign": int(signs[head.label]),
                                "anchor": anchor,
                                "weight": float(weights[head.label]),
                                "norm_z": float(head.norm_z),
                                "norm_thresh": float(head.norm_thresh),
                                "alignment_to_anchor_z": dot_q(head.d_z, head_map[anchor].d_z),
                                "alignment_to_anchor_thresh": dot_q(head.d_thresh, head_map[anchor].d_thresh),
                            }
                        )

    with (args.out_dir / "method_summary.csv").open("w", newline="") as handle:
        fieldnames = list(method_rows[0].keys()) if method_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(method_rows)

    with (args.out_dir / "head_signs.csv").open("w", newline="") as handle:
        fieldnames = list(sign_rows[0].keys()) if sign_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sign_rows)

    with (args.out_dir / "head_matrix_summary.csv").open("w", newline="") as handle:
        fieldnames = ["label", "matrix_path", "norm_z", "norm_thresh", "single_z_tau_original_diagnostic"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(head_rows)

    spec = {
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": int(ckpt.get("iter_num", -1)),
        "num_blocks": num_blocks,
        "threshold_percentile": float(args.threshold_percentile),
        "head_matrices": {
            label: {
                "path": str(head.path),
                "norm_z": float(head.norm_z),
                "norm_thresh": float(head.norm_thresh),
            }
            for label, head in head_map.items()
        },
        "sets": set_specs,
        "no_prior_selection_guarantee": {
            "original_l2r_used_for_signs": False,
            "original_tau_used_for_signs": False,
            "validation_loss_used_for_signs": False,
            "direction_source": "current attention pairwise-Q cross-head alignment",
        },
    }
    (args.out_dir / "consensus_spec.json").write_text(json.dumps(spec, indent=2))

    lines = [
        "# Signed Pairwise-Q Orders",
        "",
        "Signs are selected from current attention pairwise-Q alignment only. Original tau and validation loss are diagnostics, not sign-selection inputs.",
        "",
        "| method | heads | sign rule | q | weight | anchor | tau original diagnostic | first16 current |",
        "| --- | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in method_rows:
        first16 = " ".join(row["order_current"].split()[:16])
        lines.append(
            f"| {row['method']} | {row['num_heads']} | {row['sign_rule']} | {row['q_kind']} | "
            f"{row['weight_mode']} | {row['anchor']} | {float(row['tau_original_diagnostic']):+.4f} | `{first16}` |"
        )
    (args.out_dir / "results.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"out_dir": str(args.out_dir), "num_orders": len(list(order_dir.glob("*.json")))}, indent=2))


if __name__ == "__main__":
    main()
