#!/usr/bin/env python3
"""Evaluate layer-mean pairwise-max Laplacian orders from a checkpoint.

This tests a layer-level order-discovery path:

1. collect current-frame block attention for all heads in a layer;
2. average heads into one layer matrix A;
3. build an undirected affinity W = max(A, A.T), diag(W)=0;
4. sort blocks by the Fiedler vector of the graph Laplacian;
5. orient order vs reverse by current-model loss profile.

The script also reports controls including the largest-eigenvalue Laplacian
eigenvector, which checks whether the high-frequency end of the same graph
spectrum carries original-order signal.

Original-frame tau/order are diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.eval_layer_mean_direct_asym_eig_order import (  # noqa: E402
    order_tau,
    parse_layers,
    select_by_loss,
    write_csv,
)
from scripts.analysis.test_10k_direct_asym_eig_head_method import (  # noqa: E402
    autocast_context,
    collect_all_head_matrices,
    evaluate_loss_profiles,
    infer_record_mode,
    load_checkpoint,
    load_model,
    load_tokens,
    loss_score,
    matrix_asymmetry_metrics,
    original_frame_matrix,
    permutation_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-mean pairwise-max Fiedler order diagnostic."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--attention_split", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--loss_split", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--export_type",
        choices=("with_none", "without_none", "target_to_observed"),
        default="without_none",
    )
    parser.add_argument(
        "--attention_order_mode",
        choices=("random", "current_ar", "original_l2r"),
        default="random",
    )
    parser.add_argument("--attention_samples", type=int, default=1024)
    parser.add_argument("--attention_batch_size", type=int, default=16)
    parser.add_argument("--attention_seed", type=int, default=24681357)
    parser.add_argument("--loss_samples", type=int, default=1024)
    parser.add_argument("--loss_batch_size", type=int, default=16)
    parser.add_argument("--loss_candidate_batch_size", type=int, default=4)
    parser.add_argument("--loss_seed", type=int, default=97531)
    parser.add_argument("--prefix_k", type=int, default=16)
    parser.add_argument(
        "--loss_score",
        choices=("linear_profile", "exp_profile", "prefix", "full"),
        default="linear_profile",
    )
    parser.add_argument("--exp_tau", type=float, default=16.0)
    parser.add_argument("--layers", type=str, default="0")
    parser.add_argument(
        "--layer_reduce",
        choices=("per_layer", "all_layers_mean"),
        default="per_layer",
        help=(
            "per_layer keeps the historical behavior: mean heads within each selected layer. "
            "all_layers_mean averages all heads across the selected layers into one matrix."
        ),
    )
    parser.add_argument("--write_matrices", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def fiedler_order(affinity: np.ndarray) -> Tuple[List[int], Dict[str, float | str]]:
    values = np.nan_to_num(np.asarray(affinity, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    values = 0.5 * (values + values.T)
    np.fill_diagonal(values, 0.0)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square affinity, got shape={tuple(values.shape)}")
    if values.shape[0] <= 1:
        return list(range(values.shape[0])), {"error": "degenerate_n"}
    degree = values.sum(axis=1)
    laplacian = np.diag(degree) - values
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    eig_idx = 1 if eigvals.size > 1 else 0
    vector = np.asarray(eigvecs[:, eig_idx], dtype=np.float64)
    if not np.isfinite(vector).all() or float(np.std(vector)) <= 1e-12:
        raise ValueError("Fiedler vector is degenerate or non-finite.")
    order = [
        int(idx)
        for idx in sorted(
            range(values.shape[0]),
            key=lambda item: (float(vector[item]), int(item)),
        )
    ]
    return order, {
        "laplacian_eigval": float(eigvals[eig_idx]),
        "laplacian_zero_eigval": float(eigvals[0]),
        "fiedler_vector_std": float(np.std(vector)),
        "fiedler_vector_min": float(np.min(vector)),
        "fiedler_vector_max": float(np.max(vector)),
        "affinity_sum": float(values.sum()),
        "affinity_density": float(np.mean(values > 0.0)),
    }


def laplacian_largest_order(affinity: np.ndarray) -> Tuple[List[int], Dict[str, float | str]]:
    values = np.nan_to_num(np.asarray(affinity, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    values = 0.5 * (values + values.T)
    np.fill_diagonal(values, 0.0)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square affinity, got shape={tuple(values.shape)}")
    if values.shape[0] <= 1:
        return list(range(values.shape[0])), {"error": "degenerate_n"}
    degree = values.sum(axis=1)
    laplacian = np.diag(degree) - values
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    eig_idx = int(eigvals.size - 1)
    vector = np.asarray(eigvecs[:, eig_idx], dtype=np.float64)
    if not np.isfinite(vector).all() or float(np.std(vector)) <= 1e-12:
        raise ValueError("Largest Laplacian eigenvector is degenerate or non-finite.")
    order = [
        int(idx)
        for idx in sorted(
            range(values.shape[0]),
            key=lambda item: (float(vector[item]), int(item)),
        )
    ]
    return order, {
        "laplacian_largest_eigval": float(eigvals[eig_idx]),
        "laplacian_zero_eigval": float(eigvals[0]),
        "laplacian_largest_vector_std": float(np.std(vector)),
        "laplacian_largest_vector_min": float(np.min(vector)),
        "laplacian_largest_vector_max": float(np.max(vector)),
        "affinity_sum": float(values.sum()),
        "affinity_density": float(np.mean(values > 0.0)),
    }


def direct_laplacian_fiedler_order(matrix: np.ndarray) -> Tuple[List[int], Dict[str, float | str]]:
    values = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = values.copy()
    np.fill_diagonal(values, 0.0)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square matrix, got shape={tuple(values.shape)}")
    if values.shape[0] <= 1:
        return list(range(values.shape[0])), {"error": "degenerate_n"}
    degree = values.sum(axis=1)
    laplacian = np.diag(degree) - values
    try:
        eigvals, eigvecs = np.linalg.eig(laplacian)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"direct_laplacian_fiedler failed: {exc}") from exc
    if eigvals.size <= 1 or eigvecs.size <= 0:
        raise ValueError("direct_laplacian_fiedler produced no nontrivial eigenvectors")
    sorted_indices = sorted(
        range(int(eigvals.size)),
        key=lambda idx: (
            float(np.real(eigvals[idx])),
            float(abs(np.imag(eigvals[idx]))),
            float(abs(eigvals[idx])),
            int(idx),
        ),
    )
    eig_idx = int(sorted_indices[1])
    vector_complex = eigvecs[:, eig_idx]
    vector = np.asarray(np.real(vector_complex), dtype=np.float64)
    if not np.isfinite(vector).all() or float(np.std(vector)) <= 1e-12:
        vector = np.asarray(values.sum(axis=1) - values.sum(axis=0), dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("direct_laplacian_fiedler produced a non-finite ordering vector")
    if float(np.std(vector)) <= 1e-12:
        raise ValueError("direct_laplacian_fiedler produced a degenerate ordering vector")
    order = [
        int(idx)
        for idx in sorted(
            range(values.shape[0]),
            key=lambda item: (float(vector[item]), int(item)),
        )
    ]
    return order, {
        "direct_laplacian_eigval0_real": float(np.real(eigvals[sorted_indices[0]])),
        "direct_laplacian_eigval0_imag": float(np.imag(eigvals[sorted_indices[0]])),
        "direct_laplacian_fiedler_eigval_real": float(np.real(eigvals[eig_idx])),
        "direct_laplacian_fiedler_eigval_imag": float(np.imag(eigvals[eig_idx])),
        "direct_laplacian_fiedler_eigval_abs": float(abs(eigvals[eig_idx])),
        "direct_laplacian_vector_std": float(np.std(vector)),
        "direct_laplacian_vector_min": float(np.min(vector)),
        "direct_laplacian_vector_max": float(np.max(vector)),
        "matrix_sum": float(values.sum()),
        "matrix_density": float(np.mean(values != 0.0)),
        "note": "Directed/raw L=D_out-A; eigenvalues sorted by real part; right eigenvector real part is sorted.",
    }


def top_eigenvector_order(affinity: np.ndarray) -> Tuple[List[int], Dict[str, float]]:
    values = np.nan_to_num(np.asarray(affinity, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = 0.5 * (values + values.T)
    eigvals, eigvecs = np.linalg.eigh(values)
    eig_idx = int(np.argmax(eigvals))
    vector = np.asarray(eigvecs[:, eig_idx], dtype=np.float64)
    if np.sum(vector) < 0.0:
        vector = -vector
    order = [
        int(idx)
        for idx in sorted(
            range(values.shape[0]),
            key=lambda item: (float(vector[item]), int(item)),
        )
    ]
    return order, {
        "top_eigval": float(eigvals[eig_idx]),
        "top_eig_vector_std": float(np.std(vector)),
    }


def _plot_matrix_pair(
    path: Path,
    *,
    title: str,
    left_matrix: np.ndarray,
    left_title: str,
    right_matrix: np.ndarray,
    right_title: str,
) -> None:
    def vmax_pos(*mats: np.ndarray) -> float:
        vals = np.concatenate([m[np.isfinite(m)].reshape(-1) for m in mats])
        vals = vals[vals > 0.0]
        if vals.size == 0:
            return 1.0
        return float(np.percentile(vals, 99.0))

    vmax = vmax_pos(left_matrix, right_matrix)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)
    fig.suptitle(title)
    for ax, mat, subtitle in [
        (axes[0], left_matrix, left_title),
        (axes[1], right_matrix, right_title),
    ]:
        im = ax.imshow(mat, interpolation="nearest", aspect="equal", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("key block")
        ax.set_ylabel("query block")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_layer(out_dir: Path, layer_label: str, matrix: np.ndarray, affinity: np.ndarray, perm_state) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_matrix_pair(
        plots_dir / f"{layer_label}_layermean_pairwise_max_fiedler_input.png",
        title=f"{layer_label} mean attention and pairwise-max affinity",
        left_matrix=matrix,
        left_title="mean A, input/current coords",
        right_matrix=affinity,
        right_title="W=max(A,A.T), input/current coords",
    )

    original_matrix = original_frame_matrix(matrix, perm_state)
    original_affinity = original_frame_matrix(affinity, perm_state)
    _plot_matrix_pair(
        plots_dir / f"{layer_label}_layermean_pairwise_max_fiedler_true_original.png",
        title=f"{layer_label} true-original-frame mean attention and affinity",
        left_matrix=original_matrix,
        left_title="mean A, true original coords",
        right_matrix=original_affinity,
        right_title="W=max(A,A.T), true original coords",
    )


def format_layer_label(row: Dict) -> str:
    label = row.get("layer_label")
    if label:
        return str(label)
    return f"L{int(row['layer'])}"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.plot):
        (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.ckpt_path)
    model = load_model(ckpt, args.device)
    attention_tokens, data_dir = load_tokens(ckpt, args.attention_split, args.data_dir)
    loss_tokens, _ = load_tokens(ckpt, args.loss_split, data_dir)
    record_mode = infer_record_mode(ckpt, data_dir)
    perm_state = permutation_state(ckpt, model, int(model.config.block_size))
    ctx = autocast_context(args)

    matrices, attention_total = collect_all_head_matrices(
        args,
        model,
        attention_tokens,
        record_mode,
        perm_state,
        ctx,
    )
    layers = parse_layers(args.layers, int(matrices.shape[0]))
    if bool(args.write_matrices):
        np.savez_compressed(
            args.out_dir / "all_head_attention_matrices_current.npz",
            matrices=matrices.astype(np.float32),
            selected_layers=np.asarray(layers, dtype=np.int64),
            layer_reduce=str(args.layer_reduce),
        )

    if str(args.layer_reduce) == "all_layers_mean":
        selected = np.asarray(matrices[layers], dtype=np.float64)
        layer_inputs = [
            (
                -1,
                "all_layers",
                selected.mean(axis=(0, 1)),
            )
        ]
    else:
        layer_inputs = [
            (
                int(layer_idx),
                f"L{int(layer_idx)}",
                np.asarray(matrices[layer_idx], dtype=np.float64).mean(axis=0),
            )
            for layer_idx in layers
        ]

    payloads: List[Dict] = []
    order_requests: List[List[int]] = []
    for layer_idx, layer_label, matrix in layer_inputs:
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(matrix, 0.0)
        affinity = np.maximum(matrix, matrix.T)
        np.fill_diagonal(affinity, 0.0)
        sum_affinity = matrix + matrix.T
        np.fill_diagonal(sum_affinity, 0.0)
        if bool(args.write_matrices):
            np.save(args.out_dir / f"{layer_label}_layermean_attention_current.npy", matrix.astype(np.float32))
            np.save(args.out_dir / f"{layer_label}_pairwise_max_affinity_current.npy", affinity.astype(np.float32))
            np.save(args.out_dir / f"{layer_label}_pairwise_sum_affinity_current.npy", sum_affinity.astype(np.float32))
            np.save(
                args.out_dir / f"{layer_label}_layermean_attention_true_original.npy",
                original_frame_matrix(matrix, perm_state).astype(np.float32),
            )
            np.save(
                args.out_dir / f"{layer_label}_pairwise_max_affinity_true_original.npy",
                original_frame_matrix(affinity, perm_state).astype(np.float32),
            )
            np.save(
                args.out_dir / f"{layer_label}_pairwise_sum_affinity_true_original.npy",
                original_frame_matrix(sum_affinity, perm_state).astype(np.float32),
            )
        if bool(args.plot):
            plot_layer(args.out_dir, str(layer_label), matrix, affinity, perm_state)

        fiedler_raw, fiedler_meta = fiedler_order(affinity)
        fiedler_reverse = list(reversed(fiedler_raw))
        sum_fiedler_raw, sum_fiedler_meta = fiedler_order(sum_affinity)
        sum_fiedler_reverse = list(reversed(sum_fiedler_raw))
        direct_laplacian_raw, direct_laplacian_meta = direct_laplacian_fiedler_order(matrix)
        direct_laplacian_reverse = list(reversed(direct_laplacian_raw))
        lap_largest_raw, lap_largest_meta = laplacian_largest_order(affinity)
        lap_largest_reverse = list(reversed(lap_largest_raw))
        top_raw, top_meta = top_eigenvector_order(affinity)
        top_reverse = list(reversed(top_raw))
        fiedler_original, fiedler_tau = order_tau(fiedler_raw, perm_state)
        fiedler_rev_original, fiedler_rev_tau = order_tau(fiedler_reverse, perm_state)
        sum_fiedler_original, sum_fiedler_tau = order_tau(sum_fiedler_raw, perm_state)
        sum_fiedler_rev_original, sum_fiedler_rev_tau = order_tau(sum_fiedler_reverse, perm_state)
        direct_laplacian_original, direct_laplacian_tau = order_tau(direct_laplacian_raw, perm_state)
        direct_laplacian_rev_original, direct_laplacian_rev_tau = order_tau(
            direct_laplacian_reverse,
            perm_state,
        )
        lap_largest_original, lap_largest_tau = order_tau(lap_largest_raw, perm_state)
        lap_largest_rev_original, lap_largest_rev_tau = order_tau(lap_largest_reverse, perm_state)
        top_original, top_tau = order_tau(top_raw, perm_state)
        top_rev_original, top_rev_tau = order_tau(top_reverse, perm_state)
        metrics = matrix_asymmetry_metrics(matrix)
        payloads.append(
            {
                "layer": int(layer_idx),
                "layer_label": str(layer_label),
                "layer_reduce": str(args.layer_reduce),
                "source": f"{layer_label}_head_mean_pairwise_max_fiedler",
                "matrix": matrix,
                "affinity": affinity,
                "asym_score": float(metrics["asym_score"]),
                "upper_frac": float(metrics["upper_frac"]),
                "fiedler_order_current": fiedler_raw,
                "fiedler_order_original": fiedler_original,
                "fiedler_original_tau": float(fiedler_tau),
                "fiedler_original_abs_tau": abs(float(fiedler_tau)),
                "fiedler_original_distance": float((1.0 - fiedler_tau) / 2.0),
                "fiedler_reverse_order_original": fiedler_rev_original,
                "fiedler_reverse_original_tau": float(fiedler_rev_tau),
                "fiedler_reverse_original_abs_tau": abs(float(fiedler_rev_tau)),
                "fiedler_reverse_original_distance": float((1.0 - fiedler_rev_tau) / 2.0),
                "sum_fiedler_order_current": sum_fiedler_raw,
                "sum_fiedler_order_original": sum_fiedler_original,
                "sum_fiedler_original_tau": float(sum_fiedler_tau),
                "sum_fiedler_original_abs_tau": abs(float(sum_fiedler_tau)),
                "sum_fiedler_reverse_order_original": sum_fiedler_rev_original,
                "sum_fiedler_reverse_original_tau": float(sum_fiedler_rev_tau),
                "sum_fiedler_reverse_original_abs_tau": abs(float(sum_fiedler_rev_tau)),
                "direct_laplacian_order_current": direct_laplacian_raw,
                "direct_laplacian_order_original": direct_laplacian_original,
                "direct_laplacian_original_tau": float(direct_laplacian_tau),
                "direct_laplacian_original_abs_tau": abs(float(direct_laplacian_tau)),
                "direct_laplacian_reverse_order_original": direct_laplacian_rev_original,
                "direct_laplacian_reverse_original_tau": float(direct_laplacian_rev_tau),
                "direct_laplacian_reverse_original_abs_tau": abs(float(direct_laplacian_rev_tau)),
                "laplacian_largest_order_current": lap_largest_raw,
                "laplacian_largest_order_original": lap_largest_original,
                "laplacian_largest_original_tau": float(lap_largest_tau),
                "laplacian_largest_original_abs_tau": abs(float(lap_largest_tau)),
                "laplacian_largest_reverse_order_original": lap_largest_rev_original,
                "laplacian_largest_reverse_original_tau": float(lap_largest_rev_tau),
                "laplacian_largest_reverse_original_abs_tau": abs(float(lap_largest_rev_tau)),
                "top_eig_order_current": top_raw,
                "top_eig_order_original": top_original,
                "top_eig_original_tau": float(top_tau),
                "top_eig_reverse_order_original": top_rev_original,
                "top_eig_reverse_original_tau": float(top_rev_tau),
                "fiedler": fiedler_meta,
                "sum_fiedler": sum_fiedler_meta,
                "direct_laplacian": direct_laplacian_meta,
                "laplacian_largest": lap_largest_meta,
                "top_eig": top_meta,
            }
        )
        order_requests.extend(
            [
                fiedler_raw,
                fiedler_reverse,
                sum_fiedler_raw,
                sum_fiedler_reverse,
                direct_laplacian_raw,
                direct_laplacian_reverse,
                lap_largest_raw,
                lap_largest_reverse,
                top_raw,
                top_reverse,
            ]
        )

    loss_by_order = evaluate_loss_profiles(
        args,
        model,
        loss_tokens,
        record_mode,
        perm_state,
        order_requests,
        split_name=str(args.loss_split),
        num_samples=int(args.loss_samples),
        batch_size=int(args.loss_batch_size),
        seed=int(args.loss_seed),
        ctx=ctx,
    )

    rows: List[Dict] = []
    for payload in payloads:
        selected_order, selected_loss, other_loss, selected_reverse, score_gap = select_by_loss(
            args,
            payload["fiedler_order_current"],
            loss_by_order,
        )
        selected_original, selected_tau = order_tau(selected_order, perm_state)
        sum_selected_order, sum_selected_loss, sum_other_loss, sum_selected_reverse, sum_score_gap = select_by_loss(
            args,
            payload["sum_fiedler_order_current"],
            loss_by_order,
        )
        sum_selected_original, sum_selected_tau = order_tau(sum_selected_order, perm_state)
        top_selected_order, top_selected_loss, top_other_loss, top_selected_reverse, top_score_gap = select_by_loss(
            args,
            payload["top_eig_order_current"],
            loss_by_order,
        )
        top_selected_original, top_selected_tau = order_tau(top_selected_order, perm_state)
        lap_largest_selected_order, lap_largest_selected_loss, lap_largest_other_loss, lap_largest_selected_reverse, lap_largest_score_gap = select_by_loss(
            args,
            payload["laplacian_largest_order_current"],
            loss_by_order,
        )
        lap_largest_selected_original, lap_largest_selected_tau = order_tau(
            lap_largest_selected_order,
            perm_state,
        )
        direct_laplacian_selected_order, direct_laplacian_selected_loss, direct_laplacian_other_loss, direct_laplacian_selected_reverse, direct_laplacian_score_gap = select_by_loss(
            args,
            payload["direct_laplacian_order_current"],
            loss_by_order,
        )
        direct_laplacian_selected_original, direct_laplacian_selected_tau = order_tau(
            direct_laplacian_selected_order,
            perm_state,
        )
        row = {
            "layer": int(payload["layer"]),
            "layer_label": str(payload["layer_label"]),
            "layer_reduce": str(payload["layer_reduce"]),
            "attention_samples": int(attention_total),
            "loss_samples": int(selected_loss["count"]),
            "export_type": str(args.export_type),
            "attention_order_mode": str(args.attention_order_mode),
            "asym_score": float(payload["asym_score"]),
            "upper_frac": float(payload["upper_frac"]),
            "fiedler_laplacian_eigval": float(payload["fiedler"]["laplacian_eigval"]),
            "fiedler_vector_std": float(payload["fiedler"]["fiedler_vector_std"]),
            "fiedler_raw_original_tau": float(payload["fiedler_original_tau"]),
            "fiedler_raw_original_abs_tau": float(payload["fiedler_original_abs_tau"]),
            "fiedler_reverse_original_tau": float(payload["fiedler_reverse_original_tau"]),
            "fiedler_loss_oriented_selected_reverse": bool(selected_reverse),
            "fiedler_loss_oriented_score_gap": float(score_gap),
            "fiedler_loss_oriented_original_tau": float(selected_tau),
            "fiedler_loss_oriented_original_abs_tau": abs(float(selected_tau)),
            "fiedler_selected_linear_profile_loss": float(loss_score(selected_loss, "linear_profile")),
            "fiedler_other_linear_profile_loss": float(loss_score(other_loss, "linear_profile")),
            "fiedler_selected_prefix_loss": float(selected_loss["prefix_loss"]),
            "fiedler_selected_full_loss": float(selected_loss["full_loss"]),
            "fiedler_loss_oriented_order_original": " ".join(str(v) for v in selected_original),
            "sum_fiedler_laplacian_eigval": float(payload["sum_fiedler"]["laplacian_eigval"]),
            "sum_fiedler_vector_std": float(payload["sum_fiedler"]["fiedler_vector_std"]),
            "sum_fiedler_raw_original_tau": float(payload["sum_fiedler_original_tau"]),
            "sum_fiedler_raw_original_abs_tau": float(payload["sum_fiedler_original_abs_tau"]),
            "sum_fiedler_reverse_original_tau": float(payload["sum_fiedler_reverse_original_tau"]),
            "sum_fiedler_loss_oriented_selected_reverse": bool(sum_selected_reverse),
            "sum_fiedler_loss_oriented_score_gap": float(sum_score_gap),
            "sum_fiedler_loss_oriented_original_tau": float(sum_selected_tau),
            "sum_fiedler_loss_oriented_original_abs_tau": abs(float(sum_selected_tau)),
            "sum_fiedler_selected_linear_profile_loss": float(
                loss_score(sum_selected_loss, "linear_profile")
            ),
            "sum_fiedler_other_linear_profile_loss": float(
                loss_score(sum_other_loss, "linear_profile")
            ),
            "sum_fiedler_loss_oriented_order_original": " ".join(str(v) for v in sum_selected_original),
            "direct_laplacian_eigval0_real": float(
                payload["direct_laplacian"]["direct_laplacian_eigval0_real"]
            ),
            "direct_laplacian_eigval0_imag": float(
                payload["direct_laplacian"]["direct_laplacian_eigval0_imag"]
            ),
            "direct_laplacian_fiedler_eigval_real": float(
                payload["direct_laplacian"]["direct_laplacian_fiedler_eigval_real"]
            ),
            "direct_laplacian_fiedler_eigval_imag": float(
                payload["direct_laplacian"]["direct_laplacian_fiedler_eigval_imag"]
            ),
            "direct_laplacian_vector_std": float(
                payload["direct_laplacian"]["direct_laplacian_vector_std"]
            ),
            "direct_laplacian_raw_original_tau": float(payload["direct_laplacian_original_tau"]),
            "direct_laplacian_raw_original_abs_tau": float(
                payload["direct_laplacian_original_abs_tau"]
            ),
            "direct_laplacian_reverse_original_tau": float(
                payload["direct_laplacian_reverse_original_tau"]
            ),
            "direct_laplacian_reverse_original_abs_tau": float(
                payload["direct_laplacian_reverse_original_abs_tau"]
            ),
            "direct_laplacian_loss_oriented_selected_reverse": bool(direct_laplacian_selected_reverse),
            "direct_laplacian_loss_oriented_score_gap": float(direct_laplacian_score_gap),
            "direct_laplacian_loss_oriented_original_tau": float(direct_laplacian_selected_tau),
            "direct_laplacian_loss_oriented_original_abs_tau": abs(float(direct_laplacian_selected_tau)),
            "direct_laplacian_selected_linear_profile_loss": float(
                loss_score(direct_laplacian_selected_loss, "linear_profile")
            ),
            "direct_laplacian_other_linear_profile_loss": float(
                loss_score(direct_laplacian_other_loss, "linear_profile")
            ),
            "direct_laplacian_loss_oriented_order_original": " ".join(
                str(v) for v in direct_laplacian_selected_original
            ),
            "laplacian_largest_laplacian_eigval": float(
                payload["laplacian_largest"]["laplacian_largest_eigval"]
            ),
            "laplacian_largest_vector_std": float(
                payload["laplacian_largest"]["laplacian_largest_vector_std"]
            ),
            "laplacian_largest_raw_original_tau": float(payload["laplacian_largest_original_tau"]),
            "laplacian_largest_raw_original_abs_tau": float(
                payload["laplacian_largest_original_abs_tau"]
            ),
            "laplacian_largest_reverse_original_tau": float(
                payload["laplacian_largest_reverse_original_tau"]
            ),
            "laplacian_largest_reverse_original_abs_tau": float(
                payload["laplacian_largest_reverse_original_abs_tau"]
            ),
            "laplacian_largest_loss_oriented_selected_reverse": bool(lap_largest_selected_reverse),
            "laplacian_largest_loss_oriented_score_gap": float(lap_largest_score_gap),
            "laplacian_largest_loss_oriented_original_tau": float(lap_largest_selected_tau),
            "laplacian_largest_loss_oriented_original_abs_tau": abs(float(lap_largest_selected_tau)),
            "laplacian_largest_selected_linear_profile_loss": float(
                loss_score(lap_largest_selected_loss, "linear_profile")
            ),
            "laplacian_largest_other_linear_profile_loss": float(
                loss_score(lap_largest_other_loss, "linear_profile")
            ),
            "laplacian_largest_loss_oriented_order_original": " ".join(
                str(v) for v in lap_largest_selected_original
            ),
            "top_eig_raw_original_tau": float(payload["top_eig_original_tau"]),
            "top_eig_reverse_original_tau": float(payload["top_eig_reverse_original_tau"]),
            "top_eig_loss_oriented_selected_reverse": bool(top_selected_reverse),
            "top_eig_loss_oriented_score_gap": float(top_score_gap),
            "top_eig_loss_oriented_original_tau": float(top_selected_tau),
            "top_eig_loss_oriented_original_abs_tau": abs(float(top_selected_tau)),
            "top_eig_loss_oriented_order_original": " ".join(str(v) for v in top_selected_original),
            "note": (
                "Original tau/order are diagnostic-only. Direction selection uses current-model loss profile only."
            ),
        }
        rows.append(row)

    csv_rows = []
    for row in rows:
        csv_rows.append(dict(row))
    write_csv(args.out_dir / "layer_mean_pairwise_max_fiedler.csv", csv_rows)

    summary = {
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": int(ckpt.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "data_dir": str(data_dir),
        "attention": {
            "split": str(args.attention_split),
            "samples": int(attention_total),
            "batch_size": int(args.attention_batch_size),
            "seed": int(args.attention_seed),
            "order_mode": str(args.attention_order_mode),
            "export_type": str(args.export_type),
            "layer_reduce": str(args.layer_reduce),
            "selected_layers": [int(v) for v in layers],
        },
        "loss_orientation": {
            "split": str(args.loss_split),
            "samples": int(args.loss_samples),
            "batch_size": int(args.loss_batch_size),
            "candidate_batch_size": int(args.loss_candidate_batch_size),
            "seed": int(args.loss_seed),
            "score": str(args.loss_score),
            "prefix_k": int(args.prefix_k),
        },
        "method": (
            "Mean-reduced attention -> W=max(A,A.T), diag=0 -> graph Laplacian Fiedler vector "
            "axis -> current-model loss profile chooses axis vs reverse. Controls include the "
            "largest-eigenvalue Laplacian eigenvector and the top eigenvector of W."
        ),
        "rows": rows,
        "note": "Original-frame tau/order are diagnostic-only and not used for selection.",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Layer-Mean Pairwise-Max Fiedler Diagnostic",
        "",
        f"- checkpoint: `{args.ckpt_path}`",
        f"- checkpoint iter: `{summary['checkpoint_iter']}`",
        f"- attention split/order/export: `{args.attention_split}` / `{args.attention_order_mode}` / `{args.export_type}`",
        f"- attention samples: `{int(attention_total)}`",
        f"- loss-orientation samples: `{int(args.loss_samples)}`",
        f"- loss score: `{args.loss_score}`",
        "",
        "Original tau is diagnostic-only. The no-prior direction decision is current-model loss-profile selection between Fiedler order and reverse.",
        "",
        "| layer | raw tau | raw abs tau | reverse tau | loss-oriented tau | loss abs tau | selected reverse | score gap | selected order original |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {format_layer_label(row)} | {float(row['fiedler_raw_original_tau']):+.6f} | "
            f"{float(row['fiedler_raw_original_abs_tau']):.6f} | "
            f"{float(row['fiedler_reverse_original_tau']):+.6f} | "
            f"{float(row['fiedler_loss_oriented_original_tau']):+.6f} | "
            f"{float(row['fiedler_loss_oriented_original_abs_tau']):.6f} | "
            f"{bool(row['fiedler_loss_oriented_selected_reverse'])} | "
            f"{float(row['fiedler_loss_oriented_score_gap']):.6f} | "
            f"`[{', '.join(row['fiedler_loss_oriented_order_original'].split())}]` |"
        )
    lines.extend(
        [
            "",
            "## Pairwise-Sum Fiedler Test",
            "",
            "This replaces W=max(A,A.T) with W=A+A.T, then uses the same graph Laplacian Fiedler vector and current-model loss-profile orientation.",
            "",
            "| layer | raw tau | raw abs tau | reverse tau | loss-oriented tau | loss abs tau | selected reverse | score gap | selected order original |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {format_layer_label(row)} | {float(row['sum_fiedler_raw_original_tau']):+.6f} | "
            f"{float(row['sum_fiedler_raw_original_abs_tau']):.6f} | "
            f"{float(row['sum_fiedler_reverse_original_tau']):+.6f} | "
            f"{float(row['sum_fiedler_loss_oriented_original_tau']):+.6f} | "
            f"{float(row['sum_fiedler_loss_oriented_original_abs_tau']):.6f} | "
            f"{bool(row['sum_fiedler_loss_oriented_selected_reverse'])} | "
            f"{float(row['sum_fiedler_loss_oriented_score_gap']):.6f} | "
            f"`[{', '.join(row['sum_fiedler_loss_oriented_order_original'].split())}]` |"
        )
    lines.extend(
        [
            "",
            "## Direct-Laplacian Fiedler Test",
            "",
            "This skips W=max(A,A.T). It uses the raw directed layer-mean attention matrix A, builds L=D_out-A, sorts eigenvalues by real part, and sorts the real part of the second eigenvector.",
            "",
            "| layer | raw tau | raw abs tau | reverse tau | loss-oriented tau | loss abs tau | selected reverse | score gap | eigval real | eigval imag | selected order original |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {format_layer_label(row)} | {float(row['direct_laplacian_raw_original_tau']):+.6f} | "
            f"{float(row['direct_laplacian_raw_original_abs_tau']):.6f} | "
            f"{float(row['direct_laplacian_reverse_original_tau']):+.6f} | "
            f"{float(row['direct_laplacian_loss_oriented_original_tau']):+.6f} | "
            f"{float(row['direct_laplacian_loss_oriented_original_abs_tau']):.6f} | "
            f"{bool(row['direct_laplacian_loss_oriented_selected_reverse'])} | "
            f"{float(row['direct_laplacian_loss_oriented_score_gap']):.6f} | "
            f"{float(row['direct_laplacian_fiedler_eigval_real']):+.6f} | "
            f"{float(row['direct_laplacian_fiedler_eigval_imag']):+.6f} | "
            f"`[{', '.join(row['direct_laplacian_loss_oriented_order_original'].split())}]` |"
        )
    lines.extend(
        [
            "",
            "## Laplacian-Largest Eigenvector Test",
            "",
            "This directly replaces the Fiedler vector with the largest-eigenvalue eigenvector of the same graph Laplacian.",
            "",
            "| layer | largest raw tau | largest raw abs tau | largest reverse tau | loss-oriented tau | loss abs tau | selected reverse | score gap | selected order original |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {format_layer_label(row)} | {float(row['laplacian_largest_raw_original_tau']):+.6f} | "
            f"{float(row['laplacian_largest_raw_original_abs_tau']):.6f} | "
            f"{float(row['laplacian_largest_reverse_original_tau']):+.6f} | "
            f"{float(row['laplacian_largest_loss_oriented_original_tau']):+.6f} | "
            f"{float(row['laplacian_largest_loss_oriented_original_abs_tau']):.6f} | "
            f"{bool(row['laplacian_largest_loss_oriented_selected_reverse'])} | "
            f"{float(row['laplacian_largest_loss_oriented_score_gap']):.6f} | "
            f"`[{', '.join(row['laplacian_largest_loss_oriented_order_original'].split())}]` |"
        )
    lines.extend(
        [
            "",
            "## Top-Eigenvector Control",
            "",
            "This is the largest-eigenvalue eigenvector of the affinity matrix W, not the Laplacian.",
            "",
            "| layer | top-eig raw tau | top-eig reverse tau | top-eig loss-oriented tau |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {format_layer_label(row)} | {float(row['top_eig_raw_original_tau']):+.6f} | "
            f"{float(row['top_eig_reverse_original_tau']):+.6f} | "
            f"{float(row['top_eig_loss_oriented_original_tau']):+.6f} |"
        )
    (args.out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
