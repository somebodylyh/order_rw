#!/usr/bin/env python3
"""Diagnose teacher and readout collapse across old and new g_beta runs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import kendalltau
import torch


_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _ROOT / "block_lo_arm_order_network"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))


REPORT_FIELDS = {
    "old_vs_new_metrics.csv": [
        "run", "model", "head", "step", "input", "tau_to_L2R",
        "tau_to_teacher", "tau_to_CDL_real", "reference_frame", "status", "reason",
    ],
    "teacher_collapse_metrics.csv": [
        "run", "head", "step", "metric_scope", "reveal", "n_orders",
        "tau_meanCDL_to_L2R", "teacher_pairwise_tau_mean",
        "teacher_pairwise_tau_min", "teacher_pairwise_tau_std",
        "unique_teacher_count", "tau_per_sample_CDL_to_L2R_mean",
        "tau_per_sample_CDL_to_L2R_std", "tau_per_sample_CDL_to_teacher_mean",
        "sample_pairwise_tau_mean", "sample_pairwise_tau_std", "status", "reason",
    ],
    "input_sanity_metrics.csv": [
        "run", "model", "input_type", "tau_to_L2R", "tau_to_teacher",
        "output_pairwise_tau_to_real", "score_std_across_samples",
        "input_term_norm", "fixed_term_norm", "input_fixed_ratio",
        "final_prebias_norm", "final_bias_norm", "status", "reason",
    ],
    "output_diversity_metrics.csv": [
        "run", "input_type", "output_pairwise_tau_mean", "output_pairwise_tau_min",
        "output_unique_count", "score_variance_mean", "status", "reason",
    ],
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    architecture: str
    readout_path: Path
    source_ckpt: Path
    dataset_path: Path | None
    head: tuple[int, int]
    step: int
    seed: int
    batch_size: int
    input_frame: str
    output_frame: str
    teacher_protocol: str
    none_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "readout_path", Path(self.readout_path))
        object.__setattr__(self, "source_ckpt", Path(self.source_ckpt))
        if self.dataset_path is not None:
            object.__setattr__(self, "dataset_path", Path(self.dataset_path))


def build_default_run_specs(artifact_root: str | Path) -> dict[str, RunSpec]:
    """Return the two explicitly approved comparison artifacts."""
    root = Path(artifact_root).resolve()
    old_base = (
        root
        / "block_lo_arm_order_network/batch_readout/logs"
        / "gbeta_b1_L0H2_seed2_step10k/random_baseline_continuous_jun08_seed2/full"
    )
    specs = {
        "old_gbeta": RunSpec(
            name="old_gbeta",
            architecture="nodewise",
            readout_path=old_base / "g_beta_best.pt",
            source_ckpt=(
                root
                / "block_lo_arm_order_network/probe_results"
                / "random_baseline_continuous_jun08_seed2/ckpt_step10000.pt"
            ),
            dataset_path=old_base / "data/ds_10000.npz",
            head=(0, 2),
            step=10000,
            seed=2,
            batch_size=16,
            input_frame="physical",
            output_frame="physical",
            teacher_protocol="batch_mean_cdl_alpha_dep_0.5",
            none_mode="b1",
        ),
        "new_flatten_readout": RunSpec(
            name="new_flatten_readout",
            architecture="flatten",
            readout_path=root / "reports/uniform_label_free_v1/label_free_readout.pt",
            source_ckpt=(
                root
                / "block_lo_arm_order_network/probe_results"
                / "overnight_20260625_random_baseline/ckpt_step10000.pt"
            ),
            dataset_path=None,
            head=(1, 7),
            step=10000,
            seed=123,
            batch_size=8,
            input_frame="model",
            output_frame="model",
            teacher_protocol="batch_mean_cdl_isolated_none_C-D+L",
            none_mode="strict65_model_strip_none",
        ),
    }
    return specs


def _load_npz_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as payload:
        if "meta" not in payload.files:
            raise ValueError(f"saved dataset {path} has no meta field")
        raw = payload["meta"]
        value = raw.item() if raw.shape == () else raw.reshape(-1)[0]
        if not isinstance(value, dict):
            raise ValueError(f"saved dataset {path} meta is not a dict")
        return dict(value)


def validate_run_spec(spec: RunSpec) -> None:
    """Fail early on missing artifacts or contradictory saved metadata."""
    for label, path in (("readout", spec.readout_path), ("source checkpoint", spec.source_ckpt)):
        if not path.is_file():
            raise FileNotFoundError(f"{spec.name} {label} not found: {path}")
    if spec.input_frame not in {"physical", "model"}:
        raise ValueError(f"unknown input frame {spec.input_frame!r}")
    if spec.output_frame not in {"physical", "model"}:
        raise ValueError(f"unknown output frame {spec.output_frame!r}")
    if spec.dataset_path is None:
        return
    if not spec.dataset_path.is_file():
        raise FileNotFoundError(f"{spec.name} dataset not found: {spec.dataset_path}")
    meta = _load_npz_meta(spec.dataset_path)
    if tuple(meta.get("head", ())) != tuple(spec.head):
        raise ValueError(f"head mismatch: manifest={spec.head}, dataset={meta.get('head')}")
    if int(meta.get("seed", -1)) != spec.seed:
        raise ValueError(f"seed mismatch: manifest={spec.seed}, dataset={meta.get('seed')}")
    if int(meta.get("batch_size", -1)) != spec.batch_size:
        raise ValueError(
            f"batch size mismatch: manifest={spec.batch_size}, dataset={meta.get('batch_size')}"
        )
    dataset_ckpt = Path(str(meta.get("ckpt", ""))).name
    if dataset_ckpt != spec.source_ckpt.name:
        raise ValueError(
            f"source checkpoint mismatch: manifest={spec.source_ckpt.name}, dataset={dataset_ckpt}"
        )


def load_saved_batch_split(spec: RunSpec, split: str = "val") -> dict:
    """Load one split from a saved BR-1 dataset into independent arrays."""
    if spec.dataset_path is None:
        raise ValueError(f"run {spec.name} has no saved batch dataset")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}")
    with np.load(spec.dataset_path, allow_pickle=True) as payload:
        required = {
            "B": f"{split}_B_batch",
            "teacher_order": f"{split}_sigma_T",
            "rank": f"{split}_rank",
            "chunks": f"{split}_chunks",
        }
        missing = [key for key in required.values() if key not in payload.files]
        if missing:
            raise ValueError(f"saved dataset is missing fields: {missing}")
        result = {name: payload[key].copy() for name, key in required.items()}
    result["meta"] = _load_npz_meta(spec.dataset_path)
    return result


def map_orders_to_physical(orders: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
    """Map model-frame node ids in one or more orders to physical node ids."""
    orders = np.asarray(orders, dtype=np.int64)
    inv_perm = np.asarray(inv_perm, dtype=np.int64)
    if orders.ndim == 1:
        orders = orders[None, :]
        squeeze = True
    elif orders.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"orders must be one- or two-dimensional, got {orders.shape}")
    order_to_rank(inv_perm)  # validate lookup is a permutation
    if orders.shape[1] != len(inv_perm):
        raise ValueError("order width does not match inv_perm")
    for order in orders:
        order_to_rank(order)
    mapped = inv_perm[orders]
    return mapped[0] if squeeze else mapped


def teacher_orders_from_graphs(graphs: np.ndarray, protocol: str) -> np.ndarray:
    """Run the exact deterministic teacher associated with one training protocol."""
    graphs = np.asarray(graphs)
    if graphs.ndim != 3 or graphs.shape[1] != graphs.shape[2]:
        raise ValueError(f"graphs must have shape (M, N, N), got {graphs.shape}")
    orders = []
    if protocol == "batch_mean_cdl_alpha_dep_0.5":
        from neural_readout.teacher_labels import generate_teacher_label

        for graph in graphs:
            orders.append(generate_teacher_label(graph, alpha_dep=0.5)[0])
    elif protocol == "batch_mean_cdl_isolated_none_C-D+L":
        from none_separated_block_graph import rollout_by_method

        for graph in graphs:
            n = graph.shape[0]
            graph_with_none = np.zeros((n + 1, n + 1), dtype=np.float64)
            graph_with_none[1:, 1:] = graph
            orders.append(rollout_by_method(graph_with_none, "C-D+L"))
    else:
        raise ValueError(f"unknown teacher protocol {protocol!r}")
    return np.asarray(orders, dtype=np.int64)


def teacher_collapse_summary(
    orders: np.ndarray,
    physical_orders: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Summarize teacher proximity to physical L2R and order diversity."""
    orders = np.asarray(orders, dtype=np.int64)
    if physical_orders is None:
        physical_orders = orders
    physical_orders = np.asarray(physical_orders, dtype=np.int64)
    if orders.shape != physical_orders.shape:
        raise ValueError("orders and physical_orders must have identical shapes")
    diversity = pairwise_order_stats(orders)
    l2r = np.arange(orders.shape[1], dtype=np.int64)
    taus = np.asarray([order_tau(order, l2r) for order in physical_orders])
    return {
        "n_orders": int(len(orders)),
        "tau_to_l2r_mean": float(taus.mean()),
        "tau_to_l2r_std": float(taus.std()),
        "unique_teacher_count": int(diversity["unique_count"]),
        "teacher_pairwise_tau_mean": diversity["pairwise_tau_mean"],
        "teacher_pairwise_tau_min": diversity["pairwise_tau_min"],
        "teacher_pairwise_tau_std": diversity["pairwise_tau_std"],
    }


def _hypothesis(status: str, *evidence: str) -> dict[str, object]:
    return {"status": status, "evidence": list(evidence)}


def assess_hypotheses(evidence: dict[str, float]) -> dict[str, dict[str, object]]:
    """Classify H1-H5 from named evidence without filling missing observations."""
    result: dict[str, dict[str, object]] = {}

    tau = evidence.get("new_teacher_tau_l2r")
    unique = evidence.get("new_teacher_unique_fraction")
    unique_count = evidence.get("new_teacher_unique_count")
    if tau is None or unique is None:
        result["H1"] = _hypothesis("unresolved", "new batch-mean teacher metrics unavailable")
    elif tau >= 0.95 and (unique <= 0.05 or unique_count == 1):
        result["H1"] = _hypothesis(
            "supported",
            f"new teacher tau_to_L2R={tau:.3f}",
            f"unique_count={unique_count}, unique_fraction={unique:.3f}",
        )
    elif tau < 0.8 or unique > 0.2:
        result["H1"] = _hypothesis(
            "contradicted", f"new teacher tau_to_L2R={tau:.3f}", f"unique_fraction={unique:.3f}"
        )
    else:
        result["H1"] = _hypothesis("unresolved", f"intermediate teacher metrics tau={tau:.3f}, unique={unique:.3f}")

    per_sample = evidence.get("new_per_sample_tau_l2r")
    if tau is None or per_sample is None:
        result["H2"] = _hypothesis("unresolved", "new per-sample CDL metrics unavailable")
    elif tau >= 0.95 and abs(per_sample) <= 0.2:
        result["H2"] = _hypothesis(
            "supported", f"batch-mean tau_to_L2R={tau:.3f}", f"per-sample mean tau={per_sample:.3f}"
        )
    elif abs(per_sample) >= 0.5:
        result["H2"] = _hypothesis("contradicted", f"per-sample mean tau={per_sample:.3f}")
    else:
        result["H2"] = _hypothesis("unresolved", f"per-sample mean tau={per_sample:.3f}")

    ratio = evidence.get("new_input_fixed_ratio")
    destroyed = evidence.get("new_destroyed_tau_l2r_min")
    if ratio is None or destroyed is None:
        result["H3"] = _hypothesis("unresolved", "new input/fixed or destroyed-input metrics unavailable")
    elif ratio <= 0.1 and destroyed >= 0.9:
        result["H3"] = _hypothesis(
            "supported", f"input/fixed ratio={ratio:.3f}", f"minimum destroyed tau={destroyed:.3f}"
        )
    elif ratio >= 1.0:
        result["H3"] = _hypothesis(
            "contradicted",
            f"input/fixed ratio={ratio:.3f} rules out literal bias-only dominance",
            f"minimum in-distribution destroyed tau={destroyed:.3f}",
        )
    else:
        result["H3"] = _hypothesis("unresolved", f"ratio={ratio:.3f}, destroyed tau={destroyed:.3f}")

    old_noise = evidence.get("old_noise_tau_l2r")
    new_noise = evidence.get("new_noise_tau_l2r")
    if old_noise is None or new_noise is None:
        result["H4"] = _hypothesis("unresolved", "same-reference old/new noise metrics unavailable")
    elif abs(old_noise - new_noise) <= 0.1:
        result["H4"] = _hypothesis(
            "supported", f"same-reference old/new noise tau difference={abs(old_noise-new_noise):.3f}"
        )
    elif abs(old_noise - new_noise) >= 0.5:
        result["H4"] = _hypothesis(
            "contradicted", f"old noise tau={old_noise:.3f}", f"new noise tau={new_noise:.3f}"
        )
    else:
        result["H4"] = _hypothesis("unresolved", f"old/new noise tau={old_noise:.3f}/{new_noise:.3f}")

    old_unique = evidence.get("old_teacher_unique_fraction")
    old_unique_count = evidence.get("old_teacher_unique_count")
    old_pairwise = evidence.get("old_teacher_pairwise_tau")
    new_pairwise = evidence.get("new_teacher_pairwise_tau")
    if old_unique is None or unique is None:
        result["H5"] = _hypothesis("unresolved", "old/new teacher diversity metrics unavailable")
    elif (
        old_unique_count is not None
        and unique_count is not None
        and old_pairwise is not None
        and new_pairwise is not None
        and old_unique_count > unique_count
        and old_pairwise <= new_pairwise - 0.02
    ):
        result["H5"] = _hypothesis(
            "supported",
            f"old/new unique counts={old_unique_count}/{unique_count}",
            f"old/new pairwise tau={old_pairwise:.3f}/{new_pairwise:.3f}",
        )
    elif old_unique >= unique + 0.05 and old_unique >= 2 * max(unique, 1e-12):
        result["H5"] = _hypothesis(
            "supported", f"old unique fraction={old_unique:.3f}", f"new unique fraction={unique:.3f}"
        )
    elif old_unique <= unique + 0.01:
        result["H5"] = _hypothesis(
            "contradicted", f"old unique fraction={old_unique:.3f}", f"new unique fraction={unique:.3f}"
        )
    else:
        result["H5"] = _hypothesis("unresolved", f"old/new unique fractions={old_unique:.3f}/{unique:.3f}")
    return result


def save_graph_cache(path: str | Path, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    """Save extracted graph arrays with exact provenance metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def load_graph_cache(path: str | Path, expected_metadata: dict) -> dict[str, np.ndarray]:
    """Load a graph cache only when all provenance metadata matches exactly."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"graph cache {path} has no metadata_json")
        actual = json.loads(str(payload["metadata_json"].item()))
        if actual != expected_metadata:
            raise ValueError(
                "cache metadata mismatch: "
                f"expected={json.dumps(expected_metadata, sort_keys=True)}, "
                f"actual={json.dumps(actual, sort_keys=True)}"
            )
        return {key: payload[key].copy() for key in payload.files if key != "metadata_json"}


def write_csv_report(path: str | Path, fields: list[str], rows: list[dict]) -> None:
    """Write rows using a stable schema, retaining explicit unavailable reasons."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report_bundle(
    out_dir: str | Path,
    tables: dict[str, list[dict]],
    summary_md: str,
    config_diff_md: str,
) -> None:
    """Write the complete six-file report bundle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, fields in REPORT_FIELDS.items():
        write_csv_report(out_dir / filename, fields, tables.get(filename, []))
    (out_dir / "summary.md").write_text(summary_md)
    (out_dir / "config_diff.md").write_text(config_diff_md)


def _actual_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _cache_metadata(spec: RunSpec, M: int, reveals: int, device: str) -> dict:
    return {
        "run": spec.name,
        "source_ckpt": str(spec.source_ckpt.resolve()),
        "head": list(spec.head),
        "step": spec.step,
        "seed": spec.seed,
        "batch_size": spec.batch_size,
        "M": M,
        "reveals": reveals,
        "none_mode": spec.none_mode,
        "input_frame": spec.input_frame,
        "device": _actual_device(device),
    }


def extract_run_graphs(
    spec: RunSpec,
    M: int,
    reveals: int,
    device: str,
    cache_dir: str | Path,
    fwd_batch: int = 8,
) -> dict[str, np.ndarray]:
    """Extract and cache per-sample and batch-mean graphs for one run."""
    actual_device = _actual_device(device)
    metadata = _cache_metadata(spec, M, reveals, actual_device)
    cache_path = Path(cache_dir) / f"{spec.name}_graphs_M{M}_R{reveals}.npz"
    if cache_path.is_file():
        print(f"[cache] {spec.name}: {cache_path}", flush=True)
        return load_graph_cache(cache_path, metadata)

    from neural_readout.extract_b import _load_model_and_chunks

    total = M * spec.batch_size
    model, chunks, clean_perm, dev, _chunk_index = _load_model_and_chunks(
        str(spec.source_ckpt), total, spec.seed, actual_device, "train"
    )
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy().astype(np.int64)
    per_reveal = []

    if spec.none_mode == "b1":
        from per_head_order_scan import extract_per_head_and_heavy_A

        for reveal in range(reveals):
            reveal_seed = spec.seed + reveal * 1000
            A_lh, _ = extract_per_head_and_heavy_A(
                model,
                chunks,
                clean_perm,
                dev,
                seed=reveal_seed,
                fwd_batch=fwd_batch,
                none_mode="b1",
                head=spec.head,
            )
            A = A_lh[:, spec.head[0], spec.head[1]]
            B = np.transpose(A, (0, 2, 1)).astype(np.float32, copy=True)
            diagonal = np.arange(B.shape[-1])
            B[:, diagonal, diagonal] = 0
            per_reveal.append(B.reshape(M, spec.batch_size, B.shape[-1], B.shape[-1]))
    elif spec.none_mode == "strict65_model_strip_none":
        from analyses.uniform_label_free_v1 import _A_model_vec, _random_probe_orders
        from none_separated_block_graph import build_none_separated_B

        for reveal in range(reveals):
            B = np.zeros((total, 64, 64), dtype=np.float32)
            reveal_seed = spec.seed + reveal * 1000
            for start in range(0, total, max(1, fwd_batch)):
                stop = min(start + max(1, fwd_batch), total)
                probe = _random_probe_orders(
                    stop - start, reveal_seed * 100 + start, dev
                )
                model.eval()
                with torch.no_grad():
                    _, _, attn_list = model.forward_fn(
                        chunks[start:stop].to(dev), probe, return_attentions=True
                    )
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                attn = attn_list[spec.head[0]][:, spec.head[1]].detach().cpu().numpy()
                probe_np = probe.detach().cpu().numpy()
                for index in range(stop - start):
                    A65 = _A_model_vec(attn[index], probe_np[index])
                    B65 = build_none_separated_B(A65)
                    B[start + index] = B65[1:, 1:]
                print(
                    f"  [extract {spec.name} reveal={reveal}] {stop}/{total}", flush=True
                )
            per_reveal.append(B.reshape(M, spec.batch_size, 64, 64))
    else:
        raise ValueError(f"no extraction adapter for none_mode={spec.none_mode!r}")

    B_per_sample = np.stack(per_reveal).astype(np.float32)
    B_batch_mean = B_per_sample.astype(np.float64).mean(axis=2).astype(np.float32)
    diagonal = np.arange(B_batch_mean.shape[-1])
    B_batch_mean[:, :, diagonal, diagonal] = 0
    arrays = {
        "B_per_sample": B_per_sample,
        "B_batch_mean": B_batch_mean,
        "inv_perm": inv_perm,
    }
    save_graph_cache(cache_path, metadata, arrays)
    del model
    if actual_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return arrays


def _mean_paired_tau(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"paired order shapes differ: {left.shape} vs {right.shape}")
    return float(np.mean([order_tau(a, b) for a, b in zip(left, right)]))


def _physical_orders(spec: RunSpec, orders: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
    if spec.output_frame == "physical":
        return np.asarray(orders)
    return map_orders_to_physical(orders, inv_perm)


def _score_variance_mean(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    centered = scores - scores.mean(axis=1, keepdims=True)
    return float(np.var(centered, axis=0).mean())


def evaluate_input_variants(
    spec: RunSpec,
    model: torch.nn.Module,
    B_real: np.ndarray,
    teacher_orders: np.ndarray,
    cdl_real_orders: np.ndarray,
    inv_perm: np.ndarray,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, float]]:
    """Evaluate identical real/noise/zero/shuffle inputs for one readout."""
    variants = make_input_variants(B_real, seed=seed)
    real_scores, real_orders = predict_readout(model, variants["real"])
    real_physical = _physical_orders(spec, real_orders, inv_perm)
    l2r = np.tile(np.arange(B_real.shape[-1]), (len(B_real), 1))
    old_new_rows = []
    sanity_rows = []
    diversity_rows = []
    compact = {}

    for input_type, graphs in variants.items():
        scores, orders = predict_readout(model, graphs)
        physical = _physical_orders(spec, orders, inv_perm)
        tau_l2r = _mean_paired_tau(physical, l2r)
        tau_teacher = _mean_paired_tau(orders, teacher_orders)
        tau_cdl = _mean_paired_tau(orders, cdl_real_orders)
        tau_real = _mean_paired_tau(orders, real_orders)
        decomposition = input_fixed_decomposition(model, graphs)
        diversity = pairwise_order_stats(orders)
        reason = ""
        score_range = np.ptp(scores, axis=1)
        if float(np.max(score_range)) <= 1e-7:
            reason = "stable-argsort tie artifact: all node scores are equal"
        base = {
            "run": spec.name,
            "model": spec.architecture,
            "head": f"L{spec.head[0]}H{spec.head[1]}",
            "step": spec.step,
            "status": "ok",
            "reason": reason,
        }
        old_new_rows.append(
            {
                **base,
                "input": input_type,
                "tau_to_L2R": tau_l2r,
                "tau_to_teacher": tau_teacher,
                "tau_to_CDL_real": tau_cdl,
                "reference_frame": "physical for L2R; native run frame for teacher/CDL",
            }
        )
        sanity_rows.append(
            {
                **base,
                "input_type": input_type,
                "tau_to_L2R": tau_l2r,
                "tau_to_teacher": tau_teacher,
                "output_pairwise_tau_to_real": tau_real,
                **decomposition,
            }
        )
        diversity_rows.append(
            {
                "run": spec.name,
                "input_type": input_type,
                "output_pairwise_tau_mean": diversity["pairwise_tau_mean"],
                "output_pairwise_tau_min": diversity["pairwise_tau_min"],
                "output_unique_count": diversity["unique_count"],
                "score_variance_mean": _score_variance_mean(scores),
                "status": "ok",
                "reason": reason,
            }
        )
        compact[f"{input_type}_tau_l2r"] = tau_l2r
        compact[f"{input_type}_input_fixed_ratio"] = decomposition["input_fixed_ratio"]
    compact["real_score_variance"] = _score_variance_mean(real_scores)
    compact["real_tau_l2r"] = _mean_paired_tau(real_physical, l2r)
    return old_new_rows, sanity_rows, diversity_rows, compact


def _teacher_report_row(
    spec: RunSpec,
    orders: np.ndarray,
    physical_orders: np.ndarray,
    scope: str,
    reveal: str | int,
) -> dict:
    summary = teacher_collapse_summary(orders, physical_orders)
    return {
        "run": spec.name,
        "head": f"L{spec.head[0]}H{spec.head[1]}",
        "step": spec.step,
        "metric_scope": scope,
        "reveal": reveal,
        "n_orders": summary["n_orders"],
        "tau_meanCDL_to_L2R": summary["tau_to_l2r_mean"],
        "teacher_pairwise_tau_mean": summary["teacher_pairwise_tau_mean"],
        "teacher_pairwise_tau_min": summary["teacher_pairwise_tau_min"],
        "teacher_pairwise_tau_std": summary["teacher_pairwise_tau_std"],
        "unique_teacher_count": summary["unique_teacher_count"],
        "status": "ok",
        "reason": "",
    }


def _per_sample_teacher_row(
    spec: RunSpec,
    graph_data: dict[str, np.ndarray],
) -> tuple[dict, dict[str, float]]:
    B_per = graph_data["B_per_sample"]
    B_mean = graph_data["B_batch_mean"]
    inv_perm = graph_data["inv_perm"]
    per_orders_all = []
    per_physical_all = []
    paired_to_mean = []
    for reveal in range(B_per.shape[0]):
        mean_orders = teacher_orders_from_graphs(B_mean[reveal], spec.teacher_protocol)
        per_orders = teacher_orders_from_graphs(
            B_per[reveal].reshape(-1, B_per.shape[-1], B_per.shape[-1]),
            spec.teacher_protocol,
        ).reshape(B_per.shape[1], B_per.shape[2], B_per.shape[-1])
        for batch_index in range(B_per.shape[1]):
            for sample_order in per_orders[batch_index]:
                paired_to_mean.append(order_tau(sample_order, mean_orders[batch_index]))
        flat = per_orders.reshape(-1, per_orders.shape[-1])
        per_orders_all.append(flat)
        per_physical_all.append(_physical_orders(spec, flat, inv_perm))
    orders = np.concatenate(per_orders_all)
    physical = np.concatenate(per_physical_all)
    l2r = np.arange(orders.shape[1])
    tau_l2r = np.asarray([order_tau(order, l2r) for order in physical])
    diversity = pairwise_order_stats(orders)
    row = {
        "run": spec.name,
        "head": f"L{spec.head[0]}H{spec.head[1]}",
        "step": spec.step,
        "metric_scope": "per_sample_cdl",
        "reveal": "all",
        "n_orders": len(orders),
        "tau_per_sample_CDL_to_L2R_mean": float(tau_l2r.mean()),
        "tau_per_sample_CDL_to_L2R_std": float(tau_l2r.std()),
        "tau_per_sample_CDL_to_teacher_mean": float(np.mean(paired_to_mean)),
        "sample_pairwise_tau_mean": diversity["pairwise_tau_mean"],
        "sample_pairwise_tau_std": diversity["pairwise_tau_std"],
        "unique_teacher_count": diversity["unique_count"],
        "status": "ok",
        "reason": "",
    }
    return row, {"per_sample_tau_l2r": float(tau_l2r.mean())}


def _unavailable_teacher_row(spec: RunSpec, reason: str) -> dict:
    return {
        "run": spec.name,
        "head": f"L{spec.head[0]}H{spec.head[1]}",
        "step": spec.step,
        "metric_scope": "per_sample_cdl",
        "reveal": "all",
        "status": "unavailable",
        "reason": reason,
    }


def _first_step_entropy(orders: np.ndarray) -> float:
    values, counts = np.unique(np.asarray(orders)[:, 0], return_counts=True)
    del values
    probabilities = counts / counts.sum()
    return -float(np.sum(probabilities * np.log(probabilities)))


def render_config_diff(
    specs: dict[str, RunSpec],
    old_teacher: np.ndarray,
    new_teacher: np.ndarray | None,
) -> str:
    old = specs["old_gbeta"]
    new = specs["new_flatten_readout"]
    old_entropy = _first_step_entropy(old_teacher)
    new_entropy = _first_step_entropy(new_teacher) if new_teacher is not None else None
    rows = [
        ("head", f"L{old.head[0]}H{old.head[1]}", f"L{new.head[0]}H{new.head[1]}", "yes"),
        ("checkpoint step", old.step, new.step, "no"),
        ("seed", old.seed, new.seed, "yes"),
        ("teacher", old.teacher_protocol, new.teacher_protocol, "yes"),
        ("batch-mean B", "yes", "yes", "yes: different batch size/frame"),
        ("per-sample teacher", "no", "no", "yes"),
        ("multi-reveal average rank", "no", "no", "yes"),
        ("readout architecture", "NodewiseReadout", "FlattenReadout", "yes"),
        ("loss", "pairwise logistic", "pairwise BCE", "yes"),
        ("bias terms", "yes, shared nodewise", "yes, position-specific MLP", "yes"),
        ("input normalization", "none", "none", "no"),
        ("dropout", "0", "0", "no"),
        ("training perturbation", "random reveal only", "random reveal only", "no"),
        ("zero/noise/shuffle gate", "posthoc only", "absent during training", "yes"),
        ("label shared within source batch", "yes", "yes", "yes"),
        ("first-step target entropy", f"{old_entropy:.6f}", "unavailable" if new_entropy is None else f"{new_entropy:.6f}", "yes"),
    ]
    lines = [
        "# Old vs new configuration diff",
        "",
        "| item | old_gbeta | new_flatten_readout | suspicious? |",
        "|---|---|---|---|",
    ]
    lines.extend(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
    lines.extend(
        [
            "",
            "The old graph and output are in physical coordinates. The new L1H7 graph and readout output are in model coordinates and are mapped through `inv_perm` only for physical-L2R evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def render_summary(
    hypotheses: dict[str, dict[str, object]],
    evidence: dict[str, float],
    scope: str,
    device: str,
    extraction_errors: list[str],
) -> str:
    supported = [name for name, item in hypotheses.items() if item["status"] == "supported"]
    contradicted = [name for name, item in hypotheses.items() if item["status"] == "contradicted"]
    if "H1" in supported and "H3" in supported:
        root_cause = (
            "The new batch-mean teacher is near-constant physical L2R, making a constant solution optimal, "
            "and FlattenReadout realizes that shortcut through its fixed positional component."
        )
    elif "H1" in supported:
        root_cause = "The strongest supported cause is batch-mean teacher collapse to physical L2R."
    elif "H3" in supported:
        root_cause = "The strongest supported cause is a fixed-position FlattenReadout shortcut."
    else:
        root_cause = "The available quick-run evidence does not isolate a single root cause."
    lines = [
        "# Readout collapse diagnosis",
        "",
        f"Scope: `{scope}`. Actual device: `{device}`.",
        "",
        "## Most likely root cause",
        "",
        root_cause,
        "",
        "## Hypotheses",
        "",
        "| hypothesis | status | evidence |",
        "|---|---|---|",
    ]
    for name in ("H1", "H2", "H3", "H4", "H5"):
        item = hypotheses[name]
        lines.append(f"| {name} | {item['status']} | {'; '.join(item['evidence'])} |")
    lines.extend(
        [
            "",
            f"Supported: {', '.join(supported) if supported else 'none'}.",
            f"Contradicted: {', '.join(contradicted) if contradicted else 'none'}.",
            "",
            "## Recommended fixes",
            "",
            "1. Do not use batch-mean CDL as a sample-level readout teacher.",
            "2. Evaluate per-sample CDL stability before using it as supervision.",
            "3. Prefer multi-reveal average rank: run CDL per sample/reveal, then aggregate ranks.",
            "4. Make zero, Gaussian-noise, row/column-shuffle, and entry-shuffle checks mandatory gates.",
            "5. Reject checkpoints whose destroyed-input order remains near L2R or whose `f(B)-f(0)` contribution is negligible.",
            "6. Exclude heads with near-one batch-mean L2R tau and negligible per-sample diversity, or use a validated top-k set.",
            "",
            "## Minimal next experiment",
            "",
            "Keep L1H7, FlattenReadout, loss, seed, and optimizer fixed; change only the target from batch-mean CDL to per-sample multi-reveal average rank. Re-run the same destroyed-input table. This isolates teacher collapse before changing architecture or bias.",
            "",
            "## Metric convention",
            "",
            "All non-identity Kendall tau comparisons convert reveal orders to node-rank vectors first. L2R tau is computed after mapping model-frame outputs through `inv_perm` into physical coordinates.",
        ]
    )
    if extraction_errors:
        lines.extend(["", "## Unavailable phases", ""])
        lines.extend(f"- {error}" for error in extraction_errors)
    lines.extend(["", "## Evidence keys", "", "```json", json.dumps(evidence, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def run_diagnostic(args: argparse.Namespace) -> Path:
    artifact_root = Path(args.artifact_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    specs = build_default_run_specs(artifact_root)
    for spec in specs.values():
        validate_run_spec(spec)
    actual_device = _actual_device(args.device)
    print(f"[device] requested={args.device} actual={actual_device}", flush=True)

    old = specs["old_gbeta"]
    new = specs["new_flatten_readout"]
    old_data = load_saved_batch_split(old, split="val")
    old_model, _ = load_readout_checkpoint(old.readout_path, actual_device)
    new_model, _ = load_readout_checkpoint(new.readout_path, actual_device)

    old_B = old_data["B"][: args.input_samples]
    old_teacher_input = old_data["teacher_order"][: len(old_B)]
    old_cdl_input = teacher_orders_from_graphs(old_B, old.teacher_protocol)
    identity = np.arange(old_B.shape[-1])
    old_inv = identity.copy()

    old_teacher_all = old_data["teacher_order"]
    teacher_rows = [
        _teacher_report_row(old, old_teacher_all, old_teacher_all, "batch_mean_saved", "training")
    ]
    old_summary = teacher_collapse_summary(old_teacher_all, old_teacher_all)

    tables = {filename: [] for filename in REPORT_FIELDS}
    old_rows = evaluate_input_variants(
        old,
        old_model,
        old_B,
        old_teacher_input,
        old_cdl_input,
        old_inv,
        seed=args.perturb_seed,
    )
    for filename, rows in zip(
        ("old_vs_new_metrics.csv", "input_sanity_metrics.csv", "output_diversity_metrics.csv"),
        old_rows[:3],
    ):
        tables[filename].extend(rows)
    old_compact = old_rows[3]

    extraction_errors: list[str] = []
    new_graphs = None
    old_graphs = None
    M = args.quick_m if args.scope == "quick" else args.full_m
    reveals = args.reveals
    if not args.no_extract:
        try:
            new_graphs = extract_run_graphs(
                new, M, reveals, actual_device, out_dir / "cache", args.fwd_batch
            )
        except Exception as error:
            extraction_errors.append(f"new extraction failed: {type(error).__name__}: {error}")
        if args.scope == "full" or args.extract_old:
            try:
                old_graphs = extract_run_graphs(
                    old, M, reveals, actual_device, out_dir / "cache", args.fwd_batch
                )
            except Exception as error:
                extraction_errors.append(f"old per-sample extraction failed: {type(error).__name__}: {error}")

    new_teacher_all = None
    new_compact: dict[str, float] = {}
    per_sample_evidence: dict[str, float] = {}
    if new_graphs is not None:
        inv_perm = new_graphs["inv_perm"]
        mean_orders_by_reveal = []
        for reveal in range(new_graphs["B_batch_mean"].shape[0]):
            orders = teacher_orders_from_graphs(
                new_graphs["B_batch_mean"][reveal], new.teacher_protocol
            )
            physical = _physical_orders(new, orders, inv_perm)
            mean_orders_by_reveal.append(orders)
            teacher_rows.append(
                _teacher_report_row(new, orders, physical, "batch_mean_cdl", reveal)
            )
        new_teacher_all = np.concatenate(mean_orders_by_reveal)
        new_teacher_physical = _physical_orders(new, new_teacher_all, inv_perm)
        teacher_rows.append(
            _teacher_report_row(
                new, new_teacher_all, new_teacher_physical, "batch_mean_cdl", "all"
            )
        )
        per_sample_row, per_sample_evidence = _per_sample_teacher_row(new, new_graphs)
        teacher_rows.append(per_sample_row)

        new_B = new_graphs["B_batch_mean"][0]
        new_teacher_input = mean_orders_by_reveal[0]
        new_rows = evaluate_input_variants(
            new,
            new_model,
            new_B,
            new_teacher_input,
            new_teacher_input,
            inv_perm,
            seed=args.perturb_seed,
        )
        for filename, rows in zip(
            ("old_vs_new_metrics.csv", "input_sanity_metrics.csv", "output_diversity_metrics.csv"),
            new_rows[:3],
        ):
            tables[filename].extend(rows)
        new_compact = new_rows[3]
    else:
        reason = "new real graphs unavailable; run without --no-extract or inspect extraction error"
        extraction_errors.append(reason)
        for input_type in (
            "real", "gaussian", "gaussian_matched", "zero", "rowcol_shuffled", "entry_shuffled"
        ):
            tables["old_vs_new_metrics.csv"].append(
                {
                    "run": new.name,
                    "model": new.architecture,
                    "head": f"L{new.head[0]}H{new.head[1]}",
                    "step": new.step,
                    "input": input_type,
                    "status": "unavailable",
                    "reason": reason,
                }
            )
            tables["input_sanity_metrics.csv"].append(
                {
                    "run": new.name,
                    "model": new.architecture,
                    "input_type": input_type,
                    "status": "unavailable",
                    "reason": reason,
                }
            )
            tables["output_diversity_metrics.csv"].append(
                {"run": new.name, "input_type": input_type, "status": "unavailable", "reason": reason}
            )

    if old_graphs is not None:
        old_per_sample_row, _ = _per_sample_teacher_row(old, old_graphs)
        teacher_rows.append(old_per_sample_row)
    else:
        teacher_rows.append(
            _unavailable_teacher_row(old, "quick mode reused saved batch means; old per-sample extraction not requested")
        )
    tables["teacher_collapse_metrics.csv"] = teacher_rows

    evidence: dict[str, float] = {
        "old_teacher_unique_fraction": old_summary["unique_teacher_count"] / old_summary["n_orders"],
        "old_teacher_unique_count": old_summary["unique_teacher_count"],
        "old_teacher_pairwise_tau": old_summary["teacher_pairwise_tau_mean"],
        "old_noise_tau_l2r": old_compact["gaussian_matched_tau_l2r"],
    }
    if new_teacher_all is not None:
        # Reveal 0 exactly matches the training extractor's seed schedule. Other
        # reveals are a stability diagnostic, not the teacher distribution used
        # to fit the saved FlattenReadout.
        training_orders = mean_orders_by_reveal[0]
        training_physical = _physical_orders(new, training_orders, new_graphs["inv_perm"])
        new_summary = teacher_collapse_summary(training_orders, training_physical)
        evidence.update(
            {
                "new_teacher_tau_l2r": new_summary["tau_to_l2r_mean"],
                "new_teacher_unique_fraction": new_summary["unique_teacher_count"] / new_summary["n_orders"],
                "new_teacher_unique_count": new_summary["unique_teacher_count"],
                "new_teacher_pairwise_tau": new_summary["teacher_pairwise_tau_mean"],
                "new_per_sample_tau_l2r": per_sample_evidence["per_sample_tau_l2r"],
                "new_input_fixed_ratio": new_compact["real_input_fixed_ratio"],
                "new_noise_tau_l2r": new_compact["gaussian_matched_tau_l2r"],
                "new_destroyed_tau_l2r_min": min(
                    new_compact[f"{name}_tau_l2r"]
                    for name in ("gaussian_matched", "zero", "rowcol_shuffled", "entry_shuffled")
                ),
            }
        )
    hypotheses = assess_hypotheses(evidence)
    summary_md = render_summary(
        hypotheses, evidence, args.scope, actual_device, extraction_errors
    )
    config_md = render_config_diff(specs, old_teacher_all, new_teacher_all)
    write_report_bundle(out_dir, tables, summary_md, config_md)
    print(f"[saved] {out_dir}", flush=True)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("quick", "full"), default="quick")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact-root", default=str(_ROOT.parent.parent if _ROOT.parent.name == ".worktrees" else _ROOT))
    parser.add_argument("--out-dir", default="diagnostics/readout_collapse")
    parser.add_argument("--quick-m", type=int, default=5)
    parser.add_argument("--full-m", type=int, default=50)
    parser.add_argument("--reveals", type=int, default=2)
    parser.add_argument("--input-samples", type=int, default=100)
    parser.add_argument("--fwd-batch", type=int, default=8)
    parser.add_argument("--perturb-seed", type=int, default=20260630)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--extract-old", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.quick_m <= 0 or args.full_m <= 0 or args.reveals <= 0:
        raise ValueError("M and reveals must be positive")
    run_diagnostic(args)


def order_to_rank(order: np.ndarray) -> np.ndarray:
    """Convert a reveal-order permutation to node ranks (zero is earliest)."""
    order = np.asarray(order)
    if order.ndim != 1:
        raise ValueError(f"order must be one-dimensional, got {order.shape}")
    n = len(order)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError(f"order must be a permutation of 0..{n - 1}")
    rank = np.empty(n, dtype=np.int64)
    rank[order.astype(np.int64)] = np.arange(n, dtype=np.int64)
    return rank


def order_tau(left: np.ndarray, right: np.ndarray) -> float:
    """Kendall tau between two reveal orders, computed on node rank vectors."""
    left_rank = order_to_rank(left)
    right_rank = order_to_rank(right)
    if len(left_rank) != len(right_rank):
        raise ValueError("orders must have equal length")
    value = kendalltau(left_rank, right_rank)[0]
    if np.isnan(value):
        raise ValueError("Kendall tau is undefined for the supplied orders")
    return float(value)


def pairwise_order_stats(orders: np.ndarray) -> dict[str, float | int | None]:
    """Exact diversity statistics for a two-dimensional order array."""
    orders = np.asarray(orders)
    if orders.ndim != 2:
        raise ValueError(f"orders must have shape (M, N), got {orders.shape}")
    for order in orders:
        order_to_rank(order)
    unique_count = len({tuple(int(x) for x in order) for order in orders})
    if len(orders) < 2:
        return {
            "unique_count": unique_count,
            "pairwise_tau_mean": None,
            "pairwise_tau_min": None,
            "pairwise_tau_std": None,
        }
    taus = np.asarray(
        [order_tau(orders[i], orders[j]) for i, j in combinations(range(len(orders)), 2)],
        dtype=np.float64,
    )
    return {
        "unique_count": unique_count,
        "pairwise_tau_mean": float(taus.mean()),
        "pairwise_tau_min": float(taus.min()),
        "pairwise_tau_std": float(taus.std()),
    }


def _zero_diagonal(B: np.ndarray) -> np.ndarray:
    out = np.asarray(B).copy()
    diagonal = np.arange(out.shape[-1])
    out[:, diagonal, diagonal] = 0
    return out


def make_input_variants(B: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Create deterministic real/noise/zero/shuffle variants of graph inputs."""
    B = np.asarray(B)
    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError(f"B must have shape (M, N, N), got {B.shape}")
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=B.shape).astype(B.dtype, copy=False)
    gaussian_matched = rng.normal(
        loc=float(B.mean()), scale=float(B.std()), size=B.shape
    ).astype(B.dtype, copy=False)
    rowcol = np.empty_like(B)
    entry = np.empty_like(B)
    n = B.shape[-1]
    off_diagonal = ~np.eye(n, dtype=bool)
    for index, graph in enumerate(B):
        permutation = rng.permutation(n)
        rowcol[index] = graph[permutation][:, permutation]
        values = graph[off_diagonal].copy()
        rng.shuffle(values)
        entry[index] = 0
        entry[index][off_diagonal] = values
    return {
        "real": _zero_diagonal(B),
        "gaussian": _zero_diagonal(gaussian),
        "gaussian_matched": _zero_diagonal(gaussian_matched),
        "zero": np.zeros_like(B),
        "rowcol_shuffled": _zero_diagonal(rowcol),
        "entry_shuffled": _zero_diagonal(entry),
    }


def load_readout_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[torch.nn.Module, dict]:
    """Load one supported readout checkpoint and return its validated config."""
    from batch_readout.model import FlattenReadout, NodewiseReadout

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model" not in payload or "config" not in payload:
        raise ValueError(f"readout checkpoint {path} must contain model and config")
    config = dict(payload["config"])
    architecture = config.get("model_name")
    n = int(config.get("N", 64))
    if architecture == "flatten":
        model = FlattenReadout(N=n, hidden=tuple(config.get("hidden", (1024, 256))))
    elif architecture == "nodewise":
        model = NodewiseReadout(
            N=n,
            d_model=int(config.get("d_model", 64)),
            n_layers=int(config.get("n_layers", 2)),
            n_heads=int(config.get("n_heads", 4)),
        )
    else:
        raise ValueError(f"unknown readout architecture {architecture!r}")
    model.load_state_dict(payload["model"])
    model.to(torch.device(device))
    model.eval()
    return model, config


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def predict_readout(
    model: torch.nn.Module,
    B: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw scores and descending-score reveal orders."""
    B = np.asarray(B, dtype=np.float32)
    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError(f"B must have shape (M, N, N), got {B.shape}")
    device = _model_device(model)
    score_batches = []
    for start in range(0, len(B), batch_size):
        tensor = torch.from_numpy(B[start : start + batch_size]).to(device)
        score_batches.append(model(tensor).detach().cpu().numpy())
    scores = np.concatenate(score_batches, axis=0)
    orders = np.argsort(-scores, axis=1, kind="stable").astype(np.int64)
    return scores, orders


def centered_score_std(scores: np.ndarray) -> float:
    """Across-sample score variation after removing ranking-invariant offsets."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (M, N), got {scores.shape}")
    centered = scores - scores.mean(axis=1, keepdims=True)
    return float(centered.std(axis=0).mean())


def input_fixed_decomposition(model: torch.nn.Module, B: np.ndarray) -> dict[str, float]:
    """Decompose scores into fixed f(0) and input-dependent f(B)-f(0) terms."""
    B = np.asarray(B, dtype=np.float32)
    scores, _ = predict_readout(model, B)
    fixed_scores, _ = predict_readout(model, np.zeros_like(B))
    input_scores = scores - fixed_scores
    input_norm = float(np.linalg.norm(input_scores, axis=1).mean())
    fixed_norm = float(np.linalg.norm(fixed_scores, axis=1).mean())
    if fixed_norm <= 1e-12:
        ratio = float("inf") if input_norm > 1e-12 else 0.0
    else:
        ratio = input_norm / fixed_norm
    result = {
        "score_std_across_samples": centered_score_std(scores),
        "input_term_norm": input_norm,
        "fixed_term_norm": fixed_norm,
        "input_fixed_ratio": float(ratio),
    }

    # Additional literal last-layer decomposition for FlattenReadout.
    net = getattr(model, "net", None)
    if isinstance(net, torch.nn.Sequential) and isinstance(net[-1], torch.nn.Linear):
        device = _model_device(model)
        with torch.no_grad():
            tensor = torch.from_numpy(B).to(device)
            hidden = net[:-1](tensor.reshape(len(B), -1))
            pre_bias = torch.nn.functional.linear(hidden, net[-1].weight, bias=None)
            result["final_prebias_norm"] = float(
                torch.linalg.vector_norm(pre_bias, dim=1).mean().cpu()
            )
            result["final_bias_norm"] = float(torch.linalg.vector_norm(net[-1].bias).cpu())
    return result


if __name__ == "__main__":
    main()

