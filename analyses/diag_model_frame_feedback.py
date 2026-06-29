#!/usr/bin/env python3
"""Model-frame head-order ladder for feedback/controller checkpoints.

This keeps the recovered rollout in model-frame coordinates for the primary
scores. Physical-frame scores are emitted only as posthoc references.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts"))

from none_separated_block_graph import build_none_separated_B, rollout_by_method  # noqa: E402
from search_strict_label_free_65 import _extract_A_with_none_lh_model_frame  # noqa: E402


DEFAULT_METHODS = ["C-D+L", "L", "C-D", "C+L", "C"]


def _tau(a: np.ndarray, b: np.ndarray) -> float:
    val, _ = kendalltau(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    return float(val) if not np.isnan(val) else float("nan")


def _block_perm_from_inv(inv_perm: np.ndarray) -> np.ndarray:
    inv = np.asarray(inv_perm, dtype=np.int64)
    block = np.empty_like(inv)
    block[inv] = np.arange(inv.shape[0], dtype=np.int64)
    return block


def _parse_ckpt_spec(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        path = pathlib.Path(spec)
        return _infer_label(path), path
    label, path = spec.split("=", 1)
    return label.strip(), pathlib.Path(path)


def _infer_label(path: pathlib.Path) -> str:
    step = _infer_step(path)
    parent = path.parent.name
    return f"{parent}_step{step}" if step is not None else parent


def _infer_step(path: pathlib.Path) -> int | None:
    match = re.search(r"step(\d+)", path.name)
    return int(match.group(1)) if match else None


def _rank_rows(rows: list[dict], metric: str) -> dict[tuple[str, int, int, str], int]:
    by_ckpt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ckpt[row["label"]].append(row)

    ranks: dict[tuple[str, int, int, str], int] = {}
    for label, group in by_ckpt.items():
        ordered = sorted(group, key=lambda r: r[metric], reverse=True)
        for rank, row in enumerate(ordered, start=1):
            key = (label, int(row["layer"]), int(row["head"]), row["method"])
            ranks[key] = rank
    return ranks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", action="append", required=True, help="label=path, repeatable")
    p.add_argument("--out-dir", default=str(ROOT / "reports/model_frame_feedback_diag"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fwd-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perm-orientation", choices=["phys_to_model", "model_to_phys"], default="model_to_phys")
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    p.add_argument("--tracked-head", nargs=2, type=int, metavar=("LAYER", "HEAD"), default=[0, 2])
    p.add_argument("--gbeta-ckpt", default=None,
                   help="optional head-gated g_beta checkpoint; emits gate weights per scanned checkpoint")
    p.add_argument("--gbeta-layer", type=int, default=0,
                   help="layer to feed into --gbeta-ckpt gate diagnostics")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = [_parse_ckpt_spec(spec) for spec in args.ckpt]
    rows: list[dict] = []
    gate_rows: list[dict] = []
    metas: list[dict] = []
    gbeta_model = None
    gbeta_device = None
    if args.gbeta_ckpt:
        import torch
        from batch_readout.head_gated_gbeta import build_head_gated_gbeta

        state = torch.load(args.gbeta_ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        gbeta_model = build_head_gated_gbeta(
            cfg.get("variant", "head_gated"),
            N=int(cfg.get("N", 64)),
            H=int(cfg.get("H", 8)),
            head_idx=int(cfg.get("head_idx", 0)),
            gate_mode=cfg.get("gate_mode", "soft_all"),
            topk=int(cfg.get("topk", 2)),
        )
        gbeta_model.load_state_dict(state["model"])
        gbeta_model.eval()
        gbeta_device = torch.device(args.device)
        if gbeta_device.type == "cuda" and not torch.cuda.is_available():
            gbeta_device = torch.device("cpu")
        gbeta_model.to(gbeta_device)

    for label, ckpt_path in ckpts:
        if not ckpt_path.exists():
            print(f"[skip] missing ckpt: {label} -> {ckpt_path}", flush=True)
            continue

        run_args = argparse.Namespace(**vars(args))
        run_args.ckpt = str(ckpt_path)

        print(f"=== {label}: {ckpt_path} ===", flush=True)
        A_lh, inv_perm, meta = _extract_A_with_none_lh_model_frame(run_args)
        inv_perm = np.asarray(inv_perm, dtype=np.int64)   # model -> physical
        block_perm = _block_perm_from_inv(inv_perm)       # physical -> model
        identity = np.arange(inv_perm.shape[0], dtype=np.int64)
        tracked_layer, tracked_head = [int(x) for x in args.tracked_head]

        meta = dict(meta)
        meta["label"] = label
        meta["step"] = _infer_step(ckpt_path)
        metas.append(meta)

        L, H = A_lh.shape[:2]
        if gbeta_model is not None:
            import torch

            layer = int(args.gbeta_layer)
            if layer < 0 or layer >= L:
                raise ValueError(f"--gbeta-layer {layer} outside extracted layer range 0..{L - 1}")
            B_heads = np.asarray(A_lh[layer, :, :, 1:].transpose(0, 2, 1), dtype=np.float32)
            diag = np.arange(B_heads.shape[-1])
            B_heads[:, diag, diag] = 0.0
            with torch.no_grad():
                _, aux = gbeta_model(torch.from_numpy(B_heads[None]).to(gbeta_device))
            weights = aux["gate_weights"][0].detach().cpu().numpy()
            entropy = float(-(weights * np.log(weights + 1e-12)).sum())
            top = np.argsort(-weights)
            for rank, head in enumerate(top, start=1):
                gate_rows.append({
                    "label": label,
                    "step": meta["step"],
                    "ckpt": str(ckpt_path),
                    "gbeta_ckpt": str(args.gbeta_ckpt),
                    "layer": layer,
                    "head": int(head),
                    "rank": rank,
                    "weight": float(weights[head]),
                    "gate_entropy": entropy,
                    "top1_head": int(top[0]),
                    "top2_heads": " ".join(str(int(x)) for x in top[:2]),
                })

        for layer in range(L):
            for head in range(H):
                B65_model = build_none_separated_B(A_lh[layer, head])
                for method in args.methods:
                    sigma_model = rollout_by_method(B65_model, method)
                    sigma_phys = inv_perm[sigma_model]
                    rows.append({
                        "label": label,
                        "step": meta["step"],
                        "ckpt": str(ckpt_path),
                        "layer": layer,
                        "head": head,
                        "method": method,
                        "is_tracked_head": int(layer == tracked_layer and head == tracked_head),
                        "tau_model_vs_semantic_path": _tau(sigma_model, block_perm),
                        "tau_model_vs_identity": _tau(sigma_model, identity),
                        "tau_phys_vs_l2r": _tau(sigma_phys, identity),
                        "tau_phys_vs_imposed_shuffled_order": _tau(sigma_phys, inv_perm),
                        "sigma_model_first16": " ".join(str(int(x)) for x in sigma_model[:16]),
                        "sigma_phys_first16": " ".join(str(int(x)) for x in sigma_phys[:16]),
                    })

    if not rows:
        raise RuntimeError("no checkpoint rows produced")

    rank_sem = _rank_rows(rows, "tau_model_vs_semantic_path")
    rank_id = _rank_rows(rows, "tau_model_vs_identity")
    for row in rows:
        key = (row["label"], int(row["layer"]), int(row["head"]), row["method"])
        row["rank_model_semantic_path"] = rank_sem[key]
        row["rank_model_identity"] = rank_id[key]

    fieldnames = [
        "label", "step", "ckpt", "layer", "head", "method", "is_tracked_head",
        "tau_model_vs_semantic_path", "rank_model_semantic_path",
        "tau_model_vs_identity", "rank_model_identity",
        "tau_phys_vs_l2r", "tau_phys_vs_imposed_shuffled_order",
        "sigma_model_first16", "sigma_phys_first16",
    ]
    tsv_path = out_dir / "model_frame_feedback.tsv"
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    gate_path = None
    if gate_rows:
        gate_path = out_dir / "head_gated_gate.tsv"
        gate_fields = [
            "label", "step", "ckpt", "gbeta_ckpt", "layer", "head", "rank",
            "weight", "gate_entropy", "top1_head", "top2_heads",
        ]
        with gate_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=gate_fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(gate_rows)

    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    summary = {
        "local_convention": {
            "inv_perm": "inv_perm[model_block] = physical_block",
            "block_perm": "block_perm[physical_block] = model_block",
            "identity_model_order": "0,1,2,...,N-1",
            "semantic_model_path": "block_perm[np.arange(N)]",
            "primary_scores": ["tau_model_vs_semantic_path", "tau_model_vs_identity"],
        },
        "tracked_head": {"layer": args.tracked_head[0], "head": args.tracked_head[1]},
        "gbeta_gate": {
            "ckpt": str(args.gbeta_ckpt) if args.gbeta_ckpt else None,
            "layer": args.gbeta_layer,
            "gate_tsv": str(gate_path) if gate_path else None,
        },
        "meta": metas,
        "best_by_label": {},
        "tracked_by_label": {},
    }
    for label, group in by_label.items():
        summary["best_by_label"][label] = {
            "best_semantic_model_path": max(group, key=lambda r: r["tau_model_vs_semantic_path"]),
            "best_identity": max(group, key=lambda r: r["tau_model_vs_identity"]),
        }
        tracked = [r for r in group if r["is_tracked_head"]]
        summary["tracked_by_label"][label] = sorted(
            tracked,
            key=lambda r: r["tau_model_vs_semantic_path"],
            reverse=True,
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=== model-frame feedback diagnostic saved ===")
    print(f"TSV: {tsv_path}")
    if gate_path:
        print(f"Gate TSV: {gate_path}")
    print(f"Summary: {summary_path}")
    for label in sorted(summary["best_by_label"]):
        row = summary["best_by_label"][label]["best_semantic_model_path"]
        print(
            f"{label}: best_sem=L{row['layer']}H{row['head']} {row['method']} "
            f"tau_model_sem={row['tau_model_vs_semantic_path']:.4f} "
            f"tau_model_id={row['tau_model_vs_identity']:.4f}"
        )


if __name__ == "__main__":
    main()
