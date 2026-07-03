#!/usr/bin/env python3
"""Run full-validation PPL for a suite of fixed block-order JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from order_utils import build_fixed_block_permutation, invert_permutation  # noqa: E402


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def load_model_meta(checkpoint: Dict):
    return AOGPT(AOGPTConfig(**dict(checkpoint["model_args"])))


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


def extract_order_from_json(path: Path, num_blocks: int) -> List[int]:
    payload = json.loads(path.read_text())
    order = None
    if isinstance(payload.get("best_candidate"), dict) and payload["best_candidate"].get("order") is not None:
        order = payload["best_candidate"]["order"]
    else:
        for key in ("order", "block_order", "blocks", "segment"):
            if payload.get(key) is not None:
                order = payload[key]
                break
    if order is None:
        raise ValueError(f"Could not find order in {path}")
    order = [int(v) for v in order]
    if len(order) != int(num_blocks) or sorted(order) != list(range(int(num_blocks))):
        raise ValueError(f"Invalid order in {path}: len={len(order)}, unique={len(set(order))}")
    return order


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
    return float((concordant - discordant) / (n * (n - 1) / 2))


def to_original_order(order_current: List[int], perm_state) -> List[int]:
    if perm_state is None:
        return [int(v) for v in order_current]
    mapper = perm_state["block_perm"].to(dtype=torch.long, device="cpu")
    current = torch.as_tensor(order_current, dtype=torch.long)
    return [int(v) for v in mapper[current].tolist()]


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def parse_source(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--order_source entries must be LABEL=PATH")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path)


def iter_orders(sources: Iterable[str]) -> List[Tuple[str, str, Path]]:
    rows = []
    for raw in sources:
        label, path = parse_source(raw)
        if path.is_dir():
            for order_path in sorted(path.glob("*.json")):
                rows.append((label, order_path.stem, order_path))
        elif path.is_file():
            rows.append((label, path.stem, path))
        else:
            raise FileNotFoundError(path)
    return rows


def run_eval(command: List[str], env: Dict[str, str]):
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full-val order suite.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--order_source", action="append", default=[], help="LABEL=DIR_OR_JSON")
    parser.add_argument("--include_baselines", action="store_true")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23456)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = args.out_root
    eval_root = out_root / "evals"
    eval_root.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.ckpt_path)
    model_meta = load_model_meta(checkpoint)
    num_blocks = int(model_meta.num_blocks)
    perm_state = permutation_state(checkpoint, model_meta)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    command_lines = []
    targets = []
    if args.include_baselines:
        for mode in ("OriginalL2R", "AR", "Random"):
            name = sanitize(f"baseline__{mode}")
            out_dir = eval_root / name
            cmd = [
                sys.executable,
                "scripts/eval/eval_lm_original_order_ppl.py",
                "--ckpt_path",
                str(args.ckpt_path),
                "--out_dir",
                str(out_dir),
                "--split",
                str(args.split),
                "--batch_size",
                str(args.batch_size),
                "--num_batches",
                "0",
                "--sample_mode",
                "sequential",
                "--sequential_cover_tail",
                "--eval_mode",
                mode,
                "--seed",
                str(args.seed),
                "--device",
                str(args.device),
                "--dtype",
                str(args.dtype),
            ]
            targets.append({"kind": "baseline", "source": "baseline", "method": mode, "out_dir": out_dir, "cmd": cmd})

    for source_label, method, order_path in iter_orders(args.order_source):
        name = sanitize(f"{source_label}__{method}")
        out_dir = eval_root / name
        cmd = [
            sys.executable,
            "scripts/eval/eval_lm_original_order_ppl.py",
            "--ckpt_path",
            str(args.ckpt_path),
            "--out_dir",
            str(out_dir),
            "--split",
            str(args.split),
            "--batch_size",
            str(args.batch_size),
            "--num_batches",
            "0",
            "--sample_mode",
            "sequential",
            "--sequential_cover_tail",
            "--eval_mode",
            "BlockOrder",
            "--order_json",
            str(order_path),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--dtype",
            str(args.dtype),
        ]
        targets.append(
            {
                "kind": "order_json",
                "source": source_label,
                "method": method,
                "order_json": order_path,
                "out_dir": out_dir,
                "cmd": cmd,
            }
        )

    for target in targets:
        command_lines.append("CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 " + " ".join(str(x) for x in target["cmd"]))
    (out_root / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(command_lines) + "\n")

    for idx, target in enumerate(targets, start=1):
        summary_path = target["out_dir"] / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"[{idx}/{len(targets)}] skip existing {target['source']}::{target['method']}")
            continue
        print(f"[{idx}/{len(targets)}] eval {target['source']}::{target['method']}")
        run_eval(target["cmd"], env)

    rows = []
    for target in targets:
        summary_path = target["out_dir"] / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        row = {
            "source": target["source"],
            "method": target["method"],
            "kind": target["kind"],
            "mean_nll": float(summary["mean_nll_original_frame"]),
            "ppl": float(summary["ppl_original_frame"]),
            "num_samples": int(summary["num_samples"]),
            "num_scored_tokens": int(summary["num_scored_tokens"]),
            "summary_path": str(summary_path),
            "order_json": str(target.get("order_json", "")),
            "tau_current_diagnostic": "",
            "tau_original_diagnostic": "",
            "order_current": "",
            "order_original": "",
        }
        if target["kind"] == "order_json":
            order_current = extract_order_from_json(Path(target["order_json"]), num_blocks)
            order_original = to_original_order(order_current, perm_state)
            row["tau_current_diagnostic"] = kendall_tau(order_current)
            row["tau_original_diagnostic"] = kendall_tau(order_original)
            row["order_current"] = " ".join(str(v) for v in order_current)
            row["order_original"] = " ".join(str(v) for v in order_original)
        elif target["method"] == "OriginalL2R":
            row["tau_original_diagnostic"] = 1.0
        elif target["method"] == "AR":
            order_current = list(range(num_blocks))
            row["tau_current_diagnostic"] = 1.0
            row["tau_original_diagnostic"] = kendall_tau(to_original_order(order_current, perm_state))
        rows.append(row)

    rows.sort(key=lambda row: float(row["mean_nll"]))
    fieldnames = [
        "source",
        "method",
        "kind",
        "mean_nll",
        "ppl",
        "tau_original_diagnostic",
        "tau_current_diagnostic",
        "num_samples",
        "num_scored_tokens",
        "summary_path",
        "order_json",
        "order_current",
        "order_original",
    ]
    with (out_root / "full_val_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Full Val Order Suite",
        "",
        f"- checkpoint: `{args.ckpt_path}`",
        f"- full validation: `--sample_mode sequential --num_batches 0 --sequential_cover_tail`, batch_size={args.batch_size}",
        f"- evaluated targets: {len(rows)}",
        "- original tau is diagnostic only; no order/sign was selected by tau or validation loss.",
        "",
        "| rank | source | method | NLL | PPL | original tau diag | samples | scored tokens |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        tau = row["tau_original_diagnostic"]
        tau_text = "-" if tau == "" else f"{float(tau):+.4f}"
        lines.append(
            f"| {rank} | {row['source']} | `{row['method']}` | {float(row['mean_nll']):.6f} | "
            f"{float(row['ppl']):.3f} | {tau_text} | {row['num_samples']} | {row['num_scored_tokens']} |"
        )
    (out_root / "full_val_summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"out_root": str(out_root), "num_rows": len(rows), "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
