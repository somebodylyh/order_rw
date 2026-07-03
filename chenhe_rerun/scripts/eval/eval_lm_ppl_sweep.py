"""
Batch-evaluate AO-GPT language PPL for multiple checkpoints and eval modes.

This is a thin runner around eval_lm_original_order_ppl.py. It keeps the same
current-frame/original-frame semantics and writes one summary.json per
checkpoint/mode plus aggregate CSV/JSON files.

Example:
python scripts/eval/eval_lm_ppl_sweep.py \
  --ckpt_glob 'out/base/permute/seq256/block64/*/ckpt.pt' \
  --eval_modes Random,AR,OriginalL2R,CheckpointOnlineSpectralOrder,CheckpointOnlineSpectralDistributionMAP \
  --out_dir Report/eval/lm_ppl_sweep/online_spectral_block64 \
  --split val \
  --batch_size 64 \
  --num_batches 200 \
  --device cuda \
  --dtype bfloat16
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_lm_original_order_ppl import evaluate  # noqa: E402


EVAL_MODE_CHOICES = (
    "AR",
    "OriginalL2R",
    "Random",
    "BlockOrder",
    "SegmentGuided",
    "CheckpointAttnMLPOrder",
    "CheckpointOnlineSpectralOrder",
    "CheckpointOnlineSpectralDistributionMAP",
    "CheckpointOnlineSpectralDistributionSampled",
    "CheckpointOnlineSpectralDistributionMixSampled",
)


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def read_ckpt_file(path: Path) -> list[Path]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append(Path(text))
    return rows


def resolve_ckpt_paths(args) -> list[Path]:
    paths = []
    for raw in args.ckpt_paths or []:
        paths.append(Path(raw))
    for pattern in args.ckpt_glob or []:
        paths.extend(Path(row) for row in sorted(glob.glob(str(pattern))))
    if args.ckpt_file is not None:
        paths.extend(read_ckpt_file(args.ckpt_file))
    if not paths:
        raise ValueError("Provide at least one checkpoint via --ckpt_paths, --ckpt_glob, or --ckpt_file.")
    seen = set()
    unique = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))
    return unique


def safe_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text or "ckpt"


def ckpt_label(path: Path, used: set[str]) -> str:
    if path.name == "ckpt.pt":
        base = path.parent.name
    else:
        base = path.stem
    label = safe_label(base)
    if label not in used:
        used.add(label)
        return label
    parent_label = safe_label(f"{path.parent.name}_{path.stem}")
    label = parent_label
    idx = 2
    while label in used:
        label = f"{parent_label}_{idx}"
        idx += 1
    used.add(label)
    return label


def make_eval_args(args, ckpt_path: Path, mode: str, out_dir: Path) -> Namespace:
    return Namespace(
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        split=args.split,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        sample_mode=args.sample_mode,
        eval_mode=mode,
        order_json=args.order_json,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        ignore_first_token=bool(args.ignore_first_token),
        sequential_cover_tail=bool(args.sequential_cover_tail),
        distribution_sample_temperature=args.distribution_sample_temperature,
        distribution_random_mix_prob=args.distribution_random_mix_prob,
        distribution_num_order_samples=int(args.distribution_num_order_samples),
    )


def result_row(summary: dict, ckpt_label_value: str, status: str, error: str | None = None) -> dict:
    coverage_info = summary.get("coverage_info") or {}
    return {
        "status": status,
        "error": "" if error is None else str(error),
        "ckpt_label": ckpt_label_value,
        "ckpt_path": str(summary.get("ckpt_path", "")),
        "checkpoint_iter": summary.get("checkpoint_iter", ""),
        "checkpoint_best_val_loss": summary.get("checkpoint_best_val_loss", ""),
        "dataset": summary.get("dataset", ""),
        "split": summary.get("split", ""),
        "eval_mode": summary.get("eval_mode", ""),
        "num_samples": summary.get("num_samples", ""),
        "num_scored_tokens": summary.get("num_scored_tokens", ""),
        "batch_size": summary.get("batch_size", ""),
        "num_batches_arg": summary.get("num_batches_arg", ""),
        "coverage_total_tokens": coverage_info.get("total_tokens", ""),
        "coverage_tail_tokens": coverage_info.get("tail_tokens", ""),
        "coverage_tail_window_added": coverage_info.get("tail_window_added", ""),
        "mean_nll_original_frame": summary.get("mean_nll_original_frame", ""),
        "ppl_original_frame": summary.get("ppl_original_frame", ""),
        "mean_nll_reveal_frame": summary.get("mean_nll_reveal_frame", ""),
        "ppl_reveal_frame": summary.get("ppl_reveal_frame", ""),
        "order_source": (summary.get("order_info") or {}).get("source", ""),
        "order_name": (summary.get("order_info") or {}).get("name", ""),
        "distribution_sample_temperature": (summary.get("order_info") or {}).get("sample_temperature", ""),
        "distribution_random_mix_prob": (summary.get("order_info") or {}).get("random_mix_prob", ""),
        "permutation_applied": (summary.get("frame_mapping") or {}).get("data_permutation_applied", ""),
        "summary_path": str(summary.get("summary_path", "")),
    }


def write_outputs(out_dir: Path, rows: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aggregate_results.json"
    csv_path = out_dir / "aggregate_results.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_paths", nargs="*", default=None, help="Explicit checkpoint paths.")
    parser.add_argument("--ckpt_glob", action="append", default=None, help="Glob pattern for checkpoint paths.")
    parser.add_argument("--ckpt_file", type=Path, default=None, help="Text file with one checkpoint path per line.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--eval_modes",
        type=str,
        default="Random,AR,OriginalL2R,CheckpointOnlineSpectralOrder",
        help="Comma-separated eval modes.",
    )
    parser.add_argument("--order_json", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--sample_mode", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--ignore_first_token", action="store_true")
    parser.add_argument("--distribution_sample_temperature", type=float, default=None)
    parser.add_argument("--distribution_random_mix_prob", type=float, default=None)
    parser.add_argument("--distribution_num_order_samples", type=int, default=1)
    parser.add_argument(
        "--sequential_cover_tail",
        action="store_true",
        help="With sequential full-stream eval, add a tail window and score only previously uncovered tokens.",
    )
    parser.add_argument(
        "--skip_missing_order_modes",
        action="store_true",
        help="Skip checkpoint-order modes when a checkpoint does not contain the required cached order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_paths = resolve_ckpt_paths(args)
    eval_modes = parse_csv(args.eval_modes)
    bad_modes = [mode for mode in eval_modes if mode not in EVAL_MODE_CHOICES]
    if bad_modes:
        raise ValueError(f"Unsupported eval mode(s): {bad_modes}. Choices: {EVAL_MODE_CHOICES}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    used_labels = set()
    for ckpt_path in ckpt_paths:
        label = ckpt_label(ckpt_path, used_labels)
        for mode in eval_modes:
            mode_out_dir = args.out_dir / label / safe_label(mode)
            print(f"[lm-ppl-sweep] ckpt={ckpt_path} mode={mode} out={mode_out_dir}")
            try:
                summary = evaluate(make_eval_args(args, ckpt_path, mode, mode_out_dir))
                summary["summary_path"] = str(mode_out_dir / "summary.json")
                rows.append(result_row(summary, label, status="ok"))
            except Exception as exc:
                message = str(exc)
                if args.skip_missing_order_modes and mode.startswith("Checkpoint") and "does not contain" in message:
                    print(f"[lm-ppl-sweep] skipped: {message}")
                    rows.append(
                        result_row(
                            {
                                "ckpt_path": str(ckpt_path),
                                "eval_mode": mode,
                                "summary_path": str(mode_out_dir / "summary.json"),
                            },
                            label,
                            status="skipped",
                            error=message,
                        )
                    )
                    continue
                raise

    json_path, csv_path = write_outputs(args.out_dir, rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    print(
        json.dumps(
            {
                "num_checkpoints": len(ckpt_paths),
                "num_rows": len(rows),
                "num_ok": len(ok_rows),
                "aggregate_json": str(json_path),
                "aggregate_csv": str(csv_path),
                "best_by_ppl_original_frame": sorted(
                    [
                        {
                            "ckpt_label": row["ckpt_label"],
                            "eval_mode": row["eval_mode"],
                            "checkpoint_iter": row["checkpoint_iter"],
                            "ppl_original_frame": row["ppl_original_frame"],
                        }
                        for row in ok_rows
                        if isinstance(row.get("ppl_original_frame"), (int, float))
                        and math.isfinite(float(row["ppl_original_frame"]))
                    ],
                    key=lambda row: float(row["ppl_original_frame"]),
                )[:10],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
