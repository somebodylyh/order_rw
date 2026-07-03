from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from seq80_random_ckpt_online_order_attn_batch import (  # noqa: E402
    DEFAULT_CKPT_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_STEPS,
    build_model,
    ckpt_path_for_step,
    json_safe,
    load_checkpoint,
    load_tokens,
    loss_rerank_candidates,
    order_stats,
    parse_steps,
    resolve_data_dir,
    summarize_candidate,
    write_csv,
)


def load_attention_candidates(path: Path, top_k: int) -> list[dict]:
    candidates = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            order = [int(v) for v in json.loads(row["order_current"])]
            candidates.append(
                {
                    "name": row["candidate_name"],
                    "score": float(row["candidate_score"]),
                    "order": order,
                    "meta": {
                        "attention_rank": int(row["attention_rank"]),
                        "attention_tau_l2r": float(row["tau_l2r"]),
                    },
                }
            )
            if len(candidates) >= int(top_k):
                break
    return candidates


def write_markdown(rows: list[dict], out_dir: Path, *, label: str):
    lines = [
        f"# Seq80 Random Checkpoint {label} Loss-Rerank Diagnostic",
        "",
        "This table reuses the 4096-sample attention candidates already saved in each checkpoint directory.",
        "Only the loss-rerank step is rerun here.",
        "",
        "| step | loss tau | adj rate | prefix loss | full loss | first16 |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {step} | {tau:.4f} | {adj:.3f} | {prefix:.4f} | {full:.4f} | `{first16}` |".format(
                step=int(row["step"]),
                tau=float(row["loss_rerank_tau_l2r"]),
                adj=float(row["loss_rerank_either_adjacent_rate"]),
                prefix=float(row["loss_rerank_prefix_loss"]),
                full=float(row["loss_rerank_full_loss"]),
                first16=row["loss_rerank_first16"],
            )
        )
    (out_dir / f"README_{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Loss-rerank existing seq80 random-checkpoint attention candidates."
    )
    parser.add_argument("--ckpt_dir", type=Path, default=Path(DEFAULT_CKPT_DIR))
    parser.add_argument("--report_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--steps", type=str, default=DEFAULT_STEPS)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=96)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--prefix_k", type=int, default=16)
    parser.add_argument("--attention_weight", type=float, default=1.0)
    parser.add_argument("--prefix_weight", type=float, default=0.4)
    parser.add_argument("--full_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=32345)
    parser.add_argument("--label", type=str, default="loss_rerank_fast128")
    return parser.parse_args()


def main():
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ctx = nullcontext()
    for step in parse_steps(args.steps):
        step_dir = args.report_dir / f"ckpt_{int(step):06d}"
        candidates_path = step_dir / "attention_candidates.csv"
        if not candidates_path.exists():
            raise FileNotFoundError(f"Missing attention candidates: {candidates_path}")
        print(f"[seq80-rerank-existing] step={step} loading {candidates_path}", flush=True)
        candidates = load_attention_candidates(candidates_path, args.top_k)
        checkpoint = load_checkpoint(ckpt_path_for_step(args.ckpt_dir, step))
        model = build_model(checkpoint, str(args.device))
        data_dir = resolve_data_dir(checkpoint, args.dataset, args.data_dir)
        tokens = load_tokens(data_dir, args.split, checkpoint)
        reranked, rerank_rows = loss_rerank_candidates(
            model,
            tokens,
            candidates,
            batch_size=int(args.batch_size),
            batches=int(args.batches),
            candidate_batch_size=int(args.candidate_batch_size),
            prefix_k=int(args.prefix_k),
            split_seed=int(args.seed) + int(step),
            device=str(args.device),
            ctx=ctx,
            attention_weight=float(args.attention_weight),
            prefix_weight=float(args.prefix_weight),
            full_weight=float(args.full_weight),
        )
        write_csv(step_dir / f"{args.label}_candidates.csv", rerank_rows)
        best = reranked[0]
        row = {
            "step": int(step),
            "ckpt_path": str(ckpt_path_for_step(args.ckpt_dir, step)),
            "checkpoint_iter": int(checkpoint.get("iter_num", step)),
            "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
            "loss_rerank_top_k": int(args.top_k),
            "loss_rerank_batches": int(args.batches),
            "loss_rerank_batch_size": int(args.batch_size),
            "loss_rerank_candidate_batch_size": int(args.candidate_batch_size),
        }
        row.update(summarize_candidate(best, "loss_rerank"))
        rows.append(row)
        write_csv(args.report_dir / f"{args.label}_summary.csv", rows)
        write_markdown(rows, args.report_dir, label=args.label)
        del model
        if "cuda" in str(args.device):
            torch.cuda.empty_cache()
    config = {k: json_safe(v) for k, v in vars(args).items()}
    (args.report_dir / f"{args.label}_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps({"report_dir": str(args.report_dir), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
