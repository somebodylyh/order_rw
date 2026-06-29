#!/usr/bin/env python3
"""Summarize model-frame head drift for head-gated g_beta feedback runs.

Consumes ``analyses/diag_model_frame_feedback.py`` outputs and computes the
difference-in-differences requested by the head-gated g_beta plan:

    delta_random = metric_random_end - metric_start
    delta_feedback = metric_feedback_end - metric_start
    feedback_effect = delta_feedback - delta_random
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib


def _read_tsv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _as_float(row: dict, key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else float("nan")


def _find_row(rows: list[dict], label: str, layer: int, head: int, method: str) -> dict:
    matches = [
        r for r in rows
        if r.get("label") == label
        and int(r.get("layer", -1)) == layer
        and int(r.get("head", -1)) == head
        and r.get("method") == method
    ]
    if not matches:
        raise ValueError(
            f"no row for label={label!r}, layer={layer}, head={head}, method={method!r}"
        )
    return matches[0]


def _gate_summary(gate_rows: list[dict], label: str) -> dict | None:
    rows = [r for r in gate_rows if r.get("label") == label]
    if not rows:
        return None
    top = min(rows, key=lambda r: int(r.get("rank", 9999)))
    return {
        "top1_head": int(top["top1_head"]),
        "top2_heads": top["top2_heads"],
        "gate_entropy": float(top["gate_entropy"]),
        "weights": {
            int(r["head"]): float(r["weight"])
            for r in sorted(rows, key=lambda x: int(x["rank"]))
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-frame-tsv", required=True)
    p.add_argument("--gate-tsv", default=None)
    p.add_argument("--start-label", required=True)
    p.add_argument("--random-label", required=True)
    p.add_argument("--feedback-label", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=2)
    p.add_argument("--method", default="C-D+L")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    rows = _read_tsv(pathlib.Path(args.model_frame_tsv))
    gate_rows = _read_tsv(pathlib.Path(args.gate_tsv)) if args.gate_tsv else []

    labels = {
        "start": args.start_label,
        "random": args.random_label,
        "feedback": args.feedback_label,
    }
    selected = {
        name: _find_row(rows, label, args.layer, args.head, args.method)
        for name, label in labels.items()
    }

    metrics = ["tau_model_vs_semantic_path", "tau_model_vs_identity"]
    effects = {}
    for metric in metrics:
        start = _as_float(selected["start"], metric)
        random_end = _as_float(selected["random"], metric)
        feedback_end = _as_float(selected["feedback"], metric)
        delta_random = random_end - start
        delta_feedback = feedback_end - start
        effects[metric] = {
            "start": start,
            "random_end": random_end,
            "feedback_end": feedback_end,
            "delta_random": delta_random,
            "delta_feedback": delta_feedback,
            "feedback_effect": delta_feedback - delta_random,
        }

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "selection": {
            "layer": args.layer,
            "head": args.head,
            "method": args.method,
            "labels": labels,
        },
        "effects": effects,
        "gate": {
            name: _gate_summary(gate_rows, label)
            for name, label in labels.items()
        } if gate_rows else None,
        "rows": selected,
    }

    (out / "drift_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Head-Gated gBeta Head Drift Summary",
        "",
        f"Tracked: L{args.layer}H{args.head} / {args.method}",
        "",
        "metric\tstart\trandom_end\tfeedback_end\tdelta_random\tdelta_feedback\tfeedback_effect",
    ]
    for metric, vals in effects.items():
        lines.append(
            f"{metric}\t{vals['start']:.6f}\t{vals['random_end']:.6f}\t"
            f"{vals['feedback_end']:.6f}\t{vals['delta_random']:+.6f}\t"
            f"{vals['delta_feedback']:+.6f}\t{vals['feedback_effect']:+.6f}"
        )
    if summary["gate"]:
        lines.extend(["", "## Gate"])
        for name, gate in summary["gate"].items():
            if gate is None:
                lines.append(f"- {name}: no gate row")
            else:
                lines.append(
                    f"- {name}: top1=H{gate['top1_head']} top2={gate['top2_heads']} "
                    f"entropy={gate['gate_entropy']:.6f}"
                )
    (out / "drift_summary.md").write_text("\n".join(lines) + "\n")

    print(f"Saved {out / 'drift_summary.json'}")
    print(f"Saved {out / 'drift_summary.md'}")


if __name__ == "__main__":
    main()
