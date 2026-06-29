#!/usr/bin/env python3
"""Compare AO-GPT checkpoint weights by parameter and coarse group."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

import torch


GROUP_PATTERNS = [
    ("wte", re.compile(r"(?:^|\.)wte(?:\.|$)")),
    ("wpe", re.compile(r"(?:^|\.)wpe(?:\.|$)")),
    ("c_attn", re.compile(r"c_attn\.weight$")),
    ("c_proj", re.compile(r"c_proj\.weight$")),
    ("mlp", re.compile(r"\.mlp\.")),
    ("q_norm_k_norm", re.compile(r"(?:q_norm|k_norm)")),
    ("norm", re.compile(r"(?:ln_|norm)")),
]


def clean_state_dict(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"{path} has neither model nor model_state_dict")
    return {k.replace("_orig_mod.", ""): v.detach().cpu() for k, v in state.items()}


def group_for_key(key: str) -> str:
    for name, pat in GROUP_PATTERNS:
        if pat.search(key):
            return name
    return "other"


def stats_for_tensors(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    diff = af - bf
    dot = float(torch.dot(af, bf).item())
    an = float(torch.linalg.vector_norm(af).item())
    bn = float(torch.linalg.vector_norm(bf).item())
    cos = dot / max(an * bn, 1e-30)
    return {
        "n": int(af.numel()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rms_diff": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "mean_abs_a": float(af.abs().mean().item()),
        "mean_abs_b": float(bf.abs().mean().item()),
        "cosine": float(cos),
    }


def aggregate(chunks: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    a = torch.cat([x.float().reshape(-1) for x, _ in chunks])
    b = torch.cat([y.float().reshape(-1) for _, y in chunks])
    return stats_for_tensors(a, b)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"{pathlib.Path(args.a).parent.name}_vs_{pathlib.Path(args.b).parent.name}"

    sa = clean_state_dict(args.a)
    sb = clean_state_dict(args.b)
    keys = sorted(set(sa) & set(sb))
    missing = {"only_a": sorted(set(sa) - set(sb)), "only_b": sorted(set(sb) - set(sa))}

    per_param = []
    group_chunks: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    all_chunks = []
    for key in keys:
        if sa[key].shape != sb[key].shape or not torch.is_floating_point(sa[key]):
            continue
        st = stats_for_tensors(sa[key], sb[key])
        group = group_for_key(key)
        per_param.append({"key": key, "group": group, **st})
        group_chunks.setdefault(group, []).append((sa[key], sb[key]))
        all_chunks.append((sa[key], sb[key]))

    per_group = []
    for group, chunks in sorted(group_chunks.items()):
        per_group.append({"group": group, **aggregate(chunks)})
    total = {"label": label, **aggregate(all_chunks)}

    def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    fields = ["key", "group", "n", "mean_abs_diff", "rms_diff", "mean_abs_a", "mean_abs_b", "cosine"]
    write_csv(out_dir / f"{label}_per_param.csv", per_param, fields)
    write_csv(out_dir / f"{label}_per_group.csv", per_group, ["group"] + fields[2:])
    (out_dir / f"{label}_summary.json").write_text(json.dumps({
        "a": args.a,
        "b": args.b,
        "total": total,
        "missing": missing,
        "per_group": per_group,
    }, indent=2))

    print(json.dumps({"label": label, "total": total, "per_group": per_group}, indent=2))


if __name__ == "__main__":
    main()
