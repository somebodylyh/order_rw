#!/usr/bin/env python3
"""Build label-free candidate-set manifests from selector rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def select_candidates(rows: list[dict], top_k: int, mode: str, seed: int = 0) -> list[dict]:
    """Select candidates without inspecting oracle/post-hoc fields."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if mode == "structure":
        eligible = [r for r in rows if np.isfinite(float(r["structure_score"]))]
        return sorted(eligible, key=lambda r: float(r["structure_score"]), reverse=True)[:top_k]
    if mode == "random":
        rng = np.random.default_rng(seed)
        size = min(top_k, len(rows))
        idx = rng.choice(len(rows), size=size, replace=False)
        return [rows[int(i)] for i in idx]
    raise ValueError(f"unknown mode {mode!r}")


def parse_set_spec(spec: str) -> tuple[str, str, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"set spec must be name:mode:k, got {spec!r}")
    name, mode, k = parts
    return name, mode, int(k)


def build_candidate_sets(
    rows: list[dict],
    out_dir: str,
    sets: list[tuple[str, str, int]],
    seed: int = 0,
    selector_json: str = "inline",
) -> list[dict]:
    """Programmatic wrapper — write candidate-set JSON files.

    Args:
        rows: selector rows.
        out_dir: output root (a ``candidate_sets`` subdirectory is created).
        sets: list of (name, mode, top_k) tuples.
        seed: random seed for ``mode="random"``.
        selector_json: provenance label written into each JSON.

    Returns:
        manifest list of {name, path, mode, top_k}.
    """
    out_path = Path(out_dir) / "candidate_sets"
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, mode, k in sets:
        selected = select_candidates(rows, top_k=k, mode=mode, seed=seed)
        fname = f"{name}_seed{seed}.json" if mode == "random" else f"{name}.json"
        path = out_path / fname
        payload = {
            "name": name,
            "mode": mode,
            "top_k": k,
            "seed": seed,
            "selector_json": selector_json,
            "candidates": selected,
        }
        path.write_text(json.dumps(payload, indent=2))
        manifest.append({"name": name, "path": str(path), "mode": mode, "top_k": k})
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sets", nargs="+", required=True, help="Specs like top4:structure:4 random4:random:4")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = json.loads(Path(args.selector_json).read_text())
    triples = [parse_set_spec(spec) for spec in args.sets]
    build_candidate_sets(rows, args.out_dir, triples, seed=args.seed, selector_json=args.selector_json)
    print(f"wrote {len(triples)} candidate sets to {args.out_dir}/candidate_sets")


if __name__ == "__main__":
    main()
