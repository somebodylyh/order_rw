#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    root = Path(f"probe_results_image/imagenet32_continuous_round2_minimal_seed{seed}")
    arms = [
        "cont_random",
        "cont_Bcov_balanced",
        "cont_distance_only_coverage",
        "cont_shuffled_Bcov_balanced",
    ]
    rows = []
    for arm in arms:
        curve_path = root / arm / "eval_curve.tsv"
        if not curve_path.exists():
            continue
        df = pd.read_csv(curve_path, sep="\t")
        last = df.iloc[-1]
        vals = {
            "random": float(last["val_random"]),
            "raster": float(last["val_raster"]),
            "hilbert": float(last["val_hilbert"]),
            "Bcov": float(last["val_Bcov_balanced"]),
            "distance": float(last["val_distance_only_coverage"]),
            "shuffled": float(last["val_shuffled_Bcov_balanced"]),
            "graph_rw": float(last["val_graph_rw_top4"]),
        }
        cross = float(np.mean(list(vals.values())))
        structured = float(np.mean([vals["raster"], vals["hilbert"], vals["Bcov"], vals["distance"], vals["graph_rw"]]))
        noisy = vals["random"]
        matched = {
            "cont_random": vals["random"],
            "cont_Bcov_balanced": vals["Bcov"],
            "cont_distance_only_coverage": vals["distance"],
            "cont_shuffled_Bcov_balanced": vals["shuffled"],
        }[arm]
        rows.append({"arm": arm, "cross": cross, "structured": structured, "noisy": noisy, "matched": matched, **vals})

    out = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "aggregate.tsv", sep="\t", index=False)

    def fmt(x):
        return f"{x:.6f}"

    lines = [
        "# ImageNet32 Continuous Round-2 Minimal",
        "",
        "Phase-B extra proximity-positive graph. Lower is better. Continuous patches + MSE; this is not a VQ-token experiment.",
        "",
        "| arm | cross | structured | noisy/random | matched |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in out.iterrows():
        lines.append(f"| {r.arm} | {fmt(r.cross)} | {fmt(r.structured)} | {fmt(r.noisy)} | {fmt(r.matched)} |")
    if "cont_Bcov_balanced" in set(out.arm):
        b = out[out.arm == "cont_Bcov_balanced"].iloc[0]
        lines += ["", "## Key deltas"]
        for base in ["cont_random", "cont_distance_only_coverage", "cont_shuffled_Bcov_balanced"]:
            if base in set(out.arm):
                r = out[out.arm == base].iloc[0]
                lines.append(
                    f"- Bcov - {base}: cross={b.cross-r.cross:+.6f}, "
                    f"structured={b.structured-r.structured:+.6f}, noisy={b.noisy-r.noisy:+.6f}"
                )
    lines += [
        "",
        "## Interpretation",
        "",
        "Use this as Phase-B evidence for whether the graph-regime story extends to a continuous-patch proximity graph. "
        "The main E3-control-small claim remains based on VQ continuation; this experiment tests cross-graph scope.",
    ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "SUMMARY.md")


if __name__ == "__main__":
    main()
