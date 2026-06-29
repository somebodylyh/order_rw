#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd


def to_md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    root = Path(sys.argv[1])
    rows = []
    for path in sorted(root.glob("common_*/metrics.tsv")):
        policy = path.parent.name.replace("common_", "")
        df = pd.read_csv(path, sep="\t")
        df["common_policy"] = policy
        rows.append(df)
    if not rows:
        raise SystemExit(f"no metrics.tsv under {root}")
    all_df = pd.concat(rows, ignore_index=True)
    keep = ["common_policy", "arm", "matched_policy", "samples", "fid_vq_val", "pixel_std", "token_entropy_bits", "duplicate_image_rate"]
    out = all_df[keep].copy()
    out = out.sort_values(["common_policy", "fid_vq_val"])
    formatted = out.copy()
    for c in ["fid_vq_val", "pixel_std", "token_entropy_bits", "duplicate_image_rate"]:
        formatted[c] = formatted[c].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}")
    formatted.to_csv(root / "common_order_metrics.tsv", sep="\t", index=False)

    md = [
        "# Round-2 Common-Order Sample Quality",
        "",
        "Each section generates all checkpoints with the same readout policy. FID is the VQ-decoded validation-token proxy, not raw ImageNet FID.",
        "",
    ]
    for policy, group in formatted.groupby("common_policy"):
        md.append(f"## Common policy: {policy}")
        md.append("")
        md.append(to_md_table(group.drop(columns=["common_policy"])))
        md.append("")
    md.extend([
        "## Metadata",
        "",
        "```json",
        json.dumps({"source_root": str(root), "metric": "VQ-decoded validation proxy FID"}, indent=2),
        "```",
    ])
    (root / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print(root / "SUMMARY.md")


if __name__ == "__main__":
    main()
