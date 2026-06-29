#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/admin/lyuyuhuan/order_lyu")
OUT = ROOT / "probe_results_image/e2_vq_round2_negative_seed42"
ARMS = [
    "cont_random",
    "cont_Bcov_balanced",
    "cont_distance_only_coverage",
    "cont_shuffled_Bcov_balanced",
]
COMMON = [
    "val_random",
    "val_raster",
    "val_hilbert",
    "val_Bcov_balanced",
    "val_distance_only_coverage",
    "val_rw_top4_eps0",
    "val_rw_eps015",
    "val_rw_topk8",
]
STRUCTURED = [
    "val_raster",
    "val_hilbert",
    "val_Bcov_balanced",
    "val_distance_only_coverage",
    "val_rw_top4_eps0",
    "val_rw_topk8",
]
NOISY = ["val_random", "val_rw_eps015"]
MATCHED = {
    "cont_random": "val_random",
    "cont_Bcov_balanced": "val_Bcov_balanced",
    "cont_distance_only_coverage": "val_distance_only_coverage",
    "cont_shuffled_Bcov_balanced": "val_Bcov_balanced",
}


def git_meta():
    try:
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--short"],
            text=True,
        ).splitlines()
        return {"head": head, "dirty_count": len(status), "preview": status[:20]}
    except Exception as exc:
        return {"error": str(exc)}


def read_final(arm):
    path = OUT / arm / "eval_curve.tsv"
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    if not rows:
        raise RuntimeError(f"empty eval curve: {path}")
    row = rows[-1]
    step = int(row["step"])
    vals = {
        "step": step,
        "cross": sum(float(row[c]) for c in COMMON) / len(COMMON),
        "structured": sum(float(row[c]) for c in STRUCTURED) / len(STRUCTURED),
        "noisy": sum(float(row[c]) for c in NOISY) / len(NOISY),
        "matched": float(row[MATCHED[arm]]),
    }
    vals.update({c: float(row[c]) for c in COMMON if c in row})
    return vals


def main():
    vals = {arm: read_final(arm) for arm in ARMS}
    lines = [
        "# E2 VQ Round-2 Negative Minimal",
        "",
        "No-structure/fallback control on ImageNet32 VQ-f4 seq64 E2. Lower is better.",
        "This is a 3000-step seed-42 continuation, intended as a lightweight boundary check rather than a final multi-seed result.",
        "",
        "| arm | step | cross | structured | noisy/random | matched |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        v = vals[arm]
        lines.append(
            f"| {arm} | {v['step']} | {v['cross']:.4f} | {v['structured']:.4f} | "
            f"{v['noisy']:.4f} | {v['matched']:.4f} |"
        )
    lines.extend(["", "## Key deltas"])
    for target in ["cont_random", "cont_distance_only_coverage", "cont_shuffled_Bcov_balanced"]:
        b = vals["cont_Bcov_balanced"]
        t = vals[target]
        lines.append(
            f"- Bcov_balanced - {target}: cross={b['cross']-t['cross']:+.4f}, "
            f"structured={b['structured']-t['structured']:+.4f}, noisy={b['noisy']-t['noisy']:+.4f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this as a no-structure/fallback boundary check. The graph diagnostic classifies E2 VQ as uniform/noisy, so B-guided coverage is not expected to provide the E3-style proximity-regime benefit.",
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(
                {
                    "source_A_path": str(ROOT / "probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy"),
                    "baseline_ckpt": str(ROOT / "nanogpt-learned-order/out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256/ckpt.pt"),
                    "data_train": str(ROOT / "block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/train.bin"),
                    "data_val": str(ROOT / "block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/val.bin"),
                    "B_construction": "build_directed_graph(A_global)",
                    "physical_to_model_remap": "not present unless ckpt data_permutation exists; trainer records this in per-arm config.json",
                    "readout_parameters": {
                        "Bcov_balanced": "score=minmax(B[last,v])-minmax(manh(last,v)); gamma_B=gamma_d=1.0; deterministic argmax",
                        "distance_only_coverage": "score=0.01*minmax(B[last,v])-minmax(manh(last,v)); deterministic argmax",
                        "shuffled_Bcov": "row-wise independent shuffle of B, diagonal zeroed",
                    },
                    "seed": 42,
                    "max_steps": 3000,
                    "git": git_meta(),
                },
                indent=2,
            ),
            "```",
        ]
    )
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(OUT / "SUMMARY.md")
    for arm in ARMS:
        v = vals[arm]
        print(
            f"{arm}\tcross={v['cross']:.4f}\tstructured={v['structured']:.4f}\t"
            f"noisy={v['noisy']:.4f}\tmatched={v['matched']:.4f}"
        )


if __name__ == "__main__":
    main()
