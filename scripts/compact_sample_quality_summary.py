#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def main():
    out = Path("probe_results_image_large/grw_e3ctrlsmall_round2_sample_quality_seed42")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    try:
        status = subprocess.check_output(["git", "status", "--short"], text=True, timeout=10)
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=10).strip()
        lines = [line for line in status.splitlines() if line.strip()]
        meta["git"] = {
            "head": head,
            "dirty_count": len(lines),
            "status_short_preview": lines[:30],
            "truncated": len(lines) > 30,
        }
        (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        meta["git"] = {"error": repr(exc)}

    rows_sorted = sorted(
        metrics,
        key=lambda row: float("inf") if row.get("fid_vq_val") is None else row["fid_vq_val"],
    )
    md = [
        "# Round-2 Sample Quality / FID",
        "",
        "FID is computed against VQ-decoded validation tokens, not raw ImageNet images. "
        "Treat this as a matched-order sample-quality proxy, not the final validation criterion.",
        "",
        "| rank | arm | policy | samples | FID_vq_val ↓ | pixel_std | token_entropy_bits | duplicate_rate |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows_sorted, 1):
        fid = "NA" if row.get("fid_vq_val") is None else f"{row['fid_vq_val']:.4f}"
        md.append(
            f"| {rank} | {row['arm']} | {row['policy']} | {row['samples']} | {fid} | "
            f"{row['pixel_std']:.4f} | {row['token_entropy_bits']:.3f} | "
            f"{row['duplicate_image_rate']:.4f} |"
        )

    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This run uses matched readout at sampling time for each continuation arm.",
            "- Raster has the lowest proxy FID here, consistent with its matched-order specialization; "
            "this does not overturn Round-2 validation/cross-order robustness, where raster was less balanced.",
            "- Bcov_balanced remains a robustness/validation claim, not a claim that it always wins "
            "matched-order sample FID at small sample count.",
            "- The next stronger sample-quality run should increase samples and include common-order sampling, "
            "especially common random and common Bcov readout across all checkpoints.",
            "",
            "## Metadata",
            "",
            f"- source round2 root: `{meta.get('round2_root')}`",
            f"- source A/B path: `{meta.get('a_block_path')}`",
            f"- data dir: `{meta.get('data_dir')}`",
            f"- VQ decoder: `{meta.get('vae_path')}`",
            "- physical order is remapped via `inverse_block_perm`; generated model-frame tokens are unpermuted before VQ decode.",
            "- B construction method: frozen Stage-1 E3-control-small attention-derived A_block/B readout input.",
            f"- readout parameters: matched per arm, temperature={meta.get('temperature')}, top_k={meta.get('top_k')}",
            f"- random seed: {meta.get('seed')}",
            f"- git: `{meta.get('git')}`",
        ]
    )
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out / "SUMMARY.md")


if __name__ == "__main__":
    main()
