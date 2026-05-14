"""Aggregate image-side 3-seed CI results into mean ± 95% CI paired Δ table.

Reads:
  probe_results_image/eval_ci/<config>/seed<S>/eval_summary.tsv   for S in {0,1}
  probe_results_image/eval/<config>/eval_summary.tsv              for the legacy seed=42 location

Writes:
  probe_results_image/graph_rw_ci/ci_summary.tsv

Output columns:
  config, eval_mode, mean_mse, std_mse, n_seeds,
  paired_delta_vs_random_mean, paired_delta_vs_random_ci_lo, paired_delta_vs_random_ci_hi
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Dict, List

CONFIGS = ("cont_random", "cont_top4", "cont_eps015", "cont_top8")
EVAL_MODES = ("random", "raster", "rw_top4_eps0", "rw_eps015", "rw_topk8")


def _read_summary(path: Path) -> Dict[str, float]:
    """eval_summary.tsv → {mode: mean_mse}."""
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            mode_i = header.index("mode")
            mse_i = header.index("mean_mse")
        except ValueError:
            return out
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(mode_i, mse_i):
                continue
            try:
                out[cols[mode_i]] = float(cols[mse_i])
            except ValueError:
                continue
    return out


def _seed_paths(config: str, ci_root: Path, legacy_root: Path) -> List[Path]:
    paths = []
    for seed in (0, 1):
        p = ci_root / config / f"seed{seed}" / "eval_summary.tsv"
        if p.exists():
            paths.append(p)
    legacy = legacy_root / config / "eval_summary.tsv"
    if legacy.exists():
        paths.append(legacy)  # treated as seed=42
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    repo = Path(args.repo_root).resolve()
    ci_root = repo / "probe_results_image" / "eval_ci"
    legacy_root = repo / "probe_results_image" / "eval"
    out_path = Path(args.out) if args.out else repo / "probe_results_image" / "graph_rw_ci" / "ci_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # collect: per_config[config][mode] = list of seed values
    per_config: Dict[str, Dict[str, List[float]]] = {c: {m: [] for m in EVAL_MODES} for c in CONFIGS}
    for c in CONFIGS:
        for path in _seed_paths(c, ci_root, legacy_root):
            d = _read_summary(path)
            for m, v in d.items():
                if m in EVAL_MODES:
                    per_config[c][m].append(v)

    # compute paired Δ vs cont_random per seed, then mean ± 95% CI (z=1.96 / sqrt(n))
    z = NormalDist().inv_cdf(0.975)
    rows = ["\t".join(["config", "eval_mode", "n_seeds", "mean_mse", "std_mse",
                        "paired_delta_mean", "paired_delta_ci_lo", "paired_delta_ci_hi"])]
    random_per_mode: Dict[str, List[float]] = per_config["cont_random"]

    for c in CONFIGS:
        for m in EVAL_MODES:
            vals = per_config[c][m]
            n = len(vals)
            if n == 0:
                continue
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
            std = var ** 0.5
            # paired delta only meaningful when seeds line up; we use position
            ref = random_per_mode.get(m, [])
            paired = [a - b for a, b in zip(vals, ref)]
            np_p = len(paired)
            if c == "cont_random" or np_p == 0:
                d_mean = d_lo = d_hi = ""
            else:
                d_mean_v = sum(paired) / np_p
                d_var = sum((d - d_mean_v) ** 2 for d in paired) / max(np_p - 1, 1)
                d_se = (d_var / np_p) ** 0.5
                d_mean = f"{d_mean_v:+.6f}"
                d_lo = f"{d_mean_v - z * d_se:+.6f}"
                d_hi = f"{d_mean_v + z * d_se:+.6f}"
            rows.append("\t".join([c, m, str(n), f"{mean:.6f}", f"{std:.6f}",
                                    str(d_mean), str(d_lo), str(d_hi)]))

    out_path.write_text("\n".join(rows) + "\n")
    print(f"wrote {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
