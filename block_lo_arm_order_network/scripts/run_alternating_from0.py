#!/usr/bin/env python3
r"""Launcher for three "from-0 alternating" training arms.

Arms:
  A  random         --run-kind baseline                         (control)
  B  v3refresh      --run-kind graph_rw --rw-policy progressive_rw_v3 (v3 from-0)
  C  mlp_finetune   --run-kind graph_rw --rw-policy mlp_cdl --mlp-alternating
                    --mlp-refresh-mode finetune                 (self-bootstrap from 0)

Goal: test whether the MLP policy can bootstrap itself from scratch (arm C) — starting
from no teacher checkpoint, alternating between training the AOGPT and re-distilling the
MLP policy on the attention it produces.  Arms A and B serve as lower and upper baselines.

All three arms share the same training config (30k steps, lr=1e-3/1e-4 cosine over 50k,
batch 64 × grad-accum 2, eval every 1k, save at 5k/15k/30k, NO resume).  The order
mechanism alone differs.

Usage:
  python scripts/run_alternating_from0.py --arm random        # launch arm A
  python scripts/run_alternating_from0.py --arm v3refresh     # launch arm B
  python scripts/run_alternating_from0.py --arm mlp           # launch arm C
  python scripts/run_alternating_from0.py --report-only       # (re)write REPORT.md

If neither --arm nor --report-only is given, prints usage and exits WITHOUT launching
anything.  Running this file by accident therefore does nothing destructive.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent      # block_lo_arm_order_network/
_REPO = _HERE.parent

# helper imports (gpu_free_mib / wait_for_gpu / load_curve live in run_source_start_overnight)
sys.path.insert(0, str(_HERE / "scripts"))
import run_source_start_overnight as ov             # noqa: E402

# Output directories (repo-level, inside probe_results/attention_order_mlp/)
_MLPA = _REPO / "probe_results/attention_order_mlp"
OUT_A = _MLPA / "alt_from0_random"
OUT_B = _MLPA / "alt_from0_v3refresh"
OUT_C = _MLPA / "alt_from0_mlp_finetune"

_ARM_DIRS = {"random": OUT_A, "v3refresh": OUT_B, "mlp": OUT_C}

CONFIG_JSON = _HERE / "probe_results/clean_base_random_perm/config.json"

TARGET_STEP = 30000
REPORT_PATH = _MLPA / "alt_from0_REPORT.md"


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
_arm_log_handle = None   # set inside launch()


def wlog(msg: str, arm: str = "launcher"):
    out_dir = _ARM_DIRS.get(arm, _MLPA)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "watcher.log"
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{arm}] {msg}"
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Model-args loader
# ──────────────────────────────────────────────────────────────────────────────
def load_model_args() -> dict:
    """Read n_layer/n_head/n_embd from config.json; return {} if missing."""
    if not CONFIG_JSON.exists():
        return {}
    try:
        with CONFIG_JSON.open() as f:
            data = json.load(f)
        ma = data.get("model_args", {})
        out = {}
        for key in ("n_layer", "n_head", "n_embd"):
            if key in ma:
                out[key] = int(ma[key])
        return out
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Command builders
# ──────────────────────────────────────────────────────────────────────────────
_SHARED_FLAGS = [
    "--max-steps", str(TARGET_STEP),
    "--lr", "1e-3", "--min-lr", "1e-4", "--lr-decay-steps", "50000",
    "--batch-size", "64", "--grad-accum", "2",
    "--eval-interval", "1000", "--log-interval", "100",
    "--save-steps", "5000,15000,30000",
]

_ARM_FLAGS = {
    "random": [
        "--run-kind", "baseline",
    ],
    "v3refresh": [
        "--run-kind", "graph_rw",
        "--rw-policy", "progressive_rw_v3",
        "--rw-lam", "0.75", "--rw-rho", "0.2", "--rw-top-k", "4",
        "--refresh-interval", "5000",
        "--refresh-ema-beta", "0.0",
        "--alpha-warmup-start", "5000",
        "--alpha-warmup-steps", "10000",
        "--alpha-start", "0.0", "--alpha-target", "0.9",
    ],
    "mlp": [
        "--run-kind", "graph_rw",
        "--rw-policy", "mlp_cdl",
        "--mlp-alternating",
        "--mlp-refresh-mode", "finetune",
        "--mlp-orientation", "source_start",
        "--mlp-tau", "0.5",
        "--mlp-src-rho", "0.3",
        "--rw-top-k", "4",
        "--refresh-interval", "5000",
        "--refresh-ema-beta", "0.0",
        "--alpha-warmup-start", "5000",
        "--alpha-warmup-steps", "10000",
        "--alpha-start", "0.0", "--alpha-target", "0.9",
        "--mlp-distill-n-orders", "200",
        "--mlp-distill-epochs", "60",
    ],
}


def build_cmd(arm: str, model_args: dict) -> list:
    """Return the full argv list (after sys.executable and train_clean_aogpt.py) for arm.

    Parameters
    ----------
    arm : str
        One of 'random', 'v3refresh', 'mlp'.
    model_args : dict
        Mapping of n_layer/n_head/n_embd read from config.json (may be empty).

    Returns
    -------
    list[str]
        The argv list to pass to subprocess (does NOT include sys.executable or script name).
    """
    if arm not in _ARM_FLAGS:
        raise ValueError(f"Unknown arm {arm!r}; valid: {list(_ARM_FLAGS)}")

    out_dir = _ARM_DIRS[arm]
    cmd = list(_SHARED_FLAGS)
    cmd += ["--output-dir", str(out_dir)]
    cmd += _ARM_FLAGS[arm]

    # Model arch flags from config.json (if available)
    for key, flag in [("n_layer", "--n-layer"), ("n_head", "--n-head"), ("n_embd", "--n-embd")]:
        if key in model_args:
            cmd += [flag, str(model_args[key])]

    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# NaN guard
# ──────────────────────────────────────────────────────────────────────────────
def first_eval_nan(arm: str):
    """Return True if earliest eval has non-finite val_ori_l2r_block, False if finite, None if unavailable."""
    curve_path = _ARM_DIRS[arm] / "eval_curve.tsv"
    if not curve_path.exists():
        return None
    lines = curve_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    hdr = lines[0].split("\t")
    if "val_ori_l2r_block" not in hdr:
        return None
    j = hdr.index("val_ori_l2r_block")
    v = lines[1].split("\t")[j]
    try:
        return not np.isfinite(float(v))
    except ValueError:
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Launcher
# ──────────────────────────────────────────────────────────────────────────────
def launch(arm: str):
    """Wait for a free GPU, launch arm, NaN-guard the first eval, wait for completion."""
    out_dir = _ARM_DIRS[arm]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency: already done?
    done = ov.load_curve(out_dir / "eval_curve.tsv")
    if TARGET_STEP in done:
        wlog(f"arm {arm!r} already reached step{TARGET_STEP}; skipping launch.", arm=arm)
        write_report()
        return

    model_args = load_model_args()
    if model_args:
        wlog(f"model_args from config.json: {model_args}", arm=arm)
    else:
        wlog("config.json not found or incomplete — using train_clean_aogpt.py defaults.", arm=arm)

    wlog(f"waiting for a GPU with >= {ov.FREE_MIB} MiB free ...", arm=arm)
    gpu_idx = ov.wait_for_gpu()

    argv = [sys.executable, "-u", "train_clean_aogpt.py"] + build_cmd(arm, model_args)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu_idx),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        TOKENIZERS_PARALLELISM="false",
    )
    log_path = out_dir / "train_stdout.log"
    wlog(f"launching arm {arm!r} on physical GPU{gpu_idx}: {' '.join(argv)}", arm=arm)

    proc = subprocess.Popen(
        argv,
        cwd=str(_HERE),
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    # Early NaN guard: poll until we get the first eval row, then decide.
    checked_first = False
    while True:
        rc = proc.poll()
        if not checked_first:
            nan = first_eval_nan(arm)
            if nan is True:
                wlog(
                    "FIRST EVAL is non-finite (NaN/inf) — killing run to protect the GPU slot.",
                    arm=arm,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                wlog("ABORTED on NaN. Investigate before relaunch.", arm=arm)
                return
            elif nan is False:
                checked_first = True
                wlog("first eval finite — NaN guard passed; running to step30000.", arm=arm)
        if rc is not None:
            wlog(f"training process exited rc={rc}", arm=arm)
            break
        time.sleep(60)

    write_report()
    wlog(f"arm {arm!r} DONE.", arm=arm)


# ──────────────────────────────────────────────────────────────────────────────
# Report writer
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(v) -> str:
    """Format a float or None for markdown tables."""
    if v is None:
        return "—"
    if isinstance(v, dict):
        v = v.get("val")
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def _delta(a, b) -> str:
    """Format a - b for markdown tables."""
    try:
        fa = float(a.get("val") if isinstance(a, dict) else a)
        fb = float(b.get("val") if isinstance(b, dict) else b)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return "—"
        return f"{fa - fb:+.4f}"
    except (TypeError, ValueError, AttributeError):
        return "—"


def _curve_shape_note(curve: dict, warmup_end: int = 5000) -> str:
    """Return a one-line human-readable note about the curve shape post-warmup."""
    if not curve:
        return "no data"
    steps = sorted(curve.keys())
    post = [curve[s]["val"] for s in steps if s >= warmup_end]
    if len(post) < 2:
        return "insufficient post-warmup data"
    monotone = all(post[i + 1] <= post[i] + 1e-6 for i in range(len(post) - 1))
    early_slope = (post[min(4, len(post) - 1)] - post[0]) if len(post) > 1 else 0.0
    late_slope = (post[-1] - post[max(-5, -len(post))]) if len(post) >= 5 else (post[-1] - post[0])
    slope_note = "accelerating" if late_slope < early_slope else "decelerating"
    return (
        f"post-{warmup_end} monotone={monotone}; "
        f"{post[0]:.4f}→{post[-1]:.4f} "
        f"(early_slope={early_slope:+.4f}, late_slope={late_slope:+.4f}, {slope_note})"
    )


def _read_refresh_diag(arm_dir: Path) -> list:
    """Parse refresh_diagnostics.jsonl; return list of dicts (or empty if missing)."""
    path = arm_dir / "refresh_diagnostics.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def write_report():
    """(Re)write alt_from0_REPORT.md from whatever eval_curves + diagnostics currently exist."""
    curve_a = ov.load_curve(OUT_A / "eval_curve.tsv")
    curve_b = ov.load_curve(OUT_B / "eval_curve.tsv")
    curve_c = ov.load_curve(OUT_C / "eval_curve.tsv")

    report_steps = [5000, 15000, 30000]

    # ── Table: val_ori_l2r_block at key steps ──────────────────────────────────
    md = [
        "# alt_from0 — three arms from-0 comparison\n",
        "Arms: **A=random** (baseline), **B=v3refresh** (progressive_rw_v3 from-0), "
        "**C=mlp_finetune** (mlp_cdl alternating self-bootstrap from-0).\n",
        "Primary metric: **val_ori_l2r_block** (lower = better).\n",
        "## 1. val_ori_l2r_block summary table",
        "| step | A random | B v3refresh | C mlp | Δ C−A | Δ C−B |",
        "|---|---|---|---|---|---|",
    ]

    for st in report_steps:
        va = curve_a.get(st)
        vb = curve_b.get(st)
        vc = curve_c.get(st)
        md.append(
            f"| {st} | {_fmt(va)} | {_fmt(vb)} | {_fmt(vc)} | {_delta(vc, va)} | {_delta(vc, vb)} |"
        )

    # ── Curve shape notes ──────────────────────────────────────────────────────
    md += [
        "\n## 2. Curve shape (post-warmup monotone check, slopes)",
        f"- **A random:** {_curve_shape_note(curve_a)}",
        f"- **B v3refresh:** {_curve_shape_note(curve_b)}",
        f"- **C mlp_finetune:** {_curve_shape_note(curve_c)}",
    ]

    # ── Full curves ────────────────────────────────────────────────────────────
    md += ["\n## 3. Full eval curves"]
    for label, curve in [("A random", curve_a), ("B v3refresh", curve_b), ("C mlp_finetune", curve_c)]:
        md.append(f"\n### {label}")
        if not curve:
            md.append("*(arm not run yet)*")
            continue
        md += ["| step | alpha | val_ori_l2r_block | lr |", "|---|---|---|---|"]
        for st in sorted(curve.keys()):
            e = curve[st]
            alpha_str = f"{e.get('alpha', float('nan')):.3f}"
            val_str = _fmt(e.get("val"))
            lr_str = f"{e.get('lr', float('nan')):.3e}"
            md.append(f"| {st} | {alpha_str} | {val_str} | {lr_str} |")

    # ── Arm C refresh diagnostics ──────────────────────────────────────────────
    md += ["\n## 4. Arm C — MLP refresh trajectory (refresh_diagnostics.jsonl)"]
    refresh_rows = _read_refresh_diag(OUT_C)
    if not refresh_rows:
        md.append(
            "*(No refresh_diagnostics.jsonl found — arm C not run yet or refresh not triggered yet.)*"
        )
    else:
        md += [
            "| step | val_kl | top1 | rollout_tau_vs_l2r | rollout_entropy | rollout_unique | teacher_tau_vs_l2r |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in refresh_rows:
            def _r(k, fmt=".4f"):
                v = row.get(k)
                if v is None:
                    return "—"
                try:
                    return format(int(v) if fmt == "d" else float(v), fmt)
                except (TypeError, ValueError):
                    return str(v)

            md.append(
                f"| {row.get('step', '—')} "
                f"| {_r('val_kl')} "
                f"| {_r('top1')} "
                f"| {_r('rollout_tau_vs_l2r')} "
                f"| {_r('rollout_entropy')} "
                f"| {_r('rollout_unique', 'd') if row.get('rollout_unique') is not None else '—'} "
                f"| {_r('teacher_tau_vs_l2r')} |"
            )
        # teacher_tau sharpness note
        ttau_vals = [row["teacher_tau_vs_l2r"] for row in refresh_rows if "teacher_tau_vs_l2r" in row]
        if len(ttau_vals) >= 2:
            direction = "sharpening (↑)" if ttau_vals[-1] > ttau_vals[0] else "flattening (↓)"
            md.append(f"\n*teacher_tau_vs_l2r trend: {ttau_vals[0]:.4f} → {ttau_vals[-1]:.4f} ({direction})*")

    # ── Verdict guide ──────────────────────────────────────────────────────────
    md += [
        "\n## 5. Verdict guide",
        "- **C ≪ A and C ≤ B** ⇒ self-bootstrap works: MLP from-0 matches or beats v3.",
        "- **C ≈ B** ⇒ matches v3; alternating adds little overhead over plain progressive_rw_v3.",
        "- **C ≈ A** ⇒ bootstrap fails; check `teacher_tau_vs_l2r` in the refresh trajectory "
        "(if it stays near 0, the early teacher provides no useful signal — "
        "consider longer warmup before first refresh or higher alpha-warmup-start).",
        "- **C > A** ⇒ MLP interference; verify mlp-distill-epochs/n-orders are not overfitting "
        "the first noisy teacher.",
        "",
        "*(Report auto-generated by `scripts/run_alternating_from0.py --report-only` "
        "or on completion of each arm.)*",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(md) + "\n")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REPORT written: {REPORT_PATH}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Launcher for alt_from0 arms A/B/C.  "
            "Pass --arm <name> to launch one arm, or --report-only to (re)write the report. "
            "No --arm and no --report-only → prints this help and exits WITHOUT launching."
        )
    )
    parser.add_argument(
        "--arm",
        choices=["random", "v3refresh", "mlp"],
        default=None,
        help="Which arm to launch: random (A), v3refresh (B), or mlp (C).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Skip launching; just (re)write REPORT.md from existing eval_curves.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if not args.arm and not args.report_only:
        # No action specified — print help and exit safely; do NOT launch anything.
        _parse_args(["--help"])
        return  # unreachable (argparse calls sys.exit) but keeps linters happy

    if args.report_only:
        write_report()
        return

    # --arm was specified
    launch(args.arm)


if __name__ == "__main__":
    main()
