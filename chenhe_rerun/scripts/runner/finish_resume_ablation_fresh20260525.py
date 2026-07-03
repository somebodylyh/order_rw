"""
Finish the ImageNet256 MaskGIT-VQ resume ablation for the fresh A2a rerun.

This script is intentionally narrow and non-invasive:
  1. Wait for the fresh A2a single-process training checkpoint to reach 70k.
  2. Run full val50k teacher-forced decoded FID for A1 and fresh A2a.
  3. Write a compact markdown report under report/test_resume.

Policies are loaded from current-frame results.json. Original/grid diagnostics are
reported only as context elsewhere; they are not used here to build an eval policy.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/data/users/chenhe/conda_envs/X1/bin/python")

A1_CONFIG = REPO_ROOT / "config/Imagenet256MaskGITVQ/seq256/non_permute/block1/segment_curriculum_attention_spectral_crossaxis_modelsignal_noprior_rarb_attn4096_rerank32.py"
A2A_CONFIG = REPO_ROOT / "config/Imagenet256MaskGITVQ/seq256/non_permute/block1/segment_curriculum_attention_spectral_crossaxis_modelsignal_noprior_rarb_attn4096_rerank32_a2a_singleprocess.py"

A1_CKPT = REPO_ROOT / "out/curriculum/nonpermute/imagenet256_maskgit_vq/seq256/block1/out-imagenet256-maskgitvq-seq256-block1-random-b32-rarb-nonpermute-attn-spectral-crossaxis-modelsignal-noprior-attn4096-rerank32-4-stage-70000-iters/ckpt.pt"
A1_REPORT_ROOT = REPO_ROOT / "Report/curriculum/nonpermute/imagenet256_maskgit_vq/seq256/block1/attention_spectral_crossaxis_modelsignal_noprior_rarb_nonpermute-attn4096-rerank32-4-stage-70000-iters"
A1_STAGE4_RESULTS = A1_REPORT_ROOT / "stage_04/results.json"
A1_RUNNER_META = A1_REPORT_ROOT / "runner_meta.json"

A2A_PARTIAL_CKPT = REPO_ROOT / "out/ablation/nonpermute/imagenet256_maskgit_vq/seq256/block1/out-imagenet256-maskgitvq-seq256-block1-random-b32-rarb-nonpermute-a2a-singleprocess-attn4096-rerank32-70000-iters/ckpt.pt"
A2A_FRESH_OUT_DIR = REPO_ROOT / "out/ablation/nonpermute/imagenet256_maskgit_vq/seq256/block1/out-imagenet256-maskgitvq-seq256-block1-random-b32-rarb-nonpermute-a2a-singleprocess-attn4096-rerank32-70000-iters-fresh20260525_tmux"
A2A_FRESH_CKPT = A2A_FRESH_OUT_DIR / "ckpt.pt"
A2A_FRESH_OUT_DIR_REL = A2A_FRESH_OUT_DIR.relative_to(REPO_ROOT)
A2A_PROCESS_NEEDLES = (
    str(A2A_FRESH_OUT_DIR),
    str(A2A_FRESH_OUT_DIR_REL),
    "fresh20260525-tmux",
)

REPORT_DIR = REPO_ROOT / "report/test_resume"
LOG_DIR = REPORT_DIR / "logs"
EVAL_ROOT = REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/test_resume"
EVAL_SCRIPT = REPO_ROOT / "scripts/eval/eval_maskgit_vqgan_ckpt_decode_fid.py"

BASELINE_SUMMARIES = {
    "R0 Random eval": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/nonpermute_random_iter70000_fid_val50000_randommode_b8/summary.json",
    "R0 AR eval": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/nonpermute_random_iter70000_fid_val50000_armode_b8/summary.json",
    "AR baseline": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/nonpermute_ar_iter70000_fid_val50000_armode_b32/summary.json",
    "VQGAN reconstruction": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/maskgit_vqgan_reconstruction_fid_val50000_b32/summary.json",
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (LOG_DIR / "finish_fresh20260525.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def checkpoint_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        log(f"checkpoint read failed for {path}: {exc!r}")
        return {}
    return {
        "iter": checkpoint.get("iter_num"),
        "best_val_loss": checkpoint.get("best_val_loss"),
    }


def checkpoint_iter(path: Path) -> int | None:
    value = checkpoint_meta(path).get("iter")
    return int(value) if value is not None else None


def find_running_process(needles: tuple[str, ...]) -> list[int]:
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in cmdline for needle in needles):
            pids.append(pid)
    return sorted(pids)


def wait_for_fresh_a2a() -> None:
    log("waiting for fresh A2a single-process checkpoint to reach iter 70000")
    while True:
        iter_value = checkpoint_iter(A2A_FRESH_CKPT)
        pids = find_running_process(A2A_PROCESS_NEEDLES)
        if iter_value is not None and iter_value >= 70000:
            if pids:
                log(f"fresh A2a ckpt is complete at iter={iter_value}; waiting for train process to exit: {pids}")
                time.sleep(60)
                continue
            log(f"fresh A2a complete at iter={iter_value}")
            return
        if not pids and iter_value is not None:
            raise RuntimeError(f"fresh A2a stopped before completion: iter={iter_value}")
        log(f"fresh A2a status: iter={iter_value}, running_pids={pids}")
        time.sleep(300)


def run_command(cmd: list[object], *, env: dict[str, str], log_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update(env)
    log_path = LOG_DIR / log_name
    log("exec: " + " ".join(str(part) for part in cmd))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] exec: {' '.join(str(part) for part in cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(REPO_ROOT),
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            ret = proc.poll()
            if ret is not None:
                if ret != 0:
                    raise RuntimeError(f"command failed with exit code {ret}: {' '.join(str(part) for part in cmd)}")
                log(f"completed: {log_name}")
                return
            time.sleep(60)


def eval_summary_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(summary.get("num_images", 0)) == 50000


def run_eval(name: str, ckpt_path: Path) -> Path:
    out_dir = EVAL_ROOT / name
    summary_path = out_dir / "summary.json"
    if eval_summary_ready(summary_path):
        log(f"eval already complete: {summary_path}")
        return summary_path
    cmd = [
        PYTHON,
        EVAL_SCRIPT,
        "--ckpt_path",
        ckpt_path,
        "--dataset_dir",
        "data/Imagenet256MaskGITVQ",
        "--split",
        "val",
        "--out_dir",
        out_dir,
        "--num_images",
        "50000",
        "--batch_size",
        "32",
        "--decode_batch_size",
        "32",
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--decode_dtype",
        "float16",
        "--eval_mode",
        "SegmentGuided",
        "--segment_source_json",
        A1_STAGE4_RESULTS,
        "--segment_guided_ratio",
        "1",
        "--segment_top_k_pairs",
        "256",
        "--segment_max_len",
        "256",
        "--segment_max_units_per_order",
        "999999",
        "--segment_order_frame",
        "current",
    ]
    run_command(cmd, env={"CUDA_VISIBLE_DEVICES": "0"}, log_name=f"eval_{name}.log")
    if not eval_summary_ready(summary_path):
        raise RuntimeError(f"eval did not write a complete summary: {summary_path}")
    return summary_path


def read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def model_fid(summary: dict | None):
    if not summary:
        return None
    fid = summary.get("fid")
    if isinstance(fid, (int, float)):
        return fid
    if isinstance(fid, dict):
        return fid.get("model_argmax_decode_vs_adm_ref", {}).get("fid")
    return None


def ce(summary: dict | None):
    if not summary:
        return None
    return summary.get("model_token_metrics", {}).get("mean_ce_loss")


def acc(summary: dict | None):
    if not summary:
        return None
    return summary.get("model_token_metrics", {}).get("token_accuracy")


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_report(a1_summary_path: Path, a2a_summary_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in BASELINE_SUMMARIES.items():
        rows.append((label, read_summary(path), path))
    rows.append(("A1 current runner SegmentGuided", read_summary(a1_summary_path), a1_summary_path))
    rows.append(("A2a fresh single-process SegmentGuided", read_summary(a2a_summary_path), a2a_summary_path))

    lines: list[str] = []
    lines.append("# Resume Ablation Results")
    lines.append("")
    lines.append(f"- Generated: {now()}")
    lines.append(f"- A1 config: `{rel(A1_CONFIG)}`")
    lines.append(f"- A2a config: `{rel(A2A_CONFIG)}`")
    lines.append(f"- A1 final policy: `{rel(A1_STAGE4_RESULTS)}`")
    lines.append(f"- Full FID protocol: val50k teacher-forced argmax decode against ADM reference `data/imagenet256_adm_ref/VIRTUAL_imagenet256_labeled.npz`.")
    lines.append("- SegmentGuided eval uses current-frame policy ids with `segment_guided_ratio=1`; no original-frame/grid diagnostic field is used as policy input.")
    lines.append("- The first A2a attempt was interrupted at iter 42750 and is recorded only as an interruption, not as the A2a result.")
    lines.append("")
    lines.append("| Run | Eval mode | Iter | CE | Token acc | FID vs ADM ref | Summary |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for label, summary, path in rows:
        if summary:
            lines.append(
                f"| {label} | {summary.get('eval_mode', summary.get('protocol', '-'))} | "
                f"{summary.get('checkpoint_iter', '-')} | {fmt(ce(summary))} | "
                f"{fmt(acc(summary))} | {fmt(model_fid(summary))} | `{rel(path)}` |"
            )
        else:
            lines.append(f"| {label} | - | - | - | - | - | missing `{rel(path)}` |")
    lines.append("")
    a1_meta = checkpoint_meta(A1_CKPT)
    a2a_partial = checkpoint_meta(A2A_PARTIAL_CKPT)
    a2a_fresh = checkpoint_meta(A2A_FRESH_CKPT)
    lines.append("## Checkpoints")
    lines.append("")
    lines.append(f"- A1 checkpoint: iter `{a1_meta.get('iter')}`, best val `{fmt(a1_meta.get('best_val_loss'))}`.")
    lines.append(f"- Interrupted A2a attempt: iter `{a2a_partial.get('iter')}`, best val `{fmt(a2a_partial.get('best_val_loss'))}`; excluded from final A2a comparison.")
    lines.append(f"- Fresh A2a checkpoint: iter `{a2a_fresh.get('iter')}`, best val `{fmt(a2a_fresh.get('best_val_loss'))}`.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- A1 is the current hierarchical runner with per-stage recovery/resume.")
    lines.append("- A2a is the same schedule in one uninterrupted Python training process, using A1 stage 1/2/3 current-frame policies and reusing stage 3 for the final full-guidance slot.")
    lines.append("- This FID is decoded teacher-forced quality, not unconditional generation quality.")
    lines.append("")
    out_path = REPORT_DIR / "resume_ablation_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote report: {out_path}")


def main() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    if checkpoint_iter(A1_CKPT) != 70000:
        raise RuntimeError(f"A1 checkpoint is not complete: {A1_CKPT}")
    if not A1_RUNNER_META.exists():
        raise RuntimeError(f"A1 runner meta missing: {A1_RUNNER_META}")
    if not A1_STAGE4_RESULTS.exists():
        raise RuntimeError(f"A1 stage 4 results missing: {A1_STAGE4_RESULTS}")
    wait_for_fresh_a2a()
    a1_summary = run_eval("a1_attn4096_rerank32_segmentguided_val50000_b32", A1_CKPT)
    a2a_summary = run_eval("a2a_fresh20260525_tmux_attn4096_rerank32_segmentguided_val50000_b32", A2A_FRESH_CKPT)
    write_report(a1_summary, a2a_summary)
    log("all fresh resume ablation work completed")


if __name__ == "__main__":
    main()
