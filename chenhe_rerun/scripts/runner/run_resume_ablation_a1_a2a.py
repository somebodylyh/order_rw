"""
Run the ImageNet256 MaskGIT-VQ resume ablation after A1 finishes.

This script is intentionally narrow:
  1. Wait for the existing A1 attn4096/rerank32 curriculum runner to finish.
  2. Launch A2a single-process training on GPU 0.
  3. Run full val50k teacher-forced decoded FID for A1 and A2a.
  4. Write a compact report under report/test_resume.

It never kills or modifies existing processes. A1 is assumed to be managed by
the currently running hierarchical runner.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/data/users/chenhe/conda_envs/X1/bin/python")

A1_CONFIG = REPO_ROOT / "config/Imagenet256MaskGITVQ/seq256/non_permute/block1/segment_curriculum_attention_spectral_crossaxis_modelsignal_noprior_rarb_attn4096_rerank32.py"
A2A_CONFIG = REPO_ROOT / "config/Imagenet256MaskGITVQ/seq256/non_permute/block1/segment_curriculum_attention_spectral_crossaxis_modelsignal_noprior_rarb_attn4096_rerank32_a2a_singleprocess.py"

A1_CKPT = REPO_ROOT / "out/curriculum/nonpermute/imagenet256_maskgit_vq/seq256/block1/out-imagenet256-maskgitvq-seq256-block1-random-b32-rarb-nonpermute-attn-spectral-crossaxis-modelsignal-noprior-attn4096-rerank32-4-stage-70000-iters/ckpt.pt"
A2A_CKPT = REPO_ROOT / "out/ablation/nonpermute/imagenet256_maskgit_vq/seq256/block1/out-imagenet256-maskgitvq-seq256-block1-random-b32-rarb-nonpermute-a2a-singleprocess-attn4096-rerank32-70000-iters/ckpt.pt"

A1_REPORT_ROOT = REPO_ROOT / "Report/curriculum/nonpermute/imagenet256_maskgit_vq/seq256/block1/attention_spectral_crossaxis_modelsignal_noprior_rarb_nonpermute-attn4096-rerank32-4-stage-70000-iters"
A1_STAGE_RESULTS = [A1_REPORT_ROOT / f"stage_{idx:02d}/results.json" for idx in range(1, 5)]
A2A_POLICY_RESULTS = A1_STAGE_RESULTS[:3]
A1_RUNNER_META = A1_REPORT_ROOT / "runner_meta.json"

REPORT_DIR = REPO_ROOT / "report/test_resume"
LOG_DIR = REPORT_DIR / "logs"
EVAL_ROOT = REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/test_resume"
EVAL_SCRIPT = REPO_ROOT / "scripts/eval/eval_maskgit_vqgan_ckpt_decode_fid.py"

BASELINE_SUMMARIES = {
    "R0 Random eval": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/nonpermute_random_iter70000_fid_val50000_randommode_b8/summary.json",
    "R0 AR eval": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/nonpermute_random_iter70000_fid_val50000_armode_b8/summary.json",
    "AR baseline": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/nonpermute_ar_iter70000_fid_val50000_armode_b32/summary.json",
    "VQGAN target": REPO_ROOT / "Report/eval/imagenet256_maskgit_vq/maskgit_vqgan_reconstruction_fid_val50000_b32/summary.json",
}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (LOG_DIR / "orchestrator.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def checkpoint_iter(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
        value = checkpoint.get("iter_num")
        return int(value) if value is not None else None
    except Exception as exc:
        log(f"checkpoint read failed for {path}: {exc!r}")
        return None


def find_running_train(config_path):
    """Return PIDs for train.py processes that appear to use config_path."""
    config_path = Path(config_path)
    needles = {str(config_path), config_path.name}
    try:
        needles.add(str(config_path.relative_to(REPO_ROOT)))
    except ValueError:
        pass
    pids = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        if "train.py" not in cmdline:
            continue
        if any(needle in cmdline for needle in needles):
            pids.append(pid)
    return sorted(pids)


def all_a1_policies_ready():
    return all(path.exists() for path in A1_STAGE_RESULTS)


def a2a_policy_ready():
    return all(path.exists() for path in A2A_POLICY_RESULTS)


def a1_done():
    return checkpoint_iter(A1_CKPT) is not None and checkpoint_iter(A1_CKPT) >= 70000 and all_a1_policies_ready() and A1_RUNNER_META.exists()


def run_command(cmd, *, env=None, log_name):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    log("exec: " + " ".join(str(part) for part in cmd))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
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


def wait_for_a2a_policy():
    log("waiting for A1 unique policies needed by A2a")
    while not a2a_policy_ready():
        iter_value = checkpoint_iter(A1_CKPT)
        ready = [path.exists() for path in A1_STAGE_RESULTS]
        log(
            "A1 policy status: "
            f"iter={iter_value}, stages_ready={ready}, runner_meta={A1_RUNNER_META.exists()}"
        )
        time.sleep(300)
    log("A2a policy sources are ready; stage 4 will reuse stage 3 freeze policy")


def wait_for_a1():
    log("waiting for A1 attn4096/rerank32 runner to finish")
    while not a1_done():
        iter_value = checkpoint_iter(A1_CKPT)
        ready = [path.exists() for path in A1_STAGE_RESULTS]
        log(
            "A1 completion status: "
            f"iter={iter_value}, stages_ready={ready}, runner_meta={A1_RUNNER_META.exists()}"
        )
        time.sleep(300)
    log("A1 is complete")


def run_a2a():
    iter_value = checkpoint_iter(A2A_CKPT)
    if iter_value is not None and iter_value >= 70000:
        log(f"A2a already complete at iter={iter_value}; skipping training")
        return
    pids = find_running_train(A2A_CONFIG)
    if pids:
        log(f"A2a is already running as PID(s) {pids}; waiting instead of launching a duplicate")
        while True:
            iter_value = checkpoint_iter(A2A_CKPT)
            if iter_value is not None and iter_value >= 70000:
                log(f"A2a complete at iter={iter_value}")
                return
            pids = find_running_train(A2A_CONFIG)
            if not pids:
                log(f"A2a process exited before completion; checkpoint iter={iter_value}")
                break
            log(f"A2a running PID(s) {pids}; checkpoint iter={iter_value}")
            time.sleep(300)

    cmd = [PYTHON, "train.py", A2A_CONFIG]
    if iter_value is not None and iter_value > 0:
        cmd.append("--init_from=resume")
    run_command(
        cmd,
        env={"CUDA_VISIBLE_DEVICES": "0"},
        log_name="a2a_singleprocess_train.log",
    )
    iter_value = checkpoint_iter(A2A_CKPT)
    if iter_value is None or iter_value < 70000:
        raise RuntimeError(f"A2a checkpoint did not reach 70000 iterations: iter={iter_value}")
    log(f"A2a complete at iter={iter_value}")


def eval_summary_ready(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(payload.get("num_images", 0)) == 50000


def run_eval(name, ckpt_path):
    out_dir = EVAL_ROOT / name
    summary_path = out_dir / "summary.json"
    if eval_summary_ready(summary_path):
        log(f"eval already complete: {summary_path}")
        return summary_path
    run_command(
        [
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
            A1_STAGE_RESULTS[-1],
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
        ],
        env={"CUDA_VISIBLE_DEVICES": "0"},
        log_name=f"eval_{name}.log",
    )
    return summary_path


def read_summary(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fid_value(summary):
    if not summary:
        return None
    return (
        summary.get("fid", {})
        .get("model_argmax_decode_vs_adm_ref", {})
        .get("fid")
    )


def ce_value(summary):
    if not summary:
        return None
    return summary.get("model_token_metrics", {}).get("mean_ce_loss")


def acc_value(summary):
    if not summary:
        return None
    return summary.get("model_token_metrics", {}).get("token_accuracy")


def target_fid_value(summary):
    if not summary:
        return None
    if "fid" in summary and "target_vq_reconstruction_vs_adm_ref" in summary["fid"]:
        return summary["fid"]["target_vq_reconstruction_vs_adm_ref"].get("fid")
    if "fid" in summary and "target_vq_reconstruction_vs_adm_ref" in summary["fid"]:
        return summary["fid"]["target_vq_reconstruction_vs_adm_ref"].get("fid")
    return None


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(a1_summary_path, a2a_summary_path):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in BASELINE_SUMMARIES.items():
        summary = read_summary(path)
        rows.append((label, summary, path))
    rows.append(("A1 current runner SegmentGuided", read_summary(a1_summary_path), a1_summary_path))
    rows.append(("A2a single-process SegmentGuided", read_summary(a2a_summary_path), a2a_summary_path))

    lines = []
    lines.append("# Resume Ablation Results\n")
    lines.append(f"- Generated: {now()}")
    lines.append(f"- A1 config: `{A1_CONFIG.relative_to(REPO_ROOT)}`")
    lines.append(f"- A2a config: `{A2A_CONFIG.relative_to(REPO_ROOT)}`")
    lines.append("- Full FID protocol: `scripts/eval/eval_maskgit_vqgan_ckpt_decode_fid.py`, val50k, ADM reference `data/imagenet256_adm_ref/VIRTUAL_imagenet256_labeled.npz`.")
    lines.append("- A1/A2a FID uses the final current-frame `stage_04/results.json` policy with `eval_mode=SegmentGuided`, `segment_guided_ratio=1`.")
    lines.append("- A2a training uses A1 stage 1/2/3 policies; its final schedule slot reuses stage 3, matching A1's freeze-full-guidance stage 4 policy.")
    lines.append("")
    lines.append("| Run | Eval mode | Iter | CE | Token acc | FID vs ADM ref | Summary |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for label, summary, path in rows:
        if summary:
            mode = summary.get("eval_mode", "-")
            iter_value = summary.get("checkpoint_iter", "-")
            ce = ce_value(summary)
            acc = acc_value(summary)
            fid = fid_value(summary)
            rel = Path(path).relative_to(REPO_ROOT) if Path(path).is_absolute() else Path(path)
            lines.append(
                f"| {label} | {mode} | {iter_value} | {fmt(ce)} | {fmt(acc)} | {fmt(fid)} | `{rel}` |"
            )
        else:
            lines.append(f"| {label} | - | - | - | - | - | missing `{path}` |")
    lines.append("")

    recon = read_summary(BASELINE_SUMMARIES["VQGAN target"])
    target_fid = target_fid_value(recon) or fid_value(recon)
    lines.append(f"- VQGAN reconstruction/reference FID entry: {fmt(target_fid)}")
    lines.append(f"- A1 checkpoint iter: {checkpoint_iter(A1_CKPT)}")
    lines.append(f"- A2a checkpoint iter: {checkpoint_iter(A2A_CKPT)}")
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- This compares resume/reset behavior; A2a reuses A1's recovered current-frame policies but does not restart the Python process between stages.")
    lines.append("- No original-frame or grid diagnostic fields are used to build the policies or run SegmentGuided eval.")
    lines.append("- Existing R0/AR baselines are included from prior full val50k summaries for context.")
    lines.append("")
    (REPORT_DIR / "resume_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote report to {REPORT_DIR / 'resume_ablation_summary.md'}")


def main():
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    wait_for_a2a_policy()
    run_a2a()
    wait_for_a1()
    a1_summary = run_eval("a1_attn4096_rerank32_segmentguided_val50000_b32", A1_CKPT)
    a2a_summary = run_eval("a2a_singleprocess_attn4096_rerank32_segmentguided_val50000_b32", A2A_CKPT)
    write_report(a1_summary, a2a_summary)
    log("all requested A1/A2a resume ablation work completed")


if __name__ == "__main__":
    main()
