#!/usr/bin/env python3
r"""Overnight launcher for cont_MLP_CDL_source_start (Phase-2 text, single arm).

Waits until a GPU frees up (the co-tenant's ~13 GiB job ends), then runs the
source_start MLP-policy continuation from clean ckpt20000 → step30000 (10k steps),
matched to v3/random except the order policy. On completion, writes REPORT.md
comparing val_ori_l2r_block to the existing random/v3 eval_curves at step 25000/30000,
plus orientation diagnostics. Stops after this one arm (no original/reversed/multi-seed).

Run (background):
    python block_lo_arm_order_network/scripts/run_source_start_overnight.py
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent          # block_lo_arm_order_network/
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

ARM = "cont_MLP_CDL_source_start"
OUT = _REPO / "probe_results/attention_order_mlp" / ARM
V3_A = "probe_results/clean_method_graph_rw_v3_from20k/A_global_eval.npy"   # cwd-relative (_HERE)
MLP_PT = "../probe_results/attention_order_mlp/phase1_text_mlp_ckpt20k.pt"
RANDOM_CURVE = _HERE / "probe_results/clean_base_random_perm/eval_curve.tsv"
V3_CURVE = _HERE / "probe_results/clean_method_graph_rw_v3_from20k/eval_curve.tsv"
WATCH_LOG = OUT / "watcher.log"

FREE_MIB = 20000          # a GPU is "free" when this much memory is available (co-tenant gone)
STABLE = 2                # require this many consecutive free readings
POLL_S = 120
TAU, SRC_RHO, TOP_K = 0.5, 0.3, 4
TARGET_STEP = 30000


def wlog(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with WATCH_LOG.open("a") as f:
        f.write(line + "\n")


def gpu_free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode()
    return [int(x) for x in out.split()]


def wait_for_gpu():
    hits = 0
    while True:
        free = gpu_free_mib()
        idx = next((i for i, f in enumerate(free) if f >= FREE_MIB), None)
        if idx is not None:
            hits += 1
            wlog(f"GPU{idx} free={free[idx]}MiB (>= {FREE_MIB}); stable {hits}/{STABLE}")
            if hits >= STABLE:
                return idx
        else:
            if hits:
                wlog(f"free dropped (per-GPU {free}); reset stability counter")
            hits = 0
        time.sleep(POLL_S)


def launch_training(gpu_idx):
    cmd = [
        sys.executable, "-u", "train_clean_aogpt.py",
        "--run-kind", "graph_rw", "--rw-policy", "mlp_cdl",
        "--mlp-path", MLP_PT, "--mlp-graph", V3_A,
        "--mlp-orientation", "source_start", "--mlp-tau", str(TAU), "--mlp-src-rho", str(SRC_RHO),
        "--rw-top-k", str(TOP_K),
        "--resume-ckpt", "probe_results/clean_base_random_perm/ckpt_step20000.pt",
        "--output-dir", str(OUT),
        "--max-steps", str(TARGET_STEP),
        "--lr", "1e-3", "--min-lr", "1e-4", "--lr-decay-steps", "50000",
        "--batch-size", "64", "--grad-accum", "2",
        "--alpha-start", "0.0", "--alpha-target", "0.9", "--alpha-warmup-steps", "10000",
        "--alpha-ramp-from-resume",
        "--eval-interval", "1000", "--log-interval", "100",
        "--save-steps", "25000,30000",
        "--device", "cuda:0",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_idx),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
    log_path = OUT / "train_stdout.log"
    wlog(f"launching on physical GPU{gpu_idx}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(_HERE), env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    return proc


def first_eval_nan():
    """Return True if the earliest eval row has a non-finite val_ori_l2r_block."""
    curve = OUT / "eval_curve.tsv"
    if not curve.exists():
        return None
    lines = curve.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    hdr = lines[0].split("\t")
    j = hdr.index("val_ori_l2r_block")
    v = lines[1].split("\t")[j]
    try:
        return not np.isfinite(float(v))
    except ValueError:
        return True


def load_curve(path):
    out = {}
    if not Path(path).exists():
        return out
    lines = Path(path).read_text().strip().splitlines()
    hdr = lines[0].split("\t")
    si, vi, ai, li = (hdr.index(k) for k in ("step", "val_ori_l2r_block", "alpha", "lr"))
    for ln in lines[1:]:
        c = ln.split("\t")
        out[int(c[si])] = dict(val=float(c[vi]), alpha=float(c[ai]), lr=float(c[li]))
    return out


def orientation_diag():
    import torch
    from directed_graph_policy import build_directed_graph
    import attn_order_mlp_policy as P
    from scipy.stats import kendalltau
    A = np.load(_HERE / V3_A).astype(np.float32); np.fill_diagonal(A, 0.0)
    B = build_directed_graph(A); N = B.shape[0]
    mlp = P.load_order_mlp(_HERE.parent / "probe_results/attention_order_mlp/phase1_text_mlp_ckpt20k.pt",
                           torch.device("cpu"))
    src_node = int(np.argmax(P.readiness_vector(B)))
    orders, ent = P.sample_orders_batched_mlp(B, 128, mlp, "source_start", base_seed=20000,
                                              device=torch.device("cpu"), tau=TAU, top_k=TOP_K,
                                              src_rho=SRC_RHO, return_entropy=True)
    orders = orders.cpu().numpy()
    raster = np.arange(N)
    taus = np.array([kendalltau(o, raster).correlation for o in orders])
    sc = np.bincount(orders[:, 0], minlength=N)
    return dict(src_node=src_node, tau_vs_l2r=round(float(np.nanmean(taus)), 4),
                abs_tau=round(float(np.nanmean(np.abs(taus))), 4),
                start_mode=int(sc.argmax()), start_mode_frac=round(float(sc.max()) / 128, 3),
                avg_step_entropy=round(ent, 4),
                unique=int(len({tuple(o.tolist()) for o in orders})))


def write_report():
    ss = load_curve(OUT / "eval_curve.tsv")
    rnd = load_curve(RANDOM_CURVE)
    v3 = load_curve(V3_CURVE)
    diag = orientation_diag()

    def g(curve, step):
        return curve.get(step, {}).get("val")

    steps = sorted(ss.keys())
    nan = any(not np.isfinite(v["val"]) for v in ss.values())
    last = steps[-1] if steps else None

    rows = []
    for st in [21000, 25000, 30000]:
        rows.append((st, g(ss, st), g(rnd, st), g(v3, st)))

    md = [f"# {ARM} — Phase-2 text result (single arm)\n",
          f"- 10k continuation from `ckpt_step20000` → step{TARGET_STEP}; B = v3's `A_global_eval.npy` "
          f"(shared substrate); policy = MLP_CDL source_start (v3-style readiness anchor at t=0, then "
          f"MLP rollout). tau={TAU}, top_k={TOP_K}, src_rho={SRC_RHO}. lr cosine anchored 50000; "
          f"alpha 0→0.9 warmup 10000; eval split / clean_protocol identical to v3/random.",
          f"- Primary metric: **val_ori_l2r_block** (lower = better). NaN in curve: {nan}.\n",
          "## val_ori_l2r_block vs baselines",
          "| step | source_start | random | v3 | Δ(ss−random) | Δ(ss−v3) |",
          "|---|---|---|---|---|---|"]
    for st, a, r, v in rows:
        dr = f"{a - r:+.4f}" if (a is not None and r is not None) else "—"
        dv = f"{a - v:+.4f}" if (a is not None and v is not None) else "—"
        md.append(f"| {st} | {a if a is not None else '—'} | {r if r is not None else '—'} "
                  f"| {v if v is not None else '—'} | {dr} | {dv} |")

    # full source_start curve
    md += ["\n## source_start curve (val_ori_l2r_block)",
           "| step | alpha | val_ori_l2r_block | lr |", "|---|---|---|---|"]
    for st in steps:
        e = ss[st]
        md.append(f"| {st} | {e['alpha']:.3f} | {e['val']:.4f} | {e['lr']:.3e} |")

    # monotonic-ish check over the post-warmup region
    post = [ss[s]["val"] for s in steps if s >= 21000]
    decreasing = (len(post) >= 2 and post[-1] <= post[0])
    md += ["\n## orientation diagnostics (sampler on the fixed B)",
           f"- source node (argmax readiness) = {diag['src_node']}; start_mode = {diag['start_mode']} "
           f"(frac {diag['start_mode_frac']}); tau_vs_l2r = {diag['tau_vs_l2r']}; abs_tau = {diag['abs_tau']}; "
           f"avg_step_entropy = {diag['avg_step_entropy']}; unique = {diag['unique']}/128.\n",
           "## Read (the three checks)",
           f"1. **source_start @30000 beats random?** "
           + (f"{g(ss,30000) is not None and g(rnd,30000) is not None and g(ss,30000) < g(rnd,30000)} "
              f"(ss {g(ss,30000)} vs random {g(rnd,30000)})" if g(ss,30000) is not None else "pending"),
           f"2. **source_start @30000 close to v3?** "
           + (f"Δ={g(ss,30000)-g(v3,30000):+.4f} (ss {g(ss,30000)} vs v3 {g(v3,30000)})"
              if (g(ss,30000) is not None and g(v3,30000) is not None) else "pending"),
           f"3. **curve descends post-warmup (21k→30k)?** {decreasing} "
           f"({post[0]:.4f} → {post[-1]:.4f})" if post else "pending",
           "\n## Next (per plan)",
           "- If close to v3: add reversed/original as attribution arms.",
           "- If clearly below random/v3: do NOT burn the other arms; the gap is likely the C-D+L / MLP "
           "curriculum itself, not just direction.",
           "- Held: original / reversed / multi-seed (await this arm's verdict)."]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    json.dump(dict(source_start={s: ss[s] for s in steps}, orientation=diag,
                   at={st: dict(ss=g(ss, st), random=g(rnd, st), v3=g(v3, st)) for st in [21000, 25000, 30000]},
                   nan=nan),
              open(OUT / "report.json", "w"), indent=2)
    wlog(f"REPORT written: {OUT / 'REPORT.md'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # idempotency: if already completed, just (re)write the report.
    done = load_curve(OUT / "eval_curve.tsv")
    if TARGET_STEP in done:
        wlog(f"{ARM} already reached step{TARGET_STEP}; writing report only.")
        write_report()
        return

    wlog(f"waiting for a GPU with >= {FREE_MIB} MiB free (co-tenant to finish)...")
    gpu_idx = wait_for_gpu()
    proc = launch_training(gpu_idx)

    # early NaN guard: watch the first eval, then let it run to completion.
    checked_first = False
    while True:
        rc = proc.poll()
        if not checked_first:
            nan = first_eval_nan()
            if nan is True:
                wlog("FIRST EVAL is non-finite (NaN/inf) — killing run to protect the GPU slot.")
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                wlog("ABORTED on NaN. Investigate before relaunch.")
                return
            elif nan is False:
                checked_first = True
                wlog("first eval finite — G3 (training/eval intact) CLOSED; continuing to step30000.")
        if rc is not None:
            wlog(f"training process exited rc={rc}")
            break
        time.sleep(60)

    # cleanup the duplicate full A_global save is fine; keep ckpts (25000/30000 are intended).
    write_report()
    wlog("DONE.")


if __name__ == "__main__":
    main()
