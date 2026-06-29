#!/usr/bin/env python3
r"""Conditional Phase-2 chain: after source_start finishes, run reversed ONLY if it
beat random — then stop. Implements tonight's decision rule exactly:

    wait for cont_MLP_CDL_source_start to reach step30000
    if source_start@30000 < random@30000:   -> launch cont_MLP_CDL_reversed (attribution)
    else:                                    -> HOLD reversed; flag source_start for analysis

reversed = attribution control: if reversing the MLP rollout reaches ~v3, the gap is
orientation, not readout. original is held (run later only if reversed/source_start look good).
Single GPU, no multi-seed, no new directions.

Run (background):
    python block_lo_arm_order_network/scripts/run_phase2_chain.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "scripts"))

import run_source_start_overnight as ov   # reuse gpu_free_mib / wait_for_gpu / load_curve

SS_DIR = _REPO / "probe_results/attention_order_mlp/cont_MLP_CDL_source_start"
REV_DIR = _REPO / "probe_results/attention_order_mlp/cont_MLP_CDL_reversed"
RANDOM_CURVE = _HERE / "probe_results/clean_base_random_perm/eval_curve.tsv"
V3_CURVE = _HERE / "probe_results/clean_method_graph_rw_v3_from20k/eval_curve.tsv"
V3_A = "probe_results/clean_method_graph_rw_v3_from20k/A_global_eval.npy"
MLP_PT = "../probe_results/attention_order_mlp/phase1_text_mlp_ckpt20k.pt"
TAU, TOP_K, TARGET = 0.5, 4, 30000
MAX_WAIT_S = 8 * 3600
CHAIN_LOG = REV_DIR / "chain.log"


def clog(msg):
    REV_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with CHAIN_LOG.open("a") as f:
        f.write(line + "\n")


def wait_for_source_start():
    """Block until source_start reaches step30000; abort if its watcher logged a NaN abort."""
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_S:
        curve = ov.load_curve(SS_DIR / "eval_curve.tsv")
        if TARGET in curve:
            return True
        wl = SS_DIR / "watcher.log"
        if wl.exists() and "ABORTED" in wl.read_text():
            clog("source_start watcher reported ABORTED (NaN) — chain stops; analyze source_start.")
            return False
        time.sleep(120)
    clog(f"timeout ({MAX_WAIT_S}s) waiting for source_start@{TARGET}; stopping.")
    return False


def launch_reversed(gpu_idx):
    cmd = [
        sys.executable, "-u", "train_clean_aogpt.py",
        "--run-kind", "graph_rw", "--rw-policy", "mlp_cdl",
        "--mlp-path", MLP_PT, "--mlp-graph", V3_A,
        "--mlp-orientation", "reversed", "--mlp-tau", str(TAU),
        "--rw-top-k", str(TOP_K),
        "--resume-ckpt", "probe_results/clean_base_random_perm/ckpt_step20000.pt",
        "--output-dir", str(REV_DIR),
        "--max-steps", str(TARGET),
        "--lr", "1e-3", "--min-lr", "1e-4", "--lr-decay-steps", "50000",
        "--batch-size", "64", "--grad-accum", "2",
        "--alpha-start", "0.0", "--alpha-target", "0.9", "--alpha-warmup-steps", "10000",
        "--alpha-ramp-from-resume",
        "--eval-interval", "1000", "--log-interval", "100", "--save-steps", "25000,30000",
        "--device", "cuda:0",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_idx),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
    clog(f"launching reversed on physical GPU{gpu_idx}")
    return subprocess.Popen(cmd, cwd=str(_HERE), env=env,
                            stdout=open(REV_DIR / "train_stdout.log", "w"), stderr=subprocess.STDOUT)


def orientation_diag_reversed():
    import torch
    from directed_graph_policy import build_directed_graph
    import attn_order_mlp_policy as P
    from scipy.stats import kendalltau
    A = np.load(_HERE / V3_A).astype(np.float32); np.fill_diagonal(A, 0.0)
    B = build_directed_graph(A); N = B.shape[0]
    mlp = P.load_order_mlp(_REPO / "probe_results/attention_order_mlp/phase1_text_mlp_ckpt20k.pt",
                           torch.device("cpu"))
    orders, ent = P.sample_orders_batched_mlp(B, 128, mlp, "reversed", base_seed=20000,
                                              device=torch.device("cpu"), tau=TAU, top_k=TOP_K,
                                              return_entropy=True)
    orders = orders.cpu().numpy(); raster = np.arange(N)
    taus = np.array([kendalltau(o, raster).correlation for o in orders])
    sc = np.bincount(orders[:, 0], minlength=N)
    return dict(tau_vs_l2r=round(float(np.nanmean(taus)), 4), abs_tau=round(float(np.nanmean(np.abs(taus))), 4),
                start_mode=int(sc.argmax()), start_mode_frac=round(float(sc.max()) / 128, 3),
                avg_step_entropy=round(ent, 4), unique=int(len({tuple(o.tolist()) for o in orders})))


def write_reversed_report():
    rev = ov.load_curve(REV_DIR / "eval_curve.tsv")
    rnd = ov.load_curve(RANDOM_CURVE); v3 = ov.load_curve(V3_CURVE)
    ss = ov.load_curve(SS_DIR / "eval_curve.tsv")
    diag = orientation_diag_reversed()
    steps = sorted(rev)

    def row(st):
        return (st, rev.get(st), rnd.get(st), v3.get(st), ss.get(st))

    md = ["# cont_MLP_CDL_reversed — attribution arm\n",
          f"- reversed = MLP rollout flipped post-hoc; tests whether the learned topology is useful "
          f"once orientation is corrected. Same B / protocol / lr(anchor 50000) / alpha as source_start & v3.\n",
          "## val_ori_l2r_block",
          "| step | reversed | random | v3 | source_start | rev−random | rev−v3 |",
          "|---|---|---|---|---|---|---|"]
    for st in [21000, 25000, 30000]:
        s, a, r, v, sv = row(st)
        dr = f"{a-r:+.4f}" if (a is not None and r is not None) else "—"
        dv = f"{a-v:+.4f}" if (a is not None and v is not None) else "—"
        f = lambda x: f"{x:.4f}" if x is not None else "—"
        md.append(f"| {st} | {f(a)} | {f(r)} | {f(v)} | {f(sv)} | {dr} | {dv} |")
    md += [f"\n## orientation diagnostics (reversed, fixed B)",
           f"- tau_vs_l2r={diag['tau_vs_l2r']} abs_tau={diag['abs_tau']} start_mode={diag['start_mode']} "
           f"(frac {diag['start_mode_frac']}) entropy={diag['avg_step_entropy']} unique={diag['unique']}/128.\n",
           "## full reversed curve", "| step | val_ori_l2r_block |", "|---|---|"]
    for st in steps:
        md.append(f"| {st} | {rev[st]:.4f} |")
    (REV_DIR / "REPORT.md").write_text("\n".join(md) + "\n")
    clog(f"reversed REPORT written: {REV_DIR / 'REPORT.md'}")


def main():
    REV_DIR.mkdir(parents=True, exist_ok=True)
    clog("chain start: waiting for source_start to reach step30000 ...")
    if not wait_for_source_start():
        return

    ss30 = ov.load_curve(SS_DIR / "eval_curve.tsv").get(TARGET)
    rnd30 = ov.load_curve(RANDOM_CURVE).get(TARGET)
    v330 = ov.load_curve(V3_CURVE).get(TARGET)
    clog(f"source_start@{TARGET}={ss30}  random@{TARGET}={rnd30}  v3@{TARGET}={v330}")

    if ss30 is None or rnd30 is None:
        clog("missing source_start or random @30000 — cannot evaluate gate; stopping.")
        return
    if not (ss30 < rnd30):
        clog(f"GATE NOT MET: source_start ({ss30:.4f}) did NOT beat random ({rnd30:.4f}). "
             f"HOLDING reversed. Analyze source_start before burning more GPU.")
        return

    clog(f"GATE MET: source_start ({ss30:.4f}) beat random ({rnd30:.4f}) by {ss30-rnd30:+.4f}. "
         f"Proceeding to reversed.")
    # reversed already done?
    if TARGET in ov.load_curve(REV_DIR / "eval_curve.tsv"):
        clog("reversed already at step30000; writing report only.")
        write_reversed_report(); return

    clog("waiting for a free GPU (>= 20000 MiB) for reversed ...")
    gpu_idx = ov.wait_for_gpu()
    proc = launch_reversed(gpu_idx)
    checked = False
    while True:
        rc = proc.poll()
        if not checked:
            curve = ov.load_curve(REV_DIR / "eval_curve.tsv")
            if curve:
                first_val = curve[min(curve)]
                if not np.isfinite(first_val):
                    clog("reversed first eval non-finite — killing.")
                    proc.terminate()
                    try: proc.wait(timeout=60)
                    except subprocess.TimeoutExpired: proc.kill()
                    return
                checked = True
                clog("reversed first eval finite; continuing to step30000.")
        if rc is not None:
            clog(f"reversed training exited rc={rc}")
            break
        time.sleep(60)
    write_reversed_report()
    clog("chain DONE.")


if __name__ == "__main__":
    main()
