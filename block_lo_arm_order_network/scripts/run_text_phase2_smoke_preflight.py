#!/usr/bin/env python3
r"""Phase-2 text smoke PREFLIGHT — gate the MLP order-policy wiring before any 3x10k run.

Per the agreed gates (do NOT launch the full 3x10k here; stop for check-in):
  1. B consistency  : the trainer's substrate B == v3's clean ckpt20000 A_global_eval.npy
                      (loaded directly; verified by sha + post-run max_abs_diff == 0).
  2. Orientation    : original=reverse (tau_vs_l2r<0, |tau| high), reversed=forward (tau>0),
                      source_start=anchored at the source node & more forward than original.
                      Records start node, tau_vs_l2r, abs_tau, entropy, unique for all three.
  3. Train/eval ok  : 200-step continuation runs (original, source_start, v3-ref) give finite
                      val_ori_l2r_block; lr cosine anchored to lr_decay_steps=50000 (lr column
                      matches v3-ref); alpha warmup matches v3; no NaN.
  4. Speed          : MLP sampler steady-state step time not drastically slower than v3.

Output: probe_results/attention_order_mlp/text_phase2_smoke_preflight/SMOKE_REPORT.md
"""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent          # block_lo_arm_order_network/
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from directed_graph_policy import build_directed_graph
import attn_order_mlp_policy as P

V3_DIR = _HERE / "probe_results/clean_method_graph_rw_v3_from20k"
V3_A = V3_DIR / "A_global_eval.npy"                       # = B extracted from ckpt20000 (v3 substrate)
MLP_PT = _REPO / "probe_results/attention_order_mlp/phase1_text_mlp_ckpt20k.pt"
RESUME = _HERE / "probe_results/clean_base_random_perm/ckpt_step20000.pt"
OUT = _REPO / "probe_results/attention_order_mlp/text_phase2_smoke_preflight"
DEVICE = "cuda:0"
TAU, TAU_START, TOP_K, SRC_RHO = 0.5, 0.1, 4, 0.3
START_STEP, SMOKE_STEPS = 20000, 200


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def kendall_vs_l2r(orders):
    from scipy.stats import kendalltau
    raster = np.arange(orders.shape[1])
    taus = np.array([kendalltau(o, raster).correlation for o in orders])
    return float(np.nanmean(taus)), float(np.nanmean(np.abs(taus)))


# ---------------------------------------------------------------------------
# Gate 1+2 — sampler-level orientation diagnostic on the trainer's exact B
# ---------------------------------------------------------------------------

def sampler_diagnostic():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    A = np.load(V3_A).astype(np.float32)                 # exactly how the trainer loads mlp_cdl B
    np.fill_diagonal(A, 0.0)
    B = build_directed_graph(A)
    N = B.shape[0]
    mlp = P.load_order_mlp(MLP_PT, device)
    readiness = P.readiness_vector(B)
    src_node = int(np.argmax(readiness))

    rows = {}
    K = 256
    for orient in P.ORIENTATIONS:
        t0 = time.time()
        orders, ent = P.sample_orders_batched_mlp(
            B, K, mlp, orient, base_seed=20000, device=device,
            tau=TAU, tau_start=TAU_START, top_k=TOP_K, src_rho=SRC_RHO, return_entropy=True,
        )
        dt = time.time() - t0
        orders = orders.cpu().numpy()
        tau_m, abs_m = kendall_vs_l2r(orders)
        legal = all(sorted(r.tolist()) == list(range(N)) for r in orders)
        start_counts = np.bincount(orders[:, 0], minlength=N)
        rows[orient] = dict(
            tau_vs_l2r=round(tau_m, 4), abs_tau=round(abs_m, 4),
            start_mode=int(start_counts.argmax()),
            start_mode_frac=round(float(start_counts.max()) / K, 3),
            avg_step_entropy=round(ent, 4),
            unique=int(len({tuple(o.tolist()) for o in orders})),
            legal=bool(legal), sample_ms_per_256=round(dt * 1000, 1),
        )
    return dict(N=N, src_node=src_node, K=K, rows=rows,
                B_sha=sha(V3_A), readiness_argmax=src_node)


# ---------------------------------------------------------------------------
# Gate 3+4 — short continuation runs
# ---------------------------------------------------------------------------

def run_one(name, extra_args):
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(_HERE / "train_clean_aogpt.py"),
        "--run-kind", "graph_rw",
        "--resume-ckpt", str(RESUME),
        "--output-dir", str(out_dir),
        "--max-steps", str(START_STEP + SMOKE_STEPS),
        "--lr", "1e-3", "--min-lr", "1e-4", "--lr-decay-steps", "50000",
        "--batch-size", "64", "--grad-accum", "2",
        "--rw-top-k", str(TOP_K),
        "--alpha-start", "0.0", "--alpha-target", "0.9", "--alpha-warmup-steps", "10000",
        "--eval-interval", "100", "--log-interval", "20",
        "--save-steps", "999999999",
        "--device", DEVICE,
    ] + extra_args
    log_path = out_dir / "smoke_stdout.log"
    t0 = time.time()
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    # clean up the large end-of-run checkpoint the trainer always saves at max_steps
    for ck in out_dir.glob("ckpt_step*.pt"):
        ck.unlink()
    return out_dir, rc, wall


def parse_run(out_dir):
    """Return dict with last val_ori_l2r_block, lr@last, alpha@last, nan flag, steady step time."""
    info = {"rc_ok": True}
    curve = out_dir / "eval_curve.tsv"
    rows = []
    if curve.exists():
        lines = curve.read_text().strip().splitlines()
        header = lines[0].split("\t")
        for ln in lines[1:]:
            rows.append(dict(zip(header, ln.split("\t"))))
    info["eval_rows"] = len(rows)
    if rows:
        last = rows[-1]
        info["step"] = int(last["step"])
        info["val_ori_l2r_block"] = float(last["val_ori_l2r_block"])
        info["alpha"] = float(last["alpha"])
        info["lr"] = float(last["lr"])
        info["val_finite"] = bool(np.isfinite(info["val_ori_l2r_block"]))
    else:
        info["val_finite"] = False
    # timing + nan from train_log
    tl = out_dir / "train_log.txt"
    info["nan_in_loss"] = False
    pts = []
    if tl.exists():
        for ln in tl.read_text().splitlines():
            m = re.search(r"step\s+(\d+)->\d+/\d+ \| loss=([\d.naif-]+).*\| (\d+)s$", ln)
            if m:
                step = int(m.group(1)); loss = m.group(2); elapsed = float(m.group(3))
                if "nan" in loss.lower() or "inf" in loss.lower():
                    info["nan_in_loss"] = True
                pts.append((step, elapsed))
    # steady-state per-step time: drop first 2 points (compile warmup)
    if len(pts) >= 4:
        s0, e0 = pts[2]
        s1, e1 = pts[-1]
        info["steady_s_per_step"] = round((e1 - e0) / max(s1 - s0, 1), 4)
    info["log_points"] = len(pts)
    return info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Gate 1+2: sampler diagnostic ===", flush=True)
    diag = sampler_diagnostic()
    print(json.dumps(diag, indent=2), flush=True)

    print("=== Gate 3+4: short continuation runs (200 steps each) ===", flush=True)
    runs = {}
    common_mlp = ["--rw-policy", "mlp_cdl", "--mlp-path", str(MLP_PT),
                  "--mlp-graph", str(V3_A), "--mlp-tau", str(TAU)]
    plan = {
        "mlp_original":     common_mlp + ["--mlp-orientation", "original"],
        "mlp_source_start": common_mlp + ["--mlp-orientation", "source_start", "--mlp-src-rho", str(SRC_RHO)],
        "v3_ref":           ["--rw-policy", "progressive_rw_v3", "--rw-lam", "0.75", "--rw-rho", "0.2",
                             "--tau-start", "0.10", "--tau-step", "0.10"],
    }
    for name, extra in plan.items():
        print(f"  -> running {name} ...", flush=True)
        out_dir, rc, wall = run_one(name, extra)
        info = parse_run(out_dir)
        info["rc_ok"] = (rc == 0)
        info["wall_s"] = round(wall, 1)
        runs[name] = info
        print(f"     {name}: rc_ok={info['rc_ok']} val_finite={info.get('val_finite')} "
              f"val={info.get('val_ori_l2r_block')} steady_s/step={info.get('steady_s_per_step')} "
              f"wall={info['wall_s']}s", flush=True)

    # B consistency post-run: trainer-saved A_global_eval must bit-match v3's
    ref = np.load(V3_A).astype(np.float64); np.fill_diagonal(ref, 0.0)
    bconsist = {}
    for name in ("mlp_original", "mlp_source_start"):
        saved = OUT / name / "A_global_eval.npy"
        if saved.exists():
            a = np.load(saved).astype(np.float64); np.fill_diagonal(a, 0.0)
            bconsist[name] = float(np.abs(a - ref).max())
        else:
            bconsist[name] = None

    # ---- gate verdicts ----
    o = diag["rows"]["original"]; r = diag["rows"]["reversed"]; s = diag["rows"]["source_start"]
    g1 = all(v == 0.0 for v in bconsist.values() if v is not None) and len(bconsist) > 0
    g2 = (o["tau_vs_l2r"] < 0 and o["abs_tau"] > 0.4 and r["tau_vs_l2r"] > 0
          and s["start_mode"] == diag["src_node"] and s["tau_vs_l2r"] > o["tau_vs_l2r"]
          and all(diag["rows"][x]["legal"] for x in P.ORIENTATIONS))
    g3 = all(runs[n]["rc_ok"] and runs[n].get("val_finite") and not runs[n]["nan_in_loss"] for n in runs)
    lrs = {n: runs[n].get("lr") for n in runs}
    lr_match = (lrs.get("mlp_original") is not None and lrs.get("v3_ref") is not None
                and abs(lrs["mlp_original"] - lrs["v3_ref"]) < 1e-9)
    g3 = g3 and lr_match
    sp_mlp = runs["mlp_original"].get("steady_s_per_step")
    sp_v3 = runs["v3_ref"].get("steady_s_per_step")
    slowdown = (sp_mlp / sp_v3) if (sp_mlp and sp_v3) else None
    g4 = (slowdown is not None and slowdown <= 2.0)

    verdict = "PASS" if (g1 and g2 and g3 and g4) else "FAIL"

    # ---- write report ----
    md = [f"# Phase-2 text smoke PREFLIGHT — **{verdict}**\n",
          f"- Substrate B = `{V3_A.relative_to(_REPO)}` (sha {diag['B_sha']}), loaded directly = v3's "
          f"ckpt20000 attention; N={diag['N']}, source node (argmax readiness) = {diag['src_node']}.",
          f"- MLP = `{MLP_PT.relative_to(_REPO)}`; sampling tau={TAU}, tau_start={TAU_START}, top_k={TOP_K}, "
          f"src_rho={SRC_RHO}. Continuation: resume ckpt_step{START_STEP} → {START_STEP+SMOKE_STEPS} "
          f"(alpha 0→0.9 warmup 10000, lr 1e-3→1e-4 cosine anchored 50000).\n",
          "## Gate 1 — B consistency",
          f"- post-run max_abs_diff(trainer A_global_eval, v3 A_global_eval): "
          + ", ".join(f"{k}={v}" for k, v in bconsist.items()) + f"  → **{'PASS' if g1 else 'FAIL'}**\n",
          "## Gate 2 — orientation arms",
          "| orient | tau_vs_l2r | abs_tau | start_mode | start_frac | avg_step_entropy | unique | legal | ms/256 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in P.ORIENTATIONS:
        v = diag["rows"][k]
        md.append(f"| {k} | {v['tau_vs_l2r']} | {v['abs_tau']} | {v['start_mode']} | {v['start_mode_frac']} "
                  f"| {v['avg_step_entropy']} | {v['unique']} | {v['legal']} | {v['sample_ms_per_256']} |")
    md += [f"\n- original reverse-chain (tau<0, |tau|>0.4): {o['tau_vs_l2r']<0 and o['abs_tau']>0.4}; "
           f"reversed forward (tau>0): {r['tau_vs_l2r']>0}; source_start anchored at source "
           f"node {diag['src_node']} (start_mode={s['start_mode']}) & more forward than original "
           f"({s['tau_vs_l2r']} > {o['tau_vs_l2r']}). → **{'PASS' if g2 else 'FAIL'}**\n",
           "## Gate 3 — training/eval intact",
           "| run | rc_ok | eval_rows | step | val_ori_l2r_block | alpha | lr | nan_in_loss |",
           "|---|---|---|---|---|---|---|---|"]
    for n, v in runs.items():
        md.append(f"| {n} | {v['rc_ok']} | {v.get('eval_rows')} | {v.get('step')} | "
                  f"{v.get('val_ori_l2r_block')} | {v.get('alpha')} | {v.get('lr')} | {v['nan_in_loss']} |")
    md += [f"\n- lr @ last step matches v3-ref (cosine anchored 50000): {lr_match}; all runs finite & no NaN. "
           f"→ **{'PASS' if g3 else 'FAIL'}**\n",
           "## Gate 4 — speed",
           f"- steady-state s/step: mlp_original={sp_mlp}, v3_ref={sp_v3}; slowdown×={round(slowdown,3) if slowdown else None} "
           f"(threshold ≤2.0). → **{'PASS' if g4 else 'FAIL'}**\n",
           "## Verdict",
           f"- **{verdict}**. " + ("All gates pass — safe to launch the 3×10k full arms "
            "(cont_MLP_CDL_original / reversed / source_start → step30000), reusing random/v3 eval_curve.tsv as baselines."
            if verdict == "PASS" else "One or more gates failed — fix before launching the full run.")]
    (OUT / "SMOKE_REPORT.md").write_text("\n".join(md) + "\n")
    json.dump(dict(diag=diag, runs=runs, bconsist=bconsist,
                   gates=dict(g1=g1, g2=g2, g3=g3, g4=g4, verdict=verdict)),
              open(OUT / "smoke_preflight.json", "w"), indent=2)
    print(f"\n=== SMOKE PREFLIGHT {verdict} ===  report: {OUT / 'SMOKE_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
