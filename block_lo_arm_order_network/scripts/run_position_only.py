#!/usr/bin/env python3
r"""Launcher for cont_position_only — the entropy-matched POSITION-ONLY attribution control
for the source_start arm (Phase-2 text, single arm).

This is the key attribution control: same forward direction, same per-step entropy, same
protocol as source_start, but the order sampler uses ONLY a positional prior
(logits[v] = -v / pos_tau) — no attention-B, no MLP, no readiness. It answers:

    Is source_start's val_ori_l2r_block gain just "more L2R / lower entropy", or does the
    attention-B-conditioned MLP readout add value beyond a pure position prior?

pos_tau is read from scripts/calibration_position_only.json (produced by
calibrate_position_only_fine.py), where it was matched to source_start's recomputed
K=512 avg_step_entropy within +/-0.001. Everything else is byte-for-byte the source_start
config: resume ckpt_step20000 -> step30000 (10k), lr cosine anchored 50000, alpha 0->0.9
warmup 10000 ramp-from-resume, batch 64 x grad-accum 2, eval/log/save identical.

Run (background):
    python block_lo_arm_order_network/scripts/run_position_only.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent          # block_lo_arm_order_network/
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "scripts"))

import run_source_start_overnight as ov   # reuse gpu_free_mib / wait_for_gpu / load_curve / first_eval_nan

ARM = "cont_position_only"
OUT = _REPO / "probe_results/attention_order_mlp" / ARM
SS_CURVE = _REPO / "probe_results/attention_order_mlp/cont_MLP_CDL_source_start/eval_curve.tsv"
RANDOM_CURVE = _HERE / "probe_results/clean_base_random_perm/eval_curve.tsv"
V3_CURVE = _HERE / "probe_results/clean_method_graph_rw_v3_from20k/eval_curve.tsv"
CALIB_JSON = _HERE / "scripts/calibration_position_only.json"
WATCH_LOG = OUT / "watcher.log"

TARGET_STEP = 30000
TOP_K = 4


def wlog(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with WATCH_LOG.open("a") as f:
        f.write(line + "\n")


def load_calibration():
    if not CALIB_JSON.exists():
        raise SystemExit(f"missing {CALIB_JSON}; run calibrate_position_only_fine.py first.")
    d = json.load(open(CALIB_JSON))
    if not d.get("within_tol", False):
        raise SystemExit(f"calibration not within tol ({CALIB_JSON}); refusing to launch. {d}")
    return d


def launch_training(gpu_idx, pos_tau):
    cmd = [
        sys.executable, "-u", "train_clean_aogpt.py",
        "--run-kind", "graph_rw", "--rw-policy", "position_only",
        "--pos-tau", str(pos_tau),
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
    wlog(f"launching on physical GPU{gpu_idx} (pos_tau={pos_tau}): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(_HERE), env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    return proc


def first_eval_nan():
    curve = OUT / "eval_curve.tsv"
    if not curve.exists():
        return None
    lines = curve.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    hdr = lines[0].split("\t")
    j = hdr.index("val_ori_l2r_block")
    try:
        return not np.isfinite(float(lines[1].split("\t")[j]))
    except ValueError:
        return True


def write_report(calib):
    po = ov.load_curve(OUT / "eval_curve.tsv")
    rnd = ov.load_curve(RANDOM_CURVE)
    v3 = ov.load_curve(V3_CURVE)
    ss = ov.load_curve(SS_CURVE)

    def g(curve, step):
        return curve.get(step, {}).get("val")

    steps = sorted(po.keys())
    nan = any(not np.isfinite(v["val"]) for v in po.values())
    cdiag = calib.get("position_only", {})
    sdiag = calib.get("source_start", {})

    md = [f"# {ARM} — Phase-2 text attribution control (single arm)\n",
          f"- Entropy-matched POSITION-ONLY control for source_start. Order sampler uses ONLY "
          f"`logits[v] = -v / pos_tau` (no attention-B, no MLP, no readiness). "
          f"pos_tau={calib['pos_tau']} calibrated so avg_step_entropy="
          f"{cdiag.get('avg_step_entropy'):.5f} matches source_start's recomputed K={calib['K']} "
          f"target {sdiag.get('avg_step_entropy'):.5f} (|d|="
          f"{abs(cdiag.get('avg_step_entropy',0)-sdiag.get('avg_step_entropy',0)):.5f}, tol {calib['tol']}).",
          f"- 10k continuation from `ckpt_step20000` -> step{TARGET_STEP}; lr cosine anchored 50000; "
          f"alpha 0->0.9 warmup 10000 ramp-from-resume; eval split / clean_protocol / save steps "
          f"identical to source_start / v3 / random. top_k={TOP_K}.",
          f"- Primary metric: **val_ori_l2r_block** (lower = better). NaN in curve: {nan}.\n",
          "## val_ori_l2r_block vs source_start / v3 / random",
          "| step | position_only | source_start | v3 | random | Δ(po−ss) | Δ(po−v3) | Δ(po−rand) |",
          "|---|---|---|---|---|---|---|---|"]
    for st in [21000, 25000, 30000]:
        po_v, sv, vv, rv = g(po, st), g(ss, st), g(v3, st), g(rnd, st)
        f = lambda x: f"{x:.4f}" if x is not None else "—"
        dss = f"{po_v-sv:+.4f}" if (po_v is not None and sv is not None) else "—"
        dv3 = f"{po_v-vv:+.4f}" if (po_v is not None and vv is not None) else "—"
        dr = f"{po_v-rv:+.4f}" if (po_v is not None and rv is not None) else "—"
        md.append(f"| {st} | {f(po_v)} | {f(sv)} | {f(vv)} | {f(rv)} | {dss} | {dv3} | {dr} |")

    md += ["\n## full position_only curve (val_ori_l2r_block)",
           "| step | alpha | val_ori_l2r_block | lr |", "|---|---|---|---|"]
    for st in steps:
        e = po[st]
        md.append(f"| {st} | {e['alpha']:.3f} | {e['val']:.4f} | {e['lr']:.3e} |")

    md += ["\n## sampler diagnostics (calibration_position_only.json)",
           f"- position_only: pos_tau={calib['pos_tau']}, avg_step_entropy={cdiag.get('avg_step_entropy'):.5f}, "
           f"tau_vs_l2r={cdiag.get('tau_vs_l2r'):.4f}, start_mode={cdiag.get('start_mode')} "
           f"(frac {cdiag.get('start_mode_frac'):.3f}), unique={cdiag.get('unique')}.",
           f"- source_start (ref): avg_step_entropy={sdiag.get('avg_step_entropy'):.5f}, "
           f"tau_vs_l2r={sdiag.get('tau_vs_l2r'):.4f}, start_mode={sdiag.get('start_mode')}, "
           f"unique={sdiag.get('unique')}.\n"]

    # verdict (the agreed attribution logic)
    po30, ss30, v330, r30 = g(po, 30000), g(ss, 30000), g(v3, 30000), g(rnd, 30000)
    verdict = "pending (no step30000 row yet)"
    if po30 is not None and ss30 is not None:
        d_ss = po30 - ss30
        if abs(d_ss) < 0.010:
            verdict = (f"position_only ≈ source_start (Δ={d_ss:+.4f}): text gain is mostly L2R/position "
                       f"prior — attention-B contribution is NOT clean on text.")
        elif d_ss > 0:   # position_only worse (higher) than source_start
            close_v3 = (v330 is not None and abs(po30 - v330) < 0.010)
            if close_v3:
                verdict = (f"position_only ≈ v3 ({po30:.4f}) but clearly worse than source_start "
                           f"(Δ={d_ss:+.4f}): the prettiest outcome — L2R prior helps, AND the "
                           f"attention-conditioned MLP adds value beyond it.")
            else:
                verdict = (f"source_start clearly beats position_only (Δ={d_ss:+.4f}): attention-B "
                           f"MLP readout provides value beyond a pure position/L2R prior.")
        else:            # position_only better than source_start
            verdict = (f"position_only beats source_start (Δ={d_ss:+.4f}): the position prior alone "
                       f"is stronger than the MLP arm — reconsider source_start's value on text.")
    md += ["## Verdict (attribution)", f"- {verdict}",
           f"- @30000: position_only={po30}, source_start={ss30}, v3={v330}, random={r30}.\n",
           "## Next (per plan)",
           "- If position_only ≈ source_start: tighten the text claim (gain ≈ L2R-ness); prioritize image Phase 2.",
           "- If source_start clearly beats position_only: text attention contribution holds; then image Phase 2.",
           "- reversed / C-D+L-direct remain held (lower priority, post-hoc)."]

    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    json.dump(dict(position_only={s: po[s] for s in steps},
                   calibration=calib,
                   at={st: dict(position_only=g(po, st), source_start=g(ss, st),
                                v3=g(v3, st), random=g(rnd, st)) for st in [21000, 25000, 30000]},
                   verdict=verdict, nan=nan),
              open(OUT / "report.json", "w"), indent=2)
    wlog(f"REPORT written: {OUT / 'REPORT.md'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calib = load_calibration()
    pos_tau = float(calib["pos_tau"])
    wlog(f"calibration loaded: pos_tau={pos_tau} (within_tol={calib['within_tol']}, "
         f"po_ent={calib['position_only']['avg_step_entropy']:.5f} vs "
         f"ss_ent={calib['source_start']['avg_step_entropy']:.5f})")

    # idempotency
    done = ov.load_curve(OUT / "eval_curve.tsv")
    if TARGET_STEP in done:
        wlog(f"{ARM} already reached step{TARGET_STEP}; writing report only.")
        write_report(calib)
        return

    wlog(f"waiting for a GPU with >= {ov.FREE_MIB} MiB free ...")
    gpu_idx = ov.wait_for_gpu()
    proc = launch_training(gpu_idx, pos_tau)

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
                wlog("first eval finite — training/eval intact; continuing to step30000.")
        if rc is not None:
            wlog(f"training process exited rc={rc}")
            break
        time.sleep(60)

    write_report(calib)
    wlog("DONE.")


if __name__ == "__main__":
    main()
