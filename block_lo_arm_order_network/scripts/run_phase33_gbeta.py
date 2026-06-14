"""§3.3 + Phase 1.5 driver: selected-head g_β pretrain + generalization gate.

B1 route (spec 2026-06-01-auto-order-head-end-to-end-training-design.md):
  1. Build a selected-head (L0H0) dataset at the 5k clean_base ckpt under the
     B0 canonical extraction: B0 batch-mean B -> CDL teacher σ_T -> train/val/test.
  2. Train g_β (reuse batch_readout.train_offline, nodewise + pairwise loss).
  3. Phase 1.5 generalization gate (CDL is only the ruler, never in the hook):
       - held-out (same-step 5k val + test)
       - cross-step (10k, 20k): does the 5k-trained g_β still fit CDL-quality
         order on the L0H0 graphs of LATER checkpoints?
  4. Report descriptive targets (τ≥0.6~0.7, pairwise_acc≥0.8, diversity intact).
     NO NLL anywhere in selection — these numbers are diagnostic only.

This does NOT touch the training hook; Phase 2 is gated on this passing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from batch_readout.selected_head_dataset import build_selected_head_dataset
from batch_readout.train_offline import train, _eval_split
from batch_readout.eval_frozen_phase2 import _load_g_beta
from batch_readout.diversity_batch import teacher_diversity_stats
from batch_readout.eval_metrics import kendall_tau_batch, pairwise_acc

CKPT_DIR_DEFAULT = PKG / "probe_results" / "clean_base_random_perm"

# Phase 1.5 descriptive targets (spec §4) — reported, not hard-killed mid-run.
TAU_TARGET = 0.6
PAIRWISE_TARGET = 0.8


def _ckpt(step: int, ckpt_dir=None) -> str:
    d = pathlib.Path(ckpt_dir) if ckpt_dir else CKPT_DIR_DEFAULT
    return str(d / f"ckpt_step{step}.pt")


def _build(step, head, M, batch_size, seed, out_path, device, fwd_batch, ckpt_dir=None):
    if out_path is not None and pathlib.Path(out_path).exists():
        d = np.load(out_path, allow_pickle=True)
        sig = np.concatenate([d["train_sigma_T"], d["val_sigma_T"], d["test_sigma_T"]])
        info = {"sigma_T": sig, "M_train": len(d["train_sigma_T"]),
                "M_val": len(d["val_sigma_T"]), "M_test": len(d["test_sigma_T"])}
        print(f"  [build step={step}] reuse cached {out_path}")
    else:
        info = build_selected_head_dataset(
            ckpt_path=_ckpt(step, ckpt_dir), head=head, M=M, batch_size=batch_size, seed=seed,
            none_mode="b0", out_path=out_path, device=device, split="train",
            fwd_batch=fwd_batch,
        )
    div = teacher_diversity_stats(info["sigma_T"])
    print(f"  [build step={step}] M={M} bs={batch_size} -> "
          f"train/val/test={info['M_train']}/{info['M_val']}/{info['M_test']} "
          f"| teacher diversity mean_pairwise_τ={div.get('mean_pairwise_tau'):.3f} "
          f"unique_σ_ratio={div.get('unique_sigma_ratio'):.3f} "
          f"first_step_H={div.get('first_step_entropy'):.3f}")
    return div


def _l2r_baseline(sigma_T, rank):
    """Constant 'always output L2R identity' baseline against the same teachers.

    Since the text teacher is mostly identity, this is the prior g_β must beat to
    prove it READS order out of B rather than memorizing a constant. logits=-arange
    so pl_argsort (descending) yields 0,1,...,N-1.
    """
    M, N = sigma_T.shape
    sig_pred = np.tile(np.arange(N), (M, 1))
    z = torch.from_numpy(np.tile(-np.arange(N, dtype=np.float32), (M, 1)))
    return {"kendall_tau": kendall_tau_batch(sig_pred, sigma_T),
            "pairwise_acc": pairwise_acc(z, torch.from_numpy(rank))}


def _eval_with_baseline(g_beta, B, sig, rank, device):
    """g_β metrics + L2R-prior baseline + g_β on the non-L2R subset (teacher≠identity).

    The non-L2R subset is where a constant-output g_β necessarily fails, so g_β's
    score there is the honest measure of B-dependent order reading.
    """
    m = _eval_split(g_beta, B.astype(np.float32), sig.astype(np.int64), rank.astype(np.int64), device=device)
    base = _l2r_baseline(sig, rank)
    N = sig.shape[1]
    nonl2r = ~(sig == np.arange(N)).all(axis=1)
    sub = None
    if nonl2r.sum() > 0:
        sub = _eval_split(g_beta, B[nonl2r].astype(np.float32),
                          sig[nonl2r].astype(np.int64), rank[nonl2r].astype(np.int64), device=device)
    return {"gbeta": m, "l2r_prior": base, "n": int(sig.shape[0]),
            "n_nonL2R": int(nonl2r.sum()),
            "gbeta_nonL2R": {k: sub[k] for k in ("kendall_tau", "pairwise_acc")} if sub else None}


def _eval_npz_all(g_beta, npz_path, device):
    d = np.load(npz_path, allow_pickle=True)
    B = np.concatenate([d["train_B_batch"], d["val_B_batch"], d["test_B_batch"]])
    sig = np.concatenate([d["train_sigma_T"], d["val_sigma_T"], d["test_sigma_T"]])
    rank = np.concatenate([d["train_rank"], d["val_rank"], d["test_rank"]])
    return _eval_with_baseline(g_beta, B, sig, rank, device)


def _eval_split_keys(g_beta, npz_path, prefix, device):
    d = np.load(npz_path, allow_pickle=True)
    return _eval_with_baseline(g_beta, d[f"{prefix}_B_batch"], d[f"{prefix}_sigma_T"],
                               d[f"{prefix}_rank"], device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny M for wiring validation")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--M", type=int, default=2000, help="num batch-mean graphs (full)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fwd-batch", type=int, default=8,
                    help="forward batch for attention extraction; keep small on a shared GPU")
    ap.add_argument("--out", default=str(PKG / "batch_readout" / "logs" / "phase33_gbeta"))
    ap.add_argument("--cross-steps", type=int, nargs="*", default=[10000, 20000])
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR_DEFAULT),
                    help="checkpoint directory (default: clean_base)")
    ap.add_argument("--step", type=int, default=5000,
                    help="checkpoint step to pretrain g_β on (match the hook start point)")
    args = ap.parse_args()

    head = (args.layer, args.head)
    M = 80 if args.smoke else args.M
    epochs = 8 if args.smoke else args.epochs
    tag = "smoke" if args.smoke else "full"
    ckpt_name = pathlib.Path(args.ckpt_dir).name
    out = pathlib.Path(args.out) / ckpt_name / tag
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"=== §3.3 g_β pretrain ({tag}) head=L{head[0]}H{head[1]} B0 canonical "
          f"M={M} bs={args.batch_size} epochs={epochs} step={args.step} ===")

    # --- Phase A: build training dataset ---
    print(f"[A] build step={args.step} selected-head dataset")
    train_npz = str(data_dir / f"ds_{args.step}.npz")
    div5k = _build(args.step, head, M, args.batch_size, args.seed, train_npz, args.device, args.fwd_batch, ckpt_dir=args.ckpt_dir)

    # --- Phase B: train g_β ---
    print("[B] train g_β (nodewise + pairwise)")
    n_tr = int(round(M * 0.8))
    res = train(
        dataset_path=train_npz, model_name="nodewise", loss_name="pairwise",
        N=64, batch_size=min(64, max(8, n_tr // 4)), lr=3e-4, epochs=epochs,
        seed=args.seed, out_dir=str(out), device=args.device,
    )
    gbeta_path = str(out / "g_beta_best.pt")
    bm = res["best_metrics"]
    print(f"  best epoch={res['best_epoch']} val: τ={bm['kendall_tau']:.3f} "
          f"pairwise={bm['pairwise_acc']:.3f} ρ={bm['spearman_rho']:.3f} "
          f"top1={bm['top1']:.3f} first3={bm['first3']:.3f}")

    # --- Phase C+D: Phase 1.5 generalization gate ---
    print("[C/D] Phase 1.5 generalization gate")
    g_beta = _load_g_beta(gbeta_path).to(args.device)
    report = {"config": {"head": list(head), "M": M, "batch_size": args.batch_size,
                         "epochs": epochs, "none_mode": "b0", "tag": tag},
              "teacher_diversity_5k": div5k, "splits": {}}

    def _report_line(tag, r):
        g, b = r["gbeta"], r["l2r_prior"]
        sub = r["gbeta_nonL2R"]
        sub_s = (f" | non-L2R[n={r['n_nonL2R']}] τ={sub['kendall_tau']:.3f} "
                 f"pw={sub['pairwise_acc']:.3f}") if sub else f" | non-L2R[n=0]"
        print(f"  {tag:12s} (n={r['n']}): g_β τ={g['kendall_tau']:.3f} pw={g['pairwise_acc']:.3f}"
              f"  vs L2R-prior τ={b['kendall_tau']:.3f} pw={b['pairwise_acc']:.3f}{sub_s}")

    # held-out same-step (5k val + test)
    for prefix in ("val", "test"):
        r = _eval_split_keys(g_beta, train_npz, prefix, args.device)
        report["splits"][f"5k_{prefix}"] = r
        _report_line(f"5k/{prefix}", r)

    # cross-step (build later-ckpt datasets, eval frozen g_β on all)
    M_cross = max(M // 4, 20)
    for step in args.cross_steps:
        npz = str(data_dir / f"ds_{step}.npz")
        divc = _build(step, head, M_cross, args.batch_size, args.seed, npz, args.device, args.fwd_batch, ckpt_dir=args.ckpt_dir)
        r = _eval_npz_all(g_beta, npz, args.device)
        r["teacher_diversity"] = divc
        report["splits"][f"{step}_cross"] = r
        _report_line(f"{step}/cross", r)

    # --- Phase E: verdict ---
    held = report["splits"]
    def _ok(key):
        g = held.get(key, {}).get("gbeta", {})
        return g.get("kendall_tau", 0) >= TAU_TARGET and g.get("pairwise_acc", 0) >= PAIRWISE_TARGET
    same_ok = _ok("5k_val") and _ok("5k_test")
    cross_ok = all(_ok(f"{s}_cross") for s in args.cross_steps)
    verdict = "PASS" if (same_ok and cross_ok) else "REVIEW"
    report["verdict"] = {"same_step_ok": same_ok, "cross_step_ok": cross_ok,
                         "targets": {"kendall_tau": TAU_TARGET, "pairwise_acc": PAIRWISE_TARGET},
                         "result": verdict}
    (out / "phase15_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[E] Phase 1.5 GATE (τ≥{TAU_TARGET}, pairwise≥{PAIRWISE_TARGET}): "
          f"same-step {'✓' if same_ok else '✗'} / cross-step {'✓' if cross_ok else '✗'} -> {verdict}")
    print(f"    report -> {out / 'phase15_report.json'}")


if __name__ == "__main__":
    main()
