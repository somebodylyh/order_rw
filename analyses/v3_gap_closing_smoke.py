"""Step-1 CPU wiring smoke for the naive gap-closing experiment.

Runs the TWO arms — frozen_gbeta (control) vs joint_group m=16 (unfrozen PG) —
from the SAME 20k aligned parent for a handful of steps on CPU. It does NOT look
at loss curves / science; it only validates the pipeline wiring, metric logging,
PG gradient flow, and gradient routing that the user's checklist requires.

Only causal variable between arms: gβ requires_grad (False vs True).
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for p in (ROOT, BLOCK_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 20k aligned parent (relative to repo root → portable; override with --ckpt).
PARENT_20K = str(
    ROOT / "block_lo_arm_order_network/probe_results/"
    "gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt"
)


def _check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PARENT_20K)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-root", default="runs/v3_gap_smoke")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from analyses.v3_phase1_runner import _build_evaluator
    from analyses.v3_group_trainer import train_arm
    from analyses.p5_utility_controller import load_p5_ckpt
    from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
    from analyses.p7_gbeta_policy import GBETA_CKPT
    from analyses.v3_own_order_eval import group_policy_orders

    print(f"[smoke] parent={args.ckpt}\n[smoke] n_steps={args.n_steps} "
          f"batch_size={args.batch_size} device={args.device}", flush=True)

    # Fixed held-out set + evaluator built ONCE, shared by BOTH arms (same heldout).
    model0, chunks, clean_perm, dev = load_p5_ckpt(args.ckpt, args.batch_size,
                                                   device=args.device)
    held_chunks = [chunks[i] for i in range(args.batch_size)]
    held_stack = torch.stack(held_chunks)

    # Capture the 20k INITIAL gβ deploy orders for tau_to_init (policy-diffusion).
    wrap0 = AOGPTWithOrderHead(model0, OrderHeadModule(GBETA_CKPT, device=str(dev)),
                               clean_perm, device=str(dev))
    init_orders = group_policy_orders(wrap0, held_stack, m=16, seed=0, device=str(dev))
    evaluator = _build_evaluator(held_chunks, clean_perm, str(dev), m=16,
                                 init_orders=init_orders)

    # Only causal variable: gβ requires_grad (frozen m=batch/G=1, joint m=16/G>=2).
    arm_m = {"frozen_gbeta": args.batch_size, "joint_group": 16}
    summary = {}
    for arm in ("frozen_gbeta", "joint_group"):
        summary[arm] = train_arm(
            args.ckpt, arm,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            m=arm_m[arm],
            device=args.device,
            out_dir=args.out_root,
            tag="gap_smoke",
            eval_steps=(args.n_steps,),
            evaluator=evaluator,
            save_checkpoints=False,
            allow_batch_size_override=True,
        )

    ok = True

    # ── Frozen arm checks ────────────────────────────────────────────────────
    fz = summary["frozen_gbeta"]
    fz_ev = fz["evals"][-1]
    print("\n[frozen_gbeta] checks:")
    ok &= _check("parent loaded + ran", fz["completed_steps"] == args.n_steps,
                 f"steps={fz['completed_steps']}")
    ok &= _check("own_val logged", "val_loss" in fz_ev, f"{fz_ev.get('val_loss'):.4f}")
    ok &= _check("own_over_l2r logged", "own_over_l2r" in fz_ev,
                 f"{fz_ev.get('own_over_l2r'):+.4f}")
    ok &= _check("tau_to_l2r logged", "tau_to_l2r" in fz_ev,
                 f"{fz_ev.get('tau_to_l2r'):.3f}")
    ok &= _check("gβ NOT updated (param_delta==0)",
                 fz["orderhead_param_delta"] == 0.0,
                 f"delta={fz['orderhead_param_delta']}")
    ok &= _check("no NaN", not fz["nan"])

    # ── Joint arm checks ─────────────────────────────────────────────────────
    jg = summary["joint_group"]
    jg_ev = jg["evals"][-1]
    log = jg["log"]
    max_oh_gnorm = max(e["orderhead_grad_norm"] for e in log)
    max_bb_gnorm = max(e["backbone_grad_norm"] for e in log)
    last_entropy = log[-1]["entropy"]
    all_pg_finite = all(e["pg"] == e["pg"] for e in log)  # not NaN
    print("\n[joint_group m=16] checks:")
    ok &= _check("parent loaded + ran", jg["completed_steps"] == args.n_steps,
                 f"steps={jg['completed_steps']}")
    ok &= _check("gβ updated (param_delta>0)", jg["orderhead_param_delta"] > 0,
                 f"delta={jg['orderhead_param_delta']:.3e}")
    ok &= _check("gβ grad_norm>0 (log_prob path reaches gβ)", max_oh_gnorm > 0,
                 f"max={max_oh_gnorm:.3e}")
    ok &= _check("backbone grad_norm>0", max_bb_gnorm > 0, f"max={max_bb_gnorm:.3e}")
    ok &= _check("pg_loss finite", all_pg_finite)
    ok &= _check("entropy not collapsed", last_entropy > 1e-3,
                 f"last={last_entropy:.4f}")
    ok &= _check("own_over_l2r logged", "own_over_l2r" in jg_ev,
                 f"{jg_ev.get('own_over_l2r'):+.4f}")
    ok &= _check("tau_to_init logged", "tau_to_init" in jg_ev,
                 f"{jg_ev.get('tau_to_init'):.3f}")
    ok &= _check("adv_mean/adv_std/score_norm logged",
                 all(k in log[-1] for k in ("adv_mean", "adv_std", "score_norm")),
                 f"adv_std={log[-1].get('adv_std'):.3e} score_norm={log[-1].get('score_norm'):.3f}")
    ok &= _check("PG stays off backbone (pg_only_backbone_grad≈0)",
                 jg["pg_only_backbone_grad"] < 1e-4,
                 f"{jg['pg_only_backbone_grad']:.3e}")
    ok &= _check("no NaN", not jg["nan"])

    print(f"\n[smoke] frozen own_over_l2r={fz_ev['own_over_l2r']:+.4f}  "
          f"joint own_over_l2r={jg_ev['own_over_l2r']:+.4f}  "
          f"(20-step wiring only, NOT a science result)")
    print(f"\n[smoke] {'ALL WIRING CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
