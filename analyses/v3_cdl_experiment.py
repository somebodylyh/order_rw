"""CDL-teacher order experiment (from 10k, frozen, no gβ training).

Compares two FROZEN order sources continued from the same 10k parent, measured by
own_over_l2r (deployment gap):
  - frozen_gbeta : gβ argsort order (the distilled student)
  - cdl_teacher  : greedy C-D+L rollout order on the same B (the teacher directly)

Both frozen (no PG). Question: does the richer CDL teacher order leave a smaller /
different persistent gap than its distilled gβ? Each arm uses its own reveal-order
source in the evaluator (own_over_l2r under that arm's order); same held set.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for p in (ROOT, BLOCK_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PARENT_10K = "runs/handoff_overnight/seed123/ckpt_step10000.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PARENT_10K)
    ap.add_argument("--n-steps", type=int, default=5000)      # 10k -> 15k
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-held", type=int, default=64)
    ap.add_argument("--eval-m", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default="runs/v3_cdl_from10k")
    ap.add_argument("--allow-batch-override", action="store_true")
    args = ap.parse_args()

    import torch
    from analyses.v3_phase1_runner import _build_evaluator
    from analyses.v3_group_trainer import train_arm
    from analyses.p5_utility_controller import load_p5_ckpt
    from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
    from analyses.p7_gbeta_policy import GBETA_CKPT
    from analyses.v3_own_order_eval import group_policy_orders, cdl_order_fn

    out = pathlib.Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cdl] parent={args.ckpt}  {args.n_steps} steps eval/{args.eval_every} "
          f"device={args.device}", flush=True)

    model0, chunks, clean_perm, dev = load_p5_ckpt(args.ckpt, args.n_held,
                                                   device=args.device)
    held_chunks = [chunks[i] for i in range(args.n_held)]
    held_stack = torch.stack(held_chunks)
    wrap0 = AOGPTWithOrderHead(model0, OrderHeadModule(GBETA_CKPT, device=str(dev)),
                               clean_perm, device=str(dev))

    # Per-arm init orders (10k) + per-arm evaluator (own order source).
    gbeta_init = group_policy_orders(wrap0, held_stack, m=args.eval_m, seed=0,
                                     device=str(dev))
    cdl_init = group_policy_orders(wrap0, held_stack, m=args.eval_m, seed=0,
                                   device=str(dev), order_fn=cdl_order_fn)
    evaluators = {
        "frozen_gbeta": _build_evaluator(held_chunks, clean_perm, str(dev),
                                         m=args.eval_m, init_orders=gbeta_init),
        "cdl_teacher": _build_evaluator(held_chunks, clean_perm, str(dev),
                                        m=args.eval_m, init_orders=cdl_init,
                                        order_fn=cdl_order_fn),
    }
    del model0, wrap0

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    start = int(raw.get("global_step", raw.get("iter_num", 10000)))
    del raw
    eval_steps = tuple(range(args.eval_every, args.n_steps + 1, args.eval_every))

    summary = {}
    for arm in ("frozen_gbeta", "cdl_teacher"):
        print(f"\n[cdl] === arm {arm} ===", flush=True)
        res = train_arm(
            args.ckpt, arm,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            m=args.batch_size,                       # G=1 batch-mean deploy
            device=args.device,
            out_dir=str(out),
            tag="cdl_from10k",
            eval_steps=eval_steps,
            evaluator=evaluators[arm],
            save_checkpoints=True,
            allow_batch_size_override=args.allow_batch_override,
        )
        summary[arm] = res
        print(f"[cdl] {arm} own_over_l2r / tau_to_l2r / tau_to_init:", flush=True)
        for e in res["evals"]:
            print(f"   step {start + e['step']}: own_over_l2r={e['own_over_l2r']:+.4f} "
                  f"tau_to_l2r={e.get('tau_to_l2r', float('nan')):+.3f} "
                  f"tau_to_init={e.get('tau_to_init', float('nan')):.3f}", flush=True)

    gb, cd = summary["frozen_gbeta"]["evals"], summary["cdl_teacher"]["evals"]
    report = {
        "ckpt": args.ckpt, "start_step": start, "n_steps": args.n_steps,
        "gbeta_own_over_l2r": [e["own_over_l2r"] for e in gb],
        "cdl_own_over_l2r": [e["own_over_l2r"] for e in cd],
        "gbeta_final": gb[-1]["own_over_l2r"], "cdl_final": cd[-1]["own_over_l2r"],
        "cdl_minus_gbeta_final": cd[-1]["own_over_l2r"] - gb[-1]["own_over_l2r"],
    }
    (out / "cdl_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[cdl] gbeta final own_over_l2r={report['gbeta_final']:+.4f}  "
          f"cdl final own_over_l2r={report['cdl_final']:+.4f}  "
          f"(cdl-gbeta={report['cdl_minus_gbeta_final']:+.4f})", flush=True)
    print(f"[cdl] wrote {out}/cdl_report.json", flush=True)


if __name__ == "__main__":
    main()
