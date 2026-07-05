"""GPU pilot v0 for the naive gap-closing experiment (20k -> 25k).

Only causal variable: gβ requires_grad (frozen_gbeta control vs joint_group m=16
unfrozen PG). Everything else shared: same 20k parent, data stream, batch, backbone
optimizer/schedule, eval set + grouping, seed.

Conservative naive PG (protect the 20k gβ init, small steps near it):
  lr_orderhead = 5e-5,  entropy beta = 0 (no positive entropy pressure),  tau = 1.0,
  EMA own-loss baseline + advantage clip (standard REINFORCE variance control only).

Headline = own_over_l2r trajectory (does the deployment gap close?).
Guards   = tau_to_l2r (revert-to-L2R?) + tau_to_init (policy diffusion?).
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

# Aligned 20k gβ continuation parent (relative to repo root → portable across
# machines; override with --ckpt). The large ckpt itself must be present locally.
PARENT_20K = str(
    ROOT / "block_lo_arm_order_network/probe_results/"
    "gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PARENT_20K)
    ap.add_argument("--n-steps", type=int, default=5000)      # 20k -> 25k
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-held", type=int, default=64)
    ap.add_argument("--lr-orderhead", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.0)         # NO entropy pressure
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None,
                    help="override policy seed (for multi-seed decisive runs)")
    ap.add_argument("--out-root", default="runs/v3_gapclose_pilot_20k25k_lr5e5_noent")
    args = ap.parse_args()
    if args.seed is not None:
        args.out_root = f"{args.out_root}_seed{args.seed}"

    import torch
    from analyses.v3_phase1_runner import _build_evaluator
    from analyses.v3_group_trainer import train_arm
    from analyses.p5_utility_controller import load_p5_ckpt
    from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
    from analyses.p7_gbeta_policy import GBETA_CKPT
    from analyses.v3_own_order_eval import group_policy_orders

    out = pathlib.Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] parent={args.ckpt}", flush=True)
    print(f"[pilot] {args.n_steps} steps, eval every {args.eval_every}, "
          f"lr_orderhead={args.lr_orderhead}, beta={args.beta}, tau={args.tau}, "
          f"device={args.device}", flush=True)

    # Fixed held-out set + shared evaluator; capture 20k init gβ orders (tau_to_init).
    model0, chunks, clean_perm, dev = load_p5_ckpt(args.ckpt, args.n_held,
                                                   device=args.device)
    held_chunks = [chunks[i] for i in range(args.n_held)]
    held_stack = torch.stack(held_chunks)
    wrap0 = AOGPTWithOrderHead(model0, OrderHeadModule(GBETA_CKPT, device=str(dev)),
                               clean_perm, device=str(dev))
    init_orders = group_policy_orders(wrap0, held_stack, m=16, seed=0, device=str(dev))
    evaluator = _build_evaluator(held_chunks, clean_perm, str(dev), m=16,
                                 init_orders=init_orders)
    del model0, wrap0

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    start = int(raw.get("global_step", raw.get("iter_num", 20000)))
    del raw
    eval_steps = tuple(range(args.eval_every, args.n_steps + 1, args.eval_every))

    # frozen control uses batch-mean (m=batch, G=1); joint uses m=16 (G=4).
    arm_m = {"frozen_gbeta": args.batch_size, "joint_group": 16}
    summary = {}
    for arm in ("frozen_gbeta", "joint_group"):
        print(f"\n[pilot] === arm {arm} (m={arm_m[arm]}) ===", flush=True)
        res = train_arm(
            args.ckpt, arm,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            m=arm_m[arm],
            lr_orderhead=args.lr_orderhead,
            tau=args.tau,
            beta=args.beta,
            device=args.device,
            out_dir=str(out),
            tag="pilot",
            eval_steps=eval_steps,
            evaluator=evaluator,
            save_checkpoints=True,
            seed=args.seed,
        )
        summary[arm] = res
        # concise trajectory print
        print(f"[pilot] {arm} own_over_l2r / tau_to_l2r / tau_to_init / entropy:", flush=True)
        for e in res["evals"]:
            g = start + e["step"]
            print(f"   step {g}: own_over_l2r={e['own_over_l2r']:+.4f} "
                  f"tau_to_l2r={e.get('tau_to_l2r', float('nan')):.3f} "
                  f"tau_to_init={e.get('tau_to_init', float('nan')):.3f}", flush=True)

    # ── comparison summary ───────────────────────────────────────────────────
    fz_ev, jg_ev = summary["frozen_gbeta"]["evals"], summary["joint_group"]["evals"]
    report = {
        "ckpt": args.ckpt, "start_step": start, "n_steps": args.n_steps,
        "hyperparams": {"lr_orderhead": args.lr_orderhead, "beta": args.beta,
                        "tau": args.tau, "m": 16, "batch_size": args.batch_size},
        "frozen_own_over_l2r": [e["own_over_l2r"] for e in fz_ev],
        "joint_own_over_l2r": [e["own_over_l2r"] for e in jg_ev],
        "joint_tau_to_l2r": [e.get("tau_to_l2r") for e in jg_ev],
        "joint_tau_to_init": [e.get("tau_to_init") for e in jg_ev],
        "frozen_final_own_over_l2r": fz_ev[-1]["own_over_l2r"],
        "joint_final_own_over_l2r": jg_ev[-1]["own_over_l2r"],
        "gap_closed_vs_frozen": jg_ev[-1]["own_over_l2r"] - fz_ev[-1]["own_over_l2r"],
    }
    (out / "pilot_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[pilot] frozen final own_over_l2r={report['frozen_final_own_over_l2r']:+.4f}"
          f"  joint final own_over_l2r={report['joint_final_own_over_l2r']:+.4f}"
          f"  (joint-frozen={report['gap_closed_vs_frozen']:+.4f})", flush=True)
    print(f"[pilot] wrote {out}/pilot_report.json", flush=True)


if __name__ == "__main__":
    main()
