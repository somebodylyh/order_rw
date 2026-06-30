"""P6 B2 proper-scale run for ONE seed on a given GPU.

Stable testbed config (lr=1e-5, bs=8) found via lr sweep: b_only continuation
holds/improves val so H comparison is meaningful. Arms b_only/real/shuffle/zero.
Still a single-seed datum per call; orchestrate seeds 123/2/42 across GPUs.
"""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=str, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--M-val", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-reveals", type=int, default=4)
    ap.add_argument("--h-layer", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--refresh-every", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import sys
    sys.path.insert(0, os.getcwd())
    from analyses.p6_online_controller import run_b2_smoke

    ckpt = f"runs/handoff_overnight/seed{args.seed}/ckpt_step5000.pt"
    r = run_b2_smoke(
        ckpt, arms=("b_only", "real", "shuffle", "zero"),
        M_val=args.M_val, batch=args.batch, steps=args.steps,
        n_reveals=args.n_reveals, K=args.K, h_layer=args.h_layer, tau=args.tau,
        lr=args.lr, eval_every=args.eval_every, refresh_every=args.refresh_every,
        detach_h=True, device="cuda", seed=args.seed,
        out_dir="runs/p6/b2_full", tag=f"seed{args.seed}")
    for mode, res in r["arms"].items():
        h = res["hist"]
        print(f"[seed{args.seed} {mode}] val_fixed "
              f"{[round(x,3) for x in h['val_fixed_nll']]}", flush=True)
    print(f"SEED {args.seed} DONE", flush=True)


if __name__ == "__main__":
    main()
