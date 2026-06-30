"""P6 B2 smoke launcher: wait for a free GPU, then run the joint co-adaptation
smoke (seed123 5k -> +steps; arms b_only / real / shuffle).

Validates the joint chain on GPU (entropy / memory / loss / fixed-order val NLL),
NOT a science verdict at this step count. See
docs/superpowers/specs/2026-06-30-p6-online-coadaptive-controller-design.md.
"""
import argparse, os, subprocess, sys, time


def pick_gpu(min_free_mib):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    best, best_free = None, -1
    for line in out:
        idx, free = [x.strip() for x in line.split(",")]
        free = int(free)
        if free >= min_free_mib and free > best_free:
            best, best_free = idx, free
    return best, best_free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/handoff_overnight/seed123/ckpt_step5000.pt")
    ap.add_argument("--min-free-mib", type=int, default=18000)
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-reveals", type=int, default=4)
    ap.add_argument("--M-train", type=int, default=24)
    ap.add_argument("--M-val", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--refresh-every", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--h-layer", type=int, default=1)
    args = ap.parse_args()

    print(f"[launcher] waiting for a GPU with >= {args.min_free_mib} MiB free ...",
          flush=True)
    while True:
        gpu, free = pick_gpu(args.min_free_mib)
        if gpu is not None:
            print(f"[launcher] GPU {gpu} free={free} MiB -> launching", flush=True)
            break
        time.sleep(args.poll_sec)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sys.path.insert(0, os.getcwd())
    from analyses.p6_online_controller import run_b2_smoke

    batch = args.batch
    while batch >= 1:
        try:
            r = run_b2_smoke(
                args.ckpt, arms=("b_only", "real", "shuffle"),
                M_train=args.M_train, M_val=args.M_val, batch=batch,
                steps=args.steps, n_reveals=args.n_reveals, K=args.K,
                h_layer=args.h_layer, tau=args.tau, lr=args.lr,
                eval_every=args.eval_every, refresh_every=args.refresh_every,
                detach_h=True, device="cuda",
                out_dir="runs/p6/seed123/b2_smoke", tag=f"b2_smoke_b{batch}")
            for mode, res in r["arms"].items():
                h = res["hist"]
                print(f"[{mode}] loss {[round(x,3) for x in h['loss']]} "
                      f"ent {[round(x,2) for x in h['entropy']]} "
                      f"val_fixed {[round(x,3) for x in h['val_fixed_nll']]}", flush=True)
            print("B2 SMOKE DONE", flush=True)
            return
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch > 1:
                import torch
                torch.cuda.empty_cache()
                batch //= 2
                print(f"[launcher] OOM -> retry batch={batch}", flush=True)
            else:
                raise


if __name__ == "__main__":
    main()
