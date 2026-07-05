"""Decisive frozen room-verification: group-oracle headroom curve over m, and the
per-group best orders (BC warm-start material for PG). One pass, dumps JSON+NPZ.

Answers: does a SINGLE shared reveal order retain room to beat L2R at m=16 (the
main co-adapt arm) or does cross-sample conflict collapse it toward 0?
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for p in (ROOT, BLOCK_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyses.p5_utility_controller import load_p5_ckpt  # noqa: E402
from analyses.v3_group_headroom import group_hill_climb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/handoff_overnight/seed123/ckpt_step10000.pt")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--m-list", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/v3_room_verify")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, chunks, clean_perm, dev = load_p5_ckpt(args.ckpt, args.M, device=args.device)
    held = torch.stack([chunks[i] for i in range(args.M)])

    curve = {}
    orders = {}  # m -> (n_groups, N) best shared orders (for BC)
    for m in args.m_list:
        assert args.M % m == 0, f"M={args.M} not divisible by m={m}"
        n_groups = args.M // m
        best_orders, oracle_vals, l2r_vals = [], [], []
        for g in range(n_groups):
            rows = held[g * m:(g + 1) * m]
            best, best_nll, l2r_nll = group_hill_climb(
                model, rows, clean_perm, dev, n_steps=args.n_steps, seed=args.seed + g
            )
            best_orders.append(best)
            oracle_vals.append(best_nll)
            l2r_vals.append(l2r_nll)
        oracle_val = float(np.mean(oracle_vals))
        l2r_val = float(np.mean(l2r_vals))
        curve[m] = {
            "headroom": l2r_val - oracle_val,
            "group_oracle_val": oracle_val,
            "l2r_val": l2r_val,
            "n_groups": int(n_groups),
        }
        orders[str(m)] = np.stack(best_orders)
        print(f"[room] m={m:2d}: L2R={l2r_val:.4f} oracle={oracle_val:.4f} "
              f"headroom={l2r_val - oracle_val:+.4f} (G={n_groups})", flush=True)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"ckpt": args.ckpt, "M": args.M, "m_list": args.m_list,
            "n_steps": args.n_steps, "seed": args.seed, "curve": curve}
    (out_dir / "room_curve.json").write_text(json.dumps(meta, indent=2))
    np.savez(out_dir / "group_oracle_orders.npz", **orders)
    print(f"[room] wrote {out_dir}/room_curve.json + group_oracle_orders.npz", flush=True)


if __name__ == "__main__":
    main()
