#!/usr/bin/env python3
"""Fast B1 predictor-aligned per-head scan ladder.

This is equivalent to running run_per_head_order_scan_ladder_b1.sh, but samples
the train chunks once per seed and reuses them across all checkpoints. Output
JSONs are written to block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys

import torch

REPO = pathlib.Path("/home/admin/lyuyuhuan/order_lyu")
PKG = REPO / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from per_head_order_scan import scan_loaded  # noqa: E402
from train_clean_aogpt import build_model  # noqa: E402
from training_utils import SEQ_LEN  # noqa: E402


DEFAULT_STEPS = (0, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000)


def load_model_only(ckpt_path: pathlib.Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN
    dev = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = build_model(model_args, dev, compile_model=False)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"{ckpt_path} has neither 'model' nor 'model_state_dict'")
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(dev)
    model.eval()
    return model, dev


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, nargs="*", default=list(DEFAULT_STEPS))
    p.add_argument("--split", default="train")
    p.add_argument("--alpha-dep", type=float, default=0.5)
    p.add_argument("--ckpt-dir", default=str(PKG / "probe_results/clean_base_random_perm"))
    p.add_argument("--out-dir", default=str(PKG / "batch_readout/logs/per_head_scan_b1"))
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = pathlib.Path(args.ckpt_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(step, seed) for step in args.steps for seed in args.seeds]
    print(
        f"[b1-fast] DEVICE={args.device} M={args.M} BATCH_SIZE={args.batch_size} "
        f"SEEDS={args.seeds} STEPS={args.steps}",
        flush=True,
    )
    print(f"[b1-fast] {len(jobs)} jobs -> {out_dir}", flush=True)

    total = int(args.M) * int(args.batch_size)
    done_count = 0
    first_step = int(args.steps[0])
    first_ckpt = ckpt_dir / f"ckpt_step{first_step}.pt"
    if not first_ckpt.exists():
        raise FileNotFoundError(first_ckpt)

    for seed in args.seeds:
        print(f"[b1-fast] seed={seed}: loading chunks once ({total} samples)", flush=True)
        first_model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
            str(first_ckpt), total, int(seed), args.device, args.split
        )

        for step in args.steps:
            done_count += 1
            ckpt_path = ckpt_dir / f"ckpt_step{step}.pt"
            if not ckpt_path.exists():
                print(f"[b1-fast] ({done_count}/{len(jobs)}) missing {ckpt_path} - skip", flush=True)
                continue

            out = out_dir / f"ckpt{step}_seed{seed}.json"
            if out.exists() and out.stat().st_size > 0:
                print(f"[b1-fast] ({done_count}/{len(jobs)}) skip existing {out}", flush=True)
                continue

            if step == first_step:
                model = first_model
                model_dev = dev
            else:
                model, model_dev = load_model_only(ckpt_path, args.device)

            print(f"[b1-fast] ({done_count}/{len(jobs)}) scan step={step} seed={seed} -> {out}", flush=True)
            res = scan_loaded(
                model, chunks, clean_perm, model_dev, str(ckpt_path),
                int(args.M), int(args.batch_size), int(seed),
                alpha_dep=float(args.alpha_dep), none_mode="predictor",
            )
            with out.open("w") as f:
                json.dump(res, f, indent=1)
            top = res["per_head_layer_sorted_by_abs_tau_vs_l2r"][0]
            print(
                f"[b1-fast] saved step={step} seed={seed} "
                f"heavy_tau={res['heavy_baseline']['tau_vs_l2r']:.4f} "
                f"top=L{top['layer']}H{top['head']} tau={top['tau_vs_l2r']:.4f}",
                flush=True,
            )

            if step != first_step:
                del model
                gc.collect()
                if torch.cuda.is_available() and str(model_dev).startswith("cuda"):
                    torch.cuda.empty_cache()

        del first_model, chunks
        gc.collect()
        if torch.cuda.is_available() and str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"[b1-fast] DONE: {len(list(out_dir.glob('ckpt*_seed*.json')))} JSON files in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
