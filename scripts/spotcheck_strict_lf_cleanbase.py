#!/usr/bin/env python3
"""Quick strict label-free spot-check on clean_base key checkpoints.

Confirms that the oracle-remapped vs strict-label-free equivalence
(proven for collaborator ckpt) also holds for the clean_base ladder.

Lightweight: M=8, 3 methods (L, C-D+L, none_edge), 5 control seeds.
Target steps: 10000, 60000 (earliest strong_pass emergence + latest).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts"))

from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from none_separated_block_graph import (  # noqa: E402
    build_none_separated_B,
    classify_gate_status,
    combined_discovery_score,
    content_label_permutation_control,
    discovery_metrics,
    entry_shuffled_control,
    rollout_by_method,
)
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec  # noqa: E402
from training_utils import AOGPT, AOGPTConfig, BLOCK_LEN, N, SEQ_LEN, load_train_chunks  # noqa: E402


METHODS = ("L", "C-D+L", "none_edge")


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in model_args.items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    model.crop_block_size(SEQ_LEN)
    sd = dict(ckpt["model"])
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model.to(dev).eval()
    return ckpt, model, dev


def _get_clean_perm(ckpt):
    dp = ckpt["data_permutation"]
    bp = torch.tensor(dp["block_perm"], dtype=torch.long)
    ip = torch.tensor(dp["inverse_block_perm"], dtype=torch.long)
    # clean_base uses phys_to_model convention
    return ip.cpu().numpy()  # inv_perm_model_to_phys


def _posthoc_translate_sigma(sigma_model: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
    return inv_perm[np.asarray(sigma_model, dtype=np.int64)]


def _control_abs_tau_mean(B65: np.ndarray, method: str, inv_perm: np.ndarray, seeds: list[int]) -> float:
    vals = []
    for seed in seeds:
        for Bc in (
            entry_shuffled_control(B65, seed=seed),
            content_label_permutation_control(B65, seed=seed),
        ):
            sigma_model = rollout_by_method(Bc, method)
            sigma_phys = _posthoc_translate_sigma(sigma_model, inv_perm)
            vals.append(abs(discovery_metrics(sigma_phys)["tau_vs_l2r"]))
    return float(np.mean(vals)) if vals else float("nan")


def scan_one_ckpt(ckpt_path: str, out_dir: pathlib.Path, args):
    ckpt, model, dev = _load_model(ckpt_path, args.device)
    inv_perm = _get_clean_perm(ckpt)
    iter_num = ckpt.get("iter_num", "?")

    # Build physical→model chunks (same as _physical_chunks_to_model)
    # Use inv_perm (model→phys) as gather: position m gets physical block inv_perm[m]
    inv_tensor = torch.from_numpy(inv_perm).long()
    token_gather = expand_model_blocks_to_token_order(inv_tensor.view(1, -1), BLOCK_LEN)[0]
    total = int(args.M) * int(args.batch_size)
    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = chunks_phys[:, token_gather]

    # Random reveal orders
    token_orders = torch.empty((total, SEQ_LEN), dtype=torch.long)
    for i in range(total):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(args.seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_orders[i] = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN)[0]

    sums = None
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, total, max(1, int(args.fwd_batch))):
            stop = min(start + max(1, int(args.fwd_batch)), total)
            tokens = chunks_model[start:stop].to(dev)
            orders = token_orders[start:stop].to(dev)
            _, _, attn_list = model.forward_fn(tokens, orders, return_attentions=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            attn_batch = torch.stack(attn_list).cpu().numpy()
            for bi in range(stop - start):
                sample = attn_batch[:, bi]
                A_lh = _attn_to_A_block_loss_aligned_with_none_model_vec(
                    sample, token_orders[start + bi].numpy(),
                )  # (L,H,N,N+1) model-frame
                if sums is None:
                    sums = np.zeros_like(A_lh, dtype=np.float64)
                sums += A_lh
                count += 1
            print(f"  [extract step={iter_num}] {stop}/{total}", flush=True)
    A_mean = (sums / count).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "A_with_none_lh_mean_MODEL_FRAME.npy", A_mean)
    np.save(out_dir / "inv_perm.npy", inv_perm)
    meta = {
        "ckpt": ckpt_path,
        "iter_num": iter_num,
        "protocol": "strict_label_free_65_spotcheck",
        "M": args.M,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "shape": list(A_mean.shape),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    L, H = A_mean.shape[:2]
    rows = []
    for layer in range(L):
        for head in range(H):
            B65_model = build_none_separated_B(A_mean[layer, head])
            for method in METHODS:
                sigma_model = rollout_by_method(B65_model, method)
                sigma_phys = _posthoc_translate_sigma(sigma_model, inv_perm)
                metrics = discovery_metrics(sigma_phys)
                destroyed = _control_abs_tau_mean(B65_model, method, inv_perm, args.control_seeds)
                gate = classify_gate_status(metrics, destroyed)
                score = combined_discovery_score(metrics, destroyed)
                rows.append({
                    "layer": layer, "head": head, "method": method,
                    "first_block": metrics["first_block"],
                    "first_is_phys0": metrics["first_is_phys0"],
                    "phys0_rank": metrics["phys0_rank"],
                    "tau_vs_l2r": metrics["tau_vs_l2r"],
                    "prefix4_overlap": metrics["prefix4_overlap"],
                    "prefix8_overlap": metrics["prefix8_overlap"],
                    "abs_tau": abs(float(metrics["tau_vs_l2r"])),
                    "destroyed_abs_tau_mean": destroyed,
                    "destroyed_tau_gap": float(metrics["tau_vs_l2r"]) - destroyed,
                    "combined_score": score,
                    "gate_status": gate,
                })

    with open(out_dir / "all_head_methods_strict_label_free.json", "w") as f:
        json.dump(rows, f, indent=2)

    strong = [r for r in rows if r["gate_status"] == "strong_pass"]
    weak = [r for r in rows if r["gate_status"] == "weak_pass"]
    print(f"step={iter_num}: strong={len(strong)} weak={len(weak)} fail={len(rows)-len(strong)-len(weak)}")
    for r in strong:
        print(f"  L{r['layer']}H{r['head']} {r['method']} tau={r['tau_vs_l2r']:.4f} first={r['first_block']} p4={r['prefix4_overlap']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-base", default="block_lo_arm_order_network/probe_results/clean_base_random_perm")
    p.add_argument("--steps", type=int, nargs="*", default=[10000, 60000])
    p.add_argument("--out-base", default="reports/strict_65node_discovery_ckpt_verification_20260617/spotcheck")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fwd-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    for step in args.steps:
        ckpt_path = f"{args.ckpt_base}/ckpt_step{step}.pt"
        out_dir = pathlib.Path(args.out_base) / f"step{step}"
        print(f"\n=== Strict LF spot-check: step {step} ===")
        scan_one_ckpt(ckpt_path, out_dir, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
