"""Order eval ladder: score a frozen checkpoint under many orders, no training.

Answers: what order does a random-trained model already prefer?
Is g_beta order better than cheap orders? Is it just L2R?

Usage:
    python analyses/order_eval_ladder.py \
        --ckpt block_lo_arm_order_network/probe_results/random_baseline_continuous_jun05/ckpt_step10000.pt \
        --device cuda:0 --n-eval 500
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from train_clean_aogpt import (
    build_model, clean_model_args, clean_state_dict,
    physical_blocks_to_model_blocks, model_blocks_to_physical_blocks,
    order_loss,
)
from clean_training_protocol import CleanPermutation, SEQ_LEN, N, BLOCK_LEN
from training_utils import expand_model_blocks_to_token_order


def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = ckpt.get("model_args", {})
    model = build_model(model_args, device, compile_model=True)
    sd = ckpt.get("model", ckpt.get("model_state_dict"))
    model.load_state_dict(clean_state_dict(sd))
    model.eval()
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.from_numpy(np.load(
            pathlib.Path(ckpt_path).parent / "block_perm.npy"
        )).long(),
        inv_perm_model_to_phys=torch.from_numpy(np.load(
            pathlib.Path(ckpt_path).parent / "inv_perm.npy"
        )).long(),
        block_len=BLOCK_LEN,
    )
    return model, clean_perm, model_args


@torch.no_grad()
def eval_order(model, idx_batch, physical_order, clean_perm, device):
    """Return average loss over batch for one physical order."""
    B = idx_batch.shape[0]
    phys = physical_order.to(device)
    if phys.ndim == 1:
        phys = phys.unsqueeze(0).expand(B, -1)
    loss = order_loss(model, idx_batch, phys, clean_perm, device)
    return float(loss.item())


def build_orders(clean_perm, device):
    """Build a dict of named physical orders."""
    orders = {}

    # ori-L2R: physical order = identity (model's physical layout)
    orders["ori_l2r"] = torch.arange(N, dtype=torch.long, device=device)

    # reverse-L2R
    orders["reverse_l2r"] = torch.arange(N - 1, -1, -1, dtype=torch.long, device=device)

    # shuffled-L2R: model_ascending mapped to physical
    model_ascending = torch.arange(N, dtype=torch.long, device=device)
    orders["shuffled_l2r"] = model_blocks_to_physical_blocks(model_ascending, clean_perm)

    # random orders (3 seeds)
    for seed in [0, 42, 123]:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        orders[f"random_s{seed}"] = torch.randperm(N, generator=g)

    return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-eval", type=int, default=500,
                    help="number of eval windows (clipped to available)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.ckpt} ...")
    model, clean_perm, model_args = load_ckpt(args.ckpt, device)

    # Load val data (continuous stream or chunk pool)
    ckpt_dir = pathlib.Path(args.ckpt).parent
    config = json.load(open(ckpt_dir / "config.json"))
    data_source = config["args"].get("data_source", "chunks")

    if data_source == "continuous":
        from train_clean_aogpt import load_token_stream, sample_stream_batch
        val_bin = config["args"]["val_bin"]
        stream = load_token_stream(val_bin)
        permute_seed = config["args"].get("permute_seed", 42)
        phys = sample_stream_batch(stream, min(args.n_eval, 2000), SEQ_LEN,
                                   seed=permute_seed, step=-1, micro=0)
        # Gather to model coords
        from train_clean_aogpt import build_phys_to_model_token_gather
        g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)
        idx_eval = phys[:, g_gather].contiguous().to(device)
    else:
        from train_clean_aogpt import load_train_chunks, phys_to_model_idx_clean
        idx_phys = load_train_chunks(n_chunks=None)
        idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
        eval_indices = np.load(ckpt_dir / "eval_indices.npy")
        n_use = min(args.n_eval, len(eval_indices))
        idx_eval = idx_model[eval_indices[:n_use]].to(device)

    n_use = min(args.n_eval, idx_eval.shape[0])
    idx_eval = idx_eval[:n_use]
    print(f"Eval windows: {n_use}")

    orders = build_orders(clean_perm, device)

    # --- g_beta order (if available) ---
    gbeta_ckpt = PKG / "batch_readout/logs/phase33_gbeta/random_baseline_continuous_jun05/full/g_beta_best.pt"
    if gbeta_ckpt.exists():
        from batch_readout.integration_hook import FrozenBetaHook
        hook = FrozenBetaHook(str(gbeta_ckpt), mode="argsort", device=str(device))
        # Extract attention for g_beta
        from per_head_order_scan import _attn_to_A_block_b0_vec
        head = (0, 4)  # L0H4
        model_uncompiled = model  # already compiled; use as-is
        inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

        # Use a small batch for attention extraction
        probe_bs = min(16, n_use)
        probe_idx = idx_eval[:probe_bs]

        # Random probe orders
        probe_orders_list = []
        for b in range(probe_bs):
            g = torch.Generator(device="cpu")
            g.manual_seed(42 * 1000 + b)
            blocks = torch.randperm(N, generator=g)
            probe_orders_list.append(expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0])
        probe_orders = torch.stack(probe_orders_list).to(device)

        _, _, attn_list = model.forward_fn(probe_idx, probe_orders, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_batch = torch.stack(attn_list).cpu().numpy()

        A_batch = np.zeros((probe_bs, N, N), dtype=np.float32)
        probe_np = probe_orders.cpu().numpy()
        for bi in range(probe_bs):
            A_batch[bi] = _attn_to_A_block_b0_vec(attn_batch[head[0], bi, head[1]], probe_np[bi], inv_perm)
        A_t = torch.from_numpy(A_batch).float()
        gbeta_phys = hook.step(A_t.to(device)).cpu()
        orders["g_beta"] = gbeta_phys
        print(f"g_beta order loaded: {gbeta_phys[:16].tolist()}...")
    else:
        print(f"g_beta ckpt not found: {gbeta_ckpt}")

    # --- Eval loop ---
    results = {}
    for name, phys in orders.items():
        t0 = time.time()
        loss = eval_order(model, idx_eval, phys, clean_perm, device)
        elapsed = time.time() - t0
        results[name] = {"loss": loss, "time_s": elapsed}
        print(f"  {name:20s}  loss={loss:.4f}  ({elapsed:.1f}s)")

    # --- Output ---
    print(f"\n=== Order ranking (lower loss = better) ===")
    for name, r in sorted(results.items(), key=lambda x: x[1]["loss"]):
        print(f"  {name:20s}  {r['loss']:.4f}")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"ckpt": str(args.ckpt), "results": results}, open(out, "w"), indent=2)
        print(f"\nSaved: {out}")

    # Sanity: compute tau of each order vs L2R
    print(f"\n=== τ vs ori-L2R ===")
    l2r_phys = orders["ori_l2r"].cpu().numpy()
    from scipy.stats import kendalltau
    for name, phys in orders.items():
        p = phys.cpu().numpy() if isinstance(phys, torch.Tensor) else phys
        if p.ndim == 2:
            p = p[0]
        tau, _ = kendalltau(p, l2r_phys)
        print(f"  {name:20s}  τ={tau:+.4f}")


if __name__ == "__main__":
    main()
