"""BR-1 Task 13: post-selection AR-NLL diagnostic over 4 arms.

Arms = {random, teacher(sigma_T), mlp_argsort, mlp_sample}. The reported
numbers are observational only — they must NOT be fed back into model
selection. The corresponding train-side guard lives in
tests/test_batch_readout_selection_policy.py (which forbids this module
being imported by batch_readout.train_offline).

Reuses the AR-NLL math from neural_readout.eval_frozen_nll but feeds in
BR-1 orders: each batch-mean graph has *one* sigma per arm, replicated
across that graph's constituent chunks.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

from batch_readout.dataset_batch import load_dataset
from batch_readout.model import FlattenReadout, NodewiseReadout
from batch_readout.pl_sampling import pl_argsort, pl_sample


def _load_g_beta(path, N=64):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    name = cfg["model_name"]
    if name == "flatten":
        model = FlattenReadout(N=N, hidden=(1024, 256))
    elif name == "nodewise":
        model = NodewiseReadout(N=N, d_model=64, n_layers=2, n_heads=4)
    else:
        raise ValueError(f"unknown model {name!r}")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def compute_arms(
    ckpt_path,
    g_beta_path,
    dataset_path,
    arms=("random", "teacher", "mlp_argsort", "mlp_sample"),
    tau=0.3,
    seed=0,
    device="cuda:0",
):
    """Return {arm: {"mean_nll": float, "delta_vs_random": float}} on val split.

    Each val graph contributes ``batch_size`` AR-NLL evaluations under the
    *single* sigma chosen by that arm for that graph; we average across all
    (graph, batch-member) pairs.
    """
    ds = load_dataset(dataset_path)
    val_chunks = ds["val_chunks"]           # (M_val, batch_size)
    sigma_T = ds["val_sigma_T"]             # (M_val, N)
    B_batch = ds["val_B_batch"]             # (M_val, N, N)
    M_val, Bsz = val_chunks.shape
    N = B_batch.shape[-1]

    z = None
    if any(a.startswith("mlp_") for a in arms):
        model = _load_g_beta(g_beta_path, N=N).to(device)
        with torch.no_grad():
            z = model(torch.from_numpy(B_batch).float().to(device)).cpu()

    chunk_indices_flat = val_chunks.reshape(-1).astype(np.int64)

    orders_per_arm = {}
    rng = np.random.default_rng(seed)
    if "random" in arms:
        rand_orders = np.stack([rng.permutation(N) for _ in range(M_val)])
        orders_per_arm["random"] = np.repeat(rand_orders, Bsz, axis=0)
    if "teacher" in arms:
        orders_per_arm["teacher"] = np.repeat(sigma_T, Bsz, axis=0)
    if "mlp_argsort" in arms:
        orders_per_arm["mlp_argsort"] = np.repeat(pl_argsort(z).numpy(), Bsz, axis=0)
    if "mlp_sample" in arms:
        g = torch.Generator().manual_seed(seed)
        orders_per_arm["mlp_sample"] = np.repeat(
            pl_sample(z, tau=tau, generator=g).numpy(), Bsz, axis=0
        )

    results = {}
    for arm, orders in orders_per_arm.items():
        mean_nll = _ar_nll_under_orders(ckpt_path, chunk_indices_flat, orders, device)
        results[arm] = {"mean_nll": float(mean_nll)}

    if "random" in results:
        baseline = results["random"]["mean_nll"]
        for arm, r in results.items():
            r["delta_vs_random"] = r["mean_nll"] - baseline

    return results


def _ar_nll_under_orders(ckpt_path, chunk_indices, orders_phys, device):
    """Mirror of neural_readout.eval_frozen_nll._mean_nll_under_orders but
    accepting an explicit per-row sigma array. Token bookkeeping (wikitext
    -> physical -> model index, then per-split slice) follows NR-1 exactly
    so the numbers compare apples-to-apples."""
    _R = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_R / "block_lo_arm_order_network"))
    from train_clean_aogpt import build_model, CleanPermutation, phys_to_model_idx_clean
    from training_utils import load_train_chunks, SEQ_LEN, BLOCK_LEN
    from clean_training_protocol import physical_blocks_to_model_token_order

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(
            protocol["block_perm_phys_to_model"], dtype=torch.long
        ),
        inv_perm_model_to_phys=torch.tensor(
            protocol["inv_perm_model_to_phys"], dtype=torch.long
        ),
    )
    dev = torch.device(
        device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    )
    model = build_model(model_args, dev, compile_model=False)
    sd = ckpt.get("model") or ckpt.get("model_state_dict")
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(dev)
    model.eval()

    idx_phys = load_train_chunks(n_chunks=None)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    split_indices = np.asarray(protocol["train_indices"], dtype=np.int64)
    idx_split = idx_model[split_indices]
    chunks = idx_split[np.asarray(chunk_indices, dtype=np.int64)]

    losses = []
    with torch.no_grad():
        for i in range(len(orders_phys)):
            o = torch.from_numpy(orders_phys[i:i + 1].astype(np.int64))
            tok_order = physical_blocks_to_model_token_order(
                o, clean_perm, BLOCK_LEN
            ).to(dev)
            idx = chunks[i:i + 1].to(dev)
            _, loss = model.forward_fn(idx, tok_order)
            losses.append(float(loss.item()))
    return float(np.mean(losses))
