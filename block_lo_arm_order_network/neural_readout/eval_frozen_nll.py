"""NR-1 Task 11: frozen-theta NLL diagnostic (spec §5.2 / §8 step 5).

NEVER imported by neural_readout/train_nr1.py — that property is enforced by
tests/test_neural_readout_selection_policy.py. This script is run AFTER the
g_beta checkpoint is selected by attention-order matching metrics.

Reports NLL(sigma_hat) - NLL(sigma_T) over the held-out val batch. The
number is observational, not selective. Do not use it to choose epochs,
learning rates, dropout, or any other hyperparameter (spec §1).
"""
import sys
import pathlib
import json
import argparse

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from train_clean_aogpt import build_model, CleanPermutation, phys_to_model_idx_clean
from training_utils import load_train_chunks, SEQ_LEN, BLOCK_LEN
from clean_training_protocol import physical_blocks_to_model_token_order
from neural_readout.dataset import load_dataset
from neural_readout.graph_transformer_readout import GraphTransformerReadout


@torch.no_grad()
def _mean_nll_under_orders(model, tokens_batch, orders_phys, clean_perm, device):
    """For each row i, compute mean per-token NLL when revealing tokens_batch[i]
    in the physical-block order orders_phys[i]. Returns mean over rows.

    Single forward pass per (chunk, order). No batching across orders because
    forward_fn expects a single order shape per call; we loop for clarity.
    """
    losses = []
    M = orders_phys.shape[0]
    for i in range(M):
        order_phys_t = torch.from_numpy(orders_phys[i:i + 1].astype(np.int64))
        token_order = physical_blocks_to_model_token_order(
            order_phys_t, clean_perm, BLOCK_LEN
        ).to(device)
        idx = tokens_batch[i:i + 1].to(device)
        _, loss = model.forward_fn(idx, token_order)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def compute_frozen_nll_gap(ckpt_path, g_beta_path, dataset_path, device="cuda:0"):
    """Compute NLL(sigma_hat) - NLL(sigma_T) under frozen-theta. Diagnostic only."""
    if not pathlib.Path(g_beta_path).exists():
        raise FileNotFoundError(
            f"g_beta checkpoint not found at {g_beta_path}; this is a "
            "post-selection diagnostic — train g_beta and select by §5.1 "
            "matching metrics first."
        )

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

    dev = torch.device(device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    model = build_model(model_args, dev, compile_model=False)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(dev)
    model.eval()

    B, sigma_T, _rank, chunk_index, split_name = load_dataset(dataset_path)

    g = GraphTransformerReadout(N=B.shape[1]).to(dev)
    g_state = torch.load(g_beta_path, map_location=dev, weights_only=False)
    g.load_state_dict(g_state["model"])
    g.eval()

    with torch.no_grad():
        scores = g(torch.from_numpy(B).to(dev)).cpu().numpy()
    sigma_hat = np.argsort(-scores, axis=-1).astype(np.int64)

    idx_phys = load_train_chunks(n_chunks=None)
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    split_indices = np.asarray(protocol[f"{split_name}_indices"], dtype=np.int64)
    idx_split = idx_model[split_indices]
    chunks = idx_split[chunk_index]

    nll_T = _mean_nll_under_orders(model, chunks, sigma_T, clean_perm, dev)
    nll_hat = _mean_nll_under_orders(model, chunks, sigma_hat, clean_perm, dev)

    return {
        "nll_teacher": nll_T,
        "nll_student": nll_hat,
        "nll_gap": nll_hat - nll_T,
        "n_samples": int(B.shape[0]),
        "ckpt_path": str(ckpt_path),
        "g_beta_path": str(g_beta_path),
        "dataset_path": str(dataset_path),
        "split": split_name,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="AOGPT checkpoint (frozen theta)")
    p.add_argument("--g-beta", required=True, help="trained g_beta checkpoint")
    p.add_argument("--dataset", required=True, help="dataset .npz that produced sigma_T")
    p.add_argument("--out", required=True, help="where to save the JSON report")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    res = compute_frozen_nll_gap(args.ckpt, args.g_beta, args.dataset, args.device)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
