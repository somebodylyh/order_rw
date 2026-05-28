"""BR-1 Task 9: offline training of g_beta on a (B_batch, sigma_T, rank) dataset.

Selection is by pairwise precedence accuracy on validation. AR-NLL is NEVER
imported here — see tests/test_batch_readout_selection_policy.py which fails
if a forbidden symbol appears in the source.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from batch_readout.dataset_batch import load_dataset
from batch_readout.eval_metrics import (
    first_k_overlap, kendall_tau_batch, pairwise_acc, spearman_rho_batch, top1_acc,
)
from batch_readout.loss import pairwise_logistic_loss, plackett_luce_nll
from batch_readout.model import FlattenReadout, NodewiseReadout
from batch_readout.pl_sampling import pl_argsort


def _build_model(name: str, N: int):
    if name == "flatten":
        return FlattenReadout(N=N, hidden=(1024, 256))
    if name == "nodewise":
        return NodewiseReadout(N=N, d_model=64, n_layers=2, n_heads=4)
    raise ValueError(f"unknown model {name!r}")


def _loss_fn(name: str):
    if name == "pairwise":
        return lambda z, rank, sigma: pairwise_logistic_loss(z, rank)
    if name == "pl":
        return lambda z, rank, sigma: plackett_luce_nll(z, sigma, tau=1.0)
    if name == "pairwise+pl":
        return lambda z, rank, sigma: (
            pairwise_logistic_loss(z, rank) + 0.5 * plackett_luce_nll(z, sigma, tau=1.0)
        )
    raise ValueError(f"unknown loss {name!r}")


@torch.no_grad()
def _eval_split(model, B, sigma_T, rank, device):
    """Compute matching metrics on a (M, N, N) split. NO NLL anywhere."""
    model.eval()
    Bt = torch.from_numpy(B).float().to(device)
    z = model(Bt).cpu()
    sigma_pred = pl_argsort(z).numpy()
    return {
        "kendall_tau": kendall_tau_batch(sigma_pred, sigma_T),
        "pairwise_acc": pairwise_acc(z, torch.from_numpy(rank)),
        "spearman_rho": spearman_rho_batch(sigma_pred, sigma_T),
        "top1": top1_acc(torch.from_numpy(sigma_pred), torch.from_numpy(sigma_T)),
        "first3": first_k_overlap(torch.from_numpy(sigma_pred), torch.from_numpy(sigma_T), k=3),
    }


def train(
    dataset_path: str,
    model_name: str,
    loss_name: str,
    N: int = 64,
    batch_size: int = 64,
    lr: float = 3e-4,
    epochs: int = 40,
    seed: int = 0,
    out_dir=None,
    device: str = "cuda:0",
):
    """Train g_beta on the given dataset; select by val pairwise_acc.

    Returns dict {history, best_epoch, best_metrics, final_train_loss}.
    If out_dir is given, saves g_beta_best.pt (best-by-val-pairwise) and history.json.
    """
    torch.manual_seed(seed)
    ds = load_dataset(dataset_path)
    Btr = ds["train_B_batch"].astype(np.float32)
    sigT = ds["train_sigma_T"].astype(np.int64)
    rk = ds["train_rank"].astype(np.int64)

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = _build_model(model_name, N).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = _loss_fn(loss_name)

    out_dir_p = pathlib.Path(out_dir) if out_dir else None
    if out_dir_p is not None:
        out_dir_p.mkdir(parents=True, exist_ok=True)

    best = {"pairwise_acc": -1.0, "epoch": -1, "metrics": None}
    history = []
    rng = np.random.default_rng(seed)
    n_train = len(Btr)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            Bb = torch.from_numpy(Btr[idx]).float().to(device)
            rb = torch.from_numpy(rk[idx]).to(device)
            sb = torch.from_numpy(sigT[idx]).to(device)
            z = model(Bb)
            loss = loss_fn(z, rb, sb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        val_metrics = _eval_split(
            model, ds["val_B_batch"].astype(np.float32),
            ds["val_sigma_T"].astype(np.int64), ds["val_rank"].astype(np.int64),
            device=device,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["pairwise_acc"] > best["pairwise_acc"]:
            best = {"pairwise_acc": val_metrics["pairwise_acc"], "epoch": epoch, "metrics": val_metrics}
            if out_dir_p is not None:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {"model_name": model_name, "loss_name": loss_name, "N": N},
                        "epoch": epoch,
                        "metrics": val_metrics,
                    },
                    out_dir_p / "g_beta_best.pt",
                )

    if out_dir_p is not None:
        (out_dir_p / "history.json").write_text(json.dumps({"history": history, "best": best}, indent=2))

    return {
        "history": history,
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "final_train_loss": history[-1]["train_loss"],
    }
