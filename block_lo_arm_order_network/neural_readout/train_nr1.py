"""NR-1 Task 10: training loop for g_beta (spec §8).

Selection policy (spec §8, also enforced by
tests/test_neural_readout_selection_policy.py):
  - Checkpoints and hyperparameters are selected ONLY by attention-order
    matching metrics on same-ckpt held-out val (Kendall tau, pairwise
    precedence acc, Spearman rho).
  - Frozen-theta NLL is NEVER imported here. Any developer that adds an
    `import neural_readout.eval_frozen_nll` to this file breaks the
    selection-policy test on purpose.

NR-1 supervision boundary: this script never reads NLL, never reads an
L2R / raster / oracle label. The only training signal is the pairwise
logistic loss against the teacher rank derived from attention.
"""
import sys
import pathlib
import json
import argparse

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "block_lo_arm_order_network"))

from neural_readout.dataset import load_dataset
from neural_readout.graph_transformer_readout import GraphTransformerReadout
from neural_readout.loss import pairwise_logistic_loss
from neural_readout.eval_metrics import compute_matching_metrics
from neural_readout.ablation_inputs import apply_input_transform


@torch.no_grad()
def evaluate(model, B, rank, device, batch=64):
    """Compute §5.1 + §5.2 metrics on a (B, rank) batch with the current model."""
    model.eval()
    M = B.shape[0]
    all_s = np.zeros((M, B.shape[1]), dtype=np.float32)
    for i in range(0, M, batch):
        Bt = torch.from_numpy(B[i:i + batch]).to(device)
        all_s[i:i + batch] = model(Bt).cpu().numpy()
    return compute_matching_metrics(scores=all_s, rank=rank)


def train_nr1(
    dataset_path,
    out_dir,
    train_n,
    val_n,
    epochs,
    batch,
    lr,
    device="cuda:0",
    d_model=64,
    n_layers=2,
    n_heads=4,
    eval_every_epochs=1,
    ablation="identity",
    weight_decay=0.01,
    seed=0,
):
    """Train g_beta with pairwise logistic loss; select by Kendall tau on val.

    Args:
        dataset_path: path to .npz produced by neural_readout.dataset.
        out_dir: directory to write g_beta_best.pt and train_log.json.
        train_n, val_n: sequential split sizes (rest = test, ignored here).
        epochs, batch, lr, weight_decay, seed: standard training knobs.
        device, d_model, n_layers, n_heads: model + device hyperparams.
        eval_every_epochs: how often to evaluate + possibly snapshot.
        ablation: input transform name (identity / reverse / sym / row_shuffle / b_global).
                  Teacher labels are NEVER re-derived from the transformed B; the ablation
                  studies whether the student can still recover the ORIGINAL teacher
                  order when its input is degraded.

    Returns:
        (best_ckpt_path, log) where log is a list of per-eval metric dicts.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    B, sigma, rank, chunk_index, split_name = load_dataset(dataset_path)
    del sigma, chunk_index  # not used during training; only rank drives the loss

    # Input ablation. Teacher rank labels are NEVER re-derived from the transformed B.
    ab_rng = np.random.default_rng(seed + 13)
    B = apply_input_transform(B, ablation, ab_rng)

    # Sequential split (dataset is already deterministically ordered from build_dataset).
    B_tr, r_tr = B[:train_n], rank[:train_n]
    B_va, r_va = B[train_n:train_n + val_n], rank[train_n:train_n + val_n]

    dev = torch.device(device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    model = GraphTransformerReadout(N=B.shape[1], d_model=d_model,
                                    n_heads=n_heads, n_layers=n_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    M_tr = B_tr.shape[0]
    log = []
    best_tau = -float("inf")
    best_path = out / "g_beta_best.pt"

    for ep in range(epochs):
        model.train()
        ep_rng = np.random.default_rng(seed + ep)
        perm = ep_rng.permutation(M_tr)
        ep_loss = 0.0
        n_steps = 0
        for s_i in range(0, M_tr, batch):
            idx = perm[s_i:s_i + batch]
            Bt = torch.from_numpy(B_tr[idx]).to(dev)
            rt = torch.from_numpy(r_tr[idx]).to(dev).long()
            scores = model(Bt)
            loss = pairwise_logistic_loss(scores, rt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_steps += 1
        ep_loss /= max(n_steps, 1)

        if (ep + 1) % eval_every_epochs == 0 or ep == epochs - 1:
            metrics = evaluate(model, B_va, r_va, dev)
            entry = {"epoch": int(ep), "train_loss": float(ep_loss), **metrics}
            log.append(entry)
            print(
                f"[ep {ep:3d}] loss={ep_loss:.4f}  "
                f"tau={metrics['kendall_tau']:.4f}  "
                f"pairwise={metrics['pairwise_precedence_acc']:.4f}  "
                f"spearman={metrics['spearman_rho']:.4f}  "
                f"top1={metrics['top1_first_node_match']:.3f}  "
                f"first3={metrics['first3_set_match']:.3f}",
                flush=True,
            )
            # Selection policy: Kendall tau on val ONLY (NR-1 spec §8)
            if metrics["kendall_tau"] > best_tau:
                best_tau = metrics["kendall_tau"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "metrics": metrics,
                        "epoch": int(ep),
                        "ablation": ablation,
                        "dataset_path": str(dataset_path),
                        "split": split_name,
                    },
                    best_path,
                )

    with open(out / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return str(best_path), log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-n", type=int, required=True)
    p.add_argument("--val-n", type=int, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ablation",
        default="identity",
        choices=["identity", "reverse", "sym", "row_shuffle", "b_global"],
    )
    args = p.parse_args()
    best, _log = train_nr1(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        train_n=args.train_n,
        val_n=args.val_n,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ablation=args.ablation,
        seed=args.seed,
    )
    print(f"best ckpt: {best}")


if __name__ == "__main__":
    main()
