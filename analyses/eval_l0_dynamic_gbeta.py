#!/usr/bin/env python3
"""Label-free evaluation sanity checks for pretrained L0DynamicGBeta.

Evaluates a trained g_beta checkpoint under four conditions:
  1. normal       — raw B as extracted
  2. destroyed    — structure-preserving destroy of B
  3. remove_top   — mask the top-alpha head, renormalise
  4. uniform      — force α = 1/H, no gate

Outputs: summary.json, per_sample.tsv, alpha_summary.tsv.

Physical coordinates (inv_perm, L2R, block_perm) are never loaded,
used, or reported.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = _ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta  # noqa: E402
from batch_readout.order_distillation_metrics import order_metrics  # noqa: E402
from batch_readout.soft_pairwise import (  # noqa: E402
    entropy_floor_loss,
    gate_entropy,
    pairwise_accuracy,
    per_head_aux_loss,
    soft_pairwise_bce_loss,
)
from batch_readout.train_l0_dynamic_gbeta import (  # noqa: E402
    compute_primary_loss,
    load_pretrain_dataset,
)
from batch_readout.label_free_cdl_teacher import destroy_strict65  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p_mask(device="cpu"):
    return torch.triu(torch.ones(64, 64, dtype=torch.bool, device=device), diagonal=1)


def _evaluate_scores(
    scores,
    scores_per_head,
    alpha,
    Y_pair,
    teacher_order=None,
    loss_type="pairwise_bce",
    rank_kl_temperature=4.0,
    lambda_aux=0.05,
    lambda_ent=0.001,
    min_gate_entropy=1.5,
):
    if teacher_order is None:
        p_mask = _p_mask(device=scores.device)
        primary = soft_pairwise_bce_loss(scores, Y_pair, pair_mask=p_mask)
        auxiliary = per_head_aux_loss(scores_per_head, Y_pair)
        soft_acc = pairwise_accuracy(scores, Y_pair)
        metrics = {"pairwise_acc": soft_acc}
    else:
        primary, auxiliary, soft_acc = compute_primary_loss(
            scores,
            scores_per_head,
            teacher_order,
            Y_pair,
            loss_type,
            rank_kl_temperature,
        )
        metrics = order_metrics(scores, teacher_order)

    entropy_loss = entropy_floor_loss(alpha, min_entropy=min_gate_entropy)
    total = primary + lambda_aux * auxiliary + lambda_ent * entropy_loss
    metrics.update({
        "loss": float(total.item()),
        "loss_final": float(primary.item()),
        "primary_loss": float(primary.item()),
        "loss_aux": float(auxiliary.item()),
        "loss_ent": float(entropy_loss.item()),
        "gate_entropy_mean": float(gate_entropy(alpha).mean().item()),
        "alpha_mean": alpha.mean(dim=0).cpu().numpy().tolist(),
    })
    if soft_acc is not None:
        metrics["soft_pairwise_acc"] = float(soft_acc)
    return metrics


@torch.no_grad()
def _eval_pass(
    model,
    B_raw,
    Y_pair,
    teacher_order=None,
    loss_type="pairwise_bce",
    rank_kl_temperature=4.0,
    lambda_aux=0.05,
    lambda_ent=0.001,
    min_gate_entropy=1.5,
):
    """Single evaluation pass: returns dict of scalar metrics."""
    model.eval()
    scores, aux = model(B_raw, apply_head_dropout=False)
    return _evaluate_scores(
        scores,
        aux["scores_per_head"],
        aux["alpha"],
        Y_pair,
        teacher_order=teacher_order,
        loss_type=loss_type,
        rank_kl_temperature=rank_kl_temperature,
        lambda_aux=lambda_aux,
        lambda_ent=lambda_ent,
        min_gate_entropy=min_gate_entropy,
    )


# ---------------------------------------------------------------------------
# Sanity: remove top-alpha head
# ---------------------------------------------------------------------------

def remove_top_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Mask the head with the highest α per sample, renormalise.

    Args:
        alpha: (B, H).

    Returns:
        alpha_masked: (B, H), zero at top-α head, renormalised to sum=1.
    """
    top_idx = alpha.argmax(dim=1)  # (B,)
    alpha = alpha.clone()
    alpha[torch.arange(alpha.shape[0]), top_idx] = 0.0
    return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-9)


@torch.no_grad()
def eval_remove_top_alpha(
    model,
    B_raw,
    Y_pair,
    teacher_order=None,
    loss_type="pairwise_bce",
    rank_kl_temperature=4.0,
):
    """Evaluate with the top-α head removed and weights renormalised."""
    model.eval()
    _, aux = model(B_raw, apply_head_dropout=False)
    scores_h = aux["scores_per_head"]  # (B, H, 64)

    alpha_masked = remove_top_alpha(aux["alpha"])  # (B, H)
    scores_masked = (scores_h * alpha_masked[:, :, None]).sum(dim=1)

    return _evaluate_scores(
        scores_masked,
        scores_h,
        alpha_masked,
        Y_pair,
        teacher_order=teacher_order,
        loss_type=loss_type,
        rank_kl_temperature=rank_kl_temperature,
    )


# ---------------------------------------------------------------------------
# Sanity: uniform-alpha baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_uniform_alpha(
    model,
    B_raw,
    Y_pair,
    H=8,
    teacher_order=None,
    loss_type="pairwise_bce",
    rank_kl_temperature=4.0,
):
    """Evaluate with uniform α = 1/H (no gate, equal head weight)."""
    model.eval()
    _, aux = model(B_raw, apply_head_dropout=False)
    scores_h = aux["scores_per_head"]  # (B, H, 64)

    alpha_uniform = torch.full((B_raw.shape[0], H), 1.0 / H, device=B_raw.device)
    scores_uniform = (scores_h * alpha_uniform[:, :, None]).sum(dim=1)

    return _evaluate_scores(
        scores_uniform,
        scores_h,
        alpha_uniform,
        Y_pair,
        teacher_order=teacher_order,
        loss_type=loss_type,
        rank_kl_temperature=rank_kl_temperature,
    )


# ---------------------------------------------------------------------------
# Sanity: destroyed input
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_destroyed(
    model,
    B_raw,
    Y_pair,
    seed=0,
    teacher_order=None,
    loss_type="pairwise_bce",
    rank_kl_temperature=4.0,
):
    """Evaluate on structure-preserving destroyed B."""
    B_np = B_raw.cpu().numpy()
    B_destroyed = np.zeros_like(B_np)
    for b in range(B_np.shape[0]):
        for h in range(B_np.shape[1]):
            rng = np.random.default_rng(seed * 1000 + b * 100 + h)
            B_destroyed[b, h] = destroy_strict65(B_np[b, h], rng)

    B_d = torch.from_numpy(B_destroyed).float().to(B_raw.device)
    return _eval_pass(
        model,
        B_d,
        Y_pair,
        teacher_order=teacher_order,
        loss_type=loss_type,
        rank_kl_temperature=rank_kl_temperature,
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    dataset_path: str,
    ckpt_path: str,
    device: str = "cuda:0",
    destroy_seed: int = 0,
    out_dir: Optional[str] = None,
) -> dict:
    """Run all evaluation sanity checks on a trained checkpoint.

    Args:
        dataset_path: path to Task-3 .npz.
        ckpt_path: path to g_beta_best.pt.
        device: torch device.
        destroy_seed: seed for destroy RNG.
        out_dir: directory for output files (default: ckpt directory).

    Returns:
        dict of all evaluation results.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    # ── Load checkpoint ──
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    assert cfg.get("selection_label_free", False), (
        "Checkpoint was not selected with label-free criteria — refusing eval."
    )
    loss_type = cfg.get("loss_type", "pairwise_bce")
    rank_kl_temperature = float(cfg.get("rank_kl_temperature", 4.0))

    # ── Load dataset ──
    data = load_pretrain_dataset(
        dataset_path,
        device=str(dev),
        require_consensus_order=True,
        require_pairwise=loss_type == "pairwise_bce",
    )
    B_raw = data["B_raw"]
    Y_pair = data["teacher_pairwise"]
    if Y_pair is None:
        Y_pair = torch.empty(
            (B_raw.shape[0], 0, 0), dtype=B_raw.dtype, device=dev,
        )
    teacher_order = data["teacher_consensus_order"]
    wT = data["teacher_weights"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    H = B_raw.shape[1]
    print(f"Dataset: {B_raw.shape[0]} samples, {H} heads")
    print(f"  val: {len(val_idx)}, test: {len(test_idx)}")

    model = L0DynamicGBeta(
        heads=H, nodes=65,
        scorer_hidden=tuple(cfg.get("scorer_hidden", (256, 64))),
        gate_hidden=cfg.get("gate_hidden", 32),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch', '?')}")

    # ── Output directory ──
    if out_dir is None:
        out_dir = str(pathlib.Path(ckpt_path).parent)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Evaluate on val and test ──
    results = {}
    for split_name, split_idx in (("val", val_idx), ("test", test_idx)):
        Bs = B_raw[split_idx]
        Ys = Y_pair[split_idx]
        Os = teacher_order[split_idx]
        Ws = wT[split_idx]

        # 1. Normal
        normal = _eval_pass(
            model, Bs, Ys, Os, loss_type, rank_kl_temperature,
        )

        # 2. Destroyed
        destroyed = eval_destroyed(
            model, Bs, Ys, seed=destroy_seed,
            teacher_order=Os,
            loss_type=loss_type,
            rank_kl_temperature=rank_kl_temperature,
        )

        # 3. Remove top-α
        remove_top = eval_remove_top_alpha(
            model, Bs, Ys,
            teacher_order=Os,
            loss_type=loss_type,
            rank_kl_temperature=rank_kl_temperature,
        )

        # 4. Uniform α
        uniform = eval_uniform_alpha(
            model, Bs, Ys, H=H,
            teacher_order=Os,
            loss_type=loss_type,
            rank_kl_temperature=rank_kl_temperature,
        )

        results[split_name] = {
            "normal": normal,
            "destroyed": destroyed,
            "remove_top_alpha": remove_top,
            "uniform_alpha": uniform,
        }

        # Per-sample: alpha and teacher weight summaries
        _, aux = model(Bs, apply_head_dropout=False)
        alpha_np = aux["alpha"].detach().cpu().numpy()  # (N, H)
        wT_np = Ws.cpu().numpy()

        per_sample_rows = []
        for i in range(alpha_np.shape[0]):
            per_sample_rows.append({
                "sample": int(split_idx[i]),
                "split": split_name,
                "alpha": alpha_np[i].tolist(),
                "alpha_entropy": float(gate_entropy(aux["alpha"][i:i+1]).item()),
                "teacher_weight": wT_np[i].tolist(),
            })

        # Write per-sample and alpha summary
        with open(out / f"per_sample_{split_name}.jsonl", "w") as f:
            for row in per_sample_rows:
                f.write(json.dumps(row) + "\n")

        alpha_mean = alpha_np.mean(axis=0)
        alpha_std = alpha_np.std(axis=0)
        wT_mean = wT_np.mean(axis=0)
        alpha_summary = {
            "split": split_name,
            "alpha_mean_per_head": alpha_mean.tolist(),
            "alpha_std_per_head": alpha_std.tolist(),
            "teacher_weight_mean_per_head": wT_mean.tolist(),
        }
        with open(out / f"alpha_summary_{split_name}.json", "w") as f:
            json.dump(alpha_summary, f, indent=2)

    # ── Summary ──
    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "loss_type": loss_type,
        "rank_kl_temperature": rank_kl_temperature,
        "label_free": True,
        "results": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print key numbers ──
    for split_name in ("val", "test"):
        r = results[split_name]
        print(f"\n{split_name}:")
        print(f"  normal:        acc={r['normal']['pairwise_acc']:.4f}  "
              f"loss={r['normal']['primary_loss']:.4f}  "
              f"H_α={r['normal']['gate_entropy_mean']:.2f}")
        print(f"  destroyed:     acc={r['destroyed']['pairwise_acc']:.4f}  "
              f"loss={r['destroyed']['primary_loss']:.4f}  "
              f"(Δacc={r['normal']['pairwise_acc'] - r['destroyed']['pairwise_acc']:.4f})")
        print(f"  remove_top_α:  acc={r['remove_top_alpha']['pairwise_acc']:.4f}")
        print(f"  uniform_α:     acc={r['uniform_alpha']['pairwise_acc']:.4f}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Label-free evaluation sanity checks for L0DynamicGBeta"
    )
    p.add_argument("--dataset", required=True, help="Path to Task-3 .npz")
    p.add_argument("--ckpt", required=True, help="Path to g_beta_best.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--destroy-seed", type=int, default=0)
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: ckpt directory)")
    args = p.parse_args()

    evaluate_checkpoint(
        dataset_path=args.dataset,
        ckpt_path=args.ckpt,
        device=args.device,
        destroy_seed=args.destroy_seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
