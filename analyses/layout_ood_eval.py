"""Existing-ckpt unseen-permutation OOD evaluation (P4-lite pre-step).

Without retraining, lay the evaluation data out under unseen block permutations and
ask whether the existing fixed-layout ckpt still recovers the *new* physical order
(content/context recovery) or instead keeps emitting the *training* slot->physical
map (a fixed lookup).

Key diagnostic (same forward pass, two inv_perm labelings of the SAME attention):
  tau_test_physical = score(sigma_model under inv_test) vs physical L2R
  tau_train_map     = score(sigma_model under inv_train) vs physical L2R
A lookup signature is tau_test_physical low while tau_train_map stays high: the model
keeps producing the training-layout order regardless of the new content/layout.

See physical_signal_source.relayout_diagnostic (P2 single-number relayout) which this
generalises with the train-map diagnostic and a multi-perm sweep. Paired with
docs/superpowers/specs P4-lite K-layout training (the decisive content test).
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLOCK = ROOT / "block_lo_arm_order_network"
for _p in (str(ROOT), str(_BLOCK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from analyses.canonical_reanalysis import random_reveal_orders  # noqa: E402
from analyses.position_prior_decomp import (  # noqa: E402
    clean_perm_from_layout, make_layouts, relayout_chunks)
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec  # noqa: E402
from none_separated_block_graph import (  # noqa: E402
    build_none_separated_B, rollout_by_method, discovery_metrics)


def _classify_ood(test_physical_mean, train_map_mean, high=0.6, low=0.3):
    """Interpret the unseen-layout sweep means (absolute tau)."""
    if train_map_mean >= high and test_physical_mean < low:
        return "lookup"
    if test_physical_mean >= high and train_map_mean < low:
        return "content_recovery"
    if test_physical_mean < low and train_map_mean < low:
        return "degraded"
    return "ambiguous"


@torch.no_grad()
def _two_inv_A_means(model, chunks, inv_a, inv_b, reveals, dev, M):
    """Accumulate the loss-aligned A under two inv_perm labelings of the SAME attention.

    Returns (A_mean_a, A_mean_b), each (L, H, 64, 65)."""
    acc_a = acc_b = None
    for t in range(M):
        for reveal in reveals:
            po = torch.from_numpy(np.asarray(reveal)[None, :]).to(dev)
            _, _, attn_list = model.forward_fn(
                chunks[t:t + 1].to(dev), po, return_attentions=True)
            attn = torch.stack(attn_list, dim=0).cpu().numpy()[:, 0]   # (L,H,257,257)
            Aa = _attn_to_A_block_loss_aligned_with_none_vec(attn, reveal, inv_a)
            Ab = _attn_to_A_block_loss_aligned_with_none_vec(attn, reveal, inv_b)
            acc_a = Aa.astype(np.float64) if acc_a is None else acc_a + Aa
            acc_b = Ab.astype(np.float64) if acc_b is None else acc_b + Ab
    n = M * len(reveals)
    return acc_a / n, acc_b / n


def _head_tau(A_mean, layer, head):
    B65 = build_none_separated_B(A_mean[layer, head])
    return float(discovery_metrics(rollout_by_method(B65, "C-D+L"))["tau_vs_l2r"])


def layout_ood_sweep(ckpt_path, layer, carrier_heads, K=8, M=8, n_reveals=8,
                     fixed_reveal_seed=0, device="cpu"):
    """Sweep K unseen permutations on an existing ckpt, with the train-map diagnostic.

    K is the number of UNSEEN test perms (the anchor/training layout is handled
    separately). The carrier head is fixed to the anchor's strongest |tau| head so
    test-physical vs train-map are scored on the same head (no head-switching
    confound). Returns anchor_tau, per-perm (tau_test_physical, tau_train_map),
    sweep means/std over |tau|, and an interpretation label.
    """
    carrier_heads = [int(h) for h in carrier_heads]
    if not carrier_heads:
        raise ValueError("carrier_heads must not be empty")
    if K < 1:
        raise ValueError("K (number of unseen perms) must be >= 1")

    model, chunks, train_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)
    inv_train = train_perm.inv_perm_model_to_phys.cpu().numpy()

    # Anchor: training layout, training inv. Pick the dominant carrier head.
    A_anchor, _ = _two_inv_A_means(model, chunks, inv_train, inv_train, reveals, dev, M)
    head_taus = {h: _head_tau(A_anchor, layer, h) for h in carrier_heads}
    best_head = max(head_taus, key=lambda h: abs(head_taus[h]))
    anchor_tau = head_taus[best_head]

    layouts = make_layouts(train_perm, K=K + 1)   # layout 0 = anchor, 1..K = unseen
    per_perm = []
    for layout in layouts[1:]:
        layout_perm = clean_perm_from_layout(layout)
        inv_test = layout_perm.inv_perm_model_to_phys.cpu().numpy()
        relaid = relayout_chunks(chunks, train_perm, layout_perm)
        A_test, A_trainmap = _two_inv_A_means(
            model, relaid, inv_test, inv_train, reveals, dev, M)
        per_perm.append({
            "perm_id": int(layout["layout_id"]) if "layout_id" in layout else len(per_perm) + 1,
            "tau_test_physical": _head_tau(A_test, layer, best_head),
            "tau_train_map": _head_tau(A_trainmap, layer, best_head),
        })

    tp = np.abs([r["tau_test_physical"] for r in per_perm])
    tm = np.abs([r["tau_train_map"] for r in per_perm])
    test_physical_mean, train_map_mean = float(tp.mean()), float(tm.mean())
    return {
        "ckpt": ckpt_path,
        "layer": layer,
        "best_head": best_head,
        "anchor_tau": anchor_tau,
        "per_perm": per_perm,
        "test_physical_mean": test_physical_mean,
        "test_physical_std": float(tp.std()),
        "train_map_mean": train_map_mean,
        "train_map_std": float(tm.std()),
        "interpretation": _classify_ood(test_physical_mean, train_map_mean),
    }
