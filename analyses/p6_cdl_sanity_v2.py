"""P6 CDL sanity v2 (frozen, CPU): batch-mean CDL reference + H-only arm.

Fixes from v1 feedback:
  - per-sample CDL order is noisy / has no clear order -> use BATCH-MEAN B
    (project consensus readout) for the CDL reference order.
  - add an H-only controller arm (score from hidden state alone, no attention B).

Questions:
  (1) how noisy is per-sample CDL vs the batch-mean consensus?
  (2) does H-only produce an order DIFFERENT from the batch-mean CDL order
      (and from B-only)? free-argsort(z) order is compared by Kendall tau.
"""
import json as _json
import pathlib
import numpy as np
from scipy.stats import kendalltau

from analyses.p6_online_controller import (
    build_dataset_p6, _yl, train_controller_online, controller_scores, h_mode_list,
)
from analyses.p5_utility_controller import sigma_from_B65, N


def _free_order(controller, sample, H=None):
    z = controller_scores(controller, sample["B_feat"], H, detach_h=True)
    return np.argsort(-z.detach().cpu().numpy()).astype(np.int64)


def _mean_tau(orders_a, orders_b):
    return float(np.nanmean([kendalltau(a, b)[0] for a, b in zip(orders_a, orders_b)]))


def run_cdl_sanity_v2(ckpt_path, M=64, K=6, h_layer=1, tau=0.3, epochs=300,
                      layer=1, head=7, out_dir="runs/p6/cdl_sanity", tag="v2"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_dataset_p6(ckpt_path, M, layer=layer, head=head, heads=None,
                               K=K, h_layer=h_layer)
    Ys, Ls, labs = _yl(samples)
    phys = np.arange(N, dtype=np.int64)

    # ── batch-mean CDL reference order (consensus) ───────────────────────────
    B_mean = np.mean([s["B65"] for s in samples], axis=0)
    sigma_cdl_mean = sigma_from_B65(B_mean)
    persample_cdl = [np.asarray(s["sigma_B"]) for s in samples]
    # how noisy is per-sample CDL vs the consensus?
    consensus_tau = _mean_tau(persample_cdl, [sigma_cdl_mean] * M)

    # ── train controllers: b_only, real (B+H), h_only ────────────────────────
    g_B = train_controller_online(samples, Ys, Ls, mode="b_only", tau=tau, epochs=epochs)
    for p in g_B.parameters():
        p.requires_grad_(False)
    h_dim = samples[0]["H"].shape[1]
    sc = train_controller_online(samples, Ys, Ls, mode="real", g_B=g_B, tau=tau,
                                 epochs=epochs, h_dim=h_dim)
    h_only = train_controller_online(samples, Ys, Ls, mode="h_only", tau=tau,
                                     epochs=epochs, h_dim=h_dim)
    Hs = h_mode_list([s["H"] for s in samples], "real")

    # ── free-argsort orders per arm ──────────────────────────────────────────
    ord_b = [_free_order(g_B, s) for s in samples]
    ord_bh = [_free_order(sc, s, Hs[i]) for i, s in enumerate(samples)]
    ord_h = [_free_order(h_only, s, Hs[i]) for i, s in enumerate(samples)]

    def vs(orders):
        return {"tau_vs_cdl_mean": _mean_tau(orders, [sigma_cdl_mean] * M),
                "tau_vs_phys": _mean_tau(orders, [phys] * M)}

    # cross-arm agreement (do the produced orders differ?)
    result = {
        "ckpt": ckpt_path, "tag": tag, "M": M, "layer": layer, "head": head,
        "cdl_consensus": {
            "persample_vs_batchmean_tau": consensus_tau,
            "_note": "low = per-sample CDL is noisy; batch-mean is the stable order"},
        "free_order_vs_reference": {
            "b_only": vs(ord_b), "b_plus_h": vs(ord_bh), "h_only": vs(ord_h)},
        "cross_arm_order_tau": {
            "h_only_vs_b_only": _mean_tau(ord_h, ord_b),
            "b_plus_h_vs_b_only": _mean_tau(ord_bh, ord_b),
            "h_only_vs_b_plus_h": _mean_tau(ord_h, ord_bh)},
    }
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys
    ck = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "v2"
    print(_json.dumps(run_cdl_sanity_v2(ck, tag=tag), indent=2, default=float))
