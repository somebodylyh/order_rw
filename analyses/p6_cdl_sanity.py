"""P6 CDL sanity (frozen, CPU): does the candidate pool have real diversity, and
does adding H make the routing controller pick a DIFFERENT order than CDL?

No dedicated CDL-trained model ckpt exists; the CDL-selected order is the pool's
`sigma_cdl_b` (C-D+L rollout). We run on an existing ckpt and ask:
  (1) pool diversity / headroom — is there room to pick something other than CDL?
  (2) b_only vs B+H selection — does H route to a different candidate/order than
      CDL (and than b_only)?
"""
import json as _json
import pathlib
import numpy as np
from scipy.stats import kendalltau

from analyses.p6_online_controller import (
    build_dataset_p6, _yl, train_controller_online, routing_p, controller_scores,
    h_mode_list,
)
from analyses.p5_utility_controller import priority_matrix


def _selected_label(controller, sample, Y, tau, H=None):
    z = controller_scores(controller, sample["B_feat"], H, detach_h=True)
    p = routing_p(z, Y, tau).detach().cpu().numpy()
    labels, _ = priority_matrix(sample["cands"])
    return labels[int(p.argmax())]


def run_cdl_sanity(ckpt_path, M=64, K=6, h_layer=1, tau=0.3, epochs=300,
                   layer=0, head=1, heads=(1, 2, 3, 4),
                   out_dir="runs/p6/cdl_sanity", tag="seed123_5k"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    heads_arg = list(heads) if heads is not None else None
    samples = build_dataset_p6(ckpt_path, M, layer=layer, head=head, heads=heads_arg,
                               K=K, h_layer=h_layer)
    Ys, Ls, labs = _yl(samples)

    # ── (1) pool diversity / headroom ────────────────────────────────────────
    spreads, headrooms, best_labels, eff_K = [], [], [], []
    for s, labels in zip(samples, labs):
        nlls = np.array([s["nll_by_label"][l] for l in labels])
        spreads.append(float(nlls.max() - nlls.min()))
        cdl = s["nll_by_label"]["sigma_cdl_b"]
        headrooms.append(float(cdl - nlls.min()))           # CDL above pool-best
        best_labels.append(labels[int(nlls.argmin())])
        eff_K.append(len(labels))                            # after dedup
    best_dist = {l: best_labels.count(l) / len(best_labels) for l in set(best_labels)}

    # ── train b_only + B+H routing controllers (frozen model) ────────────────
    g_B = train_controller_online(samples, Ys, Ls, mode="b_only", tau=tau, epochs=epochs)
    for p in g_B.parameters():
        p.requires_grad_(False)
    h_dim = samples[0]["H"].shape[1]
    sc = train_controller_online(samples, Ys, Ls, mode="real", g_B=g_B, tau=tau,
                                 epochs=epochs, h_dim=h_dim)
    Hs = h_mode_list([s["H"] for s in samples], "real")

    # ── (2) selection comparison: CDL vs b_only vs B+H ───────────────────────
    diff_bh_vs_cdl = diff_bonly_vs_cdl = diff_bh_vs_bonly = 0
    tau_bh_vs_cdl, tau_bonly_vs_cdl = [], []
    sel_bonly, sel_bh = [], []
    for i, s in enumerate(samples):
        cdl_lab = "sigma_cdl_b"
        lab_b = _selected_label(g_B, s, Ys[i], tau)
        lab_h = _selected_label(sc, s, Ys[i], tau, H=Hs[i])
        sel_bonly.append(lab_b); sel_bh.append(lab_h)
        diff_bonly_vs_cdl += (lab_b != cdl_lab)
        diff_bh_vs_cdl += (lab_h != cdl_lab)
        diff_bh_vs_bonly += (lab_h != lab_b)
        cdl_order = np.asarray(s["cands"][cdl_lab])
        tau_bh_vs_cdl.append(kendalltau(s["cands"][lab_h], cdl_order)[0])
        tau_bonly_vs_cdl.append(kendalltau(s["cands"][lab_b], cdl_order)[0])

    n = len(samples)
    result = {
        "ckpt": ckpt_path, "tag": tag, "M": n, "K_requested": K,
        "pool": {"eff_K_mean": float(np.mean(eff_K)),
                 "nll_spread_mean": float(np.mean(spreads)),
                 "cdl_headroom_mean": float(np.mean(headrooms)),
                 "best_candidate_dist": best_dist},
        "selection": {
            "frac_bonly_differs_from_cdl": diff_bonly_vs_cdl / n,
            "frac_bh_differs_from_cdl": diff_bh_vs_cdl / n,
            "frac_bh_differs_from_bonly": diff_bh_vs_bonly / n,
            "kendalltau_bh_vs_cdl_mean": float(np.nanmean(tau_bh_vs_cdl)),
            "kendalltau_bonly_vs_cdl_mean": float(np.nanmean(tau_bonly_vs_cdl)),
            "bonly_selection_dist": {l: sel_bonly.count(l) / n for l in set(sel_bonly)},
            "bh_selection_dist": {l: sel_bh.count(l) / n for l in set(sel_bh)}},
    }
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/handoff_overnight/seed123/ckpt_step5000.pt"
    tag = sys.argv[2] if len(sys.argv) > 2 else "seed123_5k"
    r = run_cdl_sanity(ck, tag=tag)
    print(_json.dumps(r, indent=2, default=float))
