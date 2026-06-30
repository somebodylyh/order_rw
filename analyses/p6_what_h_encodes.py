"""P6: what does H-only encode? Position (shared across texts) vs content (per-text).

Decomposes both the raw block-hidden-state H and the trained H-only readout z_H
into a shared-across-text "position/structural" profile and a per-text residual
"content" component, and checks whether the position profile is just the physical
block order and whether z_H is redundant with the attention readout z_B.
"""
import json as _json
import pathlib
import numpy as np
from scipy.stats import kendalltau, pearsonr

from analyses.p6_online_controller import (
    build_dataset_p6, _yl, train_controller_online, controller_scores, h_mode_list,
)
from analyses.p5_utility_controller import N


def _var_decomp(X):
    """X: (M, N[, d]). Split variance into across-text-shared (position) vs per-text
    residual (content). Returns position fraction of total variance."""
    gm = X.mean(axis=0, keepdims=True)                 # (1, N[, d]) per-block mean over texts
    total = float(((X - X.mean()) ** 2).sum())
    pos = float(X.shape[0] * ((gm - X.mean()) ** 2).sum())
    return pos / (total + 1e-12)


def run_what_h_encodes(ckpt_path, M=64, K=6, h_layer=1, layer=1, head=7, tau=0.3,
                       epochs=300, out_dir="runs/p6/cdl_sanity", tag="what_h"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_dataset_p6(ckpt_path, M, layer=layer, head=head, heads=None,
                               K=K, h_layer=h_layer)
    Ys, Ls, _ = _yl(samples)
    phys = np.arange(N)

    # ── raw H: position vs content ───────────────────────────────────────────
    Hstack = np.stack([s["H"] for s in samples])        # (M, N, d)
    rawH_pos_frac = _var_decomp(Hstack)

    # ── trained readouts ─────────────────────────────────────────────────────
    g_B = train_controller_online(samples, Ys, Ls, mode="b_only", tau=tau, epochs=epochs)
    for p in g_B.parameters():
        p.requires_grad_(False)
    h_dim = samples[0]["H"].shape[1]
    h_only = train_controller_online(samples, Ys, Ls, mode="h_only", tau=tau,
                                     epochs=epochs, h_dim=h_dim)
    Hs = h_mode_list([s["H"] for s in samples], "real")

    zH = np.stack([controller_scores(h_only, s["B_feat"], Hs[i]).detach().numpy()
                   for i, s in enumerate(samples)])      # (M, N)
    zB = np.stack([controller_scores(g_B, s["B_feat"]).detach().numpy()
                   for s in samples])                    # (M, N)

    zH_pos_frac = _var_decomp(zH)
    zB_pos_frac = _var_decomp(zB)
    posH, posB = zH.mean(0), zB.mean(0)                  # per-block position profiles

    # is the H position profile just the physical block order?
    tau_posH_phys = float(kendalltau(np.argsort(-posH), phys)[0])
    corr_posH_idx = float(pearsonr(posH, phys)[0])
    # redundancy of z_H with z_B
    red = float(np.mean([pearsonr(zH[t], zB[t])[0] for t in range(M)]))
    corr_posH_posB = float(pearsonr(posH, posB)[0])
    # per-text content: does the residual order vary across texts, or is everyone
    # the same position profile? mean tau of each text's order vs the position order
    posH_order = np.argsort(-posH)
    tau_text_vs_posprofile = float(np.nanmean(
        [kendalltau(np.argsort(-zH[t]), posH_order)[0] for t in range(M)]))

    result = {
        "ckpt": ckpt_path, "tag": tag, "M": M, "layer": layer, "head": head,
        "raw_H": {"position_frac_of_var": rawH_pos_frac,
                  "content_frac_of_var": 1 - rawH_pos_frac},
        "readout_zH": {"position_frac_of_var": zH_pos_frac,
                       "content_frac_of_var": 1 - zH_pos_frac,
                       "pos_profile_tau_vs_phys": tau_posH_phys,
                       "pos_profile_corr_vs_blockindex": corr_posH_idx,
                       "per_text_order_tau_vs_pos_profile": tau_text_vs_posprofile},
        "redundancy": {"zH_vs_zB_per_text_corr": red,
                       "posH_vs_posB_corr": corr_posH_posB},
        "readout_zB_position_frac": zB_pos_frac,
        "_note": "high position_frac + pos_profile aligned to phys + high per_text_"
                 "vs_pos_profile tau => H-only order is a shared position signal, "
                 "not per-text content.",
    }
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys
    ck = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "what_h"
    print(_json.dumps(run_what_h_encodes(ck, tag=tag), indent=2, default=float))
