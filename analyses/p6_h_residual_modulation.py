"""P6 H-residual modulation: z = z_B + lambda * g_H(H^perp).

H^perp = H - H_pos (across-text position profile, estimated on TRAIN). Tests
whether the per-text residual of H (after removing the position copy that is
0.999-redundant with attention) carries an independent, useful modulation of the
B-readout. Pairwise-teacher frame (reuse P5). See spec
docs/superpowers/specs/2026-06-30-p6-h-residual-modulation-design.md.
"""
import json as _json
import pathlib
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau

from analyses.p5_utility_controller import (
    build_dataset, pairwise_loss, order_nll, BOnlyController, train_b_only,
    _split_idx, N,
)


def _corr(a, b, eps=1e-8):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + eps)


def position_profile(train_samples):
    """Per-block across-text mean hidden state (the position/structural copy)."""
    return np.mean([s["H"] for s in train_samples], axis=0).astype(np.float32)  # (N,d)


def h_residual(sample, H_pos):
    return (sample["H"] - H_pos).astype(np.float32)


def _h_variant(H_res_list, mode):
    if mode == "real":
        return H_res_list
    if mode == "zero":
        return [np.zeros_like(h) for h in H_res_list]
    if mode == "shuffle":
        idx = np.roll(np.arange(len(H_res_list)), 1)
        return [H_res_list[j] for j in idx]
    raise ValueError(mode)


class HResidualController(nn.Module):
    """z = g_B(B) + lambda * g_H(H_res). g_B frozen; lambda raw scalar (init 0)."""
    def __init__(self, g_B, h_dim, hidden=64, lambda_init=0.0, use_residual=True):
        super().__init__()
        self.g_B = g_B
        for p in self.g_B.parameters():
            p.requires_grad_(False)
        self.g_H = nn.Sequential(nn.Linear(h_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.lam = nn.Parameter(torch.tensor(float(lambda_init)))
        self.use_residual = use_residual

    def parts(self, B_feat, H):
        zB = self.g_B(torch.as_tensor(B_feat, dtype=torch.float32))
        zH = self.g_H(torch.as_tensor(H, dtype=torch.float32)).squeeze(-1)
        return zB, zH

    def forward(self, B_feat, H):
        zB, zH = self.parts(B_feat, H)
        return zB + self.lam * zH


def train_h_residual(samples, g_B, h_dim, H_list, epochs=300, lr=1e-2,
                     lambda_init=0.0, alpha_corr=0.0, alpha_pos=0.0, alpha_l2=0.0):
    sc = HResidualController(g_B, h_dim, lambda_init=lambda_init)
    phys = torch.arange(N, dtype=torch.float32)
    opt = torch.optim.Adam([p for p in sc.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s, H in zip(samples, H_list):
            zB, zH = sc.parts(s["B_feat"], H)
            z = zB + sc.lam * zH
            l = pairwise_loss(z, s["P"])
            if alpha_corr:
                l = l + alpha_corr * _corr(zH, zB.detach()) ** 2
            if alpha_pos:
                l = l + alpha_pos * _corr(zH, phys) ** 2
            if alpha_l2:
                l = l + alpha_l2 * (zH ** 2).mean()
            loss = loss + l
        (loss / len(samples)).backward(); opt.step()
    return sc


@torch.no_grad()
def _downstream_nll(samples, controller, H_list=None):
    out = []
    for i, s in enumerate(samples):
        if H_list is None:
            z = controller(torch.tensor(s["B_feat"])).cpu().numpy()
        else:
            z = controller(s["B_feat"], H_list[i]).cpu().numpy()
        sigma = np.argsort(-z).astype(np.int64)
        out.append(order_nll(s["model"], s["idx_row"], sigma, s["clean_perm"], s["dev"]))
    return float(np.mean(out))


@torch.no_grad()
def _diag(controller, samples, H_list):
    phys = np.arange(N)
    rr, czb, cph = [], [], []
    for i, s in enumerate(samples):
        zB, zH = controller.parts(s["B_feat"], H_list[i])
        zB, zH = zB.cpu().numpy(), zH.cpu().numpy()
        lam = float(controller.lam)
        rr.append(np.linalg.norm(lam * zH) / (np.linalg.norm(zB) + 1e-9))
        czb.append(float(np.corrcoef(zH, zB)[0, 1]))
        cph.append(float(np.corrcoef(zH, phys)[0, 1]))
    return {"lambda": float(controller.lam), "norm_ratio": float(np.mean(rr)),
            "corr_zH_zB": float(np.nanmean(czb)),
            "corr_zH_phys": float(np.nanmean(cph))}


def run_h_residual(ckpt_path, M=64, layer=1, head=7, T=0.3, epochs=300,
                   split=(0.7, 0.15, 0.15), alpha_corr=1.0, alpha_pos=1.0,
                   out_dir="runs/p6/h_residual", tag="cdl20k"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_dataset(ckpt_path, M, layer=layer, head=head, layers=[layer],
                            heads=None, T=T, h_context="sigma_B")
    tr, va, te = _split_idx(len(samples), split)
    train = [samples[i] for i in tr]
    test = [samples[i] for i in (list(te) or list(tr))]
    H_pos = position_profile(train)
    h_dim = samples[0]["H"].shape[1]

    g_B = train_b_only(train, epochs=epochs)
    for p in g_B.parameters():
        p.requires_grad_(False)
    nll_bonly = _downstream_nll(test, g_B)

    def Hres(group):
        return [h_residual(s, H_pos) for s in group]

    def Hraw(group):
        return [s["H"].astype(np.float32) for s in group]

    arms = {"b_only": {"downstream_nll": nll_bonly, "delta_vs_bonly": 0.0}}
    configs = {
        "b_plus_rawH":     dict(Hfn=Hraw, ac=0.0, ap=0.0),
        "b_plus_Hres":     dict(Hfn=Hres, ac=0.0, ap=0.0),
        "b_plus_Hres_pen": dict(Hfn=Hres, ac=alpha_corr, ap=alpha_pos),
    }
    for name, cfg in configs.items():
        Htr, Hte = cfg["Hfn"](train), cfg["Hfn"](test)
        sc = train_h_residual(train, g_B, h_dim, Htr, epochs=epochs, lambda_init=0.0,
                              alpha_corr=cfg["ac"], alpha_pos=cfg["ap"])
        nll_real = _downstream_nll(test, sc, Hte)
        nll_zero = _downstream_nll(test, sc, _h_variant(Hte, "zero"))
        nll_shuf = _downstream_nll(test, sc, _h_variant(Hte, "shuffle"))
        d = _diag(sc, test, Hte)
        d.update({"downstream_nll": nll_real, "delta_vs_bonly": nll_real - nll_bonly,
                  "real_vs_zero_delta": nll_real - nll_zero,
                  "real_vs_shuffle_delta": nll_real - nll_shuf})
        arms[name] = d

    result = {"ckpt": ckpt_path, "tag": tag, "M": M, "layer": layer, "head": head,
              "nll_bonly": nll_bonly, "arms": arms,
              "_note": "useful H_res => b_plus_Hres_pen delta_vs_bonly<0, real<shuffle, "
                       "corr_zH_zB<<0.935, corr_zH_phys<<0.855, lambda!=0"}
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys
    ck = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "cdl20k"
    print(_json.dumps(run_h_residual(ck, tag=tag), indent=2, default=float))
