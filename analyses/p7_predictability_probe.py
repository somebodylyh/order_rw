"""P7 Step 1 — predictability probe: which features predict the hill-climb best
reveal order sigma* (the genuine non-myopic context-dependent order from Stage 0)?

NOT training the final controller — just asking whether F(x) -> sigma* is learnable.
ALL features and sigma* are in PHYSICAL-block frame (sigma* from hill_climb is
physical; B is physical; H from block_hidden_states is MODEL-frame -> remapped;
content/one-step built directly in physical frame).

Static pairwise ranker per feature set; held-out tau-to-oracle + downstream NLL.
"""
import json as _json
import pathlib
import numpy as np
import torch
from scipy.stats import kendalltau

from analyses.p5_utility_controller import (
    load_p5_ckpt, order_nll, block_b_features, block_hidden_states,
    BOnlyController, pairwise_loss, _split_idx, sample_scaffold, N, BLOCK_LEN,
)
from analyses.p6_oracle_headroom import hill_climb
from analyses.p6_easy_first_control import per_position_nll, _tok_orders, dynamic_greedy


def _pref_matrix(sigma):
    """P[i,j]=1 if block i is revealed before j in order sigma."""
    rank = np.empty(N, dtype=np.int64); rank[np.asarray(sigma)] = np.arange(N)
    return (rank[:, None] < rank[None, :]).astype(np.float32)


def block_content_emb(model, idx_row, p2m, dev):
    """Per PHYSICAL block: mean token embedding (pure content, no forward needed)."""
    wte = model.transformer.wte
    toks = idx_row[0].to(dev)
    emb = wte(toks).detach().cpu().numpy()                 # (256, d) model-pos order
    out = np.zeros((N, emb.shape[1]), dtype=np.float32)
    for b in range(N):
        m = int(p2m[b])
        out[b] = emb[m * BLOCK_LEN:(m + 1) * BLOCK_LEN].mean(0)
    return out


def onestep_difficulty(model, idx_row, clean_perm, dev):
    """Per PHYSICAL block b: NLL of revealing b FIRST (d0). (N,1)."""
    cand = [[b] + [x for x in range(N) if x != b] for b in range(N)]
    tok = _tok_orders(np.array(cand, dtype=np.int64), clean_perm, dev)
    pp = per_position_nll(model, idx_row.expand(N, -1), tok, dev)
    return pp[:, :BLOCK_LEN].sum(1, keepdims=True).astype(np.float32)


def build_probe_data(ckpt_path, M, hc_steps=500, n_reveals=8, layer=1, head=7,
                     device="cpu"):
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    p2m = clean_perm.block_perm_phys_to_model.cpu().numpy()    # phys block -> model block
    sc = sample_scaffold(ckpt_path, M, layer=layer, head=head, n_reveals=n_reveals,
                         device=device)                        # B65 physical frame
    l2r = np.arange(N, dtype=np.int64)
    samples = []
    for t in range(M):
        idx = chunks[t:t+1]
        sigma_star, nll_best, nll_l2r = hill_climb(model, idx, clean_perm, dev,
                                                   n_steps=hc_steps, seed=t)
        sig_de = dynamic_greedy(model, idx, clean_perm, dev, "easy")
        nll_de = order_nll(model, idx, sig_de, clean_perm, dev)
        B65 = sc["B"][t]
        H_model = block_hidden_states(model, idx, l2r, clean_perm, layer, dev)  # model frame
        H_phys = H_model[p2m]                                  # -> physical frame
        feats = {
            "B": block_b_features(B65).astype(np.float32),     # (N,130) physical
            "H": H_phys.astype(np.float32),
            "content": block_content_emb(model, idx, p2m, dev),
            "onestep": onestep_difficulty(model, idx, clean_perm, dev),
        }
        samples.append({"sigma_star": sigma_star, "P": _pref_matrix(sigma_star),
                        "nll_best": nll_best, "nll_l2r": nll_l2r, "nll_dyn_easy": nll_de,
                        "feats": feats, "idx_row": idx, "clean_perm": clean_perm,
                        "model": model, "dev": dev})
    return samples


def _make_feature(samples, key, h_pos=None):
    out = []
    for s in samples:
        f = (s["feats"]["H"] - h_pos) if key == "Hres" else s["feats"][key]
        out.append(f.astype(np.float32))
    return out


def train_ranker(samples, feat_list, epochs=300, lr=1e-2):
    d = feat_list[0].shape[1]
    net = BOnlyController(b_dim=d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for s, f in zip(samples, feat_list):
            z = net(torch.tensor(f))
            loss = loss + pairwise_loss(z, s["P"])
        (loss / len(samples)).backward(); opt.step()
    return net


@torch.no_grad()
def eval_ranker(samples, net, feat_list):
    taus, nlls = [], []
    for s, f in zip(samples, feat_list):
        z = net(torch.tensor(f)).cpu().numpy()
        pred = np.argsort(-z).astype(np.int64)
        taus.append(float(kendalltau(pred, s["sigma_star"])[0]))
        nlls.append(order_nll(s["model"], s["idx_row"], pred, s["clean_perm"], s["dev"]))
    return float(np.mean(taus)), np.array(nlls)


def run_probe(ckpt_path, M=16, hc_steps=500, split=(0.7, 0.0, 0.3),
              out_dir="runs/p7/probe", tag="seed123_10k", device="cpu"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_probe_data(ckpt_path, M, hc_steps=hc_steps, device=device)
    tr, _, te = _split_idx(len(samples), split)
    train = [samples[i] for i in tr]; test = [samples[i] for i in (list(te) or list(tr))]
    h_pos = np.mean([s["feats"]["H"] for s in train], axis=0).astype(np.float32)

    nll_l2r = float(np.mean([s["nll_l2r"] for s in test]))
    nll_best = float(np.mean([s["nll_best"] for s in test]))
    nll_de = float(np.mean([s["nll_dyn_easy"] for s in test]))

    feat_sets = ["B", "H", "Hres", "content", "onestep", "content+onestep", "B+content"]
    def feats(group, key):
        if key == "content+onestep":
            return [np.concatenate([s["feats"]["content"], s["feats"]["onestep"]], 1)
                    for s in group]
        if key == "B+content":
            return [np.concatenate([s["feats"]["B"], s["feats"]["content"]], 1)
                    for s in group]
        return _make_feature(group, key, h_pos)

    arms = {}
    for key in feat_sets:
        net = train_ranker(train, feats(train, key), epochs=300)
        tau, nll = eval_ranker(test, net, feats(test, key))
        arms[key] = {"tau_to_oracle": tau,
                     "nll_pred": float(nll.mean()),
                     "nll_pred_vs_L2R": float(nll.mean()) - nll_l2r,
                     "nll_pred_vs_dyn_easy": float(nll.mean()) - nll_de,
                     "nll_pred_vs_best": float(nll.mean()) - nll_best}
        print(f"[{key:16s}] tau*={tau:+.3f} vsL2R={arms[key]['nll_pred_vs_L2R']:+.4f} "
              f"vsBest={arms[key]['nll_pred_vs_best']:+.4f}", flush=True)

    result = {"ckpt": ckpt_path, "tag": tag, "M": M, "hc_steps": hc_steps,
              "nll_l2r": nll_l2r, "nll_best": nll_best, "nll_dyn_easy": nll_de,
              "headroom_best_vs_L2R": nll_l2r - nll_best, "arms": arms,
              "_verdict": "feature usable if tau_to_oracle>0 AND nll_pred_vs_L2R<0 on "
                          "held-out. position(B/H) expected ~0; content/onestep the test."}
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    print(f"\nheadroom(L2R-best)={nll_l2r-nll_best:+.3f}  (test M={len(test)})", flush=True)
    print("PROBE DONE", flush=True)
    return result


if __name__ == "__main__":
    import sys
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/handoff_overnight/seed123/ckpt_step10000.pt"
    tag = sys.argv[2] if len(sys.argv) > 2 else "seed123_10k"
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    run_probe(ck, M=M, tag=tag)
