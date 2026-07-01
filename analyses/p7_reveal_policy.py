"""P7 loss-only: reveal MLP trained directly from AO-GPT NLL via a Plackett-Luce
policy gradient (EMA baseline). Frozen AO-GPT, train/test split. Final closure
check: can AO-loss EXPLORATION beat the supervised-probe null?

Locked caveat: loss provides a training signal, not input information. If B/H carry
only position, PG can explore but cannot generalize a content-dependent policy.
"""
import json as _json
import pathlib
import numpy as np
import torch
from scipy.stats import kendalltau

import torch.nn.functional as F
from analyses.p5_utility_controller import (
    load_p5_ckpt, order_nll, block_b_features, block_hidden_states,
    BOnlyController, _split_idx, sample_scaffold, sigma_from_B65, N,
)
from analyses.p6_oracle_headroom import hill_climb
from analyses.p7_predictability_probe import block_content_emb


def sample_pl_order(scores, tau=1.0):
    """Plackett-Luce sampling without replacement. scores: (N,) tensor w/ grad."""
    avail = torch.ones(N, dtype=torch.bool, device=scores.device)
    order, logp, ent = [], scores.new_zeros(()), scores.new_zeros(())
    for _ in range(N):
        logits = (scores / tau).masked_fill(~avail, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        i = dist.sample()
        logp = logp + dist.log_prob(i)
        ent = ent + dist.entropy()
        order.append(int(i)); avail[i] = False
    return np.array(order, dtype=np.int64), logp, ent


def build_features(ckpt_path, M, layer=1, head=7, n_reveals=8, device="cpu"):
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    p2m = clean_perm.block_perm_phys_to_model.cpu().numpy()
    sc = sample_scaffold(ckpt_path, M, layer=layer, head=head, n_reveals=n_reveals,
                         device=device)
    l2r = np.arange(N, dtype=np.int64)
    samples = []
    for t in range(M):
        idx = chunks[t:t+1]
        H_phys = block_hidden_states(model, idx, l2r, clean_perm, layer, dev)[p2m]
        sigma_B = np.asarray(sc["sigma_B"][t], dtype=np.int64)   # B-CDL order (distill target)
        samples.append({
            "feats": {"B": block_b_features(sc["B"][t]).astype(np.float32),
                      "H": H_phys.astype(np.float32),
                      "content": block_content_emb(model, idx, p2m, dev)},
            "sigma_B": sigma_B,
            "idx_row": idx, "clean_perm": clean_perm, "model": model, "dev": dev,
            "nll_l2r": order_nll(model, idx, l2r, clean_perm, dev),
            "nll_sigmaB": order_nll(model, idx, sigma_B, clean_perm, dev)})
    return samples


def pl_logprob(scores, order, tau):
    """log P_PL(order | scores) at temperature tau (for distillation to a fixed order)."""
    avail = torch.ones(N, dtype=torch.bool, device=scores.device)
    logp = scores.new_zeros(())
    for i in order:
        logits = (scores / tau).masked_fill(~avail, -1e9)
        logp = logp + F.log_softmax(logits, dim=0)[int(i)]
        avail[int(i)] = False
    return logp


_POSID = np.eye(N, dtype=np.float32)          # block-id one-hot -> global fixed order


def _feat(s, key, h_pos):
    if key == "posid":
        return _POSID                          # PG positive control (same for all x)
    if key == "B+H":
        return np.concatenate([s["feats"]["B"], s["feats"]["H"]], 1)
    if key == "B+Hres":
        return np.concatenate([s["feats"]["B"], s["feats"]["H"] - h_pos], 1)
    return s["feats"][key]


def train_policy(train, key, h_pos, epochs=300, lr=1e-2, K=2, tau0=1.0, tau1=0.5,
                 beta=3e-3, ema_decay=0.9, adv_clip=0.3):
    """PG train. K samples/x (lower-variance advantage), temperature anneal tau0->tau1,
    per-sample EMA baseline."""
    d = _feat(train[0], key, h_pos).shape[1]
    net = BOnlyController(b_dim=d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ema = {}
    hist = []
    for ep in range(epochs):
        tau = tau0 + (tau1 - tau0) * ep / max(epochs - 1, 1)
        opt.zero_grad(); loss = 0.0; nlls = []
        for si, s in enumerate(train):
            scores = net(torch.tensor(_feat(s, key, h_pos)))
            samp = [sample_pl_order(scores, tau) for _ in range(K)]
            ls = [order_nll(s["model"], s["idx_row"], o, s["clean_perm"], s["dev"])
                  for o, _, _ in samp]
            nlls.append(float(np.mean(ls)))
            for (order, logp, ent), nll in zip(samp, ls):
                ema[si] = nll if si not in ema else ema_decay * ema[si] + (1 - ema_decay) * nll
                adv = float(np.clip(ema[si] - nll, -adv_clip, adv_clip))
                loss = loss - adv * logp - beta * ent
        (loss / (len(train) * K)).backward(); opt.step()
        hist.append(float(np.mean(nlls)))
    return net, hist


def train_policy_distill(train, key, h_pos, epochs=300, lr=1e-2, K=2,
                         tau_low=0.1, tau_high=1.0, lam_distill=1.0, lam_pg=1.0,
                         beta=3e-3, ema_decay=0.9, adv_clip=0.3):
    """Dual-temperature: distill the B-CDL order into the PL policy at LOW temp
    (anchor to the good B prior) + explore via AO-NLL policy gradient at HIGH temp."""
    d = _feat(train[0], key, h_pos).shape[1]
    net = BOnlyController(b_dim=d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ema = {}
    hist = []
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0; nlls = []
        for si, s in enumerate(train):
            scores = net(torch.tensor(_feat(s, key, h_pos)))
            l_distill = -pl_logprob(scores, s["sigma_B"], tau_low)      # anchor to B order
            l_pg = 0.0; ls = []
            for _k in range(K):
                order, logp, ent = sample_pl_order(scores, tau_high)
                nll = order_nll(s["model"], s["idx_row"], order, s["clean_perm"], s["dev"])
                ls.append(nll)
                ema[si] = nll if si not in ema else ema_decay * ema[si] + (1 - ema_decay) * nll
                adv = float(np.clip(ema[si] - nll, -adv_clip, adv_clip))
                l_pg = l_pg - adv * logp - beta * ent
            loss = loss + lam_distill * l_distill + lam_pg * (l_pg / K)
            nlls.append(float(np.mean(ls)))
        (loss / len(train)).backward(); opt.step()
        hist.append(float(np.mean(nlls)))
    return net, hist


@torch.no_grad()
def eval_policy(test, net, key, h_pos):
    l2r = np.arange(N, dtype=np.int64)
    nlls, taus = [], []
    for s in test:
        scores = net(torch.tensor(_feat(s, key, h_pos))).cpu().numpy()
        order = np.argsort(-scores).astype(np.int64)              # greedy (temp->0)
        nlls.append(order_nll(s["model"], s["idx_row"], order, s["clean_perm"], s["dev"]))
        taus.append(float(kendalltau(order, l2r)[0]))
    return float(np.mean(nlls)), float(np.mean(taus))


def run_reveal_policy(ckpt_path, M=40, epochs=300, hc_steps=600,
                      split=(0.7, 0.0, 0.3), out_dir="runs/p7/policy",
                      tag="seed123_10k", device="cpu"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_features(ckpt_path, M, device=device)
    tr, _, te = _split_idx(len(samples), split)
    train = [samples[i] for i in tr]; test = [samples[i] for i in (list(te) or list(tr))]
    h_pos = np.mean([s["feats"]["H"] for s in train], axis=0).astype(np.float32)

    nll_l2r = float(np.mean([s["nll_l2r"] for s in test]))
    # references on test
    rng = np.random.default_rng(0)
    nll_rand = float(np.mean([order_nll(s["model"], s["idx_row"], rng.permutation(N),
                                        s["clean_perm"], s["dev"]) for s in test]))
    nll_oracle = float(np.mean([hill_climb(s["model"], s["idx_row"], s["clean_perm"],
                                           s["dev"], n_steps=hc_steps, seed=i)[1]
                                for i, s in enumerate(test)]))

    arms = {}
    for key in ("posid", "B", "B+H", "B+Hres"):
        net, hist = train_policy(train, key, h_pos, epochs=epochs)
        nll, tau = eval_policy(test, net, key, h_pos)
        arms[key] = {"test_nll": nll, "vs_L2R": nll - nll_l2r,
                     "vs_oracle": nll - nll_oracle, "tau_vs_L2R": tau,
                     "train_nll_first_last": [hist[0], hist[-1]]}
        print(f"[{key:7s}] test_nll={nll:.4f} vsL2R={nll-nll_l2r:+.4f} "
              f"vsOracle={nll-nll_oracle:+.4f} tau_L2R={tau:.2f} "
              f"train {hist[0]:.3f}->{hist[-1]:.3f}", flush=True)

    result = {"ckpt": ckpt_path, "tag": tag, "M": M, "epochs": epochs,
              "nll_l2r": nll_l2r, "nll_random": nll_rand, "nll_oracle": nll_oracle,
              "arms": arms,
              "_verdict": "A: B/H policy can't beat L2R (vs_L2R>=0) -> line closed. "
                          "C: B+Hres beats B held-out (multi-seed) -> reopen H."}
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    print(f"\nrefs: L2R={nll_l2r:.4f} random={nll_rand:.4f} oracle={nll_oracle:.4f} "
          f"(test M={len(test)})", flush=True)
    print("REVEAL POLICY DONE", flush=True)
    return result


def run_policy_distill(ckpt_path, M=40, epochs=300, hc_steps=600,
                       split=(0.7, 0.0, 0.3), out_dir="runs/p7/policy_distill",
                       tag="seed123_10k", device="cpu"):
    """Dual-temperature distill(B-order, low tau) + AO-NLL PG(high tau)."""
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = build_features(ckpt_path, M, device=device)
    tr, _, te = _split_idx(len(samples), split)
    train = [samples[i] for i in tr]; test = [samples[i] for i in (list(te) or list(tr))]
    h_pos = np.mean([s["feats"]["H"] for s in train], axis=0).astype(np.float32)

    nll_l2r = float(np.mean([s["nll_l2r"] for s in test]))
    nll_sigmaB = float(np.mean([s["nll_sigmaB"] for s in test]))
    nll_oracle = float(np.mean([hill_climb(s["model"], s["idx_row"], s["clean_perm"],
                                           s["dev"], n_steps=hc_steps, seed=i)[1]
                                for i, s in enumerate(test)]))

    arms = {}
    for key in ("B", "B+H", "B+Hres"):
        net, hist = train_policy_distill(train, key, h_pos, epochs=epochs)
        nll, tau = eval_policy(test, net, key, h_pos)
        arms[key] = {"test_nll": nll, "vs_L2R": nll - nll_l2r,
                     "vs_sigmaB": nll - nll_sigmaB, "vs_oracle": nll - nll_oracle,
                     "tau_vs_L2R": tau, "train_nll_first_last": [hist[0], hist[-1]]}
        print(f"[{key:7s}] test={nll:.4f} vsL2R={nll-nll_l2r:+.4f} "
              f"vsSigmaB={nll-nll_sigmaB:+.4f} vsOracle={nll-nll_oracle:+.4f} "
              f"train {hist[0]:.3f}->{hist[-1]:.3f}", flush=True)

    result = {"ckpt": ckpt_path, "tag": tag, "M": M, "epochs": epochs,
              "nll_l2r": nll_l2r, "nll_sigmaB": nll_sigmaB, "nll_oracle": nll_oracle,
              "arms": arms,
              "_verdict": "distill anchors to B-order; PG explores. Win = vs_sigmaB<0 "
                          "(AO-loss exploration beats the B prior) held-out, multi-seed."}
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    print(f"\nrefs: L2R={nll_l2r:.4f} sigmaB={nll_sigmaB:.4f} oracle={nll_oracle:.4f} "
          f"(test M={len(test)})", flush=True)
    print("POLICY DISTILL DONE", flush=True)
    return result


if __name__ == "__main__":
    import sys
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/handoff_overnight/seed123/ckpt_step10000.pt"
    tag = sys.argv[2] if len(sys.argv) > 2 else "seed123_10k"
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    mode = sys.argv[4] if len(sys.argv) > 4 else "distill"
    (run_policy_distill if mode == "distill" else run_reveal_policy)(ck, M=M, tag=tag)
