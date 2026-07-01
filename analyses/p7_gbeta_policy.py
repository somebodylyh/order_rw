"""P7 gbeta-policy: reuse the DEPLOYED gbeta parameters as a Plackett-Luce reveal
policy and fine-tune with AO-GPT NLL policy gradient (frozen AO-GPT, 20k ckpt).

gbeta currently deploys as argsort(scores). Here scores become PL logits: at low
temp PL==argsort(gbeta)==the deployed order (built-in anchor — the params ARE
gbeta, no distillation needed), and AO-NLL PG at higher temp fine-tunes the shared
gbeta params. Physical frame throughout (matches order_nll + val_ori_l2r_block).

Extraction matches the 20k model's deployed path (HookOrderProvider):
  single head L1H7, none_mode='strict65_model', probe-averaged, PHYSICAL frame.
Artifacts: model gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt (frozen);
gbeta reports/uniform_label_free_v1/nodewise_K1000.pt.
"""
import json as _json
import pathlib
import numpy as np
import torch
from scipy.stats import kendalltau

import sys
sys.path.insert(0, "block_lo_arm_order_network")
from analyses.p5_utility_controller import load_p5_ckpt, order_nll, N
from analyses.p6_oracle_headroom import hill_climb
from batch_readout.integration_hook import FrozenBetaHook
from batch_readout.hook_order_provider import (
    extract_selected_head_A_for_batch, random_probe_token_orders,
)

GBETA_CKPT = "reports/uniform_label_free_v1/nodewise_K1000.pt"
HEAD = (1, 7)
NONE_MODE = "strict65_model"


def gbeta_scores_batchmean(gbeta, aogpt, idx_batch, clean_perm, step, seed, dev):
    """BATCH-MEAN readout (where the gbeta signal lives, per BR-1): extract A over
    the whole batch, average -> ONE consensus B -> ONE score vector z (N,) with grad.
    Matches deployed FrozenBetaHook.step (mean over batch). Physical frame."""
    probe = random_probe_token_orders(idx_batch.shape[0], seed, step, dev)
    with torch.no_grad():
        A = extract_selected_head_A_for_batch(
            aogpt, idx_batch.to(dev), HEAD, clean_perm, dev, probe, none_mode=NONE_MODE)
    A = A.to(dev).float()
    B = A.transpose(1, 2).mean(dim=0, keepdim=True)        # (1,N,N) batch-mean consensus
    n = B.shape[-1]; d = torch.arange(n, device=B.device)
    B = B.clone(); B[:, d, d] = 0.0
    return gbeta(B)[0]                                      # (N,) with grad


def sample_pl(scores, tau):
    avail = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    order, logp, ent = [], scores.new_zeros(()), scores.new_zeros(())
    for _ in range(scores.shape[0]):
        logits = (scores / tau).masked_fill(~avail, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        i = dist.sample()
        logp = logp + dist.log_prob(i); ent = ent + dist.entropy()
        order.append(int(i)); avail[i] = False
    return np.array(order, dtype=np.int64), logp, ent


def run_gbeta_policy(ckpt_path, M=32, epochs=200, K=2, tau_high=1.0, hc_steps=600,
                     lr=3e-4, beta=3e-3, ema_decay=0.9, adv_clip=0.3,
                     split=(0.7, 0.0, 0.3), out_dir="runs/p7/gbeta_policy",
                     tag="K1000_20k", device="cuda", gbeta_ckpt=GBETA_CKPT):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    hook = FrozenBetaHook(gbeta_ckpt, mode="argsort", device=str(dev))
    gbeta = hook.model
    for p in gbeta.parameters():
        p.requires_grad_(True)
    gbeta.train()
    opt = torch.optim.Adam(gbeta.parameters(), lr=lr)

    ids = np.arange(M); n_tr = int(split[0] * M)
    tr, te = ids[:n_tr], ids[n_tr:]
    l2r = np.arange(N, dtype=np.int64)
    tr_idx = torch.stack([chunks[t] for t in tr])          # (n_tr, seq)
    te_idx = torch.stack([chunks[t] for t in te])

    @torch.no_grad()
    def mean_nll(order, batch_ids):
        return float(np.mean([order_nll(model, chunks[t:t+1], order, clean_perm, dev)
                              for t in batch_ids]))

    # deployed consensus (init gbeta, batch-mean over test) + refs on test
    with torch.no_grad():
        z0 = gbeta_scores_batchmean(gbeta, model, te_idx, clean_perm, 0, 0, dev)
        dep_order = np.argsort(-z0.cpu().numpy()).astype(np.int64)
    dep0 = mean_nll(dep_order, te)
    nll_l2r = mean_nll(l2r, te)
    nll_oracle = float(np.mean([hill_climb(model, chunks[t:t+1], clean_perm, dev,
                                           n_steps=hc_steps, seed=j)[1]
                                for j, t in enumerate(te)]))   # per-sample upper bound

    ema = None
    for ep in range(epochs):
        opt.zero_grad(); loss = 0.0
        z = gbeta_scores_batchmean(gbeta, model, tr_idx, clean_perm, ep, 0, dev)  # consensus
        for _k in range(K):
            order, logp, entr = sample_pl(z, tau_high)
            r = mean_nll(order, tr)                          # reward = mean NLL over train
            ema = r if ema is None else ema_decay * ema + (1 - ema_decay) * r
            adv = float(np.clip(ema - r, -adv_clip, adv_clip))
            loss = loss - adv * logp - beta * entr
        (loss / K).backward(); opt.step()

    gbeta.eval()
    with torch.no_grad():
        z = gbeta_scores_batchmean(gbeta, model, te_idx, clean_perm, 0, 0, dev).cpu().numpy()
    order = np.argsort(-z).astype(np.int64)
    nll_policy = mean_nll(order, te)
    taus = [float(kendalltau(order, l2r)[0])]

    result = {"ckpt": ckpt_path, "gbeta": gbeta_ckpt, "head": HEAD, "none_mode": NONE_MODE,
              "tag": tag, "M": M, "epochs": epochs,
              "nll_deployed_gbeta_init": dep0, "nll_l2r": nll_l2r, "nll_oracle": nll_oracle,
              "nll_policy_finetuned": nll_policy,
              "policy_vs_deployed": nll_policy - dep0,
              "policy_vs_L2R": nll_policy - nll_l2r,
              "policy_vs_oracle": nll_policy - nll_oracle,
              "tau_policy_vs_L2R": float(np.mean(taus)),
              "_verdict": "win = policy_vs_deployed<0 (PG improves shared gbeta params) "
                          "held-out; ideally policy_vs_L2R<0."}
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    print(f"deployed_gbeta={dep0:.4f} L2R={nll_l2r:.4f} oracle={nll_oracle:.4f}", flush=True)
    print(f"policy_finetuned={nll_policy:.4f} vsDeployed={nll_policy-dep0:+.4f} "
          f"vsL2R={nll_policy-nll_l2r:+.4f} vsOracle={nll_policy-nll_oracle:+.4f} "
          f"tau_L2R={np.mean(taus):.2f} (test M={len(te)})", flush=True)
    print("GBETA POLICY DONE", flush=True)
    return result


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        "block_lo_arm_order_network/probe_results/gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt"
    tag = sys.argv[2] if len(sys.argv) > 2 else "K1000_20k"
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    run_gbeta_policy(ck, M=M, tag=tag)
