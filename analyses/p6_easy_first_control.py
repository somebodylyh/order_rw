"""Easy-first artifact control for the oracle-headroom finding.

Is the +0.20-nat per-sample headroom over L2R genuine sample-specific order
structure, or just teacher-forced-NLL gaming (reveal each sample's conditionally
easy blocks first)? Compare hill-climb best vs:
  - static easy-first  : reveal blocks by first-reveal difficulty (ascending)
  - dynamic greedy easy : at each step reveal the conditionally-easiest block
  - dynamic greedy hard : at each step reveal the conditionally-hardest block

A: best ~= dynamic-easy (high tau, ~equal NLL) -> headroom is easy-first artifact.
B: best << dynamic-easy (clearly lower NLL, low/moderate tau) -> genuine
   non-myopic sample-specific structure.
"""
import json as _json
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

from analyses.p5_utility_controller import (
    load_p5_ckpt, order_nll, physical_blocks_to_model_token_order, N, BLOCK_LEN,
)
from analyses.p6_oracle_headroom import hill_climb


def _tok_orders(block_orders, clean_perm, dev):
    """block_orders: (K, 64) phys-block orders -> (K, 256) model-token orders."""
    outs = [physical_blocks_to_model_token_order(
                torch.as_tensor(bo, dtype=torch.long)[None, :], clean_perm, BLOCK_LEN)
            for bo in block_orders]
    return torch.cat(outs, dim=0).to(dev)


@torch.no_grad()
def per_position_nll(model, idx_tiled, token_orders, dev):
    """(B,T) per-reveal-position NLL. mean == order_nll's scalar loss (validated)."""
    logits, _ = model.forward_fn(idx_tiled.to(dev), token_orders.to(dev))
    shift_logits = logits[:, :-1, :]
    targets = model.shuffle(idx_tiled.to(dev), token_orders.to(dev))
    B, T = targets.shape
    pp = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                         targets.reshape(-1), reduction="none").reshape(B, T)
    return pp.cpu().numpy()


def dynamic_greedy(model, idx_row, clean_perm, dev, mode="easy"):
    """Step-wise: reveal conditionally easiest (or hardest) remaining block."""
    R, remaining = [], list(range(N))
    for _ in range(N):
        rem = list(remaining)
        # candidate block orders: R + [i] + (other remaining), causal -> rest irrelevant
        cand = [R + [i] + [b for b in rem if b != i] for i in rem]
        tok = _tok_orders(np.array(cand, dtype=np.int64), clean_perm, dev)
        pp = per_position_nll(model, idx_row.expand(len(rem), -1), tok, dev)
        L = 4 * len(R)
        d = pp[:, L:L + BLOCK_LEN].sum(axis=1)            # one-step NLL of block i | R
        pick = int(d.argmin() if mode == "easy" else d.argmax())
        R.append(rem[pick]); remaining.remove(rem[pick])
    return np.array(R, dtype=np.int64)


def static_easy(model, idx_row, clean_perm, dev):
    """Reveal-each-block-first difficulty d0(i); order = argsort ascending."""
    cand = [[i] + [b for b in range(N) if b != i] for i in range(N)]
    tok = _tok_orders(np.array(cand, dtype=np.int64), clean_perm, dev)
    pp = per_position_nll(model, idx_row.expand(N, -1), tok, dev)
    d0 = pp[:, :BLOCK_LEN].sum(axis=1)                    # block i revealed first
    return np.argsort(d0).astype(np.int64)


def run_easy_first_control(ckpt_path, M=4, hc_steps=600,
                           out_dir="runs/p6/easy_first", tag="seed123_10k", device="cpu"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    l2r = np.arange(N, dtype=np.int64)

    # sanity: per-position mean == order_nll
    tok = _tok_orders(l2r[None, :], clean_perm, dev)
    pp = per_position_nll(model, chunks[0:1], tok, dev)
    assert abs(pp.mean() - order_nll(model, chunks[0:1], l2r, clean_perm, dev)) < 1e-3

    rows = []
    for t in range(M):
        idx = chunks[t:t+1]
        nll_l2r = order_nll(model, idx, l2r, clean_perm, dev)
        sig_best, nll_best, _ = hill_climb(model, idx, clean_perm, dev, n_steps=hc_steps, seed=t)
        sig_se = static_easy(model, idx, clean_perm, dev)
        sig_de = dynamic_greedy(model, idx, clean_perm, dev, "easy")
        sig_dh = dynamic_greedy(model, idx, clean_perm, dev, "hard")
        nll_se = order_nll(model, idx, sig_se, clean_perm, dev)
        nll_de = order_nll(model, idx, sig_de, clean_perm, dev)
        nll_dh = order_nll(model, idx, sig_dh, clean_perm, dev)
        row = {"seq": t, "nll_l2r": nll_l2r, "nll_best": nll_best,
               "nll_static_easy": nll_se, "nll_dyn_easy": nll_de, "nll_dyn_hard": nll_dh,
               "tau_best_vs_dyneasy": float(kendalltau(sig_best, sig_de)[0]),
               "tau_best_vs_staticeasy": float(kendalltau(sig_best, sig_se)[0])}
        rows.append(row)
        print(f"  seq{t}: L2R={nll_l2r:.3f} best={nll_best:.3f} dyn_easy={nll_de:.3f} "
              f"static_easy={nll_se:.3f} dyn_hard={nll_dh:.3f} | "
              f"tau(best,dyneasy)={row['tau_best_vs_dyneasy']:.2f}", flush=True)

    def gain(k):
        return float(np.mean([r["nll_l2r"] - r[k] for r in rows]))
    summary = {
        "ckpt": ckpt_path, "tag": tag, "M": M, "hc_steps": hc_steps,
        "gain_best_vs_L2R": gain("nll_best"),
        "gain_dyn_easy_vs_L2R": gain("nll_dyn_easy"),
        "gain_static_easy_vs_L2R": gain("nll_static_easy"),
        "gain_dyn_hard_vs_L2R": gain("nll_dyn_hard"),
        "best_minus_dyneasy": float(np.mean([r["nll_best"] - r["nll_dyn_easy"] for r in rows])),
        "tau_best_vs_dyneasy_mean": float(np.mean([r["tau_best_vs_dyneasy"] for r in rows])),
        "rows": rows,
        "_verdict": "A(artifact): best~=dyn_easy & high tau. B(genuine): best<<dyn_easy "
                    "(best_minus_dyneasy<0) & low tau.",
    }
    _json.dump(summary, open(out / f"{tag}.json", "w"), indent=2, default=float)
    print(f"\ngain: best={summary['gain_best_vs_L2R']:+.3f} "
          f"dyn_easy={summary['gain_dyn_easy_vs_L2R']:+.3f} "
          f"static_easy={summary['gain_static_easy_vs_L2R']:+.3f} "
          f"dyn_hard={summary['gain_dyn_hard_vs_L2R']:+.3f}", flush=True)
    print(f"best_minus_dyneasy={summary['best_minus_dyneasy']:+.3f} "
          f"tau(best,dyneasy)={summary['tau_best_vs_dyneasy_mean']:.2f}", flush=True)
    print("EASY FIRST CONTROL DONE", flush=True)
    return summary


if __name__ == "__main__":
    import sys
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/handoff_overnight/seed123/ckpt_step10000.pt"
    tag = sys.argv[2] if len(sys.argv) > 2 else "seed123_10k"
    run_easy_first_control(ck, tag=tag)
