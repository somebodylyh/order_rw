"""Stage-A label-free head selection (ported from uniform_label_free_v1.py).

Scores every (layer, head) with NO physical labels and NO hard-coded head:
    score = s_split + s_reveal + s_conf + 2.0 * s_cycle
where
    s_split  = split-half cross-rollout Kendall tau (reveals 0..mid vs mid..n), clipped >=0
    s_reveal = mean pairwise Kendall tau of CDL rollouts across all reveals
    s_conf   = mean |P - 0.5| of the content preference matrix
    s_cycle  = 1 - triadic cycle rate of the preference matrix
Picks the top-1 head. On a different backbone the winner differs — re-run it.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau

from orderhead_v3.per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec as _A_model_vec
from orderhead_v3.none_separated_block_graph import build_none_separated_B, rollout_by_method
from orderhead_v3.constants import N, BLOCK_LEN


def _make_probe(model, batch_size, seed, device):
    blocks = torch.stack([
        torch.randperm(N, generator=torch.Generator(device="cpu").manual_seed(int(seed) + b))
        for b in range(batch_size)])
    return model._expand_block_orders_to_token_orders(blocks).to(device)


@torch.no_grad()
def _extract_all_B65(model, chunks, total, seed, device, n_reveal, fwd_batch=32):
    """B65_by_reveal[ri][layer][head] = (total, 65, 65) float32."""
    model.eval()
    B65_by_reveal = []
    L = H_per = None
    for ri in range(n_reveal):
        B_sum = None
        for start in range(0, total, fwd_batch):
            end = min(start + fwd_batch, total)
            probe = _make_probe(model, end - start, seed * 100000 + ri * 1000 + start, device)
            out = model.forward_fn(chunks[start:end].to(device), probe,
                                   return_attentions=True, return_logits=False)
            attn_list = out[-1]
            if B_sum is None:
                L, H_per = len(attn_list), attn_list[0].shape[1]
                B_sum = {(l, h): np.zeros((total, 65, 65), np.float64) for l in range(L) for h in range(H_per)}
            probe_np = probe.cpu().numpy()
            for li in range(L):
                a = attn_list[li].cpu().numpy()             # (bs, H, 257, 257)
                for hi in range(H_per):
                    for bi in range(end - start):
                        A65 = _A_model_vec(a[bi, hi], probe_np[bi])
                        B_sum[(li, hi)][start + bi] = build_none_separated_B(A65)
            del attn_list
        B65_ri = [[B_sum[(l, h)].astype(np.float32) for h in range(H_per)] for l in range(L)]
        B65_by_reveal.append(B65_ri)
    return B65_by_reveal, L, H_per


def _pairwise_pref(B65):
    Bc = B65[1:, 1:]
    return Bc / (Bc + Bc.T + 1e-8)


def _pairwise_confidence(B65):
    return float(np.mean(np.abs(_pairwise_pref(B65) - 0.5)))


def _cycle_rate(B65):
    pref = (_pairwise_pref(B65) > 0.5).astype(np.int8)
    n = pref.shape[0]
    total = max(1, n * (n - 1) * (n - 2) // 6)
    cycles = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if pref[i, j] == pref[j, k] == pref[k, i]:
                    cycles += 1
    return cycles / total


@torch.no_grad()
def select_best_head(model, chunks, *, n_reveal=8, seed=0, device="cpu", lam_cycle=2.0):
    """Return ((layer, head), head_scores) — the label-free top-1 head."""
    total = int(chunks.shape[0])
    B65_by_reveal, L, H_per = _extract_all_B65(model, chunks, total, seed, device, n_reveal)
    mid = n_reveal // 2
    head_scores = []
    for layer in range(L):
        for head in range(H_per):
            B_A = np.mean([B65_by_reveal[ri][layer][head] for ri in range(mid)], axis=0).mean(axis=0)
            B_B = np.mean([B65_by_reveal[ri][layer][head] for ri in range(mid, n_reveal)], axis=0).mean(axis=0)
            sA = rollout_by_method(B_A, "C-D+L")
            sB = rollout_by_method(B_B, "C-D+L")
            s_split = float(max(0.0, kendalltau(sA, sB)[0] if not np.isnan(kendalltau(sA, sB)[0]) else 0.0))

            orders = [rollout_by_method(B65_by_reveal[ri][layer][head].mean(axis=0), "C-D+L")
                      for ri in range(n_reveal)]
            taus = [max(0.0, kendalltau(orders[i], orders[j])[0])
                    for i in range(n_reveal) for j in range(i + 1, n_reveal)
                    if not np.isnan(kendalltau(orders[i], orders[j])[0])]
            s_reveal = float(np.mean(taus)) if taus else 0.0

            B_mean = np.mean([B65_by_reveal[ri][layer][head].mean(axis=0) for ri in range(n_reveal)], axis=0)
            s_conf = _pairwise_confidence(B_mean)
            s_cycle = 1.0 - _cycle_rate(B_mean)
            total_score = s_split + s_reveal + s_conf + lam_cycle * s_cycle
            head_scores.append({"layer": layer, "head": head, "s_split": s_split,
                                "s_reveal": s_reveal, "s_conf": s_conf, "s_cycle": s_cycle,
                                "total": total_score})
    head_scores.sort(key=lambda x: x["total"], reverse=True)
    best = head_scores[0]
    return (best["layer"], best["head"]), head_scores
