"""Frozen causal-hidden order probe: per-step predictor hidden, candidate embedding,
param-free dynamic scores (cos-emb + model-attn), per-step gamma-mix greedy rollout,
controls, Level-2 oracle, per-sample audit. NO training. See
docs/superpowers/specs/2026-05-26-causal-hidden-order-probe-design.md."""
import numpy as np
import torch


def _expand_blocks_to_tokens(block_order, block_len):
    """block_order: list/1d int (model-frame blocks) -> 1d token order (len*block_len)."""
    bo = np.asarray(block_order, dtype=np.int64)
    base = np.arange(block_len, dtype=np.int64)
    return (bo[:, None] * block_len + base[None, :]).reshape(-1)


@torch.no_grad()
def context_hidden_at_step(model, idx_model, prefix_blocks, completion_blocks, block_len,
                           device, chunk_size=64):
    """Predictor hidden c_t conditioned on the prefix (model-frame blocks), read at rank
    t*block_len. completion_blocks fills positions after the prefix; by causal masking c_t
    must not depend on it (see causal_invariance_check). Returns (n, E)."""
    model.eval()
    t = len(prefix_blocks)
    full = list(prefix_blocks) + list(completion_blocks)
    tok_order = torch.as_tensor(_expand_blocks_to_tokens(full, block_len),
                                dtype=torch.long, device=device)
    rank = t * block_len
    n_total = idx_model.shape[0]
    out_chunks = []
    for i in range(0, n_total, chunk_size):
        idx = idx_model[i:i + chunk_size].to(device)
        order = tok_order.unsqueeze(0).expand(idx.shape[0], -1)
        out = model.forward_fn(idx, order, return_hidden=True, hidden_return_mode="predictor")
        out_chunks.append(out[2][:, rank, :].float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def causal_invariance_check(model, idx_model, n_blocks, block_len, device,
                            t_list=(0, 1, 4, 16, 32, 48, 63), n_patterns=4, seed=0):
    """PRE-GATE: c_t must depend only on the prefix S_t, not the completion. For each t and
    several prefix patterns, compare c_t under >=2 completions. Pass iff min cosine > 0.99999."""
    rng = np.random.default_rng(seed)
    min_cos, max_absdiff, fails = 1.0, 0.0, []
    for t in t_list:
        if t >= n_blocks:
            continue
        for _ in range(n_patterns):
            perm = rng.permutation(n_blocks)
            prefix = perm[:t].tolist()
            rest = perm[t:].tolist()
            comp_a = rest
            comp_b = list(reversed(rest))
            comp_c = rng.permutation(rest).tolist() if len(rest) > 1 else rest
            c_a = context_hidden_at_step(model, idx_model, prefix, comp_a, block_len, device)
            for comp in (comp_b, comp_c):
                c_b = context_hidden_at_step(model, idx_model, prefix, comp, block_len, device)
                num = (c_a * c_b).sum(1)
                den = np.linalg.norm(c_a, axis=1) * np.linalg.norm(c_b, axis=1) + 1e-12
                cos = float((num / den).min())
                ad = float(np.abs(c_a - c_b).max())
                min_cos = min(min_cos, cos); max_absdiff = max(max_absdiff, ad)
                if cos <= 0.99999:
                    fails.append({"t": int(t), "cosine": cos, "max_absdiff": ad})
    return {"passed": len(fails) == 0, "min_cosine": float(min_cos),
            "max_absdiff": float(max_absdiff), "n_fail": len(fails), "fails": fails[:10]}
