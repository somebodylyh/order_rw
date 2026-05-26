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


@torch.no_grad()
def candidate_conditioned_hidden(model, idx_model, prefix_blocks, next_block, other_blocks,
                                 block_len, device, chunk_size=64):
    """c_t^(v): predictor hidden conditioned on (S_t=prefix_blocks, sigma(t+1)=next_block).
    Thin wrapper over context_hidden_at_step with completion=[next_block]+other_blocks; by causal
    invariance (see causal_invariance_check) other_blocks do not affect the result. Returns (n, E)."""
    return context_hidden_at_step(model, idx_model, prefix_blocks, [next_block] + list(other_blocks),
                                  block_len, device, chunk_size)


def causal_invariance_check(model, idx_model, n_blocks, block_len, device,
                            t_list=(0, 1, 4, 16, 32, 48, 63), n_patterns=4, seed=0):
    """Reframed PRE-GATE (Path X): c_t^(v) must be invariant to sigma(t+2..) holding (S_t, v=sigma(t+1))
    fixed. (It is NOT invariant to v itself — that is the intended candidate-conditioning.) For each t
    and several prefix patterns, fix v = first remaining block and compare c_t^(v) under >=2 orderings
    of the REMAINING blocks. Pass iff min cosine > 0.99999. Skips (t, pattern) with <2 remaining-tail
    blocks (no reordering possible)."""
    rng = np.random.default_rng(seed)
    min_cos, max_absdiff, fails, n_cmp = 1.0, 0.0, [], 0
    for t in t_list:
        if t >= n_blocks:
            continue
        for _ in range(n_patterns):
            perm = rng.permutation(n_blocks)
            prefix = perm[:t].tolist()
            rest = perm[t:].tolist()
            if len(rest) == 0:
                continue
            v, tail = rest[0], rest[1:]
            if len(tail) < 2:
                continue                                   # no meaningful reordering of the tail
            c_a = candidate_conditioned_hidden(model, idx_model, prefix, v, tail, block_len, device)
            variant_tails = [list(reversed(tail)), rng.permutation(tail).tolist()]
            for tl in variant_tails:
                c_b = candidate_conditioned_hidden(model, idx_model, prefix, v, tl, block_len, device)
                num = (c_a * c_b).sum(1)
                den = np.linalg.norm(c_a, axis=1) * np.linalg.norm(c_b, axis=1) + 1e-12
                cos = float((num / den).min())
                ad = float(np.abs(c_a - c_b).max())
                min_cos = min(min_cos, cos); max_absdiff = max(max_absdiff, ad); n_cmp += 1
                if cos <= 0.99999:
                    fails.append({"t": int(t), "cosine": cos, "max_absdiff": ad})
    return {"passed": len(fails) == 0 and n_cmp > 0, "min_cosine": float(min_cos),
            "max_absdiff": float(max_absdiff), "n_compare": n_cmp, "n_fail": len(fails),
            "fails": fails[:10]}


@torch.no_grad()
def pooled_context_hidden(model, idx_model, prefix_blocks, completion_blocks, block_len, device,
                          pool="mean", chunk_size=64):
    """Path Y CONTROL context: pool of partial-context ORIGINAL hiddens over the revealed prefix
    blocks S_t. Target-neutral (same vector for every candidate) and depends only on S_t
    (completion-invariant). pool in {'mean','last'}. Empty prefix -> zeros. Returns (n, E)."""
    model.eval()
    n_total = idx_model.shape[0]
    E = int(model.config.n_embd)
    if len(prefix_blocks) == 0:
        return np.zeros((n_total, E), dtype=np.float32)
    full = list(prefix_blocks) + list(completion_blocks)
    tok_order = torch.as_tensor(_expand_blocks_to_tokens(full, block_len), dtype=torch.long, device=device)
    outs = []
    for i in range(0, n_total, chunk_size):
        idx = idx_model[i:i + chunk_size].to(device)
        order = tok_order.unsqueeze(0).expand(idx.shape[0], -1)
        out = model.forward_fn(idx, order, return_hidden=True, hidden_return_mode="original")
        H = out[2]                                          # (nc, T+1, E); idx 0 = [None], 1+p = model-pos p
        blk = [H[:, 1 + b * block_len:1 + (b + 1) * block_len, :].mean(dim=1) for b in prefix_blocks]
        B = torch.stack(blk, dim=1)                         # (nc, |S_t|, E)
        p = B.mean(dim=1) if pool == "mean" else B[:, -1, :]
        outs.append(p.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def candidate_embeddings(model, idx_model, model_block_ids, block_len, mode, device):
    """Returns (n, len(model_block_ids), E) candidate embeddings in model embedding space.
    mode='content_token': mean of the block's input token embeddings (wte only, NO positional)
                          -> content-aware (sees candidate content; report labels accordingly).
    mode='content_free' : the block's first-position positional embedding (wpe), broadcast over
                          samples -> no candidate content."""
    model.eval()
    idx = idx_model.to(device)
    n = idx.shape[0]
    bids = list(model_block_ids)
    if mode == "content_token":
        cols = []
        for b in bids:
            toks = idx[:, b * block_len:(b + 1) * block_len]          # (n, BL) model-frame
            cols.append(model.transformer.wte(toks).mean(dim=1))      # (n, E) token-only
        return torch.stack(cols, dim=1).float().cpu().numpy()         # (n, M, E)
    if mode == "content_free":
        pos = torch.tensor([b * block_len + 1 for b in bids], device=device)  # +1: [None] offset
        emb = model.transformer.wpe(pos)                              # (M, E)
        return emb.unsqueeze(0).expand(n, -1, -1).float().cpu().numpy()
    raise ValueError(f"unknown mode {mode}")
