# hidden_residual_hidden.py
"""Phase-1 hidden-residual diagnostic: hidden-state extraction.
oracle h_v   = full-context block representation (hidden_return_mode='original'), model->phys.
causal h_S_t = predictor hidden at block-step t under the FIXED canonical order (guard 2)."""
import numpy as np
import torch
from clean_training_protocol import expand_model_blocks_to_token_order

def _block_len_from(model):
    return int(model.block_order_block_len)

@torch.no_grad()
def extract_oracle_hidden(model, idx_model, clean_perm, device, order_model_blocks, chunk_size=16):
    """Returns (n, N, E) hidden in PHYSICAL block frame. h_v = mean of block v's token hiddens.
    Chunked across n to avoid CUDA OOM from logits (n=256 -> ~53 GB fp32)."""
    model.eval()
    bl = _block_len_from(model)
    N = order_model_blocks.shape[1]
    n_total = idx_model.shape[0]
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    chunks = []
    for i in range(0, n_total, chunk_size):
        end = min(i + chunk_size, n_total)
        idx_chunk = idx_model[i:end].to(device)
        order_chunk = order_model_blocks[i:end].to(device)
        token_order = expand_model_blocks_to_token_order(order_chunk, bl)
        out = model.forward_fn(idx_chunk, token_order, return_hidden=True,
                               hidden_return_mode="original")
        hidden = out[2]                                             # (chunk, T+1, E)
        h_tok = hidden[:, 1:, :]                                    # drop [None]
        nc, _, E = h_tok.shape
        h_blk_model = h_tok.reshape(nc, N, bl, E).mean(dim=2).cpu().numpy()
        h_blk_phys = np.empty_like(h_blk_model)
        h_blk_phys[:, inv, :] = h_blk_model                         # guard 1: align to phys frame
        chunks.append(h_blk_phys)
    return np.concatenate(chunks, axis=0)

@torch.no_grad()
def extract_causal_hidden(model, idx_model, device, canonical_order_model_blocks, t_list, chunk_size=16):
    """Returns {t: (n, E)} predictor hidden at the start of block-step t under the FIXED canonical
    order (same order for every sample -> guard 2). predictor hidden is in reveal-rank frame.
    Chunked across n to avoid CUDA OOM from logits."""
    model.eval()
    bl = _block_len_from(model)
    n_total = idx_model.shape[0]

    pred_chunks = []
    for i in range(0, n_total, chunk_size):
        end = min(i + chunk_size, n_total)
        idx_chunk = idx_model[i:end].to(device)
        order_chunk = canonical_order_model_blocks[i:end].to(device)
        token_order = expand_model_blocks_to_token_order(order_chunk, bl)
        out = model.forward_fn(idx_chunk, token_order, return_hidden=True,
                               hidden_return_mode="predictor")
        pred_chunks.append(out[2].cpu().numpy())                    # (chunk, T, E)

    pred_full = np.concatenate(pred_chunks, axis=0)                  # (n, T, E)
    res = {}
    for t in t_list:
        rank = t * bl                                                # first token rank of block-step t
        assert rank < pred_full.shape[1], f"t={t} out of range (rank {rank} >= T {pred_full.shape[1]})"
        res[t] = pred_full[:, rank, :]
    return res
