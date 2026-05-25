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
def extract_oracle_hidden(model, idx_model, clean_perm, device, order_model_blocks):
    """Returns (n, N, E) hidden in PHYSICAL block frame. h_v = mean of block v's token hiddens."""
    model.eval()
    bl = _block_len_from(model)
    N = order_model_blocks.shape[1]
    token_order = expand_model_blocks_to_token_order(order_model_blocks, bl).to(device)
    out = model.forward_fn(idx_model.to(device), token_order, return_hidden=True,
                           hidden_return_mode="original")
    hidden = out[2]                       # (logits, loss, hidden); (n, T+1, E)
    h_tok = hidden[:, 1:, :]              # drop [None]; (n, T, E) aligned to idx_model (model frame)
    n, T, E = h_tok.shape
    h_blk_model = h_tok.reshape(n, N, bl, E).mean(dim=2).cpu().numpy()   # (n, N) model-block frame
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()                # model block -> phys block
    h_blk_phys = np.empty_like(h_blk_model)
    h_blk_phys[:, inv, :] = h_blk_model                                  # guard 1: align to phys frame
    return h_blk_phys

@torch.no_grad()
def extract_causal_hidden(model, idx_model, device, canonical_order_model_blocks, t_list):
    """Returns {t: (n, E)} predictor hidden at the start of block-step t under the FIXED canonical
    order (same order for every sample -> guard 2). predictor hidden is in reveal-rank frame."""
    model.eval()
    bl = _block_len_from(model)
    token_order = expand_model_blocks_to_token_order(canonical_order_model_blocks, bl).to(device)
    out = model.forward_fn(idx_model.to(device), token_order, return_hidden=True,
                           hidden_return_mode="predictor")
    pred = out[2]                         # (n, T, E) indexed by reveal rank
    res = {}
    for t in t_list:
        rank = t * bl                     # first token rank of block-step t
        assert rank < pred.shape[1], f"t={t} out of range (rank {rank} >= T {pred.shape[1]})"
        res[t] = pred[:, rank, :].cpu().numpy()
    return res
