"""Model-coupled I/O for the hidden-graph diagnostic:
image block-hidden extraction (text reuses hidden_residual_hidden.extract_oracle_hidden),
and frozen teacher-forced NLL under a given block order (text + image)."""
import numpy as np
import torch

# train_vq64_round2 lives in scripts/; the driver adds scripts/ to sys.path before import.
from train_vq64_round2 import _forward_with_block_orders


def _expand_block_order_to_tokens(order_blocks, block_len, device):
    """order_blocks: (b, N) int -> token order (b, N*block_len) revealing each block's tokens."""
    b, N = order_blocks.shape
    base = torch.arange(block_len, device=device)
    tok = order_blocks[:, :, None] * block_len + base[None, None, :]
    return tok.reshape(b, N * block_len)


@torch.no_grad()
def extract_image_block_hidden(model, tokens, block_len, n_blocks, device, chunk_size=16,
                               token_order=None):
    """Returns (n, N, E) full-context block hidden = mean over each block's token hiddens.
    token_order None -> identity raster reveal (physical frame). Mirrors extract_oracle_hidden."""
    model.eval()
    n_total = tokens.shape[0]
    if token_order is None:
        ident = torch.arange(n_blocks, device=device)
    chunks = []
    for i in range(0, n_total, chunk_size):
        idx = tokens[i:i + min(chunk_size, n_total - i)].to(device)
        nc = idx.shape[0]
        order_blocks = (token_order[i:i + nc] if token_order is not None
                        else ident.unsqueeze(0).expand(nc, -1)).to(device)
        tok_order = _expand_block_order_to_tokens(order_blocks, block_len, device)
        out = model.forward_fn(idx, tok_order, return_hidden=True, hidden_return_mode="original")
        h_tok = out[2][:, 1:, :]                                    # drop [None]; (nc, T, E)
        E = h_tok.shape[-1]
        h_blk = h_tok.reshape(nc, n_blocks, block_len, E).mean(dim=2)  # (nc, N, E)
        chunks.append(h_blk.cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def nll_under_order_image(model, tokens, phys_block_order, block_len, device,
                          fixed_token_perm=None, inv_block_perm=None, batch_size=16):
    """Token-avg teacher-forced NLL with every sample revealed in the same phys block order."""
    model.eval()
    N = int(phys_block_order.shape[0])
    order_t = torch.as_tensor(np.asarray(phys_block_order), dtype=torch.long)
    total, ntok = 0.0, 0
    for s in range(0, tokens.shape[0], batch_size):
        x = tokens[s:s + batch_size].to(device)
        orders = order_t.unsqueeze(0).expand(x.shape[0], -1).contiguous().to(device)
        loss = _forward_with_block_orders(model, x, orders,
                                          fixed_token_perm=fixed_token_perm,
                                          inv_block_perm=inv_block_perm)
        total += float(loss.item()) * x.shape[0]
        ntok += x.shape[0]
    return total / ntok


def nll_under_order_text(model, idx_eval_model, model_block_order, block_len, device,
                         batch_size=16):
    """Token-avg teacher-forced NLL on text; model_block_order is in MODEL block frame.
    Reuses train_clean_aogpt.compute_token_ce + expand_model_blocks_to_token_order."""
    from train_clean_aogpt import compute_token_ce, expand_model_blocks_to_token_order
    model.eval()
    device_obj = torch.device(device) if isinstance(device, str) else device
    order_t = torch.as_tensor(np.asarray(model_block_order), dtype=torch.long)
    total, ntok = 0.0, 0
    with torch.no_grad():
        for s in range(0, idx_eval_model.size(0), batch_size):
            idx = idx_eval_model[s:s + batch_size].to(device_obj)
            orders = order_t.unsqueeze(0).expand(idx.size(0), -1).to(device_obj)
            token_orders = expand_model_blocks_to_token_order(orders, block_len).to(device_obj)
            token_losses, _ = compute_token_ce(model, idx, token_orders, device_obj)
            total += float(token_losses.float().sum().item())
            ntok += int(token_losses.numel())
    return total / ntok
