"""Loss-aligned model-frame attention→block-aggregate (ported, function-level).

Only ``_attn_to_A_block_loss_aligned_with_none_model_vec`` is ported here (the
strict label-free extraction used by l0_strict65). All admin model/data-loading
helpers from the original module are intentionally dropped.
"""
import numpy as np

from orderhead_v3.constants import SEQ_LEN, N, BLOCK_LEN


def _attn_to_A_block_loss_aligned_with_none_model_vec(
    attn,
    reveal_tokens,
    seq_len=SEQ_LEN,
    num_blocks=N,
    block_len=BLOCK_LEN,
):
    """Loss-aligned AR map in model-frame coordinates, [None] separate, NO inv_perm.

    Node 0 is [None], node 1+i is model block i. inv_perm is never used here.

    Returns:
        A: (..., num_blocks, num_blocks + 1), columns = [None], model_block_0..N-1.
    """
    attn = np.asarray(attn, dtype=np.float64)
    lead = attn.shape[:-2]
    K = int(np.prod(lead)) if lead else 1
    a = attn.reshape(K, seq_len + 1, seq_len + 1)[:, :seq_len, :seq_len]

    reveal_tokens = np.asarray(reveal_tokens, dtype=np.int64)
    if reveal_tokens.shape[0] != seq_len:
        raise ValueError(f"reveal_tokens must have length {seq_len}, got {reveal_tokens.shape[0]}")

    # ── model-block labels, NO inv_perm ──
    model_blocks = reveal_tokens // block_len
    query_labels = model_blocks

    source_labels = np.empty(seq_len, dtype=np.int64)
    source_labels[0] = 0                                 # [None] → source node 0
    key_model_blocks = reveal_tokens[:-1] // block_len
    source_labels[1:] = 1 + key_model_blocks             # node 1+i = model block i

    query_counts = np.bincount(query_labels, minlength=num_blocks).astype(np.float64)
    source_counts = np.bincount(source_labels, minlength=num_blocks + 1).astype(np.float64)

    Sq = np.zeros((num_blocks, seq_len), dtype=np.float64)
    Sq[query_labels, np.arange(seq_len)] = 1.0
    Sq = Sq / np.maximum(query_counts[:, None], 1.0)

    Sk = np.zeros((num_blocks + 1, seq_len), dtype=np.float64)
    Sk[source_labels, np.arange(seq_len)] = 1.0
    Sk = Sk / np.maximum(source_counts[:, None], 1.0)

    A = np.einsum("bt,ktu,cu->kbc", Sq, a, Sk, optimize=True)
    A = A.astype(np.float32, copy=False)
    return A.reshape(lead + (num_blocks, num_blocks + 1)) if lead else A[0]
