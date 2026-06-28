"""P2: disambiguate the L0 physical-order signal source (B fixed-layout map vs
C content-dependent). Measures content-dependence of the carrier's attention/B,
NOT the order output (tautological under fixed layout). See spec
docs/superpowers/specs/2026-06-28-physical-signal-source-disambiguation-design.md."""
import pathlib, sys
import numpy as np, torch

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from neural_readout.extract_b import _load_model_and_chunks
from analyses.canonical_reanalysis import random_reveal_orders
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec
from none_separated_block_graph import build_none_separated_B, rollout_by_method, discovery_metrics

@torch.no_grad()
def carrier_b65_per_text(ckpt_path, layer, head, M=24, n_reveals=8, fixed_reveal_seed=0,
                         device="cpu", return_halves=False):
    """Per-text carrier B65 at fixed layout + shared reveal orders.

    When return_halves=True, also returns (halfA_list, halfB_list): per-text B65
    built from the FIRST n_reveals//2 reveals vs the REMAINING reveals (two disjoint
    halves) so callers can estimate the within-text sampling-noise floor.
    When return_halves=False (default), returns (B_list, tau_list) unchanged.
    """
    if n_reveals < 1:
        raise ValueError("n_reveals must be at least 1")
    if return_halves and n_reveals < 2:
        raise ValueError("n_reveals must be at least 2 when return_halves=True")

    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)   # SHARED across texts
    half = n_reveals // 2
    B_list, tau_list = [], []
    halfA_list, halfB_list = [], []
    for t in range(M):
        A_acc = None
        A_accA = None   # first half
        A_accB = None   # second half
        for ri, rev in enumerate(reveals):
            po = torch.from_numpy(rev[None, :]).to(dev)
            _, _, attn_list = model.forward_fn(chunks[t:t+1].to(dev), po, return_attentions=True)
            attn = torch.stack(attn_list, 0).cpu().numpy()[:, 0]   # (L,H,257,257)
            A = _attn_to_A_block_loss_aligned_with_none_vec(attn, rev, inv)  # (L,H,64,65)
            A_acc = A.astype(np.float64) if A_acc is None else A_acc + A
            if return_halves:
                if ri < half:
                    A_accA = A.astype(np.float64) if A_accA is None else A_accA + A
                else:
                    A_accB = A.astype(np.float64) if A_accB is None else A_accB + A
        B = build_none_separated_B((A_acc / n_reveals)[layer, head])
        B_list.append(B)
        tau_list.append(float(discovery_metrics(rollout_by_method(B, "C-D+L"))["tau_vs_l2r"]))
        if return_halves:
            nA = max(half, 1); nB = max(n_reveals - half, 1)
            halfA_list.append(build_none_separated_B((A_accA / nA)[layer, head]))
            halfB_list.append(build_none_separated_B((A_accB / nB)[layer, head]))
    if return_halves:
        return B_list, tau_list, halfA_list, halfB_list
    return B_list, tau_list


def valid_edge_mask(n=65):
    m = np.zeros((n, n), dtype=bool)
    m[0, 1:] = True                                   # None -> content
    for i in range(1, n):
        for j in range(1, i):
            m[i, j] = True                            # content causal lower-tri
    return m

def row_normalize_l1(B, mask, eps=1e-9):
    B = np.asarray(B, dtype=np.float64) * mask
    mass = np.abs(B).sum(axis=1, keepdims=True) + eps
    return B / mass


def _random_valid_B(rng, n=65):
    m = valid_edge_mask(n)
    B = np.zeros((n, n)); B[m] = rng.random(int(m.sum()))
    return B

def synthetic_content_invariant(M, seed=0):
    B = _random_valid_B(np.random.default_rng(seed))
    return [B.copy() for _ in range(M)]

def synthetic_content_randomized(M, seed=0):
    return [_random_valid_B(np.random.default_rng(seed + i)) for i in range(M)]

def _stack(B_list, mask, normalize):
    rows = []
    for B in B_list:
        Bn = row_normalize_l1(B, mask) if normalize else (np.asarray(B, float) * mask)
        rows.append(Bn[mask])
    return np.stack(rows)                                   # (M, n_valid)

def cross_text_variance(B_list, mask, normalize=True):
    X = _stack(B_list, mask, normalize)
    return float(X.var(axis=0).mean())

def pairwise_similarity(B_list, mask, normalize=True):
    X = _stack(B_list, mask, normalize)
    M = len(X); sims = []
    for a in range(M):
        for b in range(a + 1, M):
            r = np.corrcoef(X[a], X[b])[0, 1]
            if np.isfinite(r):
                sims.append(r)
    return float(np.mean(sims)) if sims else 1.0

def within_text_noise_floor(halfA_list, halfB_list, mask, normalize=True):
    XA = _stack(halfA_list, mask, normalize)
    XB = _stack(halfB_list, mask, normalize)
    # per-text per-edge half-difference variance; 0.5*mean((a-b)^2) ~ single-estimate var
    return float((0.5 * (XA - XB) ** 2).mean())

def content_variance(B_list, halfA_list, halfB_list, mask, normalize=True):
    cv = cross_text_variance(B_list, mask, normalize)
    floor = within_text_noise_floor(halfA_list, halfB_list, mask, normalize)
    return float(max(0.0, cv - floor))


def slot_only_r2(B_list, mask, n_train=None, normalize=True):
    X = _stack(B_list, mask, normalize)                     # (M, n_valid)
    M = len(X)
    n_train = n_train if n_train is not None else M // 2
    B_hat = X[:n_train].mean(axis=0)                        # content-free slot-pair table
    test = X[n_train:]
    if len(test) == 0:
        return float("nan")
    # Per-edge R²: for each valid edge j, how well does B_hat[j] predict the test values?
    # Averaged over edges, this isolates between-text variance (not between-edge variance).
    ss_res_per = ((test - B_hat) ** 2).mean(axis=0)         # (n_valid,)
    test_mean = test.mean(axis=0)                            # (n_valid,)
    ss_tot_per = ((test - test_mean) ** 2).mean(axis=0) + 1e-12  # (n_valid,)
    r2_per = 1.0 - ss_res_per / ss_tot_per
    return float(r2_per.mean())


def random_token_chunk(chunk, vocab_size, rng):
    return torch.from_numpy(rng.integers(0, vocab_size, size=tuple(chunk.shape))).to(chunk.dtype)


def block_swap_chunk(chunk, swaps, block_len=4):
    out = chunk.clone()
    for a, b in swaps:
        sa, sb = a * block_len, b * block_len
        tmp = out[sa:sa+block_len].clone()
        out[sa:sa+block_len] = out[sb:sb+block_len]
        out[sb:sb+block_len] = tmp
    return out

def cross_sample_replace(chunk, donor, blocks, block_len=4):
    out = chunk.clone()
    for b in blocks:
        s = b * block_len
        out[s:s+block_len] = donor[s:s+block_len]
    return out


def carrier_valid_filter(B_list, tau_list, thr=0.9):
    idx = [i for i, t in enumerate(tau_list) if abs(t) >= thr]
    return [B_list[i] for i in idx], idx


def synthetic_ascending_B(n=65):
    # Chain: None->1->2->...->64 in the upper triangle (B[u,v]=1 for v=u+1).
    # L-term propagates the chain and CDL rolls out [0,1,...,63], tau=+1.0.
    B = np.zeros((n, n))
    B[0, 1] = 1.0               # None -> block1
    for i in range(1, n - 1):
        B[i, i + 1] = 1.0       # block_i -> block_{i+1} (upper triangle)
    return B
