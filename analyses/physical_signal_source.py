"""P2: disambiguate the L0 physical-order signal source (B fixed-layout map vs
C content-dependent). Measures content-dependence of the carrier's attention/B,
NOT the order output (tautological under fixed layout). See spec
docs/superpowers/specs/2026-06-28-physical-signal-source-disambiguation-design.md."""
import numbers
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
    if return_halves and n_reveals % 2:
        raise ValueError("n_reveals must be even when return_halves=True")

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
    B = np.where(mask, np.asarray(B, dtype=np.float64), 0.0)
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
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a 2D boolean array")
    if not mask.any():
        raise ValueError("mask must select at least one value")
    B_list = list(B_list)
    if not B_list:
        raise ValueError("B_list must not be empty")
    rows = []
    for i, B in enumerate(B_list):
        B = np.asarray(B, dtype=np.float64)
        if B.shape != mask.shape:
            raise ValueError(
                f"B_list[{i}] shape {B.shape} does not match mask shape {mask.shape}")
        if not np.isfinite(B[mask]).all():
            raise ValueError(f"B_list[{i}] contains non-finite values selected by mask")
        Bn = row_normalize_l1(B, mask) if normalize else (B * mask)
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
    if len(halfA_list) != len(halfB_list):
        raise ValueError("halfA_list and halfB_list must have equal length")
    XA = _stack(halfA_list, mask, normalize)
    XB = _stack(halfB_list, mask, normalize)
    # Equal independent halves: Var((A+B)/2) = 0.25 Var(A-B). Centering
    # across texts removes half-specific common bias from shared reveal orders.
    return float(0.25 * (XA - XB).var(axis=0, ddof=0).mean())

def content_variance(B_list, halfA_list, halfB_list, mask, normalize=True):
    if not (len(B_list) == len(halfA_list) == len(halfB_list)):
        raise ValueError(
            "B_list, halfA_list, and halfB_list must have equal length")
    cv = cross_text_variance(B_list, mask, normalize)
    floor = within_text_noise_floor(halfA_list, halfB_list, mask, normalize)
    return float(max(0.0, cv - floor))


def slot_only_r2(B_list, mask, n_train=None, normalize=True):
    M = len(B_list)
    if M < 2:
        raise ValueError("slot_only_r2 requires at least 2 matrices")
    if n_train is None:
        n_train = M // 2
    elif isinstance(n_train, bool) or not isinstance(n_train, (int, np.integer)):
        raise ValueError("n_train must be an integer")
    n_train = int(n_train)
    if not 1 <= n_train < M:
        raise ValueError(f"n_train must satisfy 1 <= n_train < {M}")
    X = _stack(B_list, mask, normalize)                     # (M, n_valid)
    B_hat = X[:n_train].mean(axis=0)                        # content-free slot-pair table
    test = X[n_train:]
    # Global held-out R² over all held-out text × valid-edge entries.
    ss_res = ((test - B_hat) ** 2).sum()
    ss_tot = ((test - test.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def random_token_chunk(chunk, vocab_size, rng):
    """Return an independent, uniformly randomized token tensor.

    ``rng`` is deliberately an explicit NumPy Generator so callers control
    reproducibility and normal generator-state advancement.
    """
    if not isinstance(chunk, torch.Tensor):
        raise TypeError("chunk must be a torch.Tensor")
    if chunk.numel() == 0:
        raise ValueError("chunk must be nonempty")
    try:
        dtype_info = torch.iinfo(chunk.dtype)
    except TypeError as exc:
        raise TypeError("chunk must have an integral, non-bool token dtype") from exc

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, numbers.Integral):
        raise TypeError("vocab_size must be a non-bool integer")
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if vocab_size - 1 > dtype_info.max:
        raise ValueError(
            f"token range [0, {vocab_size}) is not representable by {chunk.dtype}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    values = rng.integers(0, vocab_size, size=tuple(chunk.shape))
    return torch.as_tensor(np.asarray(values), dtype=chunk.dtype, device=chunk.device)


def _validate_block_chunk(chunk, block_len):
    if not isinstance(chunk, torch.Tensor):
        raise TypeError("chunk must be a torch.Tensor")
    if chunk.ndim != 1 or chunk.numel() == 0:
        raise ValueError("chunk must be one-dimensional and nonempty")
    if (isinstance(block_len, bool)
            or not isinstance(block_len, numbers.Integral)):
        raise TypeError("block_len must be a non-bool integer")
    block_len = int(block_len)
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    if chunk.numel() % block_len:
        raise ValueError("chunk length must be divisible by block_len")
    return block_len, chunk.numel() // block_len


def _validate_block_index(index, n_blocks, name):
    if isinstance(index, bool) or not isinstance(index, numbers.Integral):
        raise TypeError(f"{name} must be a non-bool integer")
    index = int(index)
    if not 0 <= index < n_blocks:
        raise ValueError(f"{name} must be in [0, {n_blocks})")
    return index


def block_swap_chunk(chunk, swaps, block_len=4):
    block_len, n_blocks = _validate_block_chunk(chunk, block_len)
    validated = []
    used = set()
    for pair_index, pair in enumerate(swaps):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"swaps[{pair_index}] must be a length-2 pair")
        a = _validate_block_index(pair[0], n_blocks, f"swaps[{pair_index}][0]")
        b = _validate_block_index(pair[1], n_blocks, f"swaps[{pair_index}][1]")
        if a == b:
            raise ValueError("swap endpoints must be different")
        if a in used or b in used:
            raise ValueError("swap endpoints must be disjoint; endpoint reused")
        used.update((a, b))
        validated.append((a, b))

    out = chunk.clone()
    for a, b in validated:
        sa, sb = a * block_len, b * block_len
        out[sa:sa+block_len] = chunk[sb:sb+block_len]
        out[sb:sb+block_len] = chunk[sa:sa+block_len]
    return out


def cross_sample_replace(chunk, donor, blocks, block_len=4):
    block_len, n_blocks = _validate_block_chunk(chunk, block_len)
    if not isinstance(donor, torch.Tensor):
        raise TypeError("donor must be a torch.Tensor")
    if donor.shape != chunk.shape:
        raise ValueError("donor must have the same shape as chunk")
    if donor.dtype != chunk.dtype:
        raise ValueError("donor must have the same dtype as chunk")
    if donor.device != chunk.device:
        raise ValueError("donor must be on the same device as chunk")

    validated = []
    used = set()
    for position, block in enumerate(blocks):
        block = _validate_block_index(block, n_blocks, f"blocks[{position}]")
        if block in used:
            raise ValueError("blocks must be unique; duplicate block")
        used.add(block)
        validated.append(block)

    out = chunk.clone()
    for block in validated:
        start = block * block_len
        out[start:start+block_len] = donor[start:start+block_len]
    return out


def carrier_valid_filter(B_list, tau_list, thr=0.9):
    if len(B_list) != len(tau_list):
        raise ValueError("B_list and tau_list must have the same length")
    if (isinstance(thr, bool) or not isinstance(thr, numbers.Real)
            or not np.isfinite(thr) or not 0 <= thr <= 1):
        raise ValueError("thr must be a finite number in [0, 1]")
    for i, tau in enumerate(tau_list):
        if (isinstance(tau, bool) or not isinstance(tau, numbers.Real)
                or not np.isfinite(tau) or not -1 <= tau <= 1):
            raise ValueError(
                f"tau_list[{i}] must be a finite number in [-1, 1]")
    idx = [i for i, t in enumerate(tau_list) if abs(t) >= thr]
    return [B_list[i] for i in idx], idx


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a non-bool integer")
    value = int(value)
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return value


@torch.no_grad()
def relayout_diagnostic(ckpt_path, layer, carrier_heads, K=4, M=8,
                        n_reveals=8, fixed_reveal_seed=0, device="cpu"):
    """Run the E3 anchor-versus-relayout supporting diagnostic.

    ``K`` is the *total* number of layouts generated by ``make_layouts``:
    layout 0 is the training-layout anchor and layouts 1 through ``K - 1``
    are relayouts. Thus ``K`` must be at least 2. The same reveal orders are
    reused for every text and layout. Each layout's score is the signed
    physical-frame tau of the carrier head with largest absolute tau, using
    the canonical loss-aligned extraction, strict B65 construction, and
    C-D+L rollout.

    ``drop`` is positive when relayouting weakens the signal and is defined as
    ``anchor_tau - relayout_mean_tau``. E3 supports interpretation only; it is
    not used directly in the source verdict.
    """
    K = _positive_int(K, "K", minimum=2)
    M = _positive_int(M, "M")
    n_reveals = _positive_int(n_reveals, "n_reveals")
    if isinstance(layer, bool) or not isinstance(layer, numbers.Integral):
        raise TypeError("layer must be a non-bool integer")
    layer = int(layer)
    if layer < 0:
        raise ValueError("layer must be nonnegative")

    carrier_heads = list(carrier_heads)
    if not carrier_heads:
        raise ValueError("carrier_heads must not be empty")
    validated_heads = []
    for index, head in enumerate(carrier_heads):
        if isinstance(head, bool) or not isinstance(head, numbers.Integral):
            raise TypeError(f"carrier_heads[{index}] must be a non-bool integer")
        head = int(head)
        if head < 0:
            raise ValueError(f"carrier_heads[{index}] must be nonnegative")
        validated_heads.append(head)
    if len(set(validated_heads)) != len(validated_heads):
        raise ValueError("carrier_heads must be unique")

    from analyses.position_prior_decomp import (
        clean_perm_from_layout, make_layouts, relayout_chunks)

    model, chunks, training_clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, M, seed=0, device=device, split="train")
    reveals = random_reveal_orders(n_reveals, fixed_reveal_seed)
    layouts = make_layouts(training_clean_perm, K=K)
    if len(layouts) != K:
        raise RuntimeError(f"make_layouts returned {len(layouts)} layouts, expected {K}")

    def best_physical_tau(layout_chunks, inv_perm):
        A_acc = None
        for text_index in range(M):
            for reveal in reveals:
                po = torch.from_numpy(np.asarray(reveal)[None, :]).to(dev)
                _, _, attn_list = model.forward_fn(
                    layout_chunks[text_index:text_index + 1].to(dev), po,
                    return_attentions=True)
                attn = torch.stack(attn_list, dim=0).cpu().numpy()[:, 0]
                A = _attn_to_A_block_loss_aligned_with_none_vec(
                    attn, reveal, inv_perm)
                A_acc = A.astype(np.float64) if A_acc is None else A_acc + A

        A_mean = A_acc / (M * len(reveals))
        if layer >= A_mean.shape[0]:
            raise ValueError(
                f"layer {layer} is out of range for {A_mean.shape[0]} layers")
        n_heads = A_mean.shape[1]
        if any(head >= n_heads for head in validated_heads):
            raise ValueError(
                f"carrier_heads must be in [0, {n_heads}); got {validated_heads}")
        taus = []
        for head in validated_heads:
            B65 = build_none_separated_B(A_mean[layer, head])
            result = rollout_by_method(B65, "C-D+L")
            tau = float(discovery_metrics(result)["tau_vs_l2r"])
            if not np.isfinite(tau):
                raise RuntimeError(
                    f"non-finite tau for layer {layer}, head {head}")
            taus.append(tau)
        return float(max(taus, key=abs))

    anchor_inv = training_clean_perm.inv_perm_model_to_phys.cpu().numpy()
    anchor_tau = best_physical_tau(chunks, anchor_inv)

    relayout_taus = []
    for layout in layouts[1:]:
        layout_perm = clean_perm_from_layout(layout)
        layout_chunks = relayout_chunks(
            chunks, training_clean_perm, layout_perm)
        layout_inv = layout_perm.inv_perm_model_to_phys.cpu().numpy()
        relayout_taus.append(best_physical_tau(layout_chunks, layout_inv))

    relayout_mean_tau = float(np.mean(relayout_taus))
    return {
        "anchor_tau": anchor_tau,
        "relayout_mean_tau": relayout_mean_tau,
        "drop": anchor_tau - relayout_mean_tau,
    }


def synthetic_ascending_B(n=65):
    # Chain: None->1->2->...->64 in the upper triangle (B[u,v]=1 for v=u+1).
    # L-term propagates the chain and CDL rolls out [0,1,...,63], tau=+1.0.
    B = np.zeros((n, n))
    B[0, 1] = 1.0               # None -> block1
    for i in range(1, n - 1):
        B[i, i + 1] = 1.0       # block_i -> block_{i+1} (upper triangle)
    return B
