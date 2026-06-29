"""
Attention-driven AOGPT–ON alternating co-training.

Phase 1: BC-train ON on mixed025-NN teacher (A_mix input).
Phase 2: AOGPT training with ON-sampled orders, alpha warmup 0→0.7.
Phase 3: periodic refresh — extract attention, update A_global (EMA),
          regenerate teacher, fine-tune ON.

This is NOT end-to-end RL. ON is updated only via BC on NN teacher paths.
"""
import os, sys, time, math, argparse, json
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ──
_current = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_current, "..", "AO-GPT-MDM"))
sys.path.insert(0, os.path.expanduser("~/ych/nanogpt-learned-order"))
sys.path.insert(0, _current)

from AOGPT import AOGPTConfig, AOGPT
from order_utils import (
    expand_block_orders_to_token_orders,
    invert_permutation,
)
from order_network import CrossAttentionOrderNetwork, masks_to_revealed_bool

# ── Constants ──
AO_GPT_CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
A_PATH = "block_lo_arm_order_network/probe_results/A64_from_N64_50k_10k.npy"
A_GLOBAL_PATH = "block_lo_arm_order_network/probe_results/attention_curriculum_real_diag/A_global_train_first8000_n64.npy"
TEACHER_PATH = "block_lo_arm_order_network/probe_results/attention_curriculum_full10k/mixed_lam025_orders.npy"
TOKENS_PATH = "block_lo_arm_order_network/probe_results/A32_from_N64_10k.tokens.npy"
OUT_BASE = "block_lo_arm_order_network/probe_results/cotrain_aogpt_on_n64"

NUM_BLOCKS = 64
SEQ_LEN = 256
TOKENS_PER_BLOCK = SEQ_LEN // NUM_BLOCKS  # 4
LAMBDA_MIX = 0.25


# ══════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════

def compute_amix(A_x, A_global, lam=LAMBDA_MIX):
    """Per-sample: A_mix = lam * A_x + (1-lam) * A_global.
    A_x: (S, N, N) float32, A_global: (N, N) float32.
    Returns (S, N, N) float32."""
    return (lam * A_x + (1.0 - lam) * A_global.astype(np.float32)).astype(np.float32)


def compute_teacher_paths(A_mix_all):
    """NN teacher paths from A_mix via inlined old_nn_greedy.
    W_mix = 0.5*(A_mix + A_mix.T), zero diag.
    Returns (paths, weights): paths (S, N) int64, weights (S,) float32."""
    S, N = A_mix_all.shape[0], A_mix_all.shape[1]
    paths = np.zeros((S, N), dtype=np.int64)
    weights = np.zeros(S, dtype=np.float32)

    for s in range(S):
        A = A_mix_all[s].astype(np.float64)
        np.fill_diagonal(A, 0.0)
        W = 0.5 * (A + A.T)
        np.fill_diagonal(W, 0.0)

        starts = np.argsort(W.sum(axis=1))[:2]
        best_order = None
        best_w = -np.inf
        for start in starts:
            order = np.zeros(N, dtype=np.int64)
            visited = np.zeros(N, dtype=bool)
            cur = int(start)
            for t in range(N):
                order[t] = cur
                visited[cur] = True
                if t == N - 1:
                    break
                w_cur = W[cur].copy()
                w_cur[visited] = -np.inf
                cur = int(np.argmax(w_cur))
            pw = sum(float(W[order[i], order[i + 1]]) for i in range(N - 1))
            if pw > best_w:
                best_w = pw
                best_order = order

        paths[s] = best_order
        weights[s] = best_w

        if (s + 1) % 2000 == 0:
            print(f"  teacher path {s+1}/{S}", flush=True)

    return paths, weights


@torch.no_grad()
def sample_on_order_n64(on_model, A_mix, temperature=1.0, top_k=4):
    """Autoregressive sample N=64 block order from ON.
    A_mix: (B, N, N) on device. Returns (B, N) LongTensor."""
    B, N = A_mix.shape[0], A_mix.shape[1]
    device = A_mix.device
    visited = torch.zeros(B, dtype=torch.long, device=device)
    last = torch.zeros(B, dtype=torch.long, device=device)
    orders = torch.zeros(B, N, dtype=torch.long, device=device)

    for t in range(N):
        logits = on_model(A_mix, visited, last)
        visited_bool = masks_to_revealed_bool(visited, N)
        logits = logits.masked_fill(visited_bool, float('-inf'))

        if top_k > 0 and top_k < N - t:
            topk_vals, topk_idx = torch.topk(logits, k=min(top_k, N - t), dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(-1, topk_idx, topk_vals)
            logits = mask

        if temperature > 1e-8:
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            chosen = torch.multinomial(probs, 1).squeeze(-1)
        else:
            chosen = logits.argmax(dim=-1)

        orders[:, t] = chosen
        visited = visited | (1 << chosen)
        last = chosen

    return orders


@torch.no_grad()
def greedy_order_n64(on_model, A_mix):
    """Greedy (argmax) autoregressive decode. For eval comparison."""
    return sample_on_order_n64(on_model, A_mix, temperature=0.0, top_k=0)


def phys_block_order_to_model_token_order(block_orders, block_perm):
    """Convert physical block order (B, 64) to model token order (B, 256).
    block_perm: (64,) — model position → physical block mapping from ckpt.
    """
    B, N = block_orders.shape
    device = block_orders.device

    inv_bp = torch.empty_like(block_perm)
    inv_bp[block_perm] = torch.arange(N, device=device)

    model_blocks = inv_bp[block_orders]  # (B, N) — block indices in model coords
    token_order = (model_blocks.unsqueeze(-1) * TOKENS_PER_BLOCK +
                   torch.arange(TOKENS_PER_BLOCK, device=device)).reshape(B, -1)
    return token_order


def compute_diag_tau(a, b):
    """Kendall tau between two orders (1D arrays)."""
    n = len(a)
    pos_a = np.empty(n, dtype=np.int64)
    pos_a[a] = np.arange(n)
    pairs = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if (pos_a[b[i]] - pos_a[b[j]]) * (j - i) > 0:
                disc += 1
    return (pairs - 2 * disc) / max(pairs, 1)


# ══════════════════════════════════════════════════════════════════════
# Phase 1: BC-train ON on teacher paths
# ══════════════════════════════════════════════════════════════════════

def phase1_bc_train(on_model, A_mix_train, teacher_paths, A_mix_val, teacher_paths_val,
                    batch_size, n_epochs, lr, device, output_dir):
    """BC training loop. Returns best checkpoint path and diagnostics."""
    N = A_mix_train.shape[1]
    A_train_t = torch.from_numpy(A_mix_train).float().to(device)
    paths_train_t = torch.from_numpy(teacher_paths.astype(np.int64)).to(device)
    A_val_t = torch.from_numpy(A_mix_val).float().to(device)
    paths_val_t = torch.from_numpy(teacher_paths_val.astype(np.int64)).to(device)

    optimizer = torch.optim.AdamW(on_model.parameters(), lr=lr)
    n_train = A_mix_train.shape[0]
    best_val_acc = 0.0
    best_ckpt = os.path.join(output_dir, "phase1_on_bc_best.pt")

    for epoch in range(1, n_epochs + 1):
        # Train
        on_model.train()
        perm = torch.randperm(n_train)
        total_loss, total_correct, total_steps = 0.0, 0, 0
        for start in range(0, n_train, batch_size):
            idx_b = perm[start:start + batch_size]
            A_b = A_train_t[idx_b]
            paths_b = paths_train_t[idx_b]
            B = A_b.shape[0]

            visited = torch.zeros(B, dtype=torch.long, device=device)
            last = torch.zeros(B, dtype=torch.long, device=device)
            batch_loss = 0.0
            batch_correct = 0

            for t in range(N):
                logits = on_model(A_b, visited, last)
                targets = paths_b[:, t]
                batch_loss += F.cross_entropy(logits, targets)
                pred = logits.argmax(dim=-1)
                batch_correct += (pred == targets).sum().item()
                visited = visited | (1 << targets)
                last = targets

            loss = batch_loss / N
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(on_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            total_correct += batch_correct
            total_steps += B * N

        train_loss = total_loss / n_train
        train_acc = total_correct / max(total_steps, 1)

        # Val
        on_model.eval()
        val_loss, val_correct, val_steps = 0.0, 0, 0
        B_val = A_val_t.shape[0]
        visited = torch.zeros(B_val, dtype=torch.long, device=device)
        last = torch.zeros(B_val, dtype=torch.long, device=device)
        with torch.no_grad():
            for t in range(N):
                logits = on_model(A_val_t, visited, last)
                targets = paths_val_t[:, t]
                val_loss += F.cross_entropy(logits, targets, reduction='sum').item()
                pred = logits.argmax(dim=-1)
                val_correct += (pred == targets).sum().item()
                visited = visited | (1 << targets)
                last = targets
        val_loss = val_loss / (B_val * N)
        val_acc = val_correct / (B_val * N)

        is_best = "*" if val_acc > best_val_acc else " "
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": on_model.state_dict(), "val_acc": val_acc}, best_ckpt)

        print(f"  epoch {epoch:>3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{is_best}", flush=True)

    # Load best
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    on_model.load_state_dict(ckpt["model_state_dict"])
    on_model.eval()
    print(f"  Best val_acc={best_val_acc:.4f} loaded from {best_ckpt}", flush=True)

    # Diagnostics
    diag = phase1_diagnostics(on_model, A_val_t, paths_val_t, device)
    diag["best_val_acc"] = best_val_acc
    return best_ckpt, diag


@torch.no_grad()
def phase1_diagnostics(on_model, A_val_t, paths_val_t, device):
    """Compute Phase 1 quality metrics."""
    B, N = A_val_t.shape[0], A_val_t.shape[1]

    # Legal permutation rate (greedy decode must produce permutation)
    greedy_orders = greedy_order_n64(on_model, A_val_t)
    legal = torch.tensor([len(set(greedy_orders[b].tolist())) == N for b in range(B)])
    legal_rate = legal.float().mean().item()

    # Sequence tau vs teacher
    taus = []
    for b in range(B):
        taus.append(compute_diag_tau(greedy_orders[b].cpu().numpy(),
                                      paths_val_t[b].cpu().numpy()))
    mean_tau = float(np.mean(taus))

    # First-node accuracy
    first_correct = (greedy_orders[:, 0] == paths_val_t[:, 0]).float().mean().item()

    # Sample entropy (via sample_on_order_n64 with T=1.0, no top-k)
    sampled = sample_on_order_n64(on_model, A_val_t, temperature=1.0, top_k=0)
    first_nodes = sampled[:, 0].cpu().numpy()
    _, counts = np.unique(first_nodes, return_counts=True)
    probs = counts / counts.sum()
    sample_entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    # Sampled tau vs teacher
    sampled_taus = []
    for b in range(B):
        sampled_taus.append(compute_diag_tau(sampled[b].cpu().numpy(),
                                              paths_val_t[b].cpu().numpy()))
    mean_sampled_tau = float(np.mean(sampled_taus))

    return {
        "legal_permutation_rate": legal_rate,
        "greedy_tau_vs_teacher": mean_tau,
        "first_node_acc": first_correct,
        "sample_entropy": sample_entropy,
        "sampled_tau_vs_teacher": mean_sampled_tau,
    }


# ══════════════════════════════════════════════════════════════════════
# Phase 2 Evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_aogpt(model, val_dl, on_model, A_mix_val_t, block_perm, inv_perm,
                   device, max_seqs=200):
    """Evaluate AOGPT on val set. Returns dict of losses."""
    model.eval()
    losses = {"ar": [], "random": [], "on_order": [], "l2r": []}
    count = 0

    for batch in val_dl:
        if count >= max_seqs:
            break
        if isinstance(batch, (tuple, list)):
            tokens = batch[0]
        else:
            tokens = batch
        tokens = tokens.to(device)
        B = tokens.shape[0]

        # AR loss
        _, ar_loss = model(tokens, mode='AR')
        losses["ar"].append(ar_loss.item())

        # Random loss
        _, rand_loss = model(tokens, mode='Random')
        losses["random"].append(rand_loss.item())

        # ON-order loss: sample one order per seq from ON
        if on_model is not None and A_mix_val_t is not None:
            idx_start = count
            idx_end = min(count + B, A_mix_val_t.shape[0])
            amix_b = A_mix_val_t[idx_start:idx_end]
            if amix_b.shape[0] == B:
                on_block = sample_on_order_n64(on_model, amix_b, temperature=1.0, top_k=4)
                on_token = phys_block_order_to_model_token_order(on_block, block_perm)
                _, on_loss, _ = model.forward_fn(tokens, on_token, return_token_loss=True)
                losses["on_order"].append(on_loss.item())

        # L2R loss
        if inv_perm is not None:
            original_block = inv_perm.unsqueeze(0)
            l2r_token = expand_block_orders_to_token_orders(original_block, block_len=TOKENS_PER_BLOCK)
            l2r_token = l2r_token.expand(B, -1).to(device)
            _, l2r_loss, _ = model.forward_fn(tokens, l2r_token, return_token_loss=True)
            losses["l2r"].append(l2r_loss.item())

        count += B

    model.train()
    return {k: float(np.mean(v)) if v else float('nan') for k, v in losses.items()}


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Refresh
# ══════════════════════════════════════════════════════════════════════

def _build_token_order(rand_order, bp, n_blocks):
    """Build model-coordinate token order from random block permutation.

    rand_order: (n_blocks,) random permutation of [0..n_blocks-1], physical indices
    bp: (64,) numpy array, bp[phys64]=model64
    n_blocks: number of ON-action blocks (should be 64)
    Returns: (256,) LongTensor in model coordinates
    """
    tokens_per_block = SEQ_LEN // n_blocks
    n64_per_step = NUM_BLOCKS // n_blocks
    token_order = torch.zeros(SEQ_LEN, dtype=torch.long)
    for t in range(n_blocks):
        p = rand_order[t].item()
        for a in range(n64_per_step):
            phys_n64 = p * n64_per_step + a
            model_n64 = int(bp[phys_n64])
            for k in range(TOKENS_PER_BLOCK):
                pos = t * tokens_per_block + a * TOKENS_PER_BLOCK + k
                token_order[pos] = model_n64 * TOKENS_PER_BLOCK + k
    return token_order


@torch.no_grad()
def _extract_one_A_inline(model, idx_tokens, bp, model_to_phys, device):
    """Extract N64×N64 attention A matrix for one chunk.

    Inline version for BlockAOGPT. Returns (NUM_BLOCKS, NUM_BLOCKS) float32.
    """
    idx = idx_tokens.to(device).unsqueeze(0)  # (1, 256)

    # Random block order -> token order
    rand_order = torch.randperm(NUM_BLOCKS, device='cpu')
    token_order = _build_token_order(rand_order, bp, NUM_BLOCKS).to(device)
    token_order = token_order.unsqueeze(0)  # (1, 256)

    # Forward with attentions
    _, _, attn_list = model.forward_fn(idx, token_order, return_attentions=True)
    torch.cuda.synchronize(device)
    attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)

    L, H = attn_stack.shape[:2]

    # Select top-4 highest-variance heads
    head_vars = np.zeros(H)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]  # skip [None] row/col
        offdiag = content[:, ~np.eye(SEQ_LEN, dtype=bool)].reshape(L, SEQ_LEN, SEQ_LEN - 1)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-4:]
    avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)

    # Remap reveal->physical
    reveal_to_model = token_order[0].cpu().numpy()  # (256,) model token positions
    reveal_to_phys = model_to_phys[reveal_to_model]  # (256,) physical token positions

    attn_content = avg_attn[1:, 1:]  # (256, 256) content-content attention
    attn_phys = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
    for rq in range(SEQ_LEN):
        pq = reveal_to_phys[rq]
        for rk in range(SEQ_LEN):
            attn_phys[pq, reveal_to_phys[rk]] += attn_content[rq, rk]

    # Aggregate to NUM_BLOCKS×NUM_BLOCKS
    A = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float32)
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            i_s, i_e = i * TOKENS_PER_BLOCK, (i + 1) * TOKENS_PER_BLOCK
            j_s, j_e = j * TOKENS_PER_BLOCK, (j + 1) * TOKENS_PER_BLOCK
            A[i, j] = attn_phys[i_s:i_e, j_s:j_e].mean()

    # [None] signal
    none_attn = avg_attn[1:, 0]
    none_block = np.array([none_attn[i * TOKENS_PER_BLOCK:(i + 1) * TOKENS_PER_BLOCK].mean()
                           for i in range(NUM_BLOCKS)])
    A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)

    return A


def phase3_extract_attention(model, train_chunks, bp, model_to_phys, device,
                              n_chunks=None):
    """Extract N64 attention from current AOGPT on train chunks."""
    if n_chunks is None:
        n_chunks = len(train_chunks)
    n_chunks = min(n_chunks, len(train_chunks))

    A_new = np.zeros((n_chunks, NUM_BLOCKS, NUM_BLOCKS), dtype=np.float32)
    model.eval()
    t0 = time.time()

    for i in range(n_chunks):
        tokens = torch.tensor(train_chunks[i], dtype=torch.long)
        A_new[i] = _extract_one_A_inline(model, tokens, bp, model_to_phys, device)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n_chunks - i - 1) / max(rate, 1e-9)
            print(f"  extract A: {i+1}/{n_chunks} | {rate:.1f} seq/s | ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  extract A done in {elapsed:.1f}s ({n_chunks/max(elapsed,1e-9):.1f} seq/s)", flush=True)
    return A_new


def phase3_refresh(aogpt, on_model, train_chunks, bp, model_to_phys,
                   A_global_old, n_train, device, output_dir, round_idx,
                   bc_epochs=3, bc_lr=1e-3, extract_subset=None):
    """Full Phase 3 refresh cycle. Returns updated ON, A_mix, teacher_paths, A_global, diagnostics."""
    print(f"\n{'='*60}")
    print(f"Phase 3 Refresh — Round {round_idx}")
    print(f"{'='*60}", flush=True)

    # 1. Extract new attention
    print("Extracting attention from current AOGPT...", flush=True)
    A_x_new = phase3_extract_attention(aogpt, train_chunks, bp, model_to_phys, device,
                                        n_chunks=extract_subset)
    n_extracted = A_x_new.shape[0]
    A_x_new_train = A_x_new[:n_train] if n_train <= n_extracted else A_x_new

    # 2. Update A_global with EMA
    A_global_new_mean = A_x_new_train.mean(axis=0)
    beta = 0.9
    A_global_new = beta * A_global_old + (1.0 - beta) * A_global_new_mean
    drift = float(np.abs(A_global_new - A_global_old).mean())
    print(f"  A_global L1 drift: {drift:.6f}", flush=True)

    # 3. Compute new A_mix
    print("Computing A_mix and teacher paths...", flush=True)
    A_mix_new = compute_amix(A_x_new, A_global_new)
    teacher_new, teacher_weights = compute_teacher_paths(A_mix_new)

    # 4. Teacher drift: tau(old_teacher, new_teacher)
    # Need old teacher for comparison — load from disk or track
    old_teacher_path = os.path.join(output_dir, f"teacher_round{round_idx-1}.npy") if round_idx > 0 else None
    teacher_tau_drift = None
    if old_teacher_path and os.path.exists(old_teacher_path):
        old_teacher = np.load(old_teacher_path)
        n_compare = min(len(old_teacher), len(teacher_new))
        taus = [compute_diag_tau(old_teacher[i], teacher_new[i]) for i in range(n_compare)]
        teacher_tau_drift = float(np.mean(taus))
        print(f"  teacher_r vs teacher_{{r-1}} tau: {teacher_tau_drift:.4f}", flush=True)

    # First-node entropy of new teacher
    first_nodes = teacher_new[:n_extracted, 0]
    _, counts = np.unique(first_nodes, return_counts=True)
    probs = counts / counts.sum()
    first_entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    print(f"  new teacher first-node entropy: {first_entropy:.4f} (max=ln64=4.159)", flush=True)

    # Save new teacher
    np.save(os.path.join(output_dir, f"teacher_round{round_idx}.npy"), teacher_new)
    np.save(os.path.join(output_dir, f"A_mix_round{round_idx}.npy"), A_mix_new)
    np.save(os.path.join(output_dir, f"A_x_round{round_idx}.npy"), A_x_new)
    np.save(os.path.join(output_dir, f"A_global_round{round_idx}.npy"), A_global_new)

    # 5. Fine-tune ON on new teacher
    print(f"Fine-tuning ON on new teacher ({bc_epochs} epochs)...", flush=True)
    # Use first n_train for training, rest for val
    n_val = min(n_extracted - n_train, n_train // 4)
    if n_val <= 0:
        n_train_bc = int(n_extracted * 0.8)
        n_val = n_extracted - n_train_bc
    else:
        n_train_bc = n_train

    A_mix_train_bc = A_mix_new[:n_train_bc]
    teacher_train_bc = teacher_new[:n_train_bc]
    A_mix_val_bc = A_mix_new[n_train_bc:n_train_bc + n_val]
    teacher_val_bc = teacher_new[n_train_bc:n_train_bc + n_val]

    _, bc_diag = phase1_bc_train(
        on_model, A_mix_train_bc, teacher_train_bc,
        A_mix_val_bc, teacher_val_bc,
        batch_size=32, n_epochs=bc_epochs, lr=bc_lr,
        device=device, output_dir=output_dir,
    )

    # 6. Check teacher path weight vs random baseline
    # Random path weight on W_mix for reference
    W_mix_first = 0.5 * (A_mix_new[0] + A_mix_new[0].T)
    np.fill_diagonal(W_mix_first, 0.0)
    rand_order = np.random.permutation(NUM_BLOCKS)
    rand_weight = sum(float(W_mix_first[rand_order[i], rand_order[i + 1]]) for i in range(NUM_BLOCKS - 1))
    teacher_weight = float(teacher_weights[0])
    print(f"  teacher weight vs random (sample 0): {teacher_weight:.4f} vs {rand_weight:.4f} "
          f"(ratio={teacher_weight/max(rand_weight, 1e-9):.1f}x)", flush=True)

    diag = {
        "round": round_idx,
        "a_global_l1_drift": drift,
        "teacher_tau_drift": teacher_tau_drift,
        "first_node_entropy": first_entropy,
        "on_post_refresh_val_acc": bc_diag.get("best_val_acc", 0),
        "on_post_refresh_tau": bc_diag.get("greedy_tau_vs_teacher", 0),
        "on_post_refresh_legal_rate": bc_diag.get("legal_permutation_rate", 0),
        "on_post_refresh_sample_entropy": bc_diag.get("sample_entropy", 0),
    }
    for k, v in diag.items():
        if v is not None:
            print(f"  {k}: {v}", flush=True)

    return on_model, A_mix_new, teacher_new, A_global_new, diag


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--ao-gpt-ckpt", default=AO_GPT_CKPT)
    parser.add_argument("--a-matrices", default=A_PATH)
    parser.add_argument("--a-global", default=A_GLOBAL_PATH)
    parser.add_argument("--teacher-path", default=TEACHER_PATH)
    parser.add_argument("--tokens-path", default=TOKENS_PATH)
    parser.add_argument("--output-dir", default=OUT_BASE)
    # Phase control
    parser.add_argument("--max-total-steps", type=int, default=4500)
    parser.add_argument("--refresh-every", type=int, default=1500)
    parser.add_argument("--no-refresh", action="store_true",
                        help="Disable Phase 3 refresh (fixed-teacher baseline)")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase3-bc-epochs", type=int, default=3)
    parser.add_argument("--phase3-bc-lr", type=float, default=1e-3)
    parser.add_argument("--phase3-extract-subset", type=int, default=2000)
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--alpha-target", type=float, default=0.7)
    parser.add_argument("--alpha-warmup", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4)
    # ON model
    parser.add_argument("--d-edge", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    # Eval
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-eval-seqs", type=int, default=200)
    # Smoke test
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=100)
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # ── Smoke mode overrides ──
    if args.smoke:
        args.max_total_steps = 20
        args.refresh_every = 10
        args.phase1_epochs = 2
        args.phase3_bc_epochs = 1
        args.phase3_extract_subset = 10
        args.eval_interval = 5
        args.log_interval = 1
        print("SMOKE TEST MODE", flush=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    print("Loading data...", flush=True)
    A_all = np.load(args.a_matrices).astype(np.float32)
    A_global = np.load(args.a_global).astype(np.float32)
    n_total = A_all.shape[0]
    if args.smoke:
        n_total = args.smoke_samples
        A_all = A_all[:n_total]

    # Load or compute teacher
    if os.path.exists(args.teacher_path):
        teacher_all = np.load(args.teacher_path)
        if args.smoke:
            teacher_all = teacher_all[:n_total]
        print(f"Using precomputed teacher: {args.teacher_path} shape={teacher_all.shape}", flush=True)
        teacher_method = "mixed025_nn"
    else:
        raise FileNotFoundError(f"Teacher not found: {args.teacher_path}")

    print(f"  teacher_method = {teacher_method}")
    print(f"  base_A = {LAMBDA_MIX:.2f} * A_x + {1-LAMBDA_MIX:.2f} * A_global_train_only")
    print(f"  path_builder = NNBestEndpointPath")
    print(f"  A_all={A_all.shape}, A_global={A_global.shape}, n_total={n_total}", flush=True)

    # Train/val split: first 8000 train, rest val (matches A_global)
    n_train = min(8000, int(n_total * 0.8))
    n_val = n_total - n_train
    print(f"  train={n_train}, val={n_val}", flush=True)

    A_train = A_all[:n_train]
    A_val = A_all[n_train:]
    teacher_train = teacher_all[:n_train]
    teacher_val = teacher_all[n_train:]

    # Compute initial A_mix
    print("Computing initial A_mix...", flush=True)
    A_mix_train = compute_amix(A_train, A_global)
    A_mix_val = compute_amix(A_val, A_global)

    # ── Load AOGPT ──
    print(f"Loading AO-GPT from {args.ao_gpt_ckpt}...", flush=True)
    ckpt = torch.load(args.ao_gpt_ckpt, map_location=device, weights_only=False)
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(ckpt["model_args"]).items() if k in sig}
    aogpt = AOGPT(AOGPTConfig(**valid))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        clean = k.replace("_orig_mod.", "")
        if clean != k:
            sd[clean] = sd.pop(k)
    aogpt.load_state_dict(sd)
    aogpt.crop_block_size(SEQ_LEN)
    aogpt.to(device)
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long, device=device)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long, device=device)
    print(f"  AOGPT loaded, {sum(p.numel() for p in aogpt.parameters()):,} params", flush=True)

    # Build model_to_phys: model token position → physical token position (256,)
    inv_perm_np = inv_perm.cpu().numpy()  # inv_perm[model64]=phys64
    model_to_phys_np = np.zeros(SEQ_LEN, dtype=np.int64)
    bp_np = block_perm.cpu().numpy()  # bp[phys64]=model64
    for mp in range(SEQ_LEN):
        mb = mp // TOKENS_PER_BLOCK  # model N64 block (0..63)
        phys_n64 = int(inv_perm_np[mb])
        off = mp % TOKENS_PER_BLOCK
        model_to_phys_np[mp] = phys_n64 * TOKENS_PER_BLOCK + off

    # ── Phase 1: BC-train ON ──
    on_model = CrossAttentionOrderNetwork(
        num_blocks=NUM_BLOCKS, d_edge=args.d_edge, d_model=args.d_model
    ).to(device)
    print(f"ON initialized: {sum(p.numel() for p in on_model.parameters()):,} params", flush=True)

    if not args.skip_phase1:
        print(f"\n{'='*60}")
        print("Phase 1: BC-train ON on mixed025-NN teacher")
        print(f"{'='*60}", flush=True)

        best_ckpt, p1_diag = phase1_bc_train(
            on_model, A_mix_train, teacher_train,
            A_mix_val, teacher_val,
            batch_size=32, n_epochs=args.phase1_epochs, lr=args.phase1_lr,
            device=device, output_dir=args.output_dir,
        )

        print(f"\nPhase 1 Diagnostics:", flush=True)
        for k, v in p1_diag.items():
            if v is not None:
                print(f"  {k}: {v}", flush=True)

        # Check thresholds
        legal_rate = p1_diag.get("legal_permutation_rate", 0)
        acc = p1_diag.get("best_val_acc", 0)
        tau = p1_diag.get("greedy_tau_vs_teacher", 0)
        entropy = p1_diag.get("sample_entropy", 0)

        if not args.smoke:
            if legal_rate < 1.0:
                print(f"FAIL: legal_permutation_rate={legal_rate:.4f} < 1.0. Stopping.", flush=True)
                return
            if acc < 0.85:
                print(f"FAIL: val_acc={acc:.4f} < 0.85. Stopping.", flush=True)
                return
            if tau < 0.65:
                print(f"FAIL: greedy_tau_vs_teacher={tau:.4f} < 0.65. Stopping.", flush=True)
                return
            if entropy < 0.1:
                print(f"WARNING: sample_entropy={entropy:.4f} near 0. ON may be collapsed.", flush=True)
            print(f"Phase 1 PASSED. acc={acc:.4f} tau={tau:.4f} legal={legal_rate:.4f}", flush=True)
        else:
            print(f"Phase 1 smoke check (thresholds skipped): acc={acc:.4f} tau={tau:.4f} legal={legal_rate:.4f}", flush=True)
    else:
        print("Phase 1 skipped (--skip-phase1)", flush=True)

    # ── Prepare Phase 2 ──
    # Build token dataset
    print("Loading tokens...", flush=True)
    tokens_all = torch.from_numpy(np.load(args.tokens_path)).long()
    if args.smoke:
        tokens_all = tokens_all[:args.smoke_samples]
    tokens_train = tokens_all[:n_train]
    tokens_val = tokens_all[n_train:]

    train_dataset = torch.utils.data.TensorDataset(tokens_train)
    val_dataset = torch.utils.data.TensorDataset(tokens_val)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    # Pre-load A_mix to GPU
    A_mix_train_t = torch.from_numpy(A_mix_train).float().to(device)
    A_mix_val_t = torch.from_numpy(A_mix_val).float().to(device)

    # Optimizer for AOGPT
    optimizer = torch.optim.AdamW(
        aogpt.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # Token chunks for Phase 3 extraction
    train_chunks = [tokens_all[i].tolist() for i in range(n_train)]

    # ── Alternating co-training loop ──
    mode_label = "Fixed-Teacher (no refresh)" if args.no_refresh else f"Refresh every {args.refresh_every}"
    print(f"\n{'='*60}")
    print(f"Co-Training: max={args.max_total_steps} steps, {mode_label}")
    print(f"{'='*60}", flush=True)

    global_step = 0
    round_idx = 0
    eval_log = []
    best_val_on_order = float('inf')
    refresh_diags = []

    while global_step < args.max_total_steps:
        steps_this_cycle = min(args.refresh_every, args.max_total_steps - global_step)
        cycle_start_step = global_step

        print(f"\n--- Round {round_idx}: AOGPT training {global_step}→{global_step + steps_this_cycle} ---", flush=True)

        # Phase 2 training
        train_iter = iter(train_loader)
        accum_loss = 0.0
        optim_step_count = 0
        t0 = time.time()

        for _ in range(steps_this_cycle):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            tokens = batch[0].to(device)
            B = tokens.shape[0]

            # Alpha schedule: linear warmup 0→alpha_target over global steps
            alpha = min(args.alpha_target, (global_step / max(args.alpha_warmup, 1)) * args.alpha_target)

            # Sample block orders
            with torch.no_grad():
                idx_in_batch = torch.randint(0, n_train, (B,))
                amix_b = A_mix_train_t[idx_in_batch]

                on_block = sample_on_order_n64(on_model, amix_b,
                                                temperature=args.temperature, top_k=args.top_k)
                rand_block = torch.stack([torch.randperm(NUM_BLOCKS, device=device) for _ in range(B)])

                # Per-sample mixing
                use_on = torch.rand(B, device=device) < alpha
                use_on_exp = use_on.unsqueeze(-1).expand(-1, NUM_BLOCKS)
                block_orders = torch.where(use_on_exp, on_block, rand_block)

                on_fraction = use_on.float().mean().item()

            # Compute AOGPT loss
            token_orders = phys_block_order_to_model_token_order(block_orders, block_perm)
            with torch.amp.autocast('cuda', enabled=True):
                _, loss, _ = aogpt.forward_fn(tokens, token_orders, return_token_loss=True)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum

            if (optim_step_count + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(aogpt.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            optim_step_count += 1

            # Logging
            if (global_step + 1) % args.log_interval == 0:
                avg_loss = accum_loss / max(args.log_interval, 1)
                elapsed = time.time() - t0
                steps_since_start = global_step - cycle_start_step + 1
                rate = steps_since_start / max(elapsed, 1e-9)
                print(f"  step {global_step+1:>5d}  loss={avg_loss:.4f}  α={alpha:.3f}  "
                      f"ON_frac={on_fraction:.3f}  rate={rate:.1f} step/s", flush=True)
                accum_loss = 0.0

            # Evaluation
            if (global_step + 1) % args.eval_interval == 0:
                eval_metrics = evaluate_aogpt(
                    aogpt, val_loader, on_model, A_mix_val_t, block_perm, inv_perm,
                    device, max_seqs=args.max_eval_seqs,
                )
                eval_metrics["iter"] = global_step + 1
                eval_metrics["alpha"] = alpha
                eval_metrics["round"] = round_idx
                eval_log.append(eval_metrics)

                print(f"  EVAL iter={global_step+1}: ar={eval_metrics['ar']:.4f} "
                      f"on_order={eval_metrics['on_order']:.4f} "
                      f"random={eval_metrics['random']:.4f} "
                      f"l2r={eval_metrics['l2r']:.4f}", flush=True)

                # Track best
                vo = eval_metrics.get("on_order", float('inf'))
                if vo < best_val_on_order and vo == vo:
                    best_val_on_order = vo
                    torch.save({
                        "model": aogpt.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": global_step + 1,
                        "val_on_order": vo,
                    }, os.path.join(args.output_dir, "aogpt_best.pt"))

                # Save eval log
                with open(os.path.join(args.output_dir, "eval_log.jsonl"), "a") as f:
                    f.write(json.dumps(eval_metrics) + "\n")

            global_step += 1
            if global_step >= args.max_total_steps:
                break

        # Phase 3: Refresh (if more steps remain and refresh enabled)
        if not args.no_refresh and global_step < args.max_total_steps:
            on_model, A_mix_new, teacher_new, A_global_new, r_diag = phase3_refresh(
                aogpt, on_model, train_chunks, bp_np, model_to_phys_np,
                A_global, n_train, device, args.output_dir, round_idx + 1,
                bc_epochs=args.phase3_bc_epochs, bc_lr=args.phase3_bc_lr,
                extract_subset=args.phase3_extract_subset,
            )

            # Update state
            A_global = A_global_new
            A_mix_train = compute_amix(A_train, A_global_new)
            A_mix_train_t = torch.from_numpy(A_mix_train).float().to(device)
            refresh_diags.append(r_diag)

            # Check for moving target issues
            if r_diag.get("teacher_tau_drift") is not None and r_diag["teacher_tau_drift"] < 0.3:
                print(f"WARNING: teacher_tau_drift={r_diag['teacher_tau_drift']:.4f} < 0.3. "
                      f"Teacher drift may be too large.", flush=True)
            if r_diag.get("first_node_entropy", 1.0) < 0.5:
                print(f"WARNING: first_node_entropy={r_diag['first_node_entropy']:.4f}. "
                      f"Teacher may be collapsing.", flush=True)

            round_idx += 1
        elif args.no_refresh:
            # Still increment round_idx for consistent logging
            round_idx += 1

    # ── Final save ──
    final_ckpt = os.path.join(args.output_dir, "aogpt_final.pt")
    torch.save({"model": aogpt.state_dict(), "iter": global_step}, final_ckpt)
    torch.save({"model_state_dict": on_model.state_dict()},
               os.path.join(args.output_dir, "on_final.pt"))

    # Save refresh diagnostics
    if refresh_diags:
        with open(os.path.join(args.output_dir, "refresh_diags.jsonl"), "w") as f:
            for d in refresh_diags:
                f.write(json.dumps(d) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE. {global_step} steps, {round_idx} rounds"
          f"{' (no refresh)' if args.no_refresh else ', ' + str(len(refresh_diags)) + ' refreshes'}.")
    print(f"Best val_on_order: {best_val_on_order:.4f}")
    print(f"Output: {args.output_dir}/")
    if eval_log:
        print(f"Eval log: {len(eval_log)} entries")
    if refresh_diags:
        print(f"Refresh diags: {len(refresh_diags)} rounds")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
