"""BR-1 mismatch diagnostic: heavy vs lightweight B definitions.

For the SAME forwards, build three batch-mean graph variants and compare the
teacher orders they induce:

  heavy        = top-4 variance head, all-layer mean, +0.1*[None]   (= the B
                 definition extract_A_matrices uses for Phase-0/1 training data)
  light_last   = all-head mean, LAST layer only, no [None]          (lightweight
                 hook candidate)
  light_all    = all-head mean, ALL layers mean, no [None]          (isolates the
                 layer-aggregation axis)

Question this answers (user-requested, 2026-05-29): if we switch the hook to a
lightweight B, does the teacher still produce diverse, non-collapsed orders, and
how far does sigma_light drift from sigma_heavy? Drives the decision to either
unify on a lightweight B (re-run Phase-1) or keep heavy-B + Gate-2.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=block_lo_arm_order_network \
    python scripts/diag_br1_lightweight_b.py --M 100 --batch_size 32 --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from neural_readout.extract_b import _load_model_and_chunks
from train_clean_aogpt import expand_model_blocks_to_token_order, N, SEQ_LEN, BLOCK_LEN
from neural_readout.teacher_labels import generate_teacher_label
from batch_readout.diversity_batch import teacher_diversity_stats
from batch_readout.eval_metrics import kendall_tau_batch, spearman_rho_batch


def _make_A(avg_attn, phys_tokens, add_none):
    """avg_attn: (257, 257) attention incl. [None] at index 0. Remap to physical,
    aggregate 256x256 -> 64x64 block, optionally add 0.1*[None] column."""
    attn_content = avg_attn[1:, 1:]
    attn_phys = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
    np.add.at(attn_phys, (phys_tokens[:, None], phys_tokens[None, :]), attn_content)
    A = np.zeros((N, N), dtype=np.float32)
    for bi in range(N):
        i_s, i_e = bi * BLOCK_LEN, (bi + 1) * BLOCK_LEN
        for bj in range(N):
            j_s, j_e = bj * BLOCK_LEN, (bj + 1) * BLOCK_LEN
            A[bi, bj] = attn_phys[i_s:i_e, j_s:j_e].mean()
    if add_none:
        none_attn = avg_attn[1:, 0]
        none_block = np.array(
            [none_attn[b * BLOCK_LEN:(b + 1) * BLOCK_LEN].mean() for b in range(N)]
        )
        A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)
    return A


@torch.no_grad()
def extract_A_variants(model, idx_chunks, clean_perm, device, n_chunks, seed):
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    A_heavy = np.zeros((n_chunks, N, N), dtype=np.float32)
    A_light_last = np.zeros((n_chunks, N, N), dtype=np.float32)
    A_light_all = np.zeros((n_chunks, N, N), dtype=np.float32)
    model.eval()
    for i in range(n_chunks):
        tokens = idx_chunks[i:i + 1].to(device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_order = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        ).to(device)
        _, _, attn_list = model.forward_fn(tokens, token_order, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)
        L, H = attn_stack.shape[:2]

        reveal_tokens = token_order[0].cpu().numpy()
        model_blocks = reveal_tokens // BLOCK_LEN
        phys_blocks = inv_perm[model_blocks]
        phys_tokens = phys_blocks * BLOCK_LEN + (reveal_tokens % BLOCK_LEN)

        # heavy: top-4 variance head, all-layer mean, +[None]
        head_vars = np.zeros(H)
        for h in range(H):
            content = attn_stack[:, h, 1:, 1:]
            mask = ~np.eye(SEQ_LEN, dtype=bool)
            offdiag = content[:, mask].reshape(L, SEQ_LEN, SEQ_LEN - 1)
            head_vars[h] = float(np.var(offdiag))
        top_heads = np.argsort(head_vars)[-4:]
        avg_heavy = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))
        A_heavy[i] = _make_A(avg_heavy, phys_tokens, add_none=True)

        # light_last: last-layer, all-head mean, no [None]
        avg_light_last = attn_stack[-1, :, :, :].mean(axis=0)
        A_light_last[i] = _make_A(avg_light_last, phys_tokens, add_none=False)

        # light_all: all-layer, all-head mean, no [None]
        avg_light_all = attn_stack[:, :, :, :].mean(axis=(0, 1))
        A_light_all[i] = _make_A(avg_light_all, phys_tokens, add_none=False)

    return A_heavy, A_light_last, A_light_all


def _to_Bbatch(A, M, bs):
    B = np.transpose(A, (0, 2, 1)).copy()
    for i in range(len(B)):
        np.fill_diagonal(B[i], 0.0)
    Bb = B.reshape(M, bs, N, N).astype(np.float64).mean(axis=1).astype(np.float32)
    for m in range(M):
        np.fill_diagonal(Bb[m], 0.0)
    return Bb


def _teacher_sigmas(Bb):
    return np.stack([generate_teacher_label(Bb[m], alpha_dep=0.5)[0] for m in range(len(Bb))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt")
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="block_lo_arm_order_network/batch_readout/logs/diag_lightweight_b.json")
    args = ap.parse_args()

    total = args.M * args.batch_size
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        args.ckpt, total, args.seed, args.device, "train"
    )
    print(f"[diag] extracting {total} per-sample graphs (M={args.M} x B={args.batch_size}) ...", flush=True)
    A_heavy, A_light_last, A_light_all = extract_A_variants(
        model, chunks, clean_perm, dev, total, args.seed
    )

    variants = {
        "heavy": A_heavy,
        "light_last": A_light_last,
        "light_all": A_light_all,
    }
    Bb = {k: _to_Bbatch(v, args.M, args.batch_size) for k, v in variants.items()}
    sig = {k: _teacher_sigmas(v) for k, v in Bb.items()}

    report = {"config": {"M": args.M, "batch_size": args.batch_size, "seed": args.seed, "ckpt": args.ckpt}}
    report["diversity"] = {k: teacher_diversity_stats(sig[k]) for k in sig}
    # sigma agreement vs heavy (the distribution g_beta was trained on)
    report["sigma_vs_heavy"] = {
        k: {
            "kendall_tau": kendall_tau_batch(sig[k], sig["heavy"]),
            "spearman_rho": spearman_rho_batch(sig[k], sig["heavy"]),
        }
        for k in ("light_last", "light_all")
    }
    # graph-level correlation of B entries (off-diagonal) heavy vs light
    def _bcorr(a, b):
        m = ~np.eye(N, dtype=bool)
        xs = a[:, m].reshape(-1)
        ys = b[:, m].reshape(-1)
        return float(np.corrcoef(xs, ys)[0, 1])
    report["B_entry_corr_vs_heavy"] = {
        k: _bcorr(Bb["heavy"], Bb[k]) for k in ("light_last", "light_all")
    }

    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
