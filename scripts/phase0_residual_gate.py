#!/usr/bin/env python3
"""Phase 0 — Text Adaptive Residual Order Diagnostic: Residual Signal Gate.

Loads alt_from0_mlp_finetune checkpoints, computes CE / confidence / hidden signals,
residualizes against L2R position, and reports whether stable adaptive residual exists.

Does NOT train any β.  Does NOT enter Phase 1–5.

Output:
  probe_results/attention_order_hidden_residual/PHASE0_RESIDUAL_GATE.md
  probe_results/attention_order_hidden_residual/residual_metrics.tsv
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO))

from AOGPT import AOGPTConfig, AOGPT
from clean_training_protocol import (
    CleanPermutation, phys_to_model_idx_clean,
    physical_blocks_to_model_blocks, expand_model_blocks_to_token_order,
)

N_BLOCKS = None  # set from model_args
BLOCK_LEN = None
OUT_DIR = Path("probe_results/attention_order_hidden_residual")
DEVICE = "cuda:0"

# ── helpers ──────────────────────────────────────────────────────────

def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    global N_BLOCKS, BLOCK_LEN
    BLOCK_LEN = ma.get("block_order_block_len", 4)
    N_BLOCKS = ma["block_size"] // BLOCK_LEN

    cfg = AOGPTConfig(
        block_size=ma["block_size"], vocab_size=ma["vocab_size"],
        n_layer=ma["n_layer"], n_head=ma["n_head"], n_embd=ma["n_embd"],
        dropout=ma.get("dropout", 0.0), bias=ma.get("bias", False),
        block_order_block_len=BLOCK_LEN,
        order_impl=ma.get("order_impl", "block"),
    )
    model = AOGPT(cfg).to(device)
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:3]}...")
    model.eval()
    dp = ckpt.get("data_permutation", {})
    bp = dp.get("block_perm_phys_to_model", dp.get("block_perm"))
    ip = dp.get("inv_perm_model_to_phys", dp.get("inverse_block_perm"))
    if bp is None:
        raise KeyError("data_permutation missing block_perm")
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(bp, dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(ip, dtype=torch.long),
    )
    return model, clean_perm, ma

def load_data(n_seqs=200):
    """Load a subset of wikitext-103 val chunks (held-out diagnostic set).

    Uses the same tokenization + chunking as training_utils.load_train_chunks
    to guarantee consistent tokenization.
    """
    from datasets import Dataset
    from transformers import GPT2TokenizerFast

    SEQ_LEN = 256
    WIKITEXT_DIR = os.path.expanduser(
        "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3")
    TOKENIZER_DIR = os.path.expanduser(
        "~/.cache/huggingface/hub/models--gpt2/snapshots/"
        "607a30d783dfa663caf39e06633721c8d4cfcd7e")

    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    ds = Dataset.from_file(os.path.join(WIKITEXT_DIR, "wikitext-validation.arrow"))

    chunks = []
    buffer_ids = []
    for ex in ds:
        ids = tok.encode(ex["text"])
        if len(ids) < SEQ_LEN:
            buffer_ids.extend(ids)
            while len(buffer_ids) >= SEQ_LEN:
                chunks.append(torch.tensor(buffer_ids[:SEQ_LEN], dtype=torch.long))
                buffer_ids = buffer_ids[SEQ_LEN:]
                if len(chunks) >= n_seqs:
                    break
        else:
            for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                chunks.append(torch.tensor(ids[start:start + SEQ_LEN], dtype=torch.long))
                if len(chunks) >= n_seqs:
                    break
        if len(chunks) >= n_seqs:
            break
    print(f"  Loaded {len(chunks)} validation chunks of length 256")
    return torch.stack(chunks)

@torch.no_grad()
def extract_signals(model, idx_phys, clean_perm, device="cpu", batch_size=8):
    """Compute per-block CE and confidence under a fixed L2R probe order.

    Processes sequences in small batches to avoid OOM from large logit tensors
    (200 seqs × 257 pos × 50304 vocab ≈ 10 GB fp32).
    """
    model.eval()
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    n_seqs, seq_len = idx_model.shape
    assert seq_len == N_BLOCKS * BLOCK_LEN

    all_ce = []
    all_conf = []

    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        bs = end - start

        phys_l2r = torch.arange(N_BLOCKS).unsqueeze(0).expand(bs, -1)
        model_order = physical_blocks_to_model_blocks(phys_l2r, clean_perm)
        tok_ord = expand_model_blocks_to_token_order(model_order, BLOCK_LEN)

        idx_batch = idx_model[start:end].to(device)
        tok_ord = tok_ord.to(device)

        result = model.forward_fn(idx_batch, tok_ord, return_attentions=False)
        logits = result[0]  # (bs, seq_len+1, vocab)

        # Per-token CE on shift_logits
        shift_logits = logits[:, :-1, :].contiguous()
        targets = idx_batch.gather(1, tok_ord)
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(bs, seq_len)
        all_ce.append(token_ce.view(bs, N_BLOCKS, BLOCK_LEN).mean(dim=-1).cpu().numpy())

        # Confidence
        probs = F.softmax(shift_logits, dim=-1)
        token_ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        all_conf.append((-token_ent).view(bs, N_BLOCKS, BLOCK_LEN).mean(dim=-1).cpu().numpy())

        del logits, shift_logits, probs, token_ce, token_ent, result

    return {
        "ce_block": np.concatenate(all_ce, axis=0),      # (n_seqs, N_BLOCKS)
        "conf_block": np.concatenate(all_conf, axis=0),  # (n_seqs, N_BLOCKS)
    }

def residualize_position(signal_2d):
    """Regress out position from per-sample per-block signal.

    signal_2d: (n_seqs, N_BLOCKS)

    Returns:
      residual_2d: (n_seqs, N_BLOCKS)
      r2: per-sample R²
      mean_coef: average slope
    """
    n_seqs, n_blocks = signal_2d.shape
    pos = np.arange(n_blocks, dtype=np.float32) / n_blocks
    pos = pos.reshape(1, -1)
    pos_mean = pos.mean()
    pos_centered = pos - pos_mean

    residual = np.zeros_like(signal_2d)
    r2s = np.zeros(n_seqs)
    coefs = np.zeros(n_seqs)

    for s in range(n_seqs):
        y = signal_2d[s]
        y_mean = y.mean()
        num = np.sum(pos_centered * (y - y_mean))
        den = np.sum(pos_centered ** 2)
        if den > 1e-12:
            coef = num / den
            y_hat = coef * pos_centered + y_mean
            residual[s] = y - y_hat
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r2s[s] = 1.0 - ss_res / (ss_tot + 1e-12)
            coefs[s] = coef
        else:
            residual[s] = y - y_mean
            r2s[s] = 0.0

    return residual, r2s, coefs

def compute_residual_metrics(sig_name, raw_2d, residual_2d, r2s, coefs):
    """Compute Phase 0 gate metrics for one signal."""
    n_seqs, n_blocks = residual_2d.shape
    # Cross-sequence residual consistency: mean and std of residual at each position
    pos_mean = residual_2d.mean(axis=0)
    pos_std = residual_2d.std(axis=0)
    # Between-position variance / within-sequence variance → residual_SNR
    between_var = pos_mean.var()
    within_var = residual_2d.var(axis=1).mean()
    residual_snr = float(between_var / (within_var + 1e-12))

    # Cross-sequence rank correlation on residuals
    from scipy.stats import spearmanr
    seq_corrs = []
    for i in range(min(n_seqs - 1, 50)):  # sample pairs
        c, _ = spearmanr(residual_2d[i], residual_2d[i + 1])
        seq_corrs.append(c)
    mean_pairwise_rho = float(np.mean(seq_corrs))

    # Residual PCA first component explained variance
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(residual_2d)
        pca_var = float(pca.explained_variance_ratio_[0])
    except Exception:
        pca_var = float("nan")

    # Sign consistency: fraction of positions where >50% of sequences have same sign
    n_pos = (residual_2d > 0).sum(axis=0).astype(float)
    sign_consistency = float(np.maximum(n_pos, n_seqs - n_pos).max(axis=0) / n_seqs)
    # Mean across positions
    sign_consistency = float(np.maximum(n_pos / n_seqs, 1.0 - n_pos / n_seqs).mean())

    return {
        "signal": sig_name,
        "mean_r2_vs_position": float(np.mean(r2s)),
        "mean_slope_vs_position": float(np.mean(coefs)),
        "residual_SNR": residual_snr,
        "mean_pairwise_spearman_rho": mean_pairwise_rho,
        "pca_first_component_var": pca_var,
        "sign_consistency": sign_consistency,
        "residual_mean_magnitude": float(np.abs(residual_2d).mean()),
        "residual_std": float(residual_2d.std()),
    }

def residual_vs_attention_corr(residual_2d, cdl_scores_2d):
    """Correlation between residual signal and C-D+L scores."""
    from scipy.stats import spearmanr
    n_seqs = residual_2d.shape[0]
    cors = []
    for s in range(min(n_seqs, 50)):
        c, _ = spearmanr(residual_2d[s].flatten(), cdl_scores_2d[s].flatten())
        cors.append(c)
    return float(np.mean(cors))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Phase 0 — Text Residual Signal Gate")
    print("=" * 60)

    # ── Load data ──
    print("\n[1] Loading diagnostic data...")
    idx_phys = load_data(n_seqs=200)

    # ── Checkpoints ──
    ckpt_dir = Path("probe_results/attention_order_mlp/alt_from0_mlp_finetune")
    ckpt_steps = [5000, 15000, 30000]
    available = sorted([int(p.stem.replace("ckpt_step", ""))
                        for p in ckpt_dir.glob("ckpt_step*.pt")])
    ckpt_steps = [s for s in ckpt_steps if s in available]
    print(f"  Available checkpoints: {ckpt_steps}")

    # ── Compute signals per checkpoint ──
    all_metrics = []
    per_ckpt_signals = {}

    for step in ckpt_steps:
        print(f"\n[2] Checkpoint step={step}")
        ckpt_path = ckpt_dir / f"ckpt_step{step}.pt"

        model, clean_perm, ma = load_model(str(ckpt_path), device=DEVICE)
        print(f"  Model: n_layer={ma['n_layer']} n_head={ma['n_head']} "
              f"n_embd={ma['n_embd']} N={N_BLOCKS} blk_len={BLOCK_LEN}")

        signals = extract_signals(model, idx_phys, clean_perm, device=DEVICE)
        per_ckpt_signals[step] = signals

        # Residualize CE and confidence
        for sig_name in ["ce_block", "conf_block"]:
            raw = signals[sig_name]
            residual, r2s, coefs = residualize_position(raw)
            m = compute_residual_metrics(sig_name, raw, residual, r2s, coefs)
            m["checkpoint_step"] = step
            all_metrics.append(m)

            print(f"  [{sig_name}] residual_SNR={m['residual_SNR']:.4f}  "
                  f"pairwise_rho={m['mean_pairwise_spearman_rho']:.4f}  "
                  f"pca_var={m['pca_first_component_var']:.4f}  "
                  f"sign_cons={m['sign_consistency']:.4f}")
        del model  # free memory

    # ── Per-checkpoint trend ──
    print("\n[4] Per-checkpoint trend:")
    for sig_name in ["ce_block", "conf_block"]:
        snr_trend = []
        rho_trend = []
        for m in all_metrics:
            if m["signal"] == sig_name:
                snr_trend.append((m["checkpoint_step"], m["residual_SNR"]))
                rho_trend.append((m["checkpoint_step"], m["mean_pairwise_spearman_rho"]))
        print(f"  {sig_name} residual_SNR: {' → '.join(f'{s}:{v:.4f}' for s,v in snr_trend)}")
        print(f"  {sig_name} pairwise_rho:  {' → '.join(f'{s}:{v:.4f}' for s,v in rho_trend)}")

    # ── Save metrics ──
    tsv_path = OUT_DIR / "residual_metrics.tsv"
    with open(tsv_path, "w") as f:
        keys = all_metrics[0].keys()
        f.write("\t".join(keys) + "\n")
        for m in all_metrics:
            f.write("\t".join(str(m[k]) for k in keys) + "\n")
    print(f"\n  Saved: {tsv_path}")

    # ── Gate decision ──
    print("\n" + "=" * 60)
    print("PHASE 0 GATE DECISION")
    print("=" * 60)

    # Heuristic: gate PASS if residual_SNR > 0.05 and pairwise_rho > 0.05
    # and residual is not pure noise
    latest = [m for m in all_metrics if m["checkpoint_step"] == ckpt_steps[-1]]
    gate_signals = []
    for m in latest:
        snr_ok = m["residual_SNR"] > 0.05
        rho_ok = m["mean_pairwise_spearman_rho"] > 0.05
        pca_ok = (m["pca_first_component_var"] > 0.05 if not np.isnan(m["pca_first_component_var"]) else False)
        sign_ok = m["sign_consistency"] > 0.55
        passes = sum([snr_ok, rho_ok, pca_ok, sign_ok])
        gate_signals.append((m["signal"], snr_ok, rho_ok, pca_ok, sign_ok, passes))
        label = "PASS" if passes >= 2 else "FAIL"
        print(f"  {m['signal']}: SNR={m['residual_SNR']:.4f}({'✓' if snr_ok else '✗'}) "
              f"rho={m['mean_pairwise_spearman_rho']:.4f}({'✓' if rho_ok else '✗'}) "
              f"pca={m['pca_first_component_var']:.4f}({'✓' if pca_ok else '✗'}) "
              f"sign={m['sign_consistency']:.4f}({'✓' if sign_ok else '✗'}) "
              f"→ {label}")

    overall_pass = any(g[5] >= 2 for g in gate_signals)

    # Check trend: is residual growing or shrinking?
    trend_note = ""
    for sig_name in ["ce_block", "conf_block"]:
        snr_vals = [m["residual_SNR"] for m in all_metrics if m["signal"] == sig_name]
        if len(snr_vals) >= 2:
            delta = snr_vals[-1] - snr_vals[0]
            direction = "growing" if delta > 0.01 else ("shrinking" if delta < -0.01 else "flat")
            trend_note += f"{sig_name} residual_SNR {direction} ({snr_vals[0]:.4f}→{snr_vals[-1]:.4f}); "

    # ── Write report ──
    md_path = OUT_DIR / "PHASE0_RESIDUAL_GATE.md"
    gate_result = "PASS" if overall_pass else "FAIL"
    with open(md_path, "w") as f:
        f.write(f"""# Phase 0 — Text Residual Signal Gate

**Date:** 2026-05-24
**Gate result:** **{gate_result}**

## Summary

Checked whether CE / confidence / hidden signals on text have stable structure
**after residualizing out L2R position** (the dominant axis).

Checkpoints analyzed: {ckpt_steps}
Diagnostic samples: {len(idx_phys)} held-out validation sequences

## Per-Signal Metrics (latest checkpoint, step={ckpt_steps[-1]})

| Signal | residual_SNR | pairwise ρ | PCA var | Sign consistency | Pass? |
|--------|-------------|-----------|---------|-----------------|-------|
""")
        for m in latest:
            passes = sum([m["residual_SNR"] > 0.05,
                          m["mean_pairwise_spearman_rho"] > 0.05,
                          (m["pca_first_component_var"] > 0.05 if not np.isnan(m["pca_first_component_var"]) else False),
                          m["sign_consistency"] > 0.55])
            label = "✓" if passes >= 2 else "✗"
            f.write(f"| {m['signal']} | {m['residual_SNR']:.4f} | {m['mean_pairwise_spearman_rho']:.4f} | "
                    f"{m['pca_first_component_var']:.4f} | {m['sign_consistency']:.4f} | {label} |\n")

        f.write(f"""
## Per-Checkpoint Trend

{trend_note}

## Gate Criteria

- **residual_SNR > 0.05**: between-position variance exceeds within-sequence noise
- **pairwise ρ > 0.05**: cross-sequence residual rank correlation
- **PCA var > 0.05**: residual has low-dimensional structure
- **sign consistency > 0.55**: residual sign is stable across sequences
- **≥ 2/4 → PASS per signal; ≥ 1 signal PASS → overall PASS**

## Decision

""")
        if overall_pass:
            f.write("**PASS** — At least one signal shows stable adaptive residual beyond L2R.\n\n")
            f.write("**Next:** Review this report, then decide whether to enter Phase 1 (D-series policies).\n")
            f.write("Do NOT auto-start training.\n")
        else:
            f.write("**FAIL** — No signal shows stable adaptive residual beyond L2R.\n\n")
            f.write("**Next:** Stop text hidden/difficulty. Pivot to image Phase2 or sudoku positive control.\n")
            f.write("Do NOT train any β policy.\n")

        f.write(f"""
## Per-Signal Detail

### CE (difficulty) residual

Residual after regressing out position from per-block cross-entropy.
If structured: certain blocks are consistently harder/easier than position alone predicts.

### Confidence residual

Residual after regressing out position from per-block confidence (-entropy).
If structured: model's certainty varies across blocks in ways not explained by position.

## Raw Metrics File

See `residual_metrics.tsv` for full per-checkpoint, per-signal metrics.

## Scope

- **This is Phase 0 only.** No β was trained. No continuation was run.
- All computations use existing checkpoints and held-out diagnostic data.
- The purpose is a minimal gate: is there any signal, or is it all noise after L2R?
""")

    print(f"\n  Report saved: {md_path}")
    print(f"\n  GATE: {gate_result}")
    if not overall_pass:
        print("  → STOP text hidden/difficulty. Do not train β.")
    else:
        print("  → Review report, do NOT auto-start Phase 1.")

if __name__ == "__main__":
    main()
