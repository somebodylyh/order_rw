"""
Unified L2R eval across all checkpoints.
Same val split (seed=42), same metric (val_l2r_loss), same data, same coordinate logic.
"""

import os, sys, json, argparse
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2TokenizerFast

from config import Config
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

SEQ_LEN = 256
N_BLOCKS = 64
BLOCK_LEN = 4

WIKITEXT_DIR = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

# ── Checkpoint registry ──────────────────────────────────────────────────────
CKPT_REGISTRY = {
    "AR_raw": "/home/admin/ych/nanogpt-learned-order/out/base/nonpermute/seq256/block64/out-wikitext103-seq256-ar-b64/ckpt.pt",
    "AR_shuffled": "/home/admin/ych/nanogpt-learned-order/out/base/permute/seq256/block64/out-wikitext103-seq256-ar-b64-permute-block-50000-iters/ckpt.pt",
    "MDM_original": "/home/admin/ych/nanogpt-learned-order/out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt",
}

M_STEP_DIRS = {
    "v2_baseline": "probe_results/m_step/m_step_baseline.pt",
    "v2_golden":   "probe_results/m_step/m_step_golden.pt",
    "v2_mixed":    "probe_results/m_step/m_step_mixed.pt",
    "v3k_baseline": "probe_results/m_step_v3k/m_step_baseline.pt",
    "v3k_golden":   "probe_results/m_step_v3k/m_step_golden.pt",
    "v3k_mixed":    "probe_results/m_step_v3k/m_step_mixed.pt",
}


# ── Data ──────────────────────────────────────────────────────────────────────
def load_wikitext_val(n_seqs=200, seed=42, seq_len=SEQ_LEN):
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tokens_all = []
    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        arrow_file = os.path.join(WIKITEXT_DIR, shard)
        ds = Dataset.from_file(arrow_file)
        for ex in ds:
            raw = tok.encode(ex["text"])
            if len(raw) >= seq_len:
                tokens_all.append(raw[:seq_len])
    tokens_all = tokens_all[:5200]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tokens_all))
    val_indices = perm[5000:5000 + n_seqs]
    idx_phys = torch.tensor(tokens_all, dtype=torch.long)
    return idx_phys[val_indices]


# ── Coordinate conversion ────────────────────────────────────────────────────
def phys_to_model_idx(idx_phys, inv_perm):
    B, T = idx_phys.shape
    blk_size = T // len(inv_perm)
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


# ── AO-GPT loader ─────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(ckpt["model_args"]).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        clean = k.replace("_orig_mod.", "")
        if clean != k:
            sd[clean] = sd.pop(k)
    model.load_state_dict(sd)
    model.crop_block_size(SEQ_LEN)
    model.to(device)
    model.eval()
    has_perm = "data_permutation" in ckpt
    if has_perm:
        inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    else:
        inv_perm = torch.arange(N_BLOCKS, dtype=torch.long)
    return model, inv_perm


# ── L2R eval ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_val_l2r_loss(model, idx_val, batch_size=8):
    model_device = next(model.parameters()).device
    total_loss = 0.0
    n = len(idx_val)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx_b = idx_val[start:end].to(model_device)
        B = idx_b.shape[0]
        T = idx_b.shape[1]
        orders = torch.arange(T, device=model_device).unsqueeze(0).expand(B, -1)
        pos = torch.arange(0, T + 1, dtype=torch.long, device=model_device)
        batch_indices = torch.arange(B, device=model_device).unsqueeze(1).expand(-1, T)
        tok_emb = model.transformer.wte(idx_b)
        tok_emb = tok_emb[batch_indices, orders]
        none_emb = model.transformer.wnonee(torch.tensor([[0]], device=model_device)).expand(B, -1, -1)
        tok_emb = torch.cat([none_emb, tok_emb], dim=1)
        pos_emb = model.transformer.wpe(pos).unsqueeze(0).expand(B, -1, -1)
        pos_emb_prefix = pos_emb[:, :1, :]
        pos_emb_postfix = pos_emb[:, 1:, :][batch_indices, orders]
        pos_emb_final = torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)
        tgt_emb = model.transformer.wtpe(pos[:T]).unsqueeze(0).expand(B, -1, -1)
        tgt_emb_prefix = tgt_emb[batch_indices, orders]
        tgt_emb_postfix = torch.zeros(B, 1, tgt_emb.shape[-1], device=model_device)
        c = torch.cat([tgt_emb_prefix, tgt_emb_postfix], dim=1)
        targets = idx_b[batch_indices, orders]
        x = tok_emb + pos_emb_final
        x = model.transformer.drop(x)
        for block in model.transformer.h:
            x = block(x, c)
        x = model.transformer.final_layer(x, c)
        logits = model.lm_head(x)[:, :-1, :].contiguous()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean", ignore_index=-1)
        total_loss += ce.item() * B
    return total_loss / n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--val-seqs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    Config.seed = args.seed
    Config.set_seed()

    print("Loading val data (same split as M-step)...")
    idx_val_phys = load_wikitext_val(n_seqs=args.val_seqs, seed=args.seed)
    print(f"Val seqs: {idx_val_phys.shape[0]}, tokens: {idx_val_phys.shape[1]}")

    results = {}

    # ── Original checkpoints ──────────────────────────────────────────────
    for name, ckpt_path in CKPT_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"  Path: {ckpt_path}")
        model, inv_perm = load_model(ckpt_path, device)
        idx_val_model = phys_to_model_idx(idx_val_phys, inv_perm)
        val_l2r = compute_val_l2r_loss(model, idx_val_model, batch_size=args.batch_size)
        train_mode = "N/A"
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            train_mode = ck["config"].get("aogpt_train_mode", "N/A")
            has_permute = ck["config"].get("permute_data", "N/A")
        except:
            pass
        results[name] = {"val_l2r_loss": val_l2r, "train_mode": train_mode, "permute_data": has_permute}
        print(f"  val_l2r_loss = {val_l2r:.4f}  (train_mode={train_mode}, permute={has_permute})")
        del model
        torch.cuda.empty_cache()

    # ── M-step checkpoints ────────────────────────────────────────────────
    # M-step checkpoints share the same base architecture (from MDM_original)
    base_path = CKPT_REGISTRY["MDM_original"]
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_inv_perm = torch.tensor(base_ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    base_valid = {k: v for k, v in dict(base_ckpt["model_args"]).items() if k in sig}

    for name, rel_path in M_STEP_DIRS.items():
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
        if not os.path.exists(full_path):
            print(f"\n  SKIP {name}: not found at {full_path}")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        model = AOGPT(AOGPTConfig(**base_valid))
        raw = torch.load(full_path, map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict", raw)  # M-step wraps in dict
        model.load_state_dict(sd)
        model.crop_block_size(SEQ_LEN)
        model.to(device)
        model.eval()
        idx_val_model = phys_to_model_idx(idx_val_phys, base_inv_perm)
        val_l2r = compute_val_l2r_loss(model, idx_val_model, batch_size=args.batch_size)
        results[name] = {"val_l2r_loss": val_l2r, "train_mode": "M-step", "permute_data": True}
        print(f"  val_l2r_loss = {val_l2r:.4f}")
        del model
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'ALIGNED EVAL RESULTS':^60}")
    print(f"{'='*60}")
    print(f"{'Name':<20s} {'val_l2r':>8s}  {'train_mode':>10s}  {'permute':>8s}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20s} {r['val_l2r_loss']:8.4f}  {r['train_mode']:>10s}  {str(r['permute_data']):>8s}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results", "eval_alignment.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
