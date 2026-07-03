"""Stage-1 CDL producer (chenhe-native), canonical single-head + NodewiseReadout.

Mirrors analyses/uniform_label_free_v1.py:
  Stage A: label-free head selection over ALL layers/heads -> pick top-1 head.
  Stage B: selected single-head B (None stripped, 64x64) -> per-sample CDL teacher
           (hard pairwise) -> train a readout (NodewiseReadout), select by val acc.

The ONLY chenhe-specific piece is model+data loading. Head is re-selected on the
chenhe backbone (winner may differ from admin's L1H7). No physical labels used.
"""

import os
import json
import hashlib
import inspect

import numpy as np
import torch
import torch.nn.functional as F

from AOGPT_block import AOGPT, AOGPTConfig
from orderhead_v3.per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec as _A_model_vec
from orderhead_v3.none_separated_block_graph import build_none_separated_B, rollout_by_method
from orderhead_v3.cdl_teacher import order_to_rank
from orderhead_v3.head_selection import select_best_head
from orderhead_v3.readouts import NodewiseReadout
from orderhead_v3.constants import assert_layout, SEQ_LEN, N, BLOCK_LEN, PERMUTE_SEED


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_chenhe_backbone(ckpt_path, device):
    """Load a chenhe AOGPT from a chenhe ckpt; filter model_args to AOGPTConfig fields."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(ckpt["model_args"])
    valid = set(inspect.signature(AOGPTConfig).parameters)
    margs = {k: v for k, v in raw.items() if k in valid}
    model = AOGPT(AOGPTConfig(**margs))
    state = ckpt["model"]
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    assert_layout(model.num_blocks, model.block_order_block_len, model.config.n_head)
    return model, margs


def _sample_data_windows(bin_path, n_windows, seed, block_size=SEQ_LEN):
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - block_size - 1, size=n_windows)
    return torch.from_numpy(np.stack([data[s:s + block_size].astype(np.int64) for s in starts]))


def _make_probe(model, batch_size, seed, device):
    blocks = torch.stack([
        torch.randperm(N, generator=torch.Generator(device="cpu").manual_seed(int(seed) + b))
        for b in range(batch_size)])
    return model._expand_block_orders_to_token_orders(blocks).to(device)


@torch.no_grad()
def single_head_B_from_forward(model, idx_batch, layer, head, probe, device):
    """One forward -> selected single-head content B (Bsz, 64, 64), None stripped."""
    out = model.forward_fn(idx_batch.to(device), probe.to(device),
                           return_attentions=True, return_logits=False)
    a = out[-1][layer].cpu().numpy()                     # (Bsz, H, 257, 257)
    probe_np = probe.cpu().numpy()
    B = np.empty((idx_batch.shape[0], 64, 64), np.float32)
    for bi in range(idx_batch.shape[0]):
        A65 = _A_model_vec(a[bi, head], probe_np[bi])
        B[bi] = build_none_separated_B(A65)[1:, 1:]      # strip None node
    return torch.from_numpy(B).float()


@torch.no_grad()
def extract_single_head_batch_mean(model, idx_batch, layer, head, *, global_step, seed,
                                   batch_mean_probes, device, probe_mode="eval"):
    """Per-sample single-head B averaged over `batch_mean_probes` probes (Bsz,64,64)."""
    model.eval() if probe_mode == "eval" else model.train()
    B_sum = None
    for k in range(max(1, batch_mean_probes)):
        probe = _make_probe(model, idx_batch.shape[0], int(seed) * 10_000 + int(global_step) + k, device)
        Bk = single_head_B_from_forward(model, idx_batch, layer, head, probe, device)
        B_sum = Bk if B_sum is None else B_sum + Bk
    return (B_sum / max(1, batch_mean_probes)).to(device)


def _per_sample_hard_pairwise(B_content):
    """Per-sample CDL rollout -> hard pairwise T (total, 64, 64); T[i,j]=1 if i before j."""
    total = B_content.shape[0]
    T = np.zeros((total, 64, 64), np.float32)
    for si in range(total):
        B65 = np.zeros((65, 65), np.float32)
        B65[1:, 1:] = B_content[si]
        rank = order_to_rank(rollout_by_method(B65, "C-D+L"))     # rank[block]=position
        T[si] = (rank[:, None] < rank[None, :]).astype(np.float32)
        np.fill_diagonal(T[si], 0.0)
    return T


def pretrain_gbeta_cdl(parent_ckpt, out_dir, *, n_select=800, n_groups=300,
                       batch_mean_size=16, n_reveal=8, epochs=40, lr=3e-4, seed=0,
                       device="cpu", batch_mean_probes=4, probe_mode="eval",
                       data_bin="data/wikitext103/train.bin", fwd_batch=32):
    os.makedirs(out_dir, exist_ok=True)
    model, margs = load_chenhe_backbone(parent_ckpt, device)

    # ── Stage A: label-free head selection (all layers/heads) ──
    sel_chunks = _sample_data_windows(data_bin, n_select, seed)
    (layer, head), head_scores = select_best_head(
        model, sel_chunks, n_reveal=n_reveal, seed=seed, device=device)
    print(f"Stage A: selected head = L{layer}H{head} "
          f"(score={head_scores[0]['total']:.3f})", flush=True)

    # ── Stage B: single-head B, BATCH-MEAN grouped (signal lives at batch-mean;
    #    per-sample is noise). total = n_groups * batch_mean_size samples -> mean
    #    each group of batch_mean_size -> n_groups training examples. ──
    total = n_groups * batch_mean_size
    train_chunks = _sample_data_windows(data_bin, total, seed + 1)
    B_parts = []
    for s in range(0, total, fwd_batch):
        chunk = train_chunks[s:s + fwd_batch]
        probe = _make_probe(model, chunk.shape[0], seed * 777 + s, device)
        B_parts.append(single_head_B_from_forward(model, chunk, layer, head, probe, device).numpy())
    all_B = np.concatenate(B_parts, axis=0)              # (total, 64, 64)
    B_content = all_B.reshape(n_groups, batch_mean_size, 64, 64).mean(axis=1)  # (n_groups,64,64)

    T = _per_sample_hard_pairwise(B_content)             # (n_groups, 64, 64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    tr = perm[:int(n_groups * 0.8)]
    va = perm[int(n_groups * 0.8):int(n_groups * 0.9)]

    readout = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4).to(device)
    opt = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=1e-2)
    B_all = torch.from_numpy(B_content).float().to(device)
    T_all = torch.from_numpy(T).float().to(device)

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        readout.train()
        order = torch.randperm(len(tr))
        for s in range(0, len(tr), 32):
            idx = tr[order[s:s + 32].numpy()]
            scores = readout(B_all[idx])                 # (bs, 64)
            sd = scores.unsqueeze(2) - scores.unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(sd, T_all[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        readout.eval()
        with torch.no_grad():
            sv = readout(B_all[va]); sdv = sv.unsqueeze(2) - sv.unsqueeze(1)
            acc = ((sdv > 0).float() == T_all[va]).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in readout.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"[{ep+1:>3}/{epochs}] val_pairwise_acc={acc:.4f} best={best_acc:.4f}", flush=True)

    if best_state is None:                               # tiny/empty val fallback
        best_state = {k: v.clone() for k, v in readout.state_dict().items()}
    readout.load_state_dict(best_state)
    ckpt_path = os.path.join(out_dir, "g_beta_best.pt")
    config = {"model_name": "nodewise", "N": 64, "d_model": 64, "n_layers": 2,
              "n_heads": 4, "sel_layer": int(layer), "sel_head": int(head)}
    torch.save({"model_state_dict": readout.state_dict(), "config": config,
                "best_val_acc": best_acc}, ckpt_path)

    prov = {"producer": "gbeta_cdl_pretrain(single-head/nodewise)", "parent_ckpt": parent_ckpt,
            "parent_hash": _file_hash(parent_ckpt), "seq_len": SEQ_LEN, "num_blocks": N,
            "block_len": BLOCK_LEN, "sel_layer": int(layer), "sel_head": int(head),
            "readout": "nodewise", "permute_seed": PERMUTE_SEED, "none_mode": "model",
            "strict65": True, "single_head": True, "probe_mode": probe_mode,
            "batch_mean_probes": batch_mean_probes, "n_select": n_select,
            "n_groups": n_groups, "batch_mean_size": batch_mean_size,
            "n_reveal": n_reveal, "seed": seed, "best_val_acc": best_acc}
    with open(os.path.join(out_dir, "gbeta_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"GBETA: {ckpt_path}  (L{layer}H{head}, val_acc={best_acc:.4f})", flush=True)
    return ckpt_path
