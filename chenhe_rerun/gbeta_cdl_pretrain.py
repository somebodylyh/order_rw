"""Stage-1 CDL producer (chenhe-native).

Builds a label-free L0 dynamic g_beta pretrain dataset from a *chenhe* backbone
checkpoint, then trains L0DynamicGBeta to imitate the CDL consensus teacher.

Protocol (mirrors the validated admin pipeline):
  1. Load chenhe AOGPT from a chenhe ckpt (model_args + model state dict).
  2. Extract total_samples = M * batch_mean_size per-sample strict65 B (1 random
     probe each) via chenhe forward_fn, L0 attention.
  3. batch_mean_heads -> B_raw (M, 8, 65, 65).
  4. Per batch-mean sample: build_dynamic_teacher -> model-frame [0,63] teacher.
  5. Deterministic 80/10/10 split; save .npz; train(); write provenance.

The ONLY chenhe-specific piece is model+data loading; the math is the ported
orderhead_v3 package. No inv_perm / physical frame is ever used.
"""

import os
import json
import hashlib

import numpy as np
import torch

from AOGPT_block import AOGPT, AOGPTConfig
from orderhead_v3.l0_strict65 import build_model_frame_strict65, batch_mean_heads
from orderhead_v3.cdl_teacher import build_dynamic_teacher
from orderhead_v3.constants import assert_layout, SEQ_LEN, N, BLOCK_LEN, HEADS, PERMUTE_SEED
from orderhead_v3 import train_gbeta


# ---------------------------------------------------------------------------
# chenhe-native backbone + data loading
# ---------------------------------------------------------------------------

def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_chenhe_backbone(ckpt_path, device):
    """Load a chenhe AOGPT from a chenhe ckpt ({'model','model_args',...}).

    chenhe's saved model_args carries extra train.py config keys (e.g. order_impl)
    that AOGPTConfig does not accept, so filter to the config's own fields.
    """
    import inspect
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(ckpt["model_args"])
    valid = set(inspect.signature(AOGPTConfig).parameters)
    margs = {k: v for k, v in raw.items() if k in valid}
    model = AOGPT(AOGPTConfig(**margs))
    state = ckpt["model"]
    # strip a possible compile prefix
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    assert_layout(model.num_blocks, model.block_order_block_len, model.config.n_head)
    return model, margs


def _sample_data_windows(bin_path, n_windows, seed, block_size=SEQ_LEN):
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - block_size - 1, size=n_windows)
    return torch.from_numpy(
        np.stack([data[s:s + block_size].astype(np.int64) for s in starts]))


def _make_probe(model, batch_size, seed, step, device):
    """Random model-frame block orders -> token orders (batch, SEQ_LEN)."""
    blocks = torch.stack([
        torch.randperm(N, generator=torch.Generator(device="cpu").manual_seed(
            int(seed) * 100_000_000 + int(step) * 1000 + b))
        for b in range(batch_size)])
    return model._expand_block_orders_to_token_orders(blocks).to(device)


@torch.no_grad()
def strict65_from_forward(model, idx_batch, probe, device, probe_mode="eval"):
    """One forward -> per-sample strict65 B (Bsz, 8, 65, 65) torch float."""
    model.eval() if probe_mode == "eval" else model.train()
    out = model.forward_fn(idx_batch.to(device), probe.to(device), return_attentions=True)
    attn_l0 = out[-1][0].detach().cpu().numpy()          # (Bsz, 8, 257, 257)
    B = build_model_frame_strict65(attn_l0, probe.cpu().numpy())
    return torch.from_numpy(B).float()


@torch.no_grad()
def extract_strict65_batch_mean(model, idx_batch, *, global_step, seed,
                                batch_mean_probes, device, probe_mode="eval"):
    """Per-sample strict65 B averaged over `batch_mean_probes` probes (Bsz,8,65,65)."""
    B_sum = None
    for k in range(max(1, batch_mean_probes)):
        probe = _make_probe(model, idx_batch.shape[0], seed, int(global_step) + k, device)
        Bk = strict65_from_forward(model, idx_batch, probe, device, probe_mode)
        B_sum = Bk if B_sum is None else B_sum + Bk
    return (B_sum / max(1, batch_mean_probes)).to(device)


# ---------------------------------------------------------------------------
# Dataset building (real protocol: batch-mean over batch_mean_size samples)
# ---------------------------------------------------------------------------

def deterministic_split(M, seed=7):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    train_n = int(round(M * 0.8)); val_n = int(round(M * 0.1))
    return {"train": np.sort(perm[:train_n].astype(np.int64)),
            "val": np.sort(perm[train_n:train_n + val_n].astype(np.int64)),
            "test": np.sort(perm[train_n + val_n:].astype(np.int64))}


def weighted_consensus_order(ranks, weights):
    """Sort blocks by teacher-weighted mean CDL rank (model-frame 0..63)."""
    ranks = np.asarray(ranks, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    mean_rank = (ranks.astype(np.float64) * weights[:, None]).sum(axis=0)
    return np.argsort(mean_rank, kind="stable").astype(np.int64)


def build_gbeta_dataset(model, data_bin, out_npz, *, M, batch_mean_size, seed,
                        forward_batch, device, destroy_replicas=1):
    total = M * batch_mean_size
    idx = _sample_data_windows(data_bin, total, seed)
    all_B = []
    for bi in range(0, total, forward_batch):
        chunk = idx[bi:bi + forward_batch]
        probe = _make_probe(model, chunk.shape[0], seed, bi, device)
        all_B.append(strict65_from_forward(model, chunk, probe, device).numpy())
    all_B = np.concatenate(all_B, axis=0)                # (total, 8, 65, 65)
    B_raw = batch_mean_heads(all_B, batch_mean_size)     # (M, 8, 65, 65)

    H = B_raw.shape[1]
    teacher_weights = np.zeros((M, H), dtype=np.float32)
    teacher_pairwise = np.zeros((M, 64, 64), dtype=np.float32)
    teacher_consensus_order = np.zeros((M, 64), dtype=np.int64)
    for m in range(M):
        res = build_dynamic_teacher(B_raw[m], destroy_seed=seed * 1000 + m,
                                    n_destroy_replicas=destroy_replicas, mode="C-D+L")
        teacher_weights[m] = res["weights"]
        teacher_pairwise[m] = res["pairwise"]
        teacher_consensus_order[m] = weighted_consensus_order(res["ranks"], res["weights"])

    split = deterministic_split(M, seed=seed)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, B_raw=B_raw, teacher_weights=teacher_weights,
                        teacher_pairwise=teacher_pairwise,
                        teacher_consensus_order=teacher_consensus_order,
                        train_idx=split["train"], val_idx=split["val"], test_idx=split["test"])
    return out_npz


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def pretrain_gbeta_cdl(parent_ckpt, out_dir, *, M=2000, batch_mean_size=16,
                       batch_mean_probes=4, heads=8, epochs=40, seed=0,
                       device="cpu", probe_mode="eval",
                       data_bin="data/wikitext103/train.bin", forward_batch=8):
    os.makedirs(out_dir, exist_ok=True)
    model, margs = load_chenhe_backbone(parent_ckpt, device)
    npz = os.path.join(out_dir, "dataset.npz")
    build_gbeta_dataset(model, data_bin, npz, M=M, batch_mean_size=batch_mean_size,
                        seed=seed, forward_batch=forward_batch, device=device)
    train_gbeta.train(npz, out_dir, epochs=epochs, seed=seed, device=device,
                      heads=heads, loss_type="pairwise_bce")
    ckpt_path = os.path.join(out_dir, "g_beta_best.pt")
    prov = {"producer": "gbeta_cdl_pretrain", "parent_ckpt": parent_ckpt,
            "parent_hash": _file_hash(parent_ckpt), "seq_len": SEQ_LEN,
            "num_blocks": N, "block_len": BLOCK_LEN, "heads": heads,
            "permute_seed": PERMUTE_SEED, "none_mode": "model", "strict65": True,
            "probe_mode": probe_mode, "batch_mean_probes": batch_mean_probes,
            "batch_mean_size": batch_mean_size, "M": M, "loss_type": "pairwise_bce",
            "seed": seed, "dataset_path": npz}
    with open(os.path.join(out_dir, "gbeta_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return ckpt_path
