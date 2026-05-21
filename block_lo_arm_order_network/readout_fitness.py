#!/usr/bin/env python3
"""Fitness for CEM readout search (Phase 3). Two tiers:
  frozen_eval_fitness        — no training; CE under sampled orders on frozen model (fast).
  short_continuation_fitness — N-step continuation then cross eval (HELD: Task 2.4, do not use yet).
Both reuse train_imagelarge_graph_rw for model load + the inverse_block_perm remap, so the
coordinate frame is handled by the single shared utility (never re-implemented).
Higher fitness = better = NEGATIVE loss.
"""
import sys, pickle
from pathlib import Path
import numpy as np, torch
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
import train_imagelarge_graph_rw as T
import unified_readout as ur
from directed_graph_policy import build_directed_graph, compute_source

SEQ = 256
N = 64


@torch.no_grad()
def _cross_eval(model, val_tokens, B, source, coords, has_grid, w, ftp, ibp,
                device, k_batches=8, bs=16, seed=123):
    model.eval()
    losses = []
    for bi in range(k_batches):
        x = val_tokens[bi*bs:(bi+1)*bs].to(device)
        if x.shape[0] == 0:
            break
        ob = ur.sample_orders_batch(B, source, coords, w, x.shape[0], seed + bi, has_grid)
        bo = torch.as_tensor(ob, dtype=torch.long, device=device)
        loss = T._forward_with_block_orders(model, x, bo, fixed_token_perm=ftp, inv_block_perm=ibp)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def frozen_eval_fitness(ckpt, A_block_path, val_bin, meta, topology, grid, device="cuda:0"):
    """Returns fitness(w) = -cross_eval on the frozen baseline. Model loaded once."""
    model, margs, ckptd, ftp, ibp = T.load_baseline_model(ckpt, device)
    A = np.load(A_block_path).astype(np.float64)
    B = build_directed_graph(A)
    source, _, _ = compute_source(B, 0.5)
    coords, has = ur.make_coords(N, topology, grid)
    meta_d = pickle.load(open(meta, "rb"))
    assert int(meta_d["tokens_per_image"]) == SEQ
    vm = np.memmap(val_bin, dtype=np.uint16, mode="r")
    nval = min(512, len(vm) // SEQ)
    val = torch.from_numpy(np.asarray(vm[:nval*SEQ], dtype=np.int64).reshape(nval, SEQ))

    def fitness(w):
        return -_cross_eval(model, val, B, source, coords, has, ur.clip_params(w), ftp, ibp, device)
    return fitness


def short_continuation_fitness(*a, **k):
    raise NotImplementedError(
        "HELD: Task 2.4 short-continuation rerank is not released. Use frozen_eval_fitness.")
