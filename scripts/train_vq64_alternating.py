#!/usr/bin/env python3
"""Single-arm from-0 image MLP-refresh alternating trainer (self-bootstrapping feasibility).
See docs/superpowers/specs/2026-05-24-image-mlp-refresh-alternating-design.md."""
from __future__ import annotations
import argparse, json, math, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))

from AOGPT import AOGPTConfig, AOGPT
from directed_graph_policy import build_directed_graph
from extract_image_attention_e2 import extract_a_global
from attn_order_distill import distill_order_mlp
from attn_order_image_diag import image_refresh_diagnostics
import attn_order_mlp_policy as P
# reuse round2 helpers without modifying it:
from train_vq64_round2 import _forward_with_block_orders, evaluate_7orders, get_lr

N_BLOCKS = 64
DEFAULT_MODEL_ARGS = dict(block_size=64, vocab_size=8192, n_layer=4, n_head=8,
                          n_embd=256, dropout=0.0, bias=False,
                          block_order_block_len=1, order_impl="block")


def build_or_load_model_for_extraction(ckpt_path, device):
    """Test/utility: load an existing AOGPT ckpt (used only by extraction tests)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    keys = list(DEFAULT_MODEL_ARGS.keys())
    model_args = {k: ma[k] for k in keys if k in ma}
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, model_args


def extract_B_from_model(model, data_tokens, tokens_per_image, n_images, m_passes, device, seed):
    """Extract A_global from the CURRENT model (un-permuted physical coords) and return B=A^T.

    Reuses extract_image_attention_e2.extract_a_global verbatim; seeds torch so the random
    AO orders (hence the snapshot) are reproducible.
    """
    was_training = model.training
    model.eval()
    torch.manual_seed(int(seed))
    A = extract_a_global(model, data_tokens, tokens_per_image, n_images, m_passes, device)
    if was_training:
        model.train()
    B = build_directed_graph(np.ascontiguousarray(A.astype(np.float64)))
    return B
