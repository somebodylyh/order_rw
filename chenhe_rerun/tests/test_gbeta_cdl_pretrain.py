import sys, os, json
import numpy as np
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from AOGPT_block import AOGPT, AOGPTConfig
from gbeta_cdl_pretrain import (
    load_chenhe_backbone, extract_strict65_batch_mean, pretrain_gbeta_cdl,
)

TRAIN_BIN = os.path.join(CHENHE, "data/wikitext103/train.bin")


def _tmp_parent_ckpt(tmp_path):
    cfg = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=8, n_embd=128,
               dropout=0.0, bias=True, block_order_block_len=4,
               block_order_layout="contiguous", position_encoding_mode="absolute")
    m = AOGPT(AOGPTConfig(**cfg))
    p = os.path.join(tmp_path, "ckpt.pt")
    torch.save({"model": m.state_dict(), "optimizer": {}, "model_args": cfg,
                "iter_num": 10000}, p)
    return p


def test_load_backbone_and_extract(tmp_path):
    p = _tmp_parent_ckpt(str(tmp_path))
    model, margs = load_chenhe_backbone(p, device="cpu")
    assert margs["block_order_block_len"] == 4
    idx = torch.randint(0, 50304, (2, 256))
    B = extract_strict65_batch_mean(model, idx, global_step=0, seed=0,
                                    batch_mean_probes=4, device="cpu", probe_mode="eval")
    assert B.shape == (2, 8, 65, 65)
    assert torch.isfinite(B).all()


def test_producer_smoke(tmp_path):
    p = _tmp_parent_ckpt(str(tmp_path))
    out = pretrain_gbeta_cdl(p, os.path.join(str(tmp_path), "gb"), M=8,
                             batch_mean_size=2, batch_mean_probes=2, epochs=2,
                             device="cpu", data_bin=TRAIN_BIN, forward_batch=8)
    s = torch.load(out, map_location="cpu", weights_only=False)
    assert s["config"]["heads"] == 8 and s["config"]["nodes"] == 65
    prov = json.load(open(os.path.join(str(tmp_path), "gb", "gbeta_provenance.json")))
    assert prov["num_blocks"] == 64 and prov["none_mode"] == "model"
    assert "parent_hash" in prov
