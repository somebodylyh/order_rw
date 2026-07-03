import sys, os, json
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from AOGPT_block import AOGPT, AOGPTConfig
from gbeta_cdl_pretrain import (
    load_chenhe_backbone, extract_single_head_batch_mean, pretrain_gbeta_cdl,
)

TRAIN_BIN = os.path.join(CHENHE, "data/wikitext103/train.bin")


def _tmp_parent_ckpt(tmp_path):
    cfg = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=8, n_embd=128,
               dropout=0.0, bias=True, block_order_block_len=4,
               block_order_layout="contiguous", position_encoding_mode="absolute")
    m = AOGPT(AOGPTConfig(**cfg))
    p = os.path.join(tmp_path, "ckpt.pt")
    # chenhe ckpts save extra train.py keys (order_impl) not in AOGPTConfig:
    torch.save({"model": m.state_dict(), "optimizer": {},
                "model_args": {**cfg, "order_impl": "block"}, "iter_num": 10000}, p)
    return p


def test_load_backbone_and_single_head_extract(tmp_path):
    p = _tmp_parent_ckpt(str(tmp_path))
    model, margs = load_chenhe_backbone(p, device="cpu")
    assert margs["block_order_block_len"] == 4
    idx = torch.randint(0, 50304, (2, 256))
    B = extract_single_head_batch_mean(model, idx, layer=1, head=3, global_step=0,
                                       seed=0, batch_mean_probes=3, device="cpu")
    assert B.shape == (2, 64, 64)                        # single-head content B, None stripped
    assert torch.isfinite(B).all()


def test_producer_smoke(tmp_path):
    p = _tmp_parent_ckpt(str(tmp_path))
    out = pretrain_gbeta_cdl(p, os.path.join(str(tmp_path), "gb"), n_select=16, n_groups=4,
                             batch_mean_size=2, n_reveal=4, epochs=2, device="cpu",
                             data_bin=TRAIN_BIN)
    s = torch.load(out, map_location="cpu", weights_only=False)
    assert s["config"]["model_name"] == "nodewise"
    assert "sel_layer" in s["config"] and "sel_head" in s["config"]
    prov = json.load(open(os.path.join(str(tmp_path), "gb", "gbeta_provenance.json")))
    assert prov["num_blocks"] == 64 and prov["single_head"] is True
    assert prov["readout"] == "nodewise" and "parent_hash" in prov
    assert 0 <= prov["sel_layer"] < 2 and 0 <= prov["sel_head"] < 8
