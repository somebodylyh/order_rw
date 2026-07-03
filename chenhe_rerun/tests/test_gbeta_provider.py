import sys, os, json, hashlib
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from AOGPT_block import AOGPT, AOGPTConfig
from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.gbeta_provider import GBetaFrozenProvider


def _backbone_and_ckpts(tmp_path):
    cfg = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=8, n_embd=128,
               dropout=0.0, bias=True, block_order_block_len=4,
               block_order_layout="contiguous", position_encoding_mode="absolute")
    m = AOGPT(AOGPTConfig(**cfg))
    parent = os.path.join(tmp_path, "ckpt.pt")
    torch.save({"model": m.state_dict(), "model_args": cfg, "optimizer": {},
                "iter_num": 10000}, parent)
    h = hashlib.sha256(open(parent, "rb").read()).hexdigest()
    gb = L0DynamicGBeta(heads=8, nodes=65)
    gpath = os.path.join(tmp_path, "g_beta_best.pt")
    torch.save({"model_state_dict": gb.state_dict(),
                "config": {"model_name": "l0_dynamic_gbeta_v0", "heads": 8, "nodes": 65}}, gpath)
    prov = {"parent_hash": h, "num_blocks": 64, "block_len": 4, "heads": 8,
            "seq_len": 256, "permute_seed": 42, "none_mode": "model",
            "strict65": True, "probe_mode": "eval"}
    json.dump(prov, open(os.path.join(tmp_path, "gbeta_provenance.json"), "w"))
    return m, parent, gpath


def test_batch_mean_single_order_frozen(tmp_path):
    m, parent, gpath = _backbone_and_ckpts(str(tmp_path))
    prov = GBetaFrozenProvider(gpath, parent, batch_mean_probes=2,
                               refresh_every=1, device="cpu", probe_mode="eval")
    before = [p.detach().clone() for p in prov.gbeta.parameters()]
    idx = torch.randint(0, 50304, (4, 256))
    orders = prov.block_orders(m, idx, global_step=0, is_eval=False)
    assert orders.shape == (4, 64)
    # ONE order broadcast to all samples (batch-mean):
    assert (orders == orders[0:1]).all()
    # each row is a valid permutation of the 64 model-frame blocks
    assert sorted(orders[0].tolist()) == list(range(64))
    # gβ frozen: no param requires grad, unchanged:
    assert all(not p.requires_grad for p in prov.gbeta.parameters())
    after = list(prov.gbeta.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_provenance_mismatch_rejected(tmp_path):
    m, parent, gpath = _backbone_and_ckpts(str(tmp_path))
    other = os.path.join(str(tmp_path), "other.pt")
    torch.save({"model": m.state_dict(), "model_args": {}, "optimizer": {}}, other)
    import pytest
    with pytest.raises(Exception):
        GBetaFrozenProvider(gpath, other, device="cpu")  # parent hash mismatch


def test_eval_refreshes_per_batch(tmp_path):
    m, parent, gpath = _backbone_and_ckpts(str(tmp_path))
    prov = GBetaFrozenProvider(gpath, parent, refresh_every=1000,
                               device="cpu", probe_mode="eval")
    idx = torch.randint(0, 50304, (2, 256))
    prov.block_orders(m, idx, global_step=0, is_eval=False)   # train cache set
    train_sig = prov._train_sigma.clone()
    prov.block_orders(m, idx, global_step=5, is_eval=True)     # eval must recompute
    assert prov._eval_sigma is not None
    assert torch.equal(prov._train_sigma, train_sig)          # train cache unchanged
