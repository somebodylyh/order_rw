import json, pathlib, sys
import numpy as np, torch, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta


def _write_fake_gbeta(out_dir):
    m = L0DynamicGBeta(heads=8, nodes=65)
    ck = pathlib.Path(out_dir) / "g_beta_best.pt"
    torch.save({"model_state_dict": m.state_dict(),
                "config": {"heads": 8, "nodes": 65}}, ck)
    return str(ck)


def test_pretrain_orchestrates_and_writes_provenance(tmp_path, monkeypatch):
    import analyses.gbeta_cdl_pretrain as mod

    calls = {}
    def fake_build(ckpt_path, **kw):
        calls["build"] = {"ckpt_path": ckpt_path, **kw}
        npz = tmp_path / "ds.npz"
        np.savez(npz, teacher_pairwise=np.zeros((2, 64, 64), dtype=np.float32))
        return str(npz)
    def fake_train(dataset_path, out_dir, **kw):
        calls["train"] = {"dataset_path": dataset_path, "out_dir": out_dir, **kw}
        _write_fake_gbeta(out_dir)
        return {"best_metrics": {"pairwise_acc": 0.9}}

    monkeypatch.setattr(mod, "build_l0_dynamic_gbeta_dataset", fake_build)
    monkeypatch.setattr(mod, "train_l0_dynamic_gbeta_train", fake_train)

    out = mod.pretrain_gbeta_cdl(
        "FAKE_10k.pt", out_dir=str(tmp_path), loss_type="pairwise_bce",
        M=8, heads=8, epochs=1, seed=0, device="cpu")

    assert pathlib.Path(out).name == "g_beta_best.pt"
    assert calls["train"]["loss_type"] == "pairwise_bce"
    assert calls["build"]["ckpt_path"] == "FAKE_10k.pt"
    prov = json.loads((tmp_path / "gbeta_provenance.json").read_text())
    assert prov["producer"] == "build_l0_dynamic_gbeta_dataset + train_l0_dynamic_gbeta"
    assert prov["source_ckpt"] == "FAKE_10k.pt"
    assert prov["loss_type"] == "pairwise_bce"


def test_reuse_dataset_skips_build(tmp_path, monkeypatch):
    import analyses.gbeta_cdl_pretrain as mod
    hit = {"build": 0}
    def fake_build(*a, **k):
        hit["build"] += 1
        return "SHOULD_NOT_BE_CALLED"
    def fake_train(dataset_path, out_dir, **kw):
        _write_fake_gbeta(out_dir); return {}
    monkeypatch.setattr(mod, "build_l0_dynamic_gbeta_dataset", fake_build)
    monkeypatch.setattr(mod, "train_l0_dynamic_gbeta_train", fake_train)

    npz = tmp_path / "pre.npz"; np.savez(npz, x=np.zeros(1))
    mod.pretrain_gbeta_cdl("FAKE.pt", out_dir=str(tmp_path),
                           dataset_path=str(npz), device="cpu")
    assert hit["build"] == 0


@pytest.mark.slow
def test_producer_learns_cdl_pairwise(tmp_path):
    """Real small-M run: gβ agrees with teacher_pairwise above chance.
    pairwise_acc = mean[ sign(s_i - s_j) == sign(Y_ij - 0.5) ] (spec test 2)."""
    from analyses.gbeta_cdl_pretrain import pretrain_gbeta_cdl, DEFAULT_SOURCE_CKPT
    ck = pretrain_gbeta_cdl(DEFAULT_SOURCE_CKPT, out_dir=str(tmp_path),
                            M=64, epochs=5, device="cpu")
    ds = np.load(tmp_path / "cdl_dataset.npz")
    Y = torch.from_numpy(ds["teacher_pairwise"]).float()     # (M,64,64)
    B = torch.from_numpy(ds["B_raw"]).float()                # (M,8,65,65)
    state = torch.load(ck, map_location="cpu", weights_only=False)
    m = L0DynamicGBeta(heads=8, nodes=65)
    m.load_state_dict(state["model_state_dict"]); m.eval()
    with torch.no_grad():
        s, _ = m(B, apply_head_dropout=False)                # (M,64)
    sd = torch.sign(s.unsqueeze(2) - s.unsqueeze(1))
    yd = torch.sign(Y - 0.5)
    acc = (sd == yd).float().mean().item()
    assert acc > 0.55
