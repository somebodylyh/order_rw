import os, tempfile, torch, pytest

CKPT = "probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt"
VQ_BIN = "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin"


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(VQ_BIN)),
                    reason="VQ ckpt/data missing")
def test_pretrain_gbeta_cdl_on_vq_smoke():
    import sys
    sys.path.insert(0, "chenhe_rerun")
    from gbeta_cdl_pretrain import pretrain_gbeta_cdl
    with tempfile.TemporaryDirectory() as d:
        path = pretrain_gbeta_cdl(
            parent_ckpt=CKPT, out_dir=d,
            n_select=64, n_groups=16, batch_mean_size=16, n_reveal=2,
            epochs=2, data_bin=VQ_BIN, device="cpu", seed=0)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(d, "gbeta_provenance.json"))
        st = torch.load(path, map_location="cpu", weights_only=False)
        assert st["config"]["N"] == 64
        assert 0 <= st["config"]["sel_layer"] < 8
        assert 0 <= st["config"]["sel_head"] < 8
