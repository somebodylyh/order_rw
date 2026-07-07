"""gβ CDL pretrain on the 15k warmup VQ backbone (earlier = more generic signal)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbeta_cdl_pretrain import pretrain_gbeta_cdl
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step15000.pt")
VQ_BIN = os.path.join(REPO, "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin")
OUT = os.path.join(REPO, "out/rerun_vq/gbeta_vq15k_bm16")
if __name__ == "__main__":
    pretrain_gbeta_cdl(parent_ckpt=CKPT, out_dir=OUT, n_select=800, n_groups=1500,
                       batch_mean_size=16, n_reveal=8, epochs=40, lr=3e-4,
                       data_bin=VQ_BIN, device="cuda", seed=0)
