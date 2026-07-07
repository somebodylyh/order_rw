"""Run gβ CDL pretrain on the VQ image backbone (Imagenet64 VQ patch2x2).

Zero-prior: gβ reads only the model's own patch attention; CDL(C-D+L) teacher.
Reuses the text AMOR pipeline unchanged — the VQ backbone IS the text AOGPT arch
(vocab_size=8192). Head is re-selected on the VQ backbone (may differ from text L1H6).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbeta_cdl_pretrain import pretrain_gbeta_cdl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt")
VQ_BIN = os.path.join(REPO, "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin")
OUT = os.path.join(REPO, "out/rerun_vq/gbeta_vq_bm16")

if __name__ == "__main__":
    pretrain_gbeta_cdl(
        parent_ckpt=CKPT, out_dir=OUT,
        n_select=800, n_groups=1500, batch_mean_size=16, n_reveal=8,
        epochs=40, lr=3e-4, data_bin=VQ_BIN, device="cuda", seed=0)
