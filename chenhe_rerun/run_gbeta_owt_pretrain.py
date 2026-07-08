"""gβ CDL pretrain on the 10k OWT scale warmup (12L/768/8h). Head re-selected on scale backbone."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbeta_cdl_pretrain import pretrain_gbeta_cdl
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "chenhe_rerun/out/rerun_owt/scale12L768_warmup10k/ckpt.pt")
OWT  = os.path.join(REPO, "chenhe_rerun/data/openwebtext/train.bin")
OUT  = os.path.join(REPO, "chenhe_rerun/out/rerun_owt/gbeta_owt_bm16")
if __name__ == "__main__":
    pretrain_gbeta_cdl(parent_ckpt=CKPT, out_dir=OUT, n_select=800, n_groups=1500,
                       batch_mean_size=16, n_reveal=8, epochs=40, lr=3e-4,
                       data_bin=OWT, device="cuda", seed=0)
