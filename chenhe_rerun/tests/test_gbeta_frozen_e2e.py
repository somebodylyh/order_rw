import sys, os, subprocess
import torch

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from AOGPT_block import AOGPT, AOGPTConfig
from gbeta_cdl_pretrain import pretrain_gbeta_cdl

PY = sys.executable  # portable: run the subprocess with the same interpreter
TRAIN_BIN = os.path.join(CHENHE, "data/wikitext103/train.bin")


def _tiny_parent(path):
    cfg = dict(block_size=256, vocab_size=50304, n_layer=2, n_head=8, n_embd=128,
               dropout=0.0, bias=True, block_order_block_len=4,
               block_order_layout="contiguous", position_encoding_mode="absolute")
    m = AOGPT(AOGPTConfig(**cfg))
    # chenhe ckpts save extra train.py keys (order_impl) not in AOGPTConfig:
    torch.save({"model": m.state_dict(), "optimizer": {},
                "model_args": {**cfg, "order_impl": "block"},
                "iter_num": 10000, "best_val_loss": 9.9}, path)


def test_gbeta_frozen_smoke(tmp_path):
    parent = os.path.join(str(tmp_path), "ckpt.pt")
    _tiny_parent(parent)
    gb_dir = os.path.join(str(tmp_path), "gb")
    gbeta = pretrain_gbeta_cdl(parent, gb_dir, n_select=16, n_groups=12, batch_mean_size=2,
                               n_reveal=4, batch_mean_probes=2, epochs=2, device="cpu",
                               data_bin=TRAIN_BIN)
    assert os.path.exists(gbeta)

    cmd = [PY, "train.py",
           "config/WikiText103/seq256/permute/block64/gbeta_frozen_warmup.py",
           f"--init_from_ckpt={parent}", f"--gbeta_ckpt={gbeta}",
           f"--out_dir={os.path.join(str(tmp_path), 'run')}",
           "--device=cpu", "--batch_size=4", "--eval_batch_size=4",
           "--gbeta_batch_mean_probes=2",
           "--max_iters=1", "--eval_interval=1", "--eval_iters=1",
           "--wandb_log=False", "--compile=False"]
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run(cmd, cwd=CHENHE, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    # init_from='ckpt' continues the parent's step counter (iter_num=10000), so
    # the first eval/log line is "step 10000" — the GBetaFrozenOrder path ran a
    # train step, evaluated, and saved a checkpoint.
    assert "step 10000" in r.stdout, r.stdout[-2000:]
    assert "saving checkpoint" in r.stdout, r.stdout[-2000:]
