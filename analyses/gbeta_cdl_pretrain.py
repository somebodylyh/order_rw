"""Stage-1 CDL pretrain orchestrator: produce an L0DynamicGBeta gβ from a
backbone ckpt via the EXISTING two-step CDL pipeline (no CDL reimplemented).

  build_l0_dynamic_gbeta_dataset(ckpt) -> .npz  (CDL dynamic teacher)
  train_l0_dynamic_gbeta.train(.npz)   -> g_beta_best.pt  (L0DynamicGBeta)

CDL lives ONLY here; the training loop (train_clean_aogpt) never imports/runs it.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyses.build_l0_dynamic_gbeta_dataset import build_l0_dynamic_gbeta_dataset
from batch_readout.train_l0_dynamic_gbeta import train as train_l0_dynamic_gbeta_train

DEFAULT_SOURCE_CKPT = str(
    ROOT / "block_lo_arm_order_network/probe_results/"
    "overnight_20260625_random_baseline/ckpt_step10000.pt"
)


def pretrain_gbeta_cdl(ckpt_10k, *, out_dir, loss_type="pairwise_bce",
                       M=2000, heads=8, epochs=40, seed=0, device="cpu",
                       dataset_path=None):
    """CDL-pretrain an L0DynamicGBeta gβ and return the g_beta_best.pt path.

    Step 1 (skipped if ``dataset_path`` given): build a CDL dynamic-teacher
    dataset from ``ckpt_10k``. Step 2: train L0DynamicGBeta on it. A provenance
    sidecar is written to ``<out_dir>/gbeta_provenance.json``.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if dataset_path is None:
        dataset_path = build_l0_dynamic_gbeta_dataset(
            ckpt_path=ckpt_10k, M=M, seed=seed, device=device,
            out_path=str(out / "cdl_dataset.npz"))

    train_l0_dynamic_gbeta_train(
        dataset_path=dataset_path, out_dir=str(out), epochs=epochs,
        seed=seed, device=device, heads=heads, loss_type=loss_type)

    ckpt = out / "g_beta_best.pt"
    (out / "gbeta_provenance.json").write_text(json.dumps({
        "producer": "build_l0_dynamic_gbeta_dataset + train_l0_dynamic_gbeta",
        "source_ckpt": str(ckpt_10k),
        "loss_type": loss_type,
        "M": M, "heads": heads, "epochs": epochs, "seed": seed,
        "dataset_path": str(dataset_path),
    }, indent=2))
    return str(ckpt)


__all__ = ["pretrain_gbeta_cdl", "DEFAULT_SOURCE_CKPT"]
