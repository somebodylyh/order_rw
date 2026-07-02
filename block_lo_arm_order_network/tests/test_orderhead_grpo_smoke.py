"""Integration smoke test: GRPO code paths are importable and structurally correct.

This is a FAST test (< 1 second) that verifies:
  1. ``grpo_advantage`` is importable from ``orderhead_pg``.
  2. The GRPO branch in ``train_clean_aogpt`` parses without error.
  3. ``grpo_advantage`` and ``--pg-update`` are referenced in the training script.

Full-training integration smoke (subprocess) is too slow for CPU CI (~3 min/step
with data loading). That verification was done manually during implementation
(see commit 2924099: 3 steps CPU, all diagnostics logged correctly).

For GPU micro-smoke, run directly:
    cd block_lo_arm_order_network
    python train_clean_aogpt.py \\
      --run-kind frozen_beta \\
      --frozen-beta-ckpt ../reports/uniform_label_free_v1/nodewise_K1000.pt \\
      --resume-ckpt <backbone_ckpt> \\
      --batch-mean-probes 4 \\
      --unfreeze-orderhead-at-step <start_step> \\
      --pg-update grpo --pg-k 4 --pg-tau 0.05 --pg-beta 0 \\
      --pg-logp-normalize length --orderhead-lr 3e-5 --lam-pg 1e-2 \\
      --pg-adv-clip 0.1 --pg-group-m 8 \\
      --max-steps <start_step+250> --batch-size 8 \\
      --device cuda --wandb-mode online
"""

import os
import pathlib
import sys

# Ensure repo root + block_lo_arm_order_network are on sys.path (match the
# convention used by test_orderhead_pg.py and train_clean_aogpt.py).
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest


class TestGRPOImportAndConfig:

    def test_grpo_advantage_importable(self):
        """grpo_advantage is importable and callable."""
        from batch_readout.orderhead_pg import grpo_advantage
        import torch
        adv = grpo_advantage(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert adv.shape == (4,)
        assert not adv.requires_grad

    def test_grpo_branch_parseable_and_wired(self):
        """train_clean_aogpt.py parses and references grpo_advantage + --pg-update."""
        import ast
        path = os.path.join(
            os.path.dirname(__file__), "..", "train_clean_aogpt.py")
        with open(path) as f:
            source = f.read()
        # AST parse: no syntax errors.
        ast.parse(source)
        # Source-level wiring check.
        assert "grpo_advantage" in source, (
            "grpo_advantage not referenced in train_clean_aogpt.py "
            "— GRPO branch may not be wired"
        )
        assert "--pg-update" in source, (
            "--pg-update CLI flag not found in train_clean_aogpt.py"
        )
        assert "pl_argsort" in source, (
            "pl_argsort not referenced — greedy backbone order may be missing"
        )
