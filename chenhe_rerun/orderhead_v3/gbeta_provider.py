"""Batch-mean frozen gβ order provider for chenhe train.py (Stage-2 deploy).

Given (model, idx_batch, global_step, is_eval), returns block_orders[B, 64]:
  - extract batch-mean strict65 B (probe-averaged, then mean over batch samples)
  - frozen gβ(B) -> argsort -> ONE model-frame block order, broadcast to all rows
All @torch.no_grad(). gβ is frozen (requires_grad=False, never in optimizer).

Cache: first call fresh; training refreshes every `refresh_every` steps (reuse
between); eval refreshes every eval batch; train/eval caches are separate.
"""

import json
import os
import hashlib

import torch

from orderhead_v3.l0_dynamic_gbeta import L0DynamicGBeta
from orderhead_v3.constants import N, BLOCK_LEN, HEADS, SEQ_LEN, PERMUTE_SEED


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class GBetaFrozenProvider:
    def __init__(self, gbeta_ckpt, init_from_ckpt, *, batch_mean_probes=4,
                 refresh_every=1, seed=0, device="cpu", probe_mode="eval"):
        prov_path = os.path.join(os.path.dirname(gbeta_ckpt), "gbeta_provenance.json")
        prov = json.load(open(prov_path))
        # ── hard config asserts (layout / frame / strict65) ──
        assert (prov["num_blocks"], prov["block_len"], prov["heads"]) == (N, BLOCK_LEN, HEADS), \
            f"gβ layout {prov['num_blocks'], prov['block_len'], prov['heads']} != {(N, BLOCK_LEN, HEADS)}"
        assert prov["seq_len"] == SEQ_LEN and prov["permute_seed"] == PERMUTE_SEED
        assert prov["none_mode"] == "model" and prov["strict65"] is True
        # ── provenance binding: gβ was pretrained from THIS parent ──
        if prov["parent_hash"] != _file_hash(init_from_ckpt):
            raise ValueError("gβ provenance parent_hash != init_from_ckpt hash")

        st = torch.load(gbeta_ckpt, map_location="cpu", weights_only=False)
        cfg = st["config"]
        self.gbeta = L0DynamicGBeta(
            heads=cfg["heads"], nodes=cfg["nodes"],
            scorer_hidden=tuple(cfg.get("scorer_hidden", (256, 64))),
            gate_hidden=cfg.get("gate_hidden", 32),
        ).to(device)
        self.gbeta.load_state_dict(st["model_state_dict"])
        self.gbeta.eval()
        for p in self.gbeta.parameters():
            p.requires_grad_(False)

        self.batch_mean_probes = int(batch_mean_probes)
        self.refresh_every = max(1, int(refresh_every))
        self.seed, self.device, self.probe_mode = int(seed), device, probe_mode
        self._train_sigma = None
        self._train_step = None
        self._eval_sigma = None

    @torch.no_grad()
    def _compute_sigma(self, model, idx_batch, global_step):
        # import here to avoid a heavy import at module load
        from gbeta_cdl_pretrain import extract_strict65_batch_mean
        B = extract_strict65_batch_mean(
            model, idx_batch, global_step=global_step, seed=self.seed,
            batch_mean_probes=self.batch_mean_probes, device=self.device,
            probe_mode=self.probe_mode)                 # (Bsz, 8, 65, 65)
        Bmean = B.mean(dim=0, keepdim=True)             # batch-mean -> (1, 8, 65, 65)
        scores, _ = self.gbeta(Bmean, apply_head_dropout=False)  # (1, 64)
        return scores.argsort(dim=1, descending=True)[0]         # (64,)

    @torch.no_grad()
    def block_orders(self, model, idx_batch, global_step, is_eval):
        if is_eval:                                     # per-eval-batch refresh
            self._eval_sigma = self._compute_sigma(model, idx_batch, global_step)
            sigma = self._eval_sigma
        else:
            if (self._train_sigma is None or
                    global_step - self._train_step >= self.refresh_every):
                self._train_sigma = self._compute_sigma(model, idx_batch, global_step)
                self._train_step = int(global_step)
            sigma = self._train_sigma
        return sigma.unsqueeze(0).expand(idx_batch.shape[0], -1).to(idx_batch.device)
