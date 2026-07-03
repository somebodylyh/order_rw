"""Batch-mean frozen gβ order provider (canonical single-head + NodewiseReadout).

Extracts the SELECTED single head's content B (batch-mean over probes then over
batch samples), runs the frozen readout, argsorts -> ONE model-frame block order
broadcast to all rows. gβ frozen, @torch.no_grad(). Which head is read comes from
the gβ ckpt/provenance (chosen by Stage-A selection on the parent backbone).
"""

import json
import os
import hashlib

import torch

from orderhead_v3.readouts import build_readout
from orderhead_v3.constants import N, BLOCK_LEN, SEQ_LEN, PERMUTE_SEED


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class GBetaFrozenProvider:
    def __init__(self, gbeta_ckpt, init_from_ckpt, *, batch_mean_probes=4,
                 refresh_every=1, seed=0, device="cpu", probe_mode="eval"):
        prov = json.load(open(os.path.join(os.path.dirname(gbeta_ckpt), "gbeta_provenance.json")))
        assert (prov["num_blocks"], prov["block_len"]) == (N, BLOCK_LEN)
        assert prov["seq_len"] == SEQ_LEN and prov["permute_seed"] == PERMUTE_SEED
        assert prov["none_mode"] == "model" and prov["strict65"] is True and prov.get("single_head")
        if prov["parent_hash"] != _file_hash(init_from_ckpt):
            raise ValueError("gβ provenance parent_hash != init_from_ckpt hash")

        st = torch.load(gbeta_ckpt, map_location="cpu", weights_only=False)
        cfg = st["config"]
        self.layer = int(cfg["sel_layer"])
        self.head = int(cfg["sel_head"])
        self.gbeta = build_readout(cfg).to(device)
        self.gbeta.load_state_dict(st["model_state_dict"])
        self.gbeta.eval()
        for p in self.gbeta.parameters():
            p.requires_grad_(False)

        self.batch_mean_probes = int(batch_mean_probes)
        self.refresh_every = max(1, int(refresh_every))
        self.seed, self.device, self.probe_mode = int(seed), device, probe_mode
        self._train_sigma = self._train_step = None
        self._eval_sigma = None

    @torch.no_grad()
    def _compute_sigma(self, model, idx_batch, global_step):
        from gbeta_cdl_pretrain import extract_single_head_batch_mean
        B = extract_single_head_batch_mean(
            model, idx_batch, self.layer, self.head, global_step=global_step,
            seed=self.seed, batch_mean_probes=self.batch_mean_probes,
            device=self.device, probe_mode=self.probe_mode)     # (Bsz, 64, 64)
        Bmean = B.mean(dim=0, keepdim=True)                     # batch-mean -> (1, 64, 64)
        scores = self.gbeta(Bmean)                              # (1, 64)
        return scores.argsort(dim=1, descending=True)[0]        # (64,)

    @torch.no_grad()
    def block_orders(self, model, idx_batch, global_step, is_eval):
        if is_eval:
            self._eval_sigma = self._compute_sigma(model, idx_batch, global_step)
            sigma = self._eval_sigma
        else:
            if self._train_sigma is None or global_step - self._train_step >= self.refresh_every:
                self._train_sigma = self._compute_sigma(model, idx_batch, global_step)
                self._train_step = int(global_step)
            sigma = self._train_sigma
        return sigma.unsqueeze(0).expand(idx_batch.shape[0], -1).to(idx_batch.device)
