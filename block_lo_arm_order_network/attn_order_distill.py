"""Mid-run-callable C-D+L distillation of the OrderMLP from a directed graph B.

Factored out of train_attn_order_mlp.run() so the alternating trainer can (re)train
beta at each refresh. Pure / CPU (N=64 blocks is tiny). No NLL probe, no geometry —
the teacher is C-D+L on B only (attention-only), matching the validated arm.
"""
from __future__ import annotations

import numpy as np
import torch

from train_attn_order_mlp import OrderMLP, make_dataset, kl_terms, mean_teacher_entropy


def distill_order_mlp(B, *, mlp=None, n_orders=200, tau_T=0.5, tau_train=0.5,
                      epochs=60, lr=1e-3, batch_states=256, hidden=64, layers=2,
                      act="gelu", seed=0, device="cpu", val_frac=0.2):
    """Distill (or finetune) an OrderMLP to the C-D+L teacher on B.

    If `mlp` is None a fresh OrderMLP is created (scratch); otherwise the passed module
    is warm-started in place (finetune) and returned as the SAME object. Distillation runs
    on CPU; the returned module is moved to `device`. Returns (mlp, diag) where diag has
    val_kl / top1 / top4 / student_entropy / teacher_entropy / kl0_untrained / n_states.
    """
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)

    data = make_dataset(B, n_orders, tau_T, int(seed), standardize=True)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(data))
    n_val = max(1, int(val_frac * len(data)))
    val = [data[i] for i in idx[:n_val]]
    train = [data[i] for i in idx[n_val:]]

    if mlp is None:
        mlp = OrderMLP(hidden=hidden, layers=layers, act=act)
    mlp = mlp.to("cpu")
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    teach_ent = mean_teacher_entropy(val)
    mlp.eval()
    kl0, _, _, _ = kl_terms(mlp, val, tau_train)

    for _ep in range(1, int(epochs) + 1):
        mlp.train()
        order = rng.permutation(len(train))
        for b in range(0, len(train), batch_states):
            batch = [train[i] for i in order[b:b + batch_states]]
            opt.zero_grad()
            kl_acc = 0.0
            for X, pT in batch:
                logp = torch.log_softmax(mlp(X) / tau_train, dim=0)
                kl_acc = kl_acc + (pT * (torch.log(pT + 1e-12) - logp)).sum()
            (kl_acc / len(batch)).backward()
            opt.step()

    mlp.eval()
    vkl, vt1, vt4, vent = kl_terms(mlp, val, tau_train)
    diag = dict(val_kl=round(vkl, 4), top1=round(vt1, 4), top4=round(vt4, 4),
                student_entropy=round(vent, 4), teacher_entropy=round(teach_ent, 4),
                kl0_untrained=round(kl0, 4), n_states=len(data), epochs=int(epochs))
    return mlp.to(device), diag
