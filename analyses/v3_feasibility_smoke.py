"""D2 joint co-adaptation smoke. Same-batch probe-then-train two-forward path.
per_sample arm = V3 headline config ([B,T] orders); batch_level arm = broadcast
sanity/diagnostic. Feasibility sign checks only — NOT a scientific verdict."""
import sys, json, pathlib
sys.path.insert(0, "block_lo_arm_order_network")
import numpy as np
import torch
from scipy.stats import kendalltau

from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
from analyses.p5_utility_controller import load_p5_ckpt, order_nll, N
from analyses.p7_gbeta_policy import sample_pl, GBETA_CKPT
from batch_readout.hook_order_provider import random_probe_token_orders

L2R = np.arange(N, dtype=np.int64)


def _sample_orders(z, tau):
    orders, logps, ents = [], [], []
    for r in range(z.shape[0]):
        o, lp, e = sample_pl(z[r], tau)
        orders.append(o); logps.append(lp); ents.append(e)
    return np.stack(orders), torch.stack(logps), torch.stack(ents)


def denoising_metrics(wrap, held_idx, dev):
    """Per-sample consensus τ + Δ_probe vs L2R (uses no_grad order_nll)."""
    probe = random_probe_token_orders(held_idx.shape[0], 0, 0, dev)
    sig_model = wrap.order_head.argsort_order(
        wrap.extract_B(held_idx, probe), per_sample=True)          # (M,64) model-frame
    sig_model = sig_model.cpu().numpy()
    taus = [kendalltau(sig_model[i], sig_model[j]).correlation
            for i in range(len(sig_model)) for j in range(i + 1, len(sig_model))]
    tau_consensus = float(np.nanmean(taus)) if taus else float("nan")
    deltas = []
    for i in range(held_idx.shape[0]):
        row = held_idx[i:i + 1]
        phys = wrap.inv_perm[sig_model[i]]
        deltas.append(order_nll(wrap.backbone, row, phys, wrap.clean_perm, dev)
                      - order_nll(wrap.backbone, row, L2R, wrap.clean_perm, dev))
    return {"tau_consensus": tau_consensus, "delta_probe": float(np.mean(deltas))}


def run_smoke(ckpt_path, arm, n_steps=40, batch_size=16, tau=1.0, beta=3e-3,
              lam_pg=1.0, ema_decay=0.9, adv_clip=0.3, lr=3e-4, held=32,
              device="cpu", out_dir="runs/v3_feasibility"):
    assert arm in ("per_sample", "batch_level")
    per_sample = arm == "per_sample"
    M = n_steps * batch_size + held
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    oh = OrderHeadModule(GBETA_CKPT, device=str(dev))
    for p in oh.gbeta.parameters():
        p.requires_grad_(True)
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device=str(dev))
    held_idx = torch.stack([chunks[i] for i in range(held)]).to(dev)

    before = denoising_metrics(wrap, held_idx, dev) if per_sample else None
    oh_p0 = torch.cat([p.detach().flatten().clone() for p in oh.gbeta.parameters()])

    opt = torch.optim.Adam(
        list(model.parameters()) + list(oh.gbeta.parameters()), lr=lr)
    ema, log = None, []
    for step in range(n_steps):
        s = held + step * batch_size
        idx = torch.stack([chunks[s + i] for i in range(batch_size)]).to(dev)
        probe = random_probe_token_orders(idx.shape[0], 0, step, dev)
        z = wrap.compute_order_logits(idx, probe, per_sample)     # (rows,N) grad on OH
        order_bn, logp, ent = _sample_orders(z, tau)              # rows = B or 1
        if not per_sample:                                       # broadcast one order to all rows
            order_bn = np.repeat(order_bn, batch_size, axis=0)
        token_order = wrap.token_orders_from_model_blocks(order_bn).to(dev)  # (B,T)
        _, lm_loss = model.forward_fn(idx, token_order)           # grad on backbone (scalar batch loss)
        r = lm_loss.detach()
        ema = r if ema is None else ema_decay * ema + (1 - ema_decay) * r
        # Feasibility simplification: batch-scalar advantage (forward_fn returns a
        # batch-mean loss). Per-sample credit assignment is a V3-sweep refinement.
        adv = (ema - r).clamp(-adv_clip, adv_clip)
        pg = -(adv * logp.mean()) - beta * ent.mean()
        loss = lm_loss + lam_pg * pg
        opt.zero_grad(); loss.backward(); opt.step()
        log.append({"step": step, "lm_loss": float(lm_loss), "pg": float(pg),
                    "adv": float(adv), "entropy": float(ent.mean()),
                    "logp": float(logp.mean())})

    oh_p1 = torch.cat([p.detach().flatten().clone() for p in oh.gbeta.parameters()])
    after = denoising_metrics(wrap, held_idx, dev) if per_sample else None
    result = {
        "arm": arm, "n_steps": n_steps, "batch_size": batch_size,
        "orderhead_param_delta": float((oh_p1 - oh_p0).abs().sum()),
        "entropy_first": log[0]["entropy"], "entropy_last": log[-1]["entropy"],
        "nan": any(not np.isfinite(x["lm_loss"]) for x in log),
        "denoise_before": before, "denoise_after": after, "log": log,
    }
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{arm}_smoke.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/handoff_overnight/seed123/ckpt_step10000.pt",
                    help="10k parent lineage; any load_p5_ckpt-compatible ckpt works for the smoke")
    ap.add_argument("--arm", choices=["per_sample", "batch_level"], default="per_sample")
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(json.dumps(run_smoke(a.ckpt, a.arm, n_steps=a.n_steps, device=a.device)
                     ["orderhead_param_delta"]))
