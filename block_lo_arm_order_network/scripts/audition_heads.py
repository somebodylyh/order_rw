"""Label-free head audition: short-train CDL hook, compare val_ori_l2r drop vs random.

For each candidate head h:
  1. Load base checkpoint θ_{t0}
  2. Run N steps with CDL teacher order (h,+) and reversed order (h,-)
  3. Run N steps with random order (paired: same data windows, same init)
  4. Every eval_interval steps, eval val_ori_l2r on a fixed validation set
  5. Report ΔL = L_random_val_l2r - L_head_val_l2r (positive = head better)
  6. Select best head as argmax cumulative advantage

No L2R labels in training. No oracle τ. Orientation selected by audition, not convention.
"""
from __future__ import annotations
import argparse, sys, os, time, copy, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from training_utils import SEQ_LEN, N, BLOCK_LEN
from clean_training_protocol import build_phys_to_model_token_gather
from train_clean_aogpt import (
    build_model, clean_model_args, clean_state_dict,
    order_loss, alpha_for_step, checkpoint_payload, parse_step_list,
    load_token_stream, sample_stream_batch, load_or_create_protocol,
    physical_blocks_to_model_blocks, expand_model_blocks_to_token_order,
    compute_token_ce,
)
from batch_readout.cdl_order_provider import CdlOrderProvider
from attn_order_teacher import rollout_order
from per_head_order_scan import extract_per_head_and_heavy_A
from neural_readout.extract_b import _load_model_and_chunks


def _A_to_B(A):
    B = np.asarray(A.T, dtype=np.float64).copy(); np.fill_diagonal(B, 0.0); return B


def reverse_order(order_phys):
    """Reverse a physical-frame block order."""
    return order_phys.flip(0)


@torch.no_grad()
def eval_val_ori_l2r(model, idx_eval_model, clean_perm, device, eval_batch_size=32):
    """Compute val_ori_l2r_block loss on a fixed eval set.

    Returns per-token average CE loss.
    """
    was_training = model.training
    model.eval()

    physical_l2r = torch.arange(N, dtype=torch.long)
    ori_model = physical_blocks_to_model_blocks(physical_l2r, clean_perm)

    total_ce = 0.0
    total_tokens = 0
    for start in range(0, idx_eval_model.size(0), eval_batch_size):
        stop = min(start + eval_batch_size, idx_eval_model.size(0))
        idx_batch = idx_eval_model[start:stop].to(device)
        orders = ori_model.unsqueeze(0).expand(stop - start, -1).to(device)
        token_orders = expand_model_blocks_to_token_order(orders, BLOCK_LEN).to(device)
        token_losses, _ = compute_token_ce(model, idx_batch, token_orders, device)
        total_ce += float(token_losses.float().sum().item())
        total_tokens += int(token_losses.numel())

    if was_training:
        model.train()
    return total_ce / max(total_tokens, 1)


def run_short_audition(model, optimizer_state, cdl_provider, args, stream_train,
                        g_gather, clean_perm, device, n_steps, orientation,
                        idx_eval_model, eval_interval=20, eval_batch_size=32):
    """Run N training steps and return list of (step, val_ori_l2r_loss).

    orientation: '+' = CDL order, '-' = reversed CDL order, 'random' = random
    """
    model.train()
    eval_losses = []

    # Eval at step 0 (before any training)
    loss0 = eval_val_ori_l2r(model, idx_eval_model, clean_perm, device, eval_batch_size)
    eval_losses.append((0, loss0))

    for local_step in range(n_steps):
        global_step = local_step
        alpha = 1.0  # always use CDL order for audition

        optimizer = model.configure_optimizers(
            weight_decay=args.weight_decay, learning_rate=args.lr,
            betas=(args.beta1, args.beta2),
            device_type="cuda" if device.type == "cuda" else "cpu")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        optimizer.zero_grad(set_to_none=True)

        for micro in range(args.grad_accum):
            idx_batch = sample_stream_batch(
                stream_train, args.batch_size, SEQ_LEN,
                args.seed, global_step, micro
            )[:, g_gather].to(device)

            if orientation == "random":
                from train_clean_aogpt import sample_random_physical_orders
                phys = sample_random_physical_orders(
                    args.batch_size, args.seed, global_step, micro, device)
            else:
                sigma_phys = cdl_provider.physical_order(model, idx_batch, global_step).to(device)
                if orientation == "-":
                    sigma_phys = reverse_order(sigma_phys)
                phys = sigma_phys.unsqueeze(0).expand(args.batch_size, -1)

            loss = order_loss(model, idx_batch, phys, clean_perm, device)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Eval val_ori_l2r every eval_interval steps
        if (local_step + 1) % eval_interval == 0:
            val_loss = eval_val_ori_l2r(model, idx_eval_model, clean_perm, device, eval_batch_size)
            eval_losses.append((local_step + 1, val_loss))

    # Final eval
    if n_steps % eval_interval != 0:
        val_loss = eval_val_ori_l2r(model, idx_eval_model, clean_perm, device, eval_batch_size)
        eval_losses.append((n_steps, val_loss))

    return eval_losses


def main():
    ap = argparse.ArgumentParser(description="Head audition via short CDL hook training")
    ap.add_argument("--ckpt", required=True, help="base checkpoint θ_{t0}")
    ap.add_argument("--heads", nargs="+", default=["0,7", "1,1", "0,1", "0,4"],
                    help="candidate heads as 'L,H' pairs")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--cdl-refresh", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--eval-interval", type=int, default=20,
                    help="eval val_ori_l2r every N steps")
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--n-eval-windows", type=int, default=200,
                    help="number of fixed eval windows for val_ori_l2r")
    args_ap = ap.parse_args()

    device = torch.device(args_ap.device)
    out_path = args_ap.out or os.path.join(
        PKG, "probe_results", "audition",
        f"audition_{os.path.basename(args_ap.ckpt).replace('.pt','')}_{args_ap.n_steps}steps.csv")

    # ── Build minimal args namespace (avoid parse_args sys.argv conflict) ──
    from argparse import Namespace
    train_args = Namespace(
        run_kind="baseline", data_source="continuous",
        train_bin="/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin",
        val_bin="/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin",
        seed=args_ap.seed, permute_seed=args_ap.seed,
        batch_size=64, grad_accum=1, lr=1e-3, min_lr=1e-4, lr_decay_steps=50000,
        weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
        warmup_iters=2000, max_steps=60000, eval_interval=500, log_interval=50,
        vocab_size=50304, n_layer=4, n_head=8, n_embd=384, dropout=0.0, bias=False,
        a_path=None, refresh_interval=2000, refresh_n_chunks=20,
        refresh_data_source="eval", refresh_ema_beta=0.9,
        eval_order_seeds=[0,1,2], max_eval_seqs=16, eval_batch_size=32,
        save_steps="", output_dir="", resume_ckpt="", val_fraction=0.005,
        alpha_start=0.0, alpha_target=1.0, alpha_warmup_steps=5000,
        alpha_warmup_start=0, alpha_ramp_from_resume=False,
        tau_start=0.1, tau_step=0, rw_top_k=4, rw_order_bag_k=1,
        rw_policy="progressive_rw_v3", pos_tau=1.0, rw_lam=0.0, rw_rho=0.0,
        epsilon_uniform=0.0,
        mlp_path="", mlp_orientation="source_start", mlp_tau=0.1, mlp_src_rho=0.0,
        mlp_graph="source_start", mlp_alternating=False, mlp_refresh_mode="finetune",
        mlp_distill_n_orders=1000, mlp_distill_tau_t=1.0, mlp_distill_tau_train=1.0,
        mlp_distill_epochs=40, mlp_distill_lr=1e-3, mlp_distill_batch_states=64,
        frozen_beta_ckpt="", frozen_beta_head=[0,7], frozen_beta_mode="argsort",
        frozen_beta_tau=1.0, frozen_beta_refresh=10,
        cdl_teacher_head=[0,7], cdl_teacher_tau=1.0, cdl_teacher_refresh=10,
        device=args_ap.device,
    )

    # ── Load base model ──
    ckpt = torch.load(args_ap.ckpt, map_location=device, weights_only=False)
    model_args = ckpt.get("model_args") or clean_model_args(train_args)
    model = build_model(model_args, device, compile_model=False)
    base_state_dict = clean_state_dict(ckpt.get("model") or ckpt.get("model_state_dict"))
    model.load_state_dict(base_state_dict)
    optimizer_state = ckpt.get("optimizer")

    clean_perm, _ = load_or_create_protocol(train_args, os.path.dirname(args_ap.ckpt), ckpt)
    g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)
    stream_train = load_token_stream(train_args.train_bin)
    stream_val = load_token_stream(train_args.val_bin)

    # ── Fixed eval set for val_ori_l2r ──
    eval_phys = sample_stream_batch(
        stream_val, args_ap.n_eval_windows, SEQ_LEN,
        seed=args_ap.seed + 9999, step=-1, micro=0,
    )
    idx_eval_model = eval_phys[:, g_gather].contiguous().to(device)
    print(f"Fixed eval set: {idx_eval_model.size(0)} windows × {SEQ_LEN} tokens")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Paired random baseline first ──
    print(f"Audition: {len(args_ap.heads)} heads × ({args_ap.n_steps} steps + reverse + random)")
    print(f"Base: {args_ap.ckpt}")
    print(f"Metric: val_ori_l2r (CE loss on fixed eval set, lower = better)")

    # Reload model for random baseline
    model.load_state_dict(base_state_dict)
    dummy_provider = CdlOrderProvider(
        head=(0, 0), clean_perm=clean_perm, refresh_every=99999,
        tau_T=1.0, mode="C-D+L", seed=args_ap.seed, device=str(device),
    )
    random_losses = run_short_audition(
        model, optimizer_state, dummy_provider, train_args, stream_train,
        g_gather, clean_perm, device, args_ap.n_steps, orientation="random",
        idx_eval_model=idx_eval_model, eval_interval=args_ap.eval_interval,
        eval_batch_size=args_ap.eval_batch_size,
    )
    print(f"  random baseline val_ori_l2r: step0={random_losses[0][1]:.4f} "
          f"→ step{random_losses[-1][0]}={random_losses[-1][1]:.4f}")

    # ── Per-head audition ──
    results = []
    for head_str in args_ap.heads:
        l, h = map(int, head_str.split(","))
        head = (l, h)

        for orient, orient_label in [("+", "fwd"), ("-", "rev")]:
            # Reload model fresh
            model.load_state_dict(base_state_dict)

            cdl_provider = CdlOrderProvider(
                head=head, clean_perm=clean_perm, refresh_every=args_ap.cdl_refresh,
                tau_T=1.0, mode="C-D+L", seed=args_ap.seed, device=str(device),
            )

            t0 = time.time()
            losses = run_short_audition(
                model, optimizer_state, cdl_provider, train_args, stream_train,
                g_gather, clean_perm, device, args_ap.n_steps, orientation=orient,
                idx_eval_model=idx_eval_model, eval_interval=args_ap.eval_interval,
                eval_batch_size=args_ap.eval_batch_size,
            )
            elapsed = time.time() - t0

            # Compute advantage: L_random_val_l2r - L_head_val_l2r at each eval point
            # (positive = head better than random)
            advantage = []
            for (step_r, lr), (step_h, lh) in zip(random_losses, losses):
                assert step_r == step_h, f"step mismatch: {step_r} vs {step_h}"
                advantage.append((step_r, lr - lh))

            total_adv = sum(a for _, a in advantage)
            final_adv = advantage[-1][1] if advantage else 0.0
            step0_loss = losses[0][1]
            final_loss = losses[-1][1]

            print(f"  L{l}H{h} {orient_label}: val_ori_l2r {step0_loss:.4f} → {final_loss:.4f}  "
                  f"final_adv={final_adv:+.4f}  total_adv={total_adv:+.4f}  "
                  f"{elapsed:.0f}s", flush=True)

            results.append({
                "head": f"L{l}H{h}", "orientation": orient_label,
                "losses": losses, "advantage": advantage,
                "final_advantage": final_adv, "total_advantage": total_adv,
                "step0_val_l2r": step0_loss, "final_val_l2r": final_loss,
                "elapsed_s": elapsed,
            })

    # ── Select best ──
    results.sort(key=lambda x: -x["total_advantage"])
    best = results[0]
    print(f"\n  Best: {best['head']} {best['orientation']} "
          f"total_adv={best['total_advantage']:+.4f} final_adv={best['final_advantage']:+.4f} "
          f"val_ori_l2r: {best['step0_val_l2r']:.4f} → {best['final_val_l2r']:.4f}")

    # ── Save CSV ──
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "orientation", "step0_val_l2r", "final_val_l2r",
                     "final_advantage", "total_advantage", "elapsed_s"])
        for r in results:
            w.writerow([r["head"], r["orientation"], r["step0_val_l2r"], r["final_val_l2r"],
                        r["final_advantage"], r["total_advantage"], r["elapsed_s"]])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
