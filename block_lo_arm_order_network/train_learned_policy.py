"""
Stage C: Learned policy joint training.

Resumes from RW-warmed checkpoint (v2 30k), jointly trains:
  - AOGPT: CE loss under MLP-augmented policy order
  - MLP policy: pairwise NLL ranking loss + weak KL regularization

Key design:
  - AOGPT and MLP have SEPARATE optimizer steps
  - Pairwise ranking uses detached NLL values
  - top-k masking consistent between sampling and differentiable replay
  - Policy unrestricted: allowed to learn L2R if NLL favors it
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── local imports ──
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from train_clean_aogpt import (
    N, BLOCK_LEN, SEQ_LEN,
    load_train_chunks, get_lr, alpha_for_step,
    evaluate_orders, refresh_rw_graph, order_loss, compute_token_ce,
    checkpoint_payload, write_config,
    clean_model_args, build_model, clean_state_dict,
    load_or_create_protocol, phys_to_model_idx_clean,
    sample_random_physical_orders, sample_rw_physical_orders,
    physical_blocks_to_model_token_order, physical_blocks_to_model_blocks,
    model_blocks_to_physical_blocks, expand_model_blocks_to_token_order,
    batch_indices_for_step, parse_step_list,
    build_fixed_split_and_shuffle, save_protocol_files,
)
from directed_graph_policy import build_directed_graph, _softmax, compute_source, progressive_rw_step
from mlp_residual_policy import (
    MLPResidualPolicy,
    sample_order_with_mlp,
    compute_logprob_for_order_mlp,
    compute_policy_logprob_batch,
)

ORDER_NAMES = ["policy", "rw", "l2r", "random", "perturbed_rw"]


# ──────────────────────────────────────────────────────────────────────
# Candidate generation
# ──────────────────────────────────────────────────────────────────────

def generate_candidate_orders(
    B, rw_params, batch_size, seed, global_step, micro_step,
    mlp, mlp_device, device,
):
    """Generate 5 candidate order sets for a batch.

    Returns dict: {name: (B, N) tensor on `device`}
    """
    N = B.shape[0]
    candidates = {}

    # 1. Policy (MLP-augmented RW)
    policy_rows = []
    for b in range(batch_size):
        s = seed * 1000000 + global_step * 1000 + micro_step * 100 + b
        order, _, _, _ = sample_order_with_mlp(
            B, rw_params, seed=s, mlp=mlp, mlp_device=mlp_device,
        )
        policy_rows.append(order)
    candidates["policy"] = torch.tensor(np.stack(policy_rows), dtype=torch.long, device=device)

    # 2. Pure RW (same seeds for reproducibility)
    rw_rows = []
    for b in range(batch_size):
        s = seed * 1000000 + global_step * 1000 + micro_step * 100 + b + 100000
        order, _ = sample_rw_single(B, rw_params, seed=s)
        rw_rows.append(order)
    candidates["rw"] = torch.tensor(np.stack(rw_rows), dtype=torch.long, device=device)

    # 3. L2R (physical order [0,1,...,N-1])
    candidates["l2r"] = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

    # 4. Random
    rand_rows = []
    rng_rand = np.random.default_rng(seed * 1000000 + global_step * 1000 + micro_step * 100 + 99999)
    for b in range(batch_size):
        rand_rows.append(rng_rand.permutation(N))
    candidates["random"] = torch.tensor(np.stack(rand_rows), dtype=torch.long, device=device)

    # 5. Perturbed RW (RW with epsilon_uniform=0.1 noise)
    pert_params = dict(rw_params)
    pert_params["epsilon_uniform"] = 0.1
    pert_rows = []
    for b in range(batch_size):
        s = seed * 1000000 + global_step * 1000 + micro_step * 100 + b + 200000
        order, _ = sample_rw_single(B, pert_params, seed=s)
        pert_rows.append(order)
    candidates["perturbed_rw"] = torch.tensor(np.stack(pert_rows), dtype=torch.long, device=device)

    return candidates


def sample_rw_single(B, params, seed):
    """Sample one RW order (thin wrapper for convenience)."""
    from directed_graph_policy import sample_order
    return sample_order(B, "progressive_rw", params, seed=seed)


# ──────────────────────────────────────────────────────────────────────
# Detached NLL computation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_detached_nlls(model, idx_batch, candidate_orders, clean_perm, device):
    """Compute per-sample token-avg CE (NLL) for each candidate order set.

    Batched: all K order types in one AOGPT forward pass (K× speedup).

    Args:
        model: AOGPT model
        idx_batch: (B, SEQ_LEN) model-coordinate tokens
        candidate_orders: dict {name: (B, N) physical block orders}
        clean_perm: CleanPermutation for coordinate transform
        device: torch device

    Returns:
        nlls: dict {name: (B,) float tensor}
    """
    B = idx_batch.size(0)
    order_names = list(candidate_orders.keys())
    K = len(order_names)

    # Stack all token orders into one batch: (B*K, SEQ_LEN)
    all_token_orders = []
    for name in order_names:
        tok = physical_blocks_to_model_token_order(
            candidate_orders[name], clean_perm, BLOCK_LEN,
        ).to(device)
        all_token_orders.append(tok)
    combined_orders = torch.cat(all_token_orders, dim=0)  # (B*K, SEQ_LEN)
    combined_idx = idx_batch.repeat_interleave(K, dim=0)   # (B*K, SEQ_LEN)

    token_losses, _ = compute_token_ce(model, combined_idx, combined_orders, device)
    # token_losses: (B*K, SEQ_LEN) → (B, K, SEQ_LEN) → mean over seq
    nll = token_losses.float().view(B, K, -1).mean(dim=-1)  # (B, K)

    results = {}
    for i, name in enumerate(order_names):
        results[name] = nll[:, i]
    return results


# ──────────────────────────────────────────────────────────────────────
# Pairwise ranking loss
# ──────────────────────────────────────────────────────────────────────

def compute_pairwise_ranking_loss(logprobs, nlls):
    """Pairwise ranking: lower NLL → higher logprob.

    For all pairs (i,j) where NLL_i < NLL_j:
      loss += -log(sigmoid(logprob_i - logprob_j))

    Args:
        logprobs: (B, K) policy logprobs (differentiable)
        nlls: (B, K) detached NLL values

    Returns:
        scalar loss
    """
    B, K = logprobs.shape
    total = torch.tensor(0.0, device=logprobs.device)
    n_pairs = 0

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            better = nlls[:, i] < nlls[:, j]  # (B,)
            if not better.any():
                continue
            diff = logprobs[better, i] - logprobs[better, j]  # (n_better,)
            total = total - F.logsigmoid(diff).mean()
            n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=logprobs.device, requires_grad=True)
    return total / n_pairs


# ──────────────────────────────────────────────────────────────────────
# Policy diagnostics
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_policy_diagnostics(B, rw_params, mlp, mlp_device, n_samples=100, seed=42):
    """Compute tau(policy, L2R), tau(policy, RW), and policy entropy."""
    N = B.shape[0]
    L2R = np.arange(N, dtype=np.int64)

    orders_rw = np.zeros((n_samples, N), dtype=np.int64)
    orders_mlp = np.zeros((n_samples, N), dtype=np.int64)

    for k in range(n_samples):
        s = seed * 10000 + k
        orders_rw[k], _ = sample_rw_single(B, rw_params, seed=s)
        orders_mlp[k], _, _, _ = sample_order_with_mlp(
            B, rw_params, seed=s, mlp=mlp, mlp_device=mlp_device,
        )

    from order_diagnostics import _kendall_tau

    tau_l2r = np.mean([_kendall_tau(orders_mlp[i], L2R) for i in range(n_samples)])
    tau_rw = np.mean([_kendall_tau(orders_mlp[i], orders_rw[i]) for i in range(n_samples)])

    # entropy
    from directed_graph_policy import _policy_step_entropy_replay
    H_all = np.zeros((n_samples, N), dtype=np.float64)
    for i in range(n_samples):
        H_all[i] = _policy_step_entropy_replay(B, 'progressive_rw', rw_params, orders_mlp[i],
                                                seed * 10000 + i)
    H_mean = float(H_all.mean())
    H_early = float(H_all[:, :21].mean())
    H_mid = float(H_all[:, 21:42].mean())
    H_late = float(H_all[:, 42:].mean())

    return {
        "tau_policy_l2r": float(tau_l2r),
        "tau_policy_rw": float(tau_rw),
        "H_mean": H_mean,
        "H_early": H_early,
        "H_mid": H_mid,
        "H_late": H_late,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage C: Learned policy joint training")
    p.add_argument("--resume-ckpt", required=True, help="Path to RW-warmed checkpoint (e.g., v2 30k)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--lr-decay-steps", type=int, default=50000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--max-eval-seqs", type=int, default=200)
    p.add_argument("--save-steps", default="")
    # Alpha (fixed at 0.9 since already warmed)
    p.add_argument("--alpha-start", type=float, default=0.9)
    p.add_argument("--alpha-target", type=float, default=0.9)
    p.add_argument("--alpha-warmup-steps", type=int, default=0)
    # RW params
    p.add_argument("--tau-start", type=float, default=0.1)
    p.add_argument("--tau-step", type=float, default=0.1)
    p.add_argument("--rw-top-k", type=int, default=4)
    p.add_argument("--epsilon-uniform", type=float, default=0.0)
    # Refresh
    p.add_argument("--refresh-interval", type=int, default=2000)
    p.add_argument("--refresh-n-chunks", type=int, default=200)
    p.add_argument("--refresh-ema-beta", type=float, default=0.9)
    p.add_argument("--refresh-data-source", choices=["eval", "random_train"], default="eval")
    # MLP policy
    p.add_argument("--mlp-lr", type=float, default=1e-4)
    p.add_argument("--mlp-gamma-start", type=float, default=0.0)
    p.add_argument("--mlp-hidden", type=str, default="32,16,8")
    # Loss weights
    p.add_argument("--lambda-rank", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=0.01)
    # Eval seeds
    p.add_argument("--eval-order-seeds", type=int, nargs="+", default=[42, 123, 456])
    # Model args (needed for clean_model_args / write_config)
    p.add_argument("--run-kind", default="graph_rw")  # eval uses this for train_objective
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--vocab-size", type=int, default=50304)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"Stage C: Learned policy joint training")
    log(f"Output dir: {output_dir}")
    log(f"Device: {device}")
    log(f"Lambda rank={args.lambda_rank}, kl={args.lambda_kl}, MLP lr={args.mlp_lr}")

    # ── Load checkpoint ──
    ckpt = torch.load(os.path.expanduser(args.resume_ckpt), map_location=device, weights_only=False)
    log(f"Loaded checkpoint: {args.resume_ckpt}")

    # ── Data protocol ──
    clean_perm, loaded_split = load_or_create_protocol(args, output_dir, ckpt)
    log("Loading train-arrow chunks...")
    idx_phys = load_train_chunks(n_chunks=None)
    if loaded_split is None:
        fixed = build_fixed_split_and_shuffle(
            total_chunks=idx_phys.size(0), seed=args.seed,
            val_fraction=0.05, max_eval_seqs=args.max_eval_seqs,
        )
        split = {
            "train_indices": fixed.train_indices,
            "val_indices": fixed.val_indices,
            "train_shuffle_order": fixed.train_shuffle_order,
            "eval_indices": fixed.eval_indices,
        }
    else:
        split = loaded_split
    save_protocol_files(output_dir, args, clean_perm, split)

    log(f"Train: {len(split['train_indices'])}, val: {len(split['val_indices'])}, eval: {len(split['eval_indices'])}")
    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)

    # ── Build model ──
    model_args = ckpt.get("model_args") or clean_model_args(args)
    model = build_model(model_args, device)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError("checkpoint lacks model state")
    model.load_state_dict(clean_state_dict(state_dict))
    start_step = int(ckpt.get("global_step", ckpt.get("iter_num", 0)))
    train_losses = list(ckpt.get("train_losses", []))
    log(f"Start step: {start_step}, max steps: {args.max_steps}")

    # ── Graph-RW (re-extract A from model) ──
    rw_params = {
        "tau_start": args.tau_start, "tau_step": args.tau_step,
        "alpha_dep": 0.5, "alpha_pr": 0.85,
        "beta_sup": 1.0, "beta_fut": 0.5, "beta_src": 0.2, "beta_loc": 0.5,
    }
    if args.rw_top_k > 0:
        rw_params["top_k"] = int(args.rw_top_k)
    if args.epsilon_uniform > 0.0:
        rw_params["epsilon_uniform"] = float(args.epsilon_uniform)

    idx_eval_model = idx_model[split["eval_indices"]]
    idx_train = idx_model[split["train_indices"]]
    if args.refresh_data_source == "random_train":
        extract_n = min(args.refresh_n_chunks, len(idx_train))
        rng_ref = np.random.RandomState(start_step + args.seed)
        extract_chunks = idx_train[rng_ref.choice(len(idx_train), size=extract_n, replace=False)]
    else:
        extract_n = min(args.refresh_n_chunks, len(idx_eval_model))
        extract_chunks = idx_eval_model

    B, A_global, elapsed, _ = refresh_rw_graph(
        model, extract_chunks, clean_perm, device, None,
        n_chunks=extract_n, ema_beta=0.0,
    )
    log(f"Extracted A_global in {elapsed:.1f}s")
    np.save(output_dir / "A_global_eval.npy", A_global)

    # ── Create MLP policy ──
    hidden_dims = tuple(int(x) for x in args.mlp_hidden.split(","))
    mlp = MLPResidualPolicy(gamma=args.mlp_gamma_start, hidden_dims=hidden_dims).to(device)
    mlp_device = device
    log(f"MLP policy: {sum(p.numel() for p in mlp.parameters())} params, gamma={mlp.gamma.item():.4f}")

    # ── Optimizers ──
    ao_optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )
    if ckpt is not None and "optimizer" in ckpt:
        ao_optimizer.load_state_dict(ckpt["optimizer"])
        log("Resumed AOGPT optimizer state.")
    policy_optimizer = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr, weight_decay=0.0)

    # ── Write config ──
    write_config(output_dir, args, split, clean_perm, "progressive_rw", rw_params)
    # also save extra args
    extra_cfg = {
        "mlp_lr": args.mlp_lr, "mlp_gamma_start": args.mlp_gamma_start,
        "mlp_hidden": args.mlp_hidden, "lambda_rank": args.lambda_rank,
        "lambda_kl": args.lambda_kl, "run_kind": "learned_policy",
    }
    with (output_dir / "extra_config.json").open("w") as f:
        json.dump(extra_cfg, f, indent=2)

    # ── Eval curve header ──
    eval_curve_path = output_dir / "eval_curve.tsv"
    with eval_curve_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "step", "alpha",
            "train_loss",
            "val_train_objective", "val_ori_l2r_block", "val_ar_l2r",
            "val_model_order", "val_unstructured_order", "val_rw_order",
            "lr",
            "gamma", "kl_est", "rank_loss",
            "tau_policy_l2r", "tau_policy_rw",
            "H_policy_mean", "H_policy_early", "H_policy_mid", "H_policy_late",
        ])

    # ── Eval helper ──
    def run_eval_and_save(global_step, avg_loss, lr, alpha):
        model.eval()
        mlp.eval()

        # standard eval
        metrics = evaluate_orders(
            model, idx_eval_model, clean_perm, B, "progressive_rw", rw_params, args, alpha,
        )

        # policy diagnostics
        with torch.no_grad():
            diag = compute_policy_diagnostics(B, rw_params, mlp, str(mlp_device), n_samples=100, seed=42)
            gamma_val = float(mlp.gamma.item())

            # Estimate KL using replay
            L2R = np.arange(N, dtype=np.int64)
            l2r_lp_mlp = compute_logprob_for_order_mlp(B, rw_params, L2R, mlp=mlp, mlp_device=str(mlp_device))
            l2r_lp_rw = compute_logprob_for_order_mlp(B, rw_params, L2R, mlp=None, mlp_device="cpu")
            kl_est = l2r_lp_mlp - l2r_lp_rw  # positive if MLP gives higher prob to L2R

        # write eval curve
        with eval_curve_path.open("a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                global_step, f"{alpha:.6f}",
                f"{avg_loss:.6f}",
                f"{metrics['val_train_objective']['loss_token_avg']:.6f}",
                f"{metrics['val_ori_l2r_block']['loss_token_avg']:.6f}",
                f"{metrics['val_ar_l2r']['loss_token_avg']:.6f}",
                f"{metrics['val_model_order']['loss_token_avg']:.6f}",
                f"{metrics['val_unstructured_order']['loss_token_avg']:.6f}",
                f"{metrics['val_rw_order']['loss_token_avg']:.6f}",
                f"{lr:.8e}",
                f"{gamma_val:.6f}", f"{kl_est:.6f}", "0",  # rank_loss placeholder
                f"{diag['tau_policy_l2r']:.4f}", f"{diag['tau_policy_rw']:.4f}",
                f"{diag['H_mean']:.4f}", f"{diag['H_early']:.4f}",
                f"{diag['H_mid']:.4f}", f"{diag['H_late']:.4f}",
            ])

        log(
            f"[Eval @ {global_step}] train_obj={metrics['val_train_objective']['loss_token_avg']:.4f} | "
            f"ori_l2r={metrics['val_ori_l2r_block']['loss_token_avg']:.4f} | "
            f"rw={metrics['val_rw_order']['loss_token_avg']:.4f} | "
            f"gamma={gamma_val:.4f} | KL={kl_est:.4f} | "
            f"tau(L2R)={diag['tau_policy_l2r']:.4f} tau(RW)={diag['tau_policy_rw']:.4f}"
        )

        model.train()
        mlp.train()
        return metrics

    # ── Training loop ──
    model.train()
    mlp.train()
    t0 = time.time()
    save_steps = parse_step_list(args.save_steps)
    next_refresh_step = start_step + args.refresh_interval if args.refresh_interval > 0 else None

    # Cumulative rank loss tracking
    cum_rank_loss = 0.0
    rank_loss_count = 0

    for global_step in range(start_step, args.max_steps):
        alpha = alpha_for_step(global_step, start_step, args)

        for micro_step in range(args.grad_accum):
            batch_indices = batch_indices_for_step(
                split["train_shuffle_order"], global_step, micro_step, args.batch_size, args.grad_accum,
            )
            idx_batch = idx_model[batch_indices].to(device)

            # 1. Generate candidate orders
            candidates = generate_candidate_orders(
                B, rw_params, args.batch_size, args.seed, global_step, micro_step,
                mlp, str(mlp_device), device,
            )

            # 2. Compute detached NLLs
            nlls_dict = compute_detached_nlls(model, idx_batch, candidates, clean_perm, device)
            nlls_tensor = torch.stack([nlls_dict[k] for k in ORDER_NAMES], dim=1)  # (B, K)
            nlls_detached = nlls_tensor.detach()

            # 3. Compute AOGPT CE loss (policy-sampled order)
            loss_ao = order_loss(model, idx_batch, candidates["policy"], clean_perm, device)
            loss_ao = loss_ao / args.grad_accum

            # 4. Backward AOGPT
            loss_ao.backward()

            # 5. Compute differentiable policy logprobs
            candidate_np = np.stack(
                [candidates[k].cpu().numpy() for k in ORDER_NAMES], axis=1,
            )  # (B, K, N)
            policy_logprobs = compute_policy_logprob_batch(
                B, rw_params, candidate_np, mlp, str(mlp_device),
            )  # (B, K)

            # 6. Pairwise ranking loss
            loss_rank = compute_pairwise_ranking_loss(policy_logprobs, nlls_detached)
            cum_rank_loss += float(loss_rank.item())
            rank_loss_count += 1

            # 7. KL regularization (policy vs RW)
            kl_loss = (policy_logprobs[:, 0] - policy_logprobs[:, 1].detach()).mean()

            # 8. Total policy loss
            loss_policy = (args.lambda_rank * loss_rank + args.lambda_kl * kl_loss) / args.grad_accum

            # 9. Backward policy (separate from AOGPT!)
            loss_policy.backward()

        # ── AOGPT optimizer step ──
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(global_step, args)
        for group in ao_optimizer.param_groups:
            group["lr"] = lr
        ao_optimizer.step()
        ao_optimizer.zero_grad(set_to_none=True)

        # ── MLP optimizer step ──
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), args.grad_clip)
        policy_optimizer.step()
        policy_optimizer.zero_grad(set_to_none=True)

        # ── Logging ──
        avg_loss = float(loss_ao.item() * args.grad_accum)
        train_losses.append(avg_loss)
        next_step = global_step + 1

        if global_step % args.log_interval == 0:
            elapsed = time.time() - t0
            avg_rank = cum_rank_loss / max(rank_loss_count, 1) if rank_loss_count > 0 else 0.0
            log(
                f"step {global_step:5d}->{next_step:5d}/{args.max_steps} | "
                f"loss={avg_loss:.4f} | rank={avg_rank:.4f} | "
                f"gamma={mlp.gamma.item():.4f} | lr={lr:.2e} | {elapsed:.0f}s"
            )

        # ── Eval ──
        if next_step % args.eval_interval == 0 or next_step in save_steps or next_step == args.max_steps:
            run_eval_and_save(next_step, avg_loss, lr, alpha_for_step(next_step, start_step, args))

        # ── Refresh ──
        if (next_refresh_step is not None
                and next_step >= next_refresh_step):
            if args.refresh_data_source == "random_train":
                extract_n = min(args.refresh_n_chunks, len(idx_train))
                rng = np.random.RandomState(next_step + args.seed)
                refresh_chunks = idx_train[rng.choice(len(idx_train), size=extract_n, replace=False)]
                log(f"[Refresh @ {next_step}] Extracting A from model on {extract_n} random train chunks...")
            else:
                extract_n = min(args.refresh_n_chunks, len(idx_eval_model))
                refresh_chunks = idx_eval_model
                log(f"[Refresh @ {next_step}] Extracting A from model on {extract_n} chunks...")
            B, A_global, elapsed, n_extracted = refresh_rw_graph(
                model, refresh_chunks, clean_perm, device, A_global,
                n_chunks=extract_n, ema_beta=args.refresh_ema_beta,
            )
            log(f"[Refresh @ {next_step}] Done in {elapsed:.1f}s, A_global saved")
            np.save(output_dir / f"A_global_step{next_step}.npy", A_global)
            np.save(output_dir / "A_global_eval.npy", A_global)
            next_refresh_step = next_step + args.refresh_interval

        # ── Save checkpoint ──
        if next_step in save_steps or next_step == args.max_steps:
            payload = checkpoint_payload(
                model, ao_optimizer, args, clean_perm, split, next_step,
                train_losses, {}, "progressive_rw", rw_params,
            )
            # Also save MLP state
            payload["mlp_policy"] = mlp.state_dict()
            payload["mlp_optimizer"] = policy_optimizer.state_dict()
            payload["mlp_gamma"] = float(mlp.gamma.item())
            path = output_dir / f"ckpt_step{next_step}.pt"
            torch.save(payload, path)
            log(f"Saved checkpoint: {path}")

        cum_rank_loss = 0.0
        rank_loss_count = 0

    log(f"Done. eval_curve={eval_curve_path}")


if __name__ == "__main__":
    main()
