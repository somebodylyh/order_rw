#!/usr/bin/env python3
"""Generic 8x8 VQ Graph-RW continuation trainer for nanogpt-learned-order AOGPT.

Round-2 policies: random | graph_rw | hilbert | Bcov_balanced | raster.
Loads a ckpt with 64 block positions, samples block-level orders (N=64), expands
to token orders via model._expand_block_orders_to_token_orders,
trains under standard CE with an alpha-mixed random-order regularizer.

Interface notes (verified against source):
- model.forward(idx, mode=None, orders=token_orders) → tuple (logits, loss, ...)
  orders must be token-level (batch, 256), NOT block-level.
  _expand_block_orders_to_token_orders converts (batch,64) → (batch,256).
- sample_image_orders_batch returns (batch_size, 64) block-level orders.
- The baseline ckpt was trained with permute_data=True (permute_mode='block').
  Before each forward pass, input tokens must be reordered by fixed_token_perm
  (derived from block_perm stored in ckpt['data_permutation']). Without this
  the model sees out-of-distribution input and val_loss is ~0.6 nats higher.
- COORDINATE FRAME: A_block (and therefore B and all RW/raster orders built
  from it) live in PHYSICAL raster coordinates. The permuted-data model lives
  in MODEL coordinates (model block position p holds physical block block_perm[p]).
  So any physically-defined block order MUST be remapped to model coordinates
  via inverse_block_perm before being fed to the model:
      model_block_order = inverse_block_perm[phys_block_order]
  This matches train.py's own original-L2R eval (it feeds inverse_block_perm
  as the block order to reveal the physical raster sequence) and the codebase
  helper phys_n64_block_order_to_model_token_order (inv_perm[phys]=model).
  Skipping this remap silently scrambles raster/graph_rw structure (random
  orders are frame-invariant, so val_random would still look fine — which is
  why this bug is invisible to the val_random gate).
"""

import argparse, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "image_order"))
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))

from AOGPT import AOGPT, AOGPTConfig
from directed_graph_policy import build_directed_graph
from graph_rw_image import (
    IMAGE_RW_PARAMS_DEFAULT, IMAGE_RW_PARAMS_V2, IMAGE_RW_PARAMS_V3,
    sample_image_orders_batch,
)
from order_utils import block_permutation_to_token_permutation
from readout_order_diagnostic import b_coverage_order, hilbert_order


N_BLOCKS = 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["random", "graph_rw", "hilbert", "Bcov_balanced", "distance_only_coverage", "shuffled_Bcov_balanced", "raster"],
                   required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--a-block-path", required=True,
                   help="Path to A_block_8x8.npy (raster, 64x64)")
    p.add_argument("--data-train", required=True)
    p.add_argument("--data-val", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--alpha-warmup", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--max-eval-batches", type=int, default=16)
    p.add_argument("--save-steps", type=str, default="1000,3000,5000")
    p.add_argument("--rw-top-k", type=int, default=4)
    p.add_argument("--rw-epsilon", type=float, default=0.0)
    p.add_argument("--rw-tau-start", type=float, default=0.10)
    p.add_argument("--rw-tau-step", type=float, default=0.10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_baseline_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ma = ckpt["model_args"]
    keys = ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
            "dropout", "bias", "block_order_block_len", "order_impl"]
    model_args = {k: ma[k] for k in keys if k in ma}
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading ckpt: {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys ignored: {unexpected[:5]}")
    model.crop_block_size(model_args["block_size"])
    model.to(device)

    # Reconstruct fixed_token_perm from data_permutation stored in ckpt.
    # The baseline was trained with permute_data=True (permute_mode='block'):
    # tokens were reordered by block_perm before each forward pass.
    # We must apply the same permutation here.
    fixed_token_perm = None
    inv_block_perm = None
    dp = ckpt.get("data_permutation")
    if dp is not None and dp.get("permute_mode") == "block":
        block_len = model_args.get("block_order_block_len", 4)
        block_perm_list = dp["block_perm"]
        block_perm = torch.tensor(block_perm_list, dtype=torch.long)
        fixed_token_perm = block_permutation_to_token_permutation(block_perm, block_len)
        # inverse_block_perm maps a PHYSICAL block index -> the MODEL block
        # position that holds it; used to remap physically-defined block orders
        # into the model coordinate frame. See module docstring.
        inv_block_perm = torch.tensor(dp["inverse_block_perm"], dtype=torch.long)
        print(f"  [data_perm] loaded block_perm (len={len(block_perm_list)}), "
              f"fixed_token_perm (len={len(fixed_token_perm)}), "
              f"inv_block_perm (len={len(inv_block_perm)})")
    else:
        print("  [data_perm] no data_permutation found in ckpt; feeding raw token order")

    return model, model_args, ckpt, fixed_token_perm, inv_block_perm


def get_lr(step, args):
    if step < args.warmup_iters:
        return args.lr * step / max(args.warmup_iters, 1)
    if step >= args.max_steps:
        return args.min_lr
    ratio = (step - args.warmup_iters) / max(args.max_steps - args.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def get_alpha(step, args):
    if args.alpha_warmup <= 0:
        return args.alpha
    return min(args.alpha, args.alpha * step / max(args.alpha_warmup, 1))


def _forward_with_block_orders(model, x, block_orders, fixed_token_perm=None,
                               inv_block_perm=None):
    """Run model forward with PHYSICAL-coordinate block-level orders.

    If fixed_token_perm is provided, x is first reordered by it (to match the
    data permutation used during baseline training), putting the model in its
    permuted (model) coordinate frame.

    If inv_block_perm is provided, block_orders are assumed to be in PHYSICAL
    raster coordinates and are remapped into model coordinates via
    model_block_order = inv_block_perm[phys_block_order] BEFORE expansion. This
    is required whenever the data is permuted; otherwise structured orders
    (raster/graph_rw) traverse scrambled positions. (Harmless for random orders,
    which stay random under any bijection.)

    Expands block_orders to token_orders, then calls model.
    then calls model(x, mode=None, orders=token_orders).
    Returns loss scalar.
    """
    if fixed_token_perm is not None:
        perm_idx = fixed_token_perm.to(x.device)
        x = x[:, perm_idx]
    if inv_block_perm is not None:
        ibp = inv_block_perm.to(block_orders.device)
        block_orders = ibp[block_orders]
    token_orders = model._expand_block_orders_to_token_orders(block_orders)
    result = model(x, mode=None, orders=token_orders)
    loss = result[1]
    return loss


def _as_torch_B(B_np, device):
    return torch.as_tensor(B_np, dtype=torch.float32, device=device)


def _mask_top_k(scores, top_k):
    if top_k > 0 and top_k < scores.size(-1):
        topk_vals, _ = torch.topk(scores, top_k, dim=-1)
        thresh = topk_vals[:, -1:].contiguous()
        scores = torch.where(scores >= thresh, scores, torch.full_like(scores, float("-inf")))
    return scores


def sample_progressive_rw_v1_batched_torch(B_np, params, batch_size, seed_base, step, device):
    """Online GPU-batched v1 progressive Graph-RW sampler.

    Same readout terms as directed_graph_policy.progressive_rw:
    support - beta_fut * future + beta_src * source + beta_loc * local.
    This keeps online sampling, but parallelizes all batch rows at each reveal step.
    """
    B = _as_torch_B(B_np, device)
    N = B.size(0)
    tau_start = float(params.get("tau_start", 1.0))
    tau_step = float(params.get("tau_step", 1.0))
    alpha_dep = float(params.get("alpha_dep", 0.5))
    top_k = int(params.get("top_k", 0) or 0)
    epsilon_uniform = float(params.get("epsilon_uniform", 0.0))
    beta_sup = float(params.get("beta_sup", 1.0))
    beta_fut = float(params.get("beta_fut", 0.5))
    beta_src = float(params.get("beta_src", 0.2))
    beta_loc = float(params.get("beta_loc", 0.5))

    out_deg = B.sum(dim=1)
    in_deg = B.sum(dim=0)
    source = out_deg - alpha_dep * in_deg

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed_base * 100000 + step * batch_size) % (2**63 - 1))

    orders = torch.empty(batch_size, N, dtype=torch.long, device=device)
    selected = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
    last_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    for t in range(N):
        if t == 0:
            scores = source.unsqueeze(0).expand(batch_size, -1).clone()
            tau = tau_start
        else:
            sel_f = selected.float()
            unsel_f = (~selected).float()
            support = torch.mm(sel_f, B)
            future = torch.mm(unsel_f, B)
            local = B[last_idx]
            scores = beta_sup * support - beta_fut * future + beta_src * source.unsqueeze(0) + beta_loc * local
            tau = tau_step

        scores = scores.masked_fill(selected, float("-inf"))
        scores = _mask_top_k(scores, top_k)
        finite = torch.isfinite(scores)
        fmax = scores.masked_fill(~finite, float("-inf")).max(dim=-1).values
        fmin = scores.masked_fill(~finite, float("inf")).min(dim=-1).values
        degenerate = (fmax - fmin).abs() < 1e-15
        if degenerate.any():
            scores = scores + torch.rand(scores.shape, generator=gen, device=device) * 1e-9 * degenerate.unsqueeze(-1).float()
        probs = torch.softmax(scores / tau, dim=-1).masked_fill(selected, 0.0)
        if epsilon_uniform > 0.0:
            uniform = (~selected).float() / float(N - t)
            probs = (1.0 - epsilon_uniform) * probs + epsilon_uniform * uniform
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        choice = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
        orders[:, t] = choice
        selected[torch.arange(batch_size, device=device), choice] = True
        last_idx = choice
    return orders


def sample_coverage_batched_torch(B_np, batch_size, step, seed_base, device, gamma_B=1.0, gamma_d=1.0):
    """Online GPU-batched coverage sampler.

    score = gamma_B * minmax_U(B[last,v]) - gamma_d * minmax_U(manh(last,v)),
    deterministic argmax. Starts vary cyclically across batch/step; no order bank is used.
    distance_only_coverage uses gamma_B=0.01 as a deterministic B tie-break.
    """
    B = _as_torch_B(B_np, device)
    N = B.size(0)
    G = int(round(N ** 0.5))
    idx = torch.arange(N, device=device)
    row = idx // G
    col = idx % G
    D = (row[:, None] - row[None, :]).abs() + (col[:, None] - col[None, :]).abs()
    D = D.float()

    starts = (torch.arange(batch_size, device=device) + int(seed_base) + int(step) * batch_size) % N
    orders = torch.empty(batch_size, N, dtype=torch.long, device=device)
    selected = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
    last = starts.long()
    orders[:, 0] = last
    selected[torch.arange(batch_size, device=device), last] = True

    eps = 1e-12
    for t in range(1, N):
        bs = B[last]
        ds = D[last]
        b_min = bs.masked_fill(selected, float("inf")).min(dim=-1).values.unsqueeze(-1)
        b_max = bs.masked_fill(selected, float("-inf")).max(dim=-1).values.unsqueeze(-1)
        d_min = ds.masked_fill(selected, float("inf")).min(dim=-1).values.unsqueeze(-1)
        d_max = ds.masked_fill(selected, float("-inf")).max(dim=-1).values.unsqueeze(-1)
        bn = (bs - b_min) / (b_max - b_min + eps)
        dn = (ds - d_min) / (d_max - d_min + eps)
        scores = (float(gamma_B) * bn - float(gamma_d) * dn).masked_fill(selected, float("-inf"))
        nxt = torch.argmax(scores, dim=-1)
        orders[:, t] = nxt
        selected[torch.arange(batch_size, device=device), nxt] = True
        last = nxt
    return orders


def sample_bcov_balanced_batched_torch(B_np, batch_size, step, seed_base, device):
    return sample_coverage_batched_torch(B_np, batch_size, step, seed_base, device, gamma_B=1.0, gamma_d=1.0)


def sample_distance_only_coverage_batched_torch(B_np, batch_size, step, seed_base, device):
    return sample_coverage_batched_torch(B_np, batch_size, step, seed_base, device, gamma_B=0.01, gamma_d=1.0)


def make_shuffled_Bcov_control(B_np, seed=424242):
    """Shuffle each source row independently, preserving row value multiset.

    This destroys physical target semantics while preserving row-wise B value distribution,
    matching the Stage-1 shuffled_B style used as a perturbation/control. Diagonal is
    zeroed after shuffling for legality.
    """
    rng = np.random.default_rng(seed)
    B = np.asarray(B_np, dtype=np.float64)
    out = np.empty_like(B)
    for i in range(B.shape[0]):
        out[i] = B[i, rng.permutation(B.shape[1])]
    np.fill_diagonal(out, 0.0)
    return out

def sample_block_orders(policy, B_global, batch_size, step, rw_params, raster_order, device, rng):
    """Return (batch_size, 64) block-order tensor in physical raster order."""
    if policy == "random":
        return torch.stack([torch.randperm(N_BLOCKS, device=device) for _ in range(batch_size)])
    if policy == "raster":
        return raster_order.unsqueeze(0).expand(batch_size, -1).contiguous()
    if policy == "hilbert":
        order = torch.from_numpy(hilbert_order().astype(np.int64)).long().to(device)
        return order.unsqueeze(0).expand(batch_size, -1).contiguous()
    if policy == "Bcov_balanced":
        seed_base = int(rng.integers(1 << 31))
        return sample_bcov_balanced_batched_torch(B_global, batch_size, step, seed_base, device)
    if policy == "distance_only_coverage":
        seed_base = int(rng.integers(1 << 31))
        return sample_distance_only_coverage_batched_torch(B_global, batch_size, step, seed_base, device)
    if policy == "shuffled_Bcov_balanced":
        seed_base = int(rng.integers(1 << 31))
        return sample_bcov_balanced_batched_torch(B_global, batch_size, step, seed_base, device)
    # graph_rw uses the frozen Stage-1 v1 image policy, sampled online in parallel on GPU.
    seed_base = int(rng.integers(1 << 31))
    return sample_progressive_rw_v1_batched_torch(B_global, rw_params, batch_size, seed_base, step, device)


@torch.no_grad()
def evaluate_7orders(model, val_tokens, B_real, raster_order, device,
                     batch_size, max_eval_batches, step, fixed_token_perm=None,
                     inv_block_perm=None):
    model.eval()
    V = val_tokens.shape[0]
    n_batches = min(max_eval_batches, math.ceil(V / batch_size))

    cols = ["random", "raster", "hilbert", "Bcov_balanced", "distance_only_coverage",
            "rw_top4_eps0", "rw_eps015", "rw_topk8"]
    results = {c: [] for c in cols}
    hilbert_b = torch.from_numpy(hilbert_order().astype(np.int64)).long().to(device)

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, V)
        x = val_tokens[start:end].to(device)
        B_actual = x.shape[0]

        rand_b = torch.stack([torch.randperm(N_BLOCKS, device=device) for _ in range(B_actual)])
        ras_b = raster_order.unsqueeze(0).expand(B_actual, -1).contiguous()
        hil_b = hilbert_b.unsqueeze(0).expand(B_actual, -1).contiguous()
        seed_base = 9999_000 + step

        # B-dependent orders (Bcov/distance/rw) require a learned graph; fixed-order
        # baselines pass B_real=None and only care about random/raster/hilbert.
        if B_real is None:
            order_map = {"random": rand_b, "raster": ras_b, "hilbert": hil_b}
        else:
            bcov_b = sample_bcov_balanced_batched_torch(B_real, B_actual, step + bi, seed_base, device)
            dist_b = sample_distance_only_coverage_batched_torch(B_real, B_actual, step + bi, seed_base + 7, device)
            rw4 = sample_progressive_rw_v1_batched_torch(B_real,
                {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4, "epsilon_uniform": 0.0},
                B_actual, seed_base=seed_base, step=0, device=device)
            rweps = sample_progressive_rw_v1_batched_torch(B_real,
                {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 0, "epsilon_uniform": 0.15},
                B_actual, seed_base=seed_base + 1, step=0, device=device)
            rwk8 = sample_progressive_rw_v1_batched_torch(B_real,
                {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 8, "epsilon_uniform": 0.0},
                B_actual, seed_base=seed_base + 2, step=0, device=device)
            order_map = dict(zip(cols, [rand_b, ras_b, hil_b, bcov_b, dist_b, rw4, rweps, rwk8]))

        for name in cols:
            if name not in order_map:
                results[name].append(float("nan"))
                continue
            loss = _forward_with_block_orders(model, x, order_map[name],
                                              fixed_token_perm=fixed_token_perm,
                                              inv_block_perm=inv_block_perm)
            results[name].append(float(loss.item()))

    model.train()
    return {k: float(np.mean(v)) for k, v in results.items()}

def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    # ----- A → B -----
    A_block = np.load(args.a_block_path).astype(np.float32)
    assert A_block.shape == (64, 64)
    B_real = build_directed_graph(A_block)
    if args.policy == "shuffled_Bcov_balanced":
        shuffle_seed = int(args.seed) + 424242
        B_used = make_shuffled_Bcov_control(B_real, seed=shuffle_seed)
    else:
        shuffle_seed = None
        B_used = B_real

    config_out = vars(args).copy()
    config_out.update({
        "B_construction": "B = build_directed_graph(A_block) = A_block.T with diagonal zeroed",
        "shuffled_Bcov_control": args.policy == "shuffled_Bcov_balanced",
        "shuffle_seed": shuffle_seed,
        "shuffle_method": "row-wise independent permutation preserving each source row value multiset; diagonal zeroed after shuffling",
        "Bcov_balanced_formula": "score = minmax_U(B[last,v]) - minmax_U(manh(last,v)); gamma_B=gamma_d=1.0; deterministic argmax",
        "distance_only_formula": "score = 0.01*minmax_U(B[last,v]) - minmax_U(manh(last,v)); deterministic argmax",
        "frame_note": "orders are physical 8x8 raster frame and remapped via inverse_block_perm before model forward when ckpt has permute_data=True",
    })
    with open(out / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # ----- RW params -----
    rw_params = {**IMAGE_RW_PARAMS_DEFAULT,
                 "top_k": args.rw_top_k, "epsilon_uniform": args.rw_epsilon,
                 "tau_start": args.rw_tau_start, "tau_step": args.rw_tau_step}

    # ----- model -----
    model, model_args, ckpt, fixed_token_perm, inv_block_perm = load_baseline_model(args.baseline_ckpt, device)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr,
                                           (args.beta1, args.beta2), device.split(":")[0])
    model.train()

    # ----- data -----
    seq_len = int(model_args["block_size"])
    with open(args.meta, "rb") as f: meta = pickle.load(f)
    tokens_per_image = int(meta["tokens_per_image"])
    assert tokens_per_image == seq_len, f"meta says {tokens_per_image}, expected {seq_len}"
    train_mm = np.memmap(args.data_train, dtype=np.uint16, mode="r")
    val_mm = np.memmap(args.data_val, dtype=np.uint16, mode="r")
    n_train = len(train_mm) // seq_len
    n_val = min(2000, len(val_mm) // seq_len)
    val_tokens = torch.from_numpy(
        np.asarray(val_mm[:n_val * seq_len], dtype=np.int64).reshape(n_val, seq_len)
    )

    raster_order = torch.arange(N_BLOCKS, device=device)

    # ----- TSV -----
    tsv_path = out / "eval_curve.tsv"
    header = "step\ttrain_loss\talpha\tlr\tval_random\tval_raster\tval_hilbert\tval_Bcov_balanced\tval_distance_only_coverage\tval_rw_top4_eps0\tval_rw_eps015\tval_rw_topk8\n"
    if not tsv_path.exists():
        tsv_path.write_text(header)
    log_path = out / "train_log.txt"
    log_f = open(log_path, "a")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    log(f"[start] policy={args.policy} max_steps={args.max_steps} eff_batch={args.batch_size*args.grad_accum} ckpt={args.baseline_ckpt}")

    save_steps = {int(x) for x in args.save_steps.split(",") if x}
    t0 = time.time()
    running_loss = []

    for step in range(args.max_steps + 1):
        lr_now = get_lr(step, args)
        for g in optimizer.param_groups: g["lr"] = lr_now
        alpha_now = get_alpha(step, args)

        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum):
                idxs = rng.integers(0, n_train, size=args.batch_size)
                tokens_np = np.stack(
                    [np.asarray(train_mm[i*seq_len:(i+1)*seq_len], dtype=np.int64) for i in idxs]
                )
                x = torch.from_numpy(tokens_np).to(device, non_blocking=True)

                if args.policy == "random" or rng.random() > alpha_now:
                    block_orders = torch.stack([torch.randperm(N_BLOCKS, device=device)
                                                for _ in range(args.batch_size)])
                else:
                    block_orders = sample_block_orders(
                        args.policy, B_used, args.batch_size, step, rw_params,
                        raster_order, device, rng,
                    )

                loss = _forward_with_block_orders(model, x, block_orders,
                                                  fixed_token_perm=fixed_token_perm,
                                                  inv_block_perm=inv_block_perm)
                (loss / args.grad_accum).backward()
                running_loss.append(float(loss.item()))

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.eval_interval == 0:
            vr = evaluate_7orders(model, val_tokens, B_real, raster_order, device,
                                  args.batch_size, args.max_eval_batches, step,
                                  fixed_token_perm=fixed_token_perm,
                                  inv_block_perm=inv_block_perm)
            tl = float(np.mean(running_loss[-100:])) if running_loss else float("nan")
            log(f"[eval] step={step:5d} train={tl:.4f} α={alpha_now:.3f} lr={lr_now:.2e} "
                f"rnd={vr['random']:.4f} ras={vr['raster']:.4f} "
                f"hil={vr['hilbert']:.4f} bcov={vr['Bcov_balanced']:.4f} dist={vr['distance_only_coverage']:.4f} "
                f"rw4={vr['rw_top4_eps0']:.4f} eps={vr['rw_eps015']:.4f} k8={vr['rw_topk8']:.4f} "
                f"elapsed={time.time()-t0:.1f}s")
            with open(tsv_path, "a") as f:
                f.write(f"{step}\t{tl:.4f}\t{alpha_now:.4f}\t{lr_now:.2e}\t"
                        f"{vr['random']:.4f}\t{vr['raster']:.4f}\t{vr['hilbert']:.4f}\t"
                        f"{vr['Bcov_balanced']:.4f}\t{vr['distance_only_coverage']:.4f}\t{vr['rw_top4_eps0']:.4f}\t"
                        f"{vr['rw_eps015']:.4f}\t{vr['rw_topk8']:.4f}\n")

        if step > 0 and step in save_steps:
            torch.save({"model": model.state_dict(), "model_args": model_args,
                        "step": step, "config": vars(args)},
                       out / f"ckpt_step{step}.pt")
            log(f"[save] ckpt_step{step}.pt")

    torch.save({"model": model.state_dict(), "model_args": model_args,
                "step": args.max_steps, "config": vars(args)},
               out / "ckpt_final.pt")
    log(f"[done] final ckpt saved at step {args.max_steps}")
    log_f.close()


if __name__ == "__main__":
    main()
