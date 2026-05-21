"""Per-sample g(B) extraction for graph diversity analysis.

Extracts per-sample attention B and computes full g(B) diagnostic vector
for each sample across multiple setups.

Usage:
    python extract_per_sample_gB.py --setup e3_ctrl_small --n-samples 300 --device cuda:0
    python extract_per_sample_gB.py --setup text --n-samples 300 --device cuda:0
    python extract_per_sample_gB.py --setup e2_small --n-samples 300 --device cuda:0
    python extract_per_sample_gB.py --setup e2_large --n-samples 300 --device cuda:0
    python extract_per_sample_gB.py --setup synthetic --n-samples 300
"""

import os, sys, argparse, time, json
from pathlib import Path
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent.parent
_NANO = _REPO / "nanogpt-learned-order"
_BLOCK = _REPO / "block_lo_arm_order_network"

sys.path.insert(0, str(_BLOCK))
sys.path.insert(0, str(_NANO))
sys.path.insert(0, str(_REPO / "scripts"))

from directed_graph_policy import build_directed_graph, compute_source
# Correct spatial token->patch aggregation for patch2x2 image setups (16x16 token
# grid -> 8x8 patch grid). The previous block_len=4 contiguous pooling SCRAMBLES
# the 2D spatial structure (global A: p_nbr<=1=0.047 vs correct 0.984) and was the
# cause of the spurious "per-sample E3 ~= random" finding.
from diagnose_e3_control_dual_level import aggregate_token_to_block as spatial_patch2x2_agg

# ─── g(B) diagnostic (from graph_regime_diagnostic.py, topology-aware) ───

def readiness_strength(B):
    source, _, _ = compute_source(B, alpha_dep=0.5)
    return float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))

def asymmetry(B):
    return float(np.mean(np.abs(B - B.T)) / (np.mean(np.abs(B)) + 1e-12))

def row_entropy_norm(B):
    P = np.abs(B).copy()
    np.fill_diagonal(P, 0.0)
    P = P / (P.sum(1, keepdims=True) + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    return float(np.mean(H) / np.log(B.shape[0] - 1))

def topk_mass(B, k):
    P = np.abs(B).copy()
    np.fill_diagonal(P, 0.0)
    P = P / (P.sum(1, keepdims=True) + 1e-12)
    part = np.sort(P, axis=1)[:, -k:].sum(1)
    return float(part.mean())

def _coords(N, topology, grid):
    if topology == "grid2d":
        idx = np.arange(N)
        return np.stack([idx // grid, idx % grid], 1).astype(float)
    return np.arange(N).reshape(N, 1).astype(float)

def _pair_dist(coords, i, js):
    return np.abs(coords[js] - coords[i]).sum(1)

def directionality(B, topology, grid):
    N = B.shape[0]
    coords = _coords(N, topology, grid)
    bins = {}
    for i in range(N):
        j = int(np.argmax(B[i]))
        off = coords[j] - coords[i]
        key = tuple(np.sign(off).astype(int))
        if all(v == 0 for v in key):
            continue
        bins[key] = bins.get(key, 0) + 1
    if not bins:
        return 0.0
    return float(max(bins.values()) / sum(bins.values()))

def locality_metrics(B, topology, grid):
    N = B.shape[0]
    coords = _coords(N, topology, grid)
    argmax = np.array([int(np.argmax(B[i])) for i in range(N)])
    amd = np.array([_pair_dist(coords, i, [argmax[i]])[0] for i in range(N)])
    all_d = np.array([_pair_dist(coords, i, [j for j in range(N) if j != i]).mean()
                      for i in range(N)])
    rand_dist = float(all_d.mean())
    # local-greedy traversal
    vis = [0]; seen = {0}
    for _ in range(N - 1):
        last = vis[-1]
        cand = [j for j in range(N) if j not in seen]
        nxt = cand[int(np.argmax(B[last, cand]))]
        vis.append(nxt); seen.add(nxt)
    o = np.array(vis)
    steps = np.array([_pair_dist(coords, o[t], [o[t + 1]])[0] for t in range(N - 1)])
    greedy_dist = float(steps.mean())
    locality_score = float(np.clip((rand_dist - amd.mean()) / (rand_dist + 1e-12), 0, 1))
    return dict(
        argmax_dist=float(amd.mean()),
        p_nbr_le1=float((amd <= 1).mean()),
        rand_dist=rand_dist,
        local_greedy_dist=greedy_dist,
        locality_score=locality_score,
    )

def degree_variance(B):
    out_deg = B.sum(axis=1)
    in_deg = B.sum(axis=0)
    return float(np.std(out_deg)), float(np.std(in_deg))

def diagnose_full(B, topology, grid):
    """Full g(B) vector: 12 dimensions."""
    d = dict(
        readiness_strength=readiness_strength(B),
        asymmetry=asymmetry(B),
        row_entropy=row_entropy_norm(B),
        top1_mass=topk_mass(B, 1),
        top4_mass=topk_mass(B, 4),
    )
    d.update(locality_metrics(B, topology, grid))
    d["directionality"] = directionality(B, topology, grid)
    out_std, in_std = degree_variance(B)
    d["out_degree_std"] = out_std
    d["in_degree_std"] = in_std
    return d


# ─── Model loading ───

def load_model(ckpt_path, device):
    from AOGPT import AOGPTConfig, AOGPT
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_args = ckpt.get("model_args", {})
    cfg = ckpt.get("config", {})
    model_args = {}
    for k in ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
              "dropout", "bias", "block_order_block_len", "order_impl"]:
        if k in m_args:
            model_args[k] = m_args[k]
        elif k in cfg:
            model_args[k] = cfg[k]
    model_args.setdefault("force_manual_attention", True)
    model = AOGPT(AOGPTConfig(**model_args))
    state_dict = ckpt["model"]
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


# ─── Per-sample attention extraction ───

def extract_single_sample_attention(model, tokens_1d, device, T, M_ORDERS=3):
    """Return A_token[T, T] for one sample, averaged over M_ORDERS random orders.
    Uses last 4 layers, all heads."""
    tokens = torch.from_numpy(tokens_1d.astype(np.int64)).to(device).unsqueeze(0)
    A_sum = torch.zeros(T, T, device=device)

    for _ in range(M_ORDERS):
        rand_order = torch.randperm(T, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(tokens, rand_order, return_attentions=True)
        n_layers = len(attn_list)
        use_layers = attn_list[-4:] if n_layers >= 4 else attn_list
        attn_stack = torch.stack(use_layers, dim=0)  # (L, 1, nh, T+1, T+1)
        attn = attn_stack.mean(dim=[0, 2])[0]  # (T+1, T+1)
        attn_content = attn[1:, 1:]  # (T, T) in model order
        inv_order = torch.argsort(rand_order[0])
        attn_phys = attn_content[inv_order][:, inv_order]
        A_sum += attn_phys

    return (A_sum / M_ORDERS).cpu().numpy()


def aggregate_token_to_block(A_token, block_len, num_blocks):
    """A_token[T,T] -> A_block[num_blocks, num_blocks] via block mean pooling."""
    A_block = np.zeros((num_blocks, num_blocks), dtype=np.float64)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = A_token[i*block_len:(i+1)*block_len, j*block_len:(j+1)*block_len]
            A_block[i, j] = block.mean()
    return A_block


# ─── Setup configs ───

SETUPS = {
    "e3_ctrl_small": {
        "ckpt": str(_NANO / "out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt"),
        "data": str(_NANO / "data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin"),
        "T": 256,
        "block_len": 4,
        "num_blocks": 64,
        "topology": "grid2d",
        "grid": 8,
        "needs_aggregation": True,
        "agg_mode": "spatial_patch2x2",   # CORRECT: 16x16 token -> 8x8 patch spatial map
    },
    "text": {
        "ckpt": str(_BLOCK / "probe_results/clean_method_graph_rw_a10_to_a095_30k60k/ckpt_step50000.pt"),
        "data": "/home/admin/lyuyuhuan/order-shakespeare/nanoGPT/data/wikitext103/train.bin",
        "T": 256,
        "block_len": 4,
        "num_blocks": 64,
        "topology": "seq1d",
        "grid": 0,
        "needs_aggregation": True,
        "agg_mode": "contiguous",   # text 1D: 4 consecutive tokens = one block (correct)
    },
    "e2_small": {
        "ckpt": str(_NANO / "out/image_alignment/e2_imagenet32_vqf4_seq64_l4h8e256/ckpt.pt"),
        "data": str(_BLOCK / "data/Imagenet32VQ_f4_800k_seq64/train.bin"),
        "T": 64,
        "block_len": 1,
        "num_blocks": 64,
        "topology": "grid2d",
        "grid": 8,
        "needs_aggregation": False,
    },
    "e2_large": {
        "ckpt": str(_NANO / "out/image_alignment/e2_large_imagenet32_vqf4_seq64_l8h8e512/ckpt.pt"),
        "data": str(_BLOCK / "data/Imagenet32VQ_f4_800k_seq64/train.bin"),
        "T": 64,
        "block_len": 1,
        "num_blocks": 64,
        "topology": "grid2d",
        "grid": 8,
        "needs_aggregation": False,
    },
}

G_B_COLS = [
    "readiness_strength", "asymmetry", "row_entropy", "top1_mass", "top4_mass",
    "argmax_dist", "p_nbr_le1", "locality_score", "local_greedy_dist",
    "directionality", "out_degree_std", "in_degree_std",
]


def run_extraction(setup_name, n_samples, device, out_dir, seed=42):
    cfg = SETUPS[setup_name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Setup: {setup_name}")
    print(f"  ckpt: {cfg['ckpt']}")
    print(f"  data: {cfg['data']}")
    print(f"  T={cfg['T']}, block_len={cfg['block_len']}, num_blocks={cfg['num_blocks']}")
    print(f"  topology={cfg['topology']}, grid={cfg['grid']}")
    print(f"  n_samples={n_samples}, device={device}")
    print(f"{'='*70}\n")

    model, model_args = load_model(cfg["ckpt"], device)
    print(f"  Model loaded: n_layer={model_args.get('n_layer')}, n_head={model_args.get('n_head')}")

    T = cfg["T"]
    data = np.memmap(cfg["data"], dtype=np.uint16, mode="r")
    n_total = len(data) // T
    print(f"  Data: {n_total} samples available, sampling {n_samples}")
    assert n_total >= n_samples

    rng = np.random.RandomState(seed)
    sample_ids = rng.permutation(n_total)[:n_samples]

    results = []
    t0 = time.time()

    for i, sid in enumerate(sample_ids):
        tokens_1d = data[sid * T:(sid + 1) * T]
        A_token = extract_single_sample_attention(model, tokens_1d, device, T)

        if cfg.get("agg_mode") == "spatial_patch2x2":
            A_block = spatial_patch2x2_agg(A_token.astype(np.float32))
        elif cfg["needs_aggregation"]:
            A_block = aggregate_token_to_block(A_token, cfg["block_len"], cfg["num_blocks"])
        else:
            A_block = A_token

        B = build_directed_graph(A_block)
        gB = diagnose_full(B, cfg["topology"], cfg["grid"])
        gB["setup"] = setup_name
        gB["sample_id"] = int(sid)
        gB["sample_idx"] = i
        results.append(gB)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_samples} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.0f}s total")

    # Save TSV
    tsv_path = out_dir / f"{setup_name}_per_sample_gB.tsv"
    with open(tsv_path, "w") as f:
        header = ["setup", "sample_idx", "sample_id"] + G_B_COLS
        f.write("\t".join(header) + "\n")
        for r in results:
            row = [r["setup"], str(r["sample_idx"]), str(r["sample_id"])]
            row += [f"{r[c]:.6f}" for c in G_B_COLS]
            f.write("\t".join(row) + "\n")
    print(f"  Saved: {tsv_path}")

    # Save summary stats
    summary = {"setup": setup_name, "n_samples": n_samples, "elapsed_s": elapsed}
    for c in G_B_COLS:
        vals = np.array([r[c] for r in results])
        summary[c] = {"mean": float(vals.mean()), "std": float(vals.std()),
                      "min": float(vals.min()), "max": float(vals.max()),
                      "q25": float(np.percentile(vals, 25)),
                      "q75": float(np.percentile(vals, 75))}
    with open(out_dir / f"{setup_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def run_synthetic(n_samples, out_dir, seed=42):
    """Generate synthetic controls: random uniform B and shuffled B."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    N = 64
    topology = "grid2d"
    grid = 8

    for ctrl_name in ["random_uniform", "shuffled_rows"]:
        print(f"\n  Synthetic control: {ctrl_name}")
        results = []
        for i in range(n_samples):
            if ctrl_name == "random_uniform":
                B = rng.random((N, N))
                np.fill_diagonal(B, 0.0)
            else:
                # Shuffle a "structured" reference (use E3 global as template)
                A_ref_path = _REPO / "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy"
                if A_ref_path.exists():
                    A_ref = np.load(A_ref_path).astype(np.float64)
                else:
                    A_ref = rng.random((N, N))
                B_ref = build_directed_graph(A_ref)
                B = np.stack([B_ref[row, rng.permutation(N)] for row in range(N)])
                np.fill_diagonal(B, 0.0)

            gB = diagnose_full(B, topology, grid)
            gB["setup"] = ctrl_name
            gB["sample_id"] = i
            gB["sample_idx"] = i
            results.append(gB)

        tsv_path = out_dir / f"{ctrl_name}_per_sample_gB.tsv"
        with open(tsv_path, "w") as f:
            header = ["setup", "sample_idx", "sample_id"] + G_B_COLS
            f.write("\t".join(header) + "\n")
            for r in results:
                row = [r["setup"], str(r["sample_idx"]), str(r["sample_id"])]
                row += [f"{r[c]:.6f}" for c in G_B_COLS]
                f.write("\t".join(row) + "\n")
        print(f"  Saved: {tsv_path}")

    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", type=str, required=True,
                   choices=list(SETUPS.keys()) + ["synthetic", "all"])
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=str,
                   default=str(_SCRIPT_DIR))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.setup == "synthetic":
        run_synthetic(args.n_samples, args.out_dir, args.seed)
    elif args.setup == "all":
        for s in SETUPS:
            run_extraction(s, args.n_samples, args.device, args.out_dir, args.seed)
        run_synthetic(args.n_samples, args.out_dir, args.seed)
    else:
        run_extraction(args.setup, args.n_samples, args.device, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
