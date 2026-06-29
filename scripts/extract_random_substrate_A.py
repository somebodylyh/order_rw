"""Extract per-step A_global on clean_base_random_perm with FIXED torch seed
so the internal `torch.randperm(N, device='cpu')` inside extract_A_matrices is
reproducible across ckpts (no cross-step variance from sampling noise).

Outputs A_global_step{N}.npy alongside the ckpts in clean_base_random_perm/.
No metrics computed here — that is the next stage (apples-to-apples diagnostic
against alt α=0.9 / alt α=1 substrates).
"""
import sys, os, time, argparse
import numpy as np
import torch

REPO = "/home/admin/lyuyuhuan/order_lyu"
sys.path.insert(0, os.path.join(REPO, "block_lo_arm_order_network"))

from training_utils import load_train_chunks, SEQ_LEN, N
from train_clean_aogpt import (
    extract_A_matrices, build_model, phys_to_model_idx_clean, CleanPermutation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str,
                        default=os.path.join(REPO, "block_lo_arm_order_network/probe_results/clean_base_random_perm"))
    parser.add_argument("--steps", type=str, default="0,1000,5000,10000,20000,30000,40000,50000,60000")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--n-chunks", type=int, default=64)  # more chunks → lower sampling noise
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  n_chunks={args.n_chunks}  seed={args.seed}")

    print("Loading wikitext-103 chunks...")
    idx_phys = load_train_chunks(n_chunks=None)
    print(f"  idx_phys: {idx_phys.shape}")

    ckpt0_path = os.path.join(args.ckpt_dir, "ckpt_step0.pt")
    ckpt0 = torch.load(ckpt0_path, map_location="cpu", weights_only=False)
    protocol = ckpt0["clean_protocol"]
    model_args_dict = ckpt0["model_args"]
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    print(f"  inv_perm[:8]: {inv_perm[:8].tolist()}")

    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    eval_indices = np.asarray(protocol["eval_indices"], dtype=np.int64)
    idx_eval = idx_model[eval_indices]
    print(f"  eval chunks: {len(idx_eval)}")

    # FIXED selection of extract chunks across all steps
    rng = np.random.RandomState(args.seed)
    extract_chunks = idx_eval[rng.choice(len(idx_eval),
                                         size=min(args.n_chunks, len(idx_eval)),
                                         replace=False)]
    model_args_dict['block_size'] = SEQ_LEN
    steps = [int(s) for s in args.steps.split(",")]

    for step in steps:
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_step{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP step {step}: ckpt not found")
            continue
        out_path = os.path.join(args.ckpt_dir, f"A_global_step{step}_seed{args.seed}.npy")
        print(f"\n--- step {step} ---")
        t0 = time.time()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(model_args_dict, device, compile_model=False)
        sd = ckpt.get("model") or ckpt.get("model_state_dict")
        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
        model.to(device).eval()

        # Reset CPU rng before each extraction so torch.randperm inside
        # extract_A_matrices is identical across ckpts.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        A_all = extract_A_matrices(model, extract_chunks, clean_perm, device,
                                    n_chunks=args.n_chunks)
        A_global = A_all.mean(axis=0).astype(np.float32)
        np.fill_diagonal(A_global, 0.0)

        np.save(out_path, A_global)
        print(f"  saved {out_path}  shape={A_global.shape}  sum={A_global.sum():.4f}  "
              f"({time.time() - t0:.1f}s)")

        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
