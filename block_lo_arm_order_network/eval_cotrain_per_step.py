"""Per-step eval for co-train checkpoints (minimal ckpt without model_args/config)."""
import sys, json, os
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path("/home/admin/ych/nanogpt-learned-order")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from eval_ckpt_modes import run_evaluation, EvalConfig

BASE_CKPT = "/home/admin/ych/nanogpt-learned-order/out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--num_random_seeds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load base ckpt for metadata, cotrain ckpt for weights
    base = torch.load(BASE_CKPT, map_location="cpu")
    cotrain = torch.load(args.ckpt_path, map_location="cpu")

    # Merge: base metadata + cotrain weights
    merged = {
        "model_args": base["model_args"],
        "config": base["config"],
        "model": cotrain["model"],
    }
    merged["config"]["out_dir"] = str(args.out_dir)

    # Save merged ckpt temporarily
    tmp_path = args.out_dir / "_merged_ckpt.pt"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, tmp_path)

    config = EvalConfig(
        ckpt_path=tmp_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_random_seeds=args.num_random_seeds,
        device=args.device,
    )
    result = run_evaluation(config)

    print(json.dumps({
        "ar_curve_mean": float(np.mean(result["ar_mean"])),
        "random_curve_mean": float(np.mean(result["random_mean"])),
        "out_dir": str(args.out_dir),
    }, indent=2))

    # Cleanup
    tmp_path.unlink()
