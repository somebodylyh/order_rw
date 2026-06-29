"""
Run ON-mixed training experiments.

Three runs:
  1. Random baseline (alpha=0)
  2. Edge-greedy mixed (alpha ramp 0→0.5)
  3. ON32 mixed (alpha ramp 0→0.5, requires ON32 ckpt)

Usage:
    python run_experiments.py --device cuda:0  # runs 1+2 (edge_greedy no ON needed)
    python run_experiments.py --device cuda:0 --on-ckpt probe_results/on32_soft_edge_best.pt --all  # runs 1+2+3
"""

import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_on_mixed import main as train_main

RUN_CONFIGS = {
    "random_baseline": {
        "alpha": 0.0,
        "order_source": "random",
        "temperature": 1.0,
        "alpha_warmup": 0,
        "description": "Random baseline (alpha=0, pure random orders)",
    },
    "edge_greedy_mixed": {
        "alpha": 0.5,
        "order_source": "edge_greedy",
        "temperature": 1.0,
        "alpha_warmup": 500,
        "description": "Edge-greedy mixed (alpha ramp 0→0.5, τ=1.0)",
    },
    "on32_mixed": {
        "alpha": 0.5,
        "order_source": "on32",
        "temperature": 1.0,
        "alpha_warmup": 500,
        "description": "ON32 mixed (alpha ramp 0→0.5, τ=1.0)",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--on-ckpt", type=str, default="probe_results/on32_soft_edge_best.pt")
    parser.add_argument("--runs", type=str, nargs="+",
                        default=["random_baseline", "edge_greedy_mixed"],
                        help="Which runs to execute")
    parser.add_argument("--all", action="store_true",
                        help="Run all three (including ON32)")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if args.all:
        args.runs = ["random_baseline", "edge_greedy_mixed", "on32_mixed"]

    for run_name in args.runs:
        cfg = RUN_CONFIGS[run_name]
        out_dir = os.path.join("probe_results/on_mixed_training", run_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Run: {run_name} — {cfg['description']}")
        print(f"{'='*60}")

        sys.argv = [
            "train_on_mixed.py",
            "--device", args.device,
            "--alpha", str(cfg["alpha"]),
            "--alpha-warmup", str(cfg["alpha_warmup"]),
            "--order-source", cfg["order_source"],
            "--temperature", str(cfg["temperature"]),
            "--max-iters", str(args.max_iters),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--lr", str(args.lr),
            "--out-dir", out_dir,
        ]
        if cfg["order_source"] == "on32":
            sys.argv.extend(["--on-ckpt", args.on_ckpt])

        train_main()

        print(f"\nRun {run_name} complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
