"""Driver: run multi-seed short training for 4 configs × 3 seeds on dual GPUs."""
import subprocess, sys, os, time

AO_GPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
TOKENS = "block_lo_arm_order_network/probe_results/A32_from_N64_10k.tokens.npy"
BASE = "block_lo_arm_order_network/probe_results/attention_curriculum_full10k/st_multi"
ORDERS_OLD = f"{BASE}/../old_nn_orders.npy"
ORDERS_MIX = f"{BASE}/../mixed_lam025_orders.npy"

CONFIGS = [
    # (label, order_source, precomputed_path, alpha, alpha_warmup)
    ("random_s42", "random", "", 0.0, 1),
    ("random_s123", "random", "", 0.0, 1),
    ("random_s456", "random", "", 0.0, 1),
    ("oldnn_s42", "precomputed", ORDERS_OLD, 0.7, 1500),
    ("oldnn_s123", "precomputed", ORDERS_OLD, 0.7, 1500),
    ("oldnn_s456", "precomputed", ORDERS_OLD, 0.7, 1500),
    ("mixed025_s42", "precomputed", ORDERS_MIX, 0.7, 1500),
    ("mixed025_s123", "precomputed", ORDERS_MIX, 0.7, 1500),
    ("mixed025_s456", "precomputed", ORDERS_MIX, 0.7, 1500),
    ("mixed025_a05_s42", "precomputed", ORDERS_MIX, 0.5, 1500),
    ("mixed025_a05_s123", "precomputed", ORDERS_MIX, 0.5, 1500),
    ("mixed025_a05_s456", "precomputed", ORDERS_MIX, 0.5, 1500),
]

SEED_MAP = {"s42": 42, "s123": 123, "s456": 456}
NUM_BLOCKS = 64
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 3e-5


def run_one(config, gpu):
    label, order_source, precomp, alpha, warmup = config
    seed = SEED_MAP[label.split("_")[-1]]
    out_dir = f"{BASE}/{label}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-u", "block_lo_arm_order_network/train_on_mixed.py",
        "--ao-gpt-ckpt", AO_GPT,
        "--tokens-path", TOKENS,
        "--num-blocks", str(NUM_BLOCKS),
        "--order-source", order_source,
        "--tokens-only",
        "--alpha", str(alpha),
        "--alpha-warmup", str(warmup),
        "--max-iters", str(MAX_ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum", str(GRAD_ACCUM),
        "--lr", str(LR),
        "--eval-interval", str(EVAL_INTERVAL),
        "--eval-iters", str(EVAL_ITERS),
        "--device", f"cuda:{gpu}",
        "--out-dir", out_dir,
        "--eval-log", f"{out_dir}/eval_log.jsonl",
        "--save-metric", "random",
        "--seed", str(seed),
    ]
    if precomp:
        cmd += ["--precomputed-orders-path", precomp]

    log_file = f"{out_dir}/train_log.txt"
    print(f"\n{'='*60}")
    print(f"[{label}] GPU={gpu} seed={seed} α={alpha} → {out_dir}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"[{label}] DONE in {elapsed:.0f}s", flush=True)
    else:
        print(f"[{label}] FAILED (exit {proc.returncode}) in {elapsed:.0f}s", flush=True)
    return proc.returncode


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    failed = []
    t_start = time.time()

    def worker(cfg, gpu):
        label = cfg[0]
        rc = run_one(cfg, gpu)
        return label, rc, gpu

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for i, cfg in enumerate(CONFIGS):
            gpu = i % 2
            futures.append(pool.submit(worker, cfg, gpu))

        for f in as_completed(futures):
            label, rc, gpu = f.result()
            if rc != 0:
                failed.append(label)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.0f}s. {len(CONFIGS)} runs, {len(failed)} failed.")
    if failed:
        print(f"Failed: {failed}")
