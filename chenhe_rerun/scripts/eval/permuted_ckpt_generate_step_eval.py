"""
Purpose:
Evaluate a permuted-data checkpoint with per-step loss curves for Random mode,
current-frame l2r, and original-frame l2r mapped back into the checkpoint's
current frame.

Typical usage:
python scripts/eval/permuted_ckpt_generate_step_eval.py \
  --ckpt_path out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt \
  --out_dir Report/eval/eval_pic_b32_permute_block_example
"""

import argparse
import json
import pickle
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    block_permutation_to_token_permutation,
    build_ascending_block_orders,
    build_fixed_block_permutation,
    evaluate_block_order_quality,
    get_order_unit_axis_label,
    get_order_unit_name,
    invert_permutation,
    token_losses_to_block_losses,
)
from path_layout import default_eval_out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a permuted-data checkpoint with three block-loss curves: "
            "Random mode, current-frame l2r, and original-frame l2r mapped into current frame."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["float32", "float16", "bfloat16"],
    )
    return parser.parse_args()


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in device or dtype == "float32":
        return nullcontext()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(ckpt_path: Path, device: str):
    return torch.load(ckpt_path, map_location=device)


def build_model(checkpoint, device: str):
    model = AOGPT(AOGPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def resolve_data_dir(args, checkpoint):
    if args.data_dir is not None:
        return args.data_dir
    dataset = args.dataset or checkpoint.get("config", {}).get("dataset")
    if dataset is None:
        raise ValueError("Could not infer dataset. Pass --dataset or --data_dir.")
    return REPO_ROOT / "data" / dataset


def load_tokens(data_dir: Path, split: str):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def infer_data_record_mode(data_dir: Path, checkpoint) -> str:
    checkpoint_mode = checkpoint.get("config", {}).get("data_record_mode")
    if checkpoint_mode is not None:
        return str(checkpoint_mode)
    meta_path = data_dir / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        return str(meta.get("data_record_mode", "stream"))
    return "stream"


def load_block_permutation_from_checkpoint(
    checkpoint,
    num_blocks,
    block_len,
    *,
    block_order_layout="contiguous",
    image_size=0,
    image_block_size=0,
    image_block_height=0,
    image_block_width=0,
):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    model_args = checkpoint.get("model_args", {})
    block_order_layout = str(model_args.get("block_order_layout", config.get("block_order_layout", block_order_layout)))
    image_size = int(model_args.get("image_size", config.get("image_size", image_size)))
    image_block_size = int(model_args.get("image_block_size", config.get("image_block_size", image_block_size)))
    image_block_height = int(model_args.get("image_block_height", config.get("image_block_height", image_block_height)))
    image_block_width = int(model_args.get("image_block_width", config.get("image_block_width", image_block_width)))
    perm_state = checkpoint.get("data_permutation")
    if perm_state is not None and perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
        block_order_layout = str(perm_state.get("block_order_layout", block_order_layout))
        image_size = int(perm_state.get("image_size", image_size))
        image_block_size = int(perm_state.get("image_block_size", image_block_size))
        image_block_height = int(perm_state.get("image_block_height", image_block_height))
        image_block_width = int(perm_state.get("image_block_width", image_block_width))
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(
        block_perm,
        block_len=block_len,
        block_order_layout=block_order_layout,
        image_size=image_size,
        image_block_size=image_block_size,
        image_block_height=image_block_height,
        image_block_width=image_block_width,
    )
    return {
        "permute_mode": "block",
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
        "block_order_layout": block_order_layout,
        "image_size": image_size,
        "image_block_size": image_block_size,
        "image_block_height": image_block_height,
        "image_block_width": image_block_width,
    }


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str, token_perm=None, data_record_mode="stream"):
    data_record_mode = str(data_record_mode or "stream")
    if data_record_mode == "fixed":
        num_records = len(tokens) // block_size
        if num_records <= 0:
            raise ValueError("Dataset split is shorter than one fixed block_size record.")
        starts = (rng.integers(0, num_records, size=batch_size) * block_size).tolist()
    elif data_record_mode == "stream":
        max_start = len(tokens) - block_size
        if max_start <= 0:
            raise ValueError("Dataset split is shorter than block_size.")
        starts = rng.integers(0, max_start, size=batch_size).tolist()
    else:
        raise ValueError(f"Unsupported data_record_mode={data_record_mode!r}.")
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    ).to(device)
    if token_perm is not None:
        batch = batch[:, token_perm.to(device)]
    return batch


def build_curve_figure(curve, title, xlabel: str, ylabel: str, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    steps = np.arange(len(curve))
    ax.plot(steps, curve, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = default_eval_out_dir(REPO_ROOT, args.ckpt_path, "permuted_ckpt_generate_step_eval")
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    data_record_mode = infer_data_record_mode(data_dir, checkpoint)
    model = build_model(checkpoint, args.device)
    autocast_context = get_autocast_context(args.device, args.dtype)
    rng = np.random.default_rng(12345)
    unit_name = get_order_unit_name(model.block_order_block_len)
    unit_axis_label = get_order_unit_axis_label(model.block_order_block_len)

    num_blocks = model.num_blocks
    permutation_state = load_block_permutation_from_checkpoint(
        checkpoint,
        num_blocks=num_blocks,
        block_len=model.block_order_block_len,
        block_order_layout=getattr(model, "block_order_layout", "contiguous"),
        image_size=getattr(model, "image_size", 0),
        image_block_size=getattr(model, "image_block_size", 0),
        image_block_height=getattr(model, "image_block_height", 0),
        image_block_width=getattr(model, "image_block_width", 0),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    inverse_block_perm = None if permutation_state is None else permutation_state["inverse_block_perm"].to(args.device)

    current_l2r = build_ascending_block_orders(1, num_blocks, args.device)
    original_l2r_in_current_frame = None if inverse_block_perm is None else inverse_block_perm.view(1, -1)

    random_block_curves = []
    current_l2r_curves = []
    original_l2r_curves = []

    for _ in range(int(args.num_batches)):
        idx = sample_batch(
            tokens,
            batch_size=int(args.batch_size),
            block_size=model.config.block_size,
            rng=rng,
            device=args.device,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
        )

        with autocast_context:
            _, _, random_token_losses = model(
                idx,
                mode="Random",
                return_token_loss=True,
            )
        random_block_losses = token_losses_to_block_losses(
            random_token_losses,
            block_len=model.block_order_block_len,
        )
        random_block_curves.append(random_block_losses.mean(dim=0).float().cpu())

        current_orders = current_l2r.expand(idx.size(0), -1)
        current_metrics = evaluate_block_order_quality(
            model,
            idx,
            current_orders,
            prefix_k=num_blocks,
            block_len=model.block_order_block_len,
            autocast_context=autocast_context,
        )
        current_l2r_curves.append(current_metrics["block_losses"].mean(dim=0).float().cpu())

        if original_l2r_in_current_frame is not None:
            original_orders = original_l2r_in_current_frame.expand(idx.size(0), -1)
            original_metrics = evaluate_block_order_quality(
                model,
                idx,
                original_orders,
                prefix_k=num_blocks,
                block_len=model.block_order_block_len,
                autocast_context=autocast_context,
            )
            original_l2r_curves.append(original_metrics["block_losses"].mean(dim=0).float().cpu())

    random_curve = torch.stack(random_block_curves, dim=0).mean(dim=0).numpy()
    current_l2r_curve = torch.stack(current_l2r_curves, dim=0).mean(dim=0).numpy()
    original_l2r_curve = None
    if original_l2r_curves:
        original_l2r_curve = torch.stack(original_l2r_curves, dim=0).mean(dim=0).numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_curve_figure(
        random_curve,
        f"{args.split} mean per-step {unit_name} loss: Random mode ({args.num_batches} batches)",
        unit_axis_label,
        f"Mean {unit_name.title()} Loss",
        args.out_dir / "random_generate_step_block_loss.png",
    )
    build_curve_figure(
        current_l2r_curve,
        f"{args.split} mean per-step {unit_name} loss: Current-frame l2r ({args.num_batches} batches)",
        unit_axis_label,
        f"Mean {unit_name.title()} Loss",
        args.out_dir / "current_l2r_block_loss.png",
    )
    if original_l2r_curve is not None:
        build_curve_figure(
            original_l2r_curve,
            f"{args.split} mean per-step {unit_name} loss: Original-frame l2r mapped into current frame ({args.num_batches} batches)",
            unit_axis_label,
            f"Mean {unit_name.title()} Loss",
            args.out_dir / "original_l2r_in_current_frame_block_loss.png",
        )

    payload = {
        "run_meta": {
            "ckpt_path": str(args.ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": args.split,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "device": args.device,
            "dtype": args.dtype,
            "order_unit": unit_name,
            "data_record_mode": data_record_mode,
            "num_blocks": int(num_blocks),
            "block_order_block_len": int(model.block_order_block_len),
            "block_order_layout": str(getattr(model, "block_order_layout", "contiguous")),
            "image_size": int(getattr(model, "image_size", 0)),
            "image_block_size": int(getattr(model, "image_block_size", 0)),
            "image_block_height": int(getattr(model, "image_block_height", 0)),
            "image_block_width": int(getattr(model, "image_block_width", 0)),
            "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
            "permute_mode": checkpoint.get("config", {}).get("permute_mode", ""),
            "permute_seed": checkpoint.get("config", {}).get("permute_seed", None),
        },
        "curves": {
            "random_mode": [float(v) for v in random_curve.tolist()],
            "current_l2r": [float(v) for v in current_l2r_curve.tolist()],
            "original_l2r_in_current_frame": None
            if original_l2r_curve is None
            else [float(v) for v in original_l2r_curve.tolist()],
        },
    }
    if permutation_state is not None:
        payload["permute_map"] = {
            "block_perm": [int(v) for v in permutation_state["block_perm"].tolist()],
            "inverse_block_perm": [int(v) for v in permutation_state["inverse_block_perm"].tolist()],
        }
    save_json(args.out_dir / "curves.json", payload)
    print(f"saved permuted checkpoint generate-step eval to {args.out_dir}")


if __name__ == "__main__":
    main()
