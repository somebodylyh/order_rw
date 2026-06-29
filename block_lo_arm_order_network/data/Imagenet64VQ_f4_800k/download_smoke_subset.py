"""Download a small ImageNet-64 subset from HF datasets and save as .npz files
in the original ImageNet-64 format (compatible with prepare.py's loader).

Usage:
    python download_smoke_subset.py --num-images 2048 --output-dir /tmp/imagenet64_smoke
"""

import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-images", type=int, default=2048)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Stream train split to avoid downloading full dataset
    dataset = load_dataset(
        "sradc/imagenet_resized_64x64",
        split="train",
        streaming=True,
    )

    images = []
    print(f"Streaming {args.num_images} images from sradc/imagenet_resized_64x64 ...")
    for i, example in enumerate(dataset):
        img = np.array(example["image"])  # (64, 64, 3) HWC uint8
        img = img.transpose(2, 0, 1)      # (3, 64, 64) CHW uint8
        img = img.ravel()                  # (12288,) flat
        images.append(img)
        if (i + 1) % 512 == 0:
            print(f"  {i + 1} / {args.num_images}")
        if i + 1 >= args.num_images:
            break

    images = np.stack(images, axis=0)  # (N, 12288)
    assert images.shape == (args.num_images, 3 * 64 * 64), f"Unexpected shape: {images.shape}"
    assert images.dtype == np.uint8

    # Save in batches of 512 like original ImageNet-64 format
    batch_size = 512
    n_batches = (args.num_images + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, args.num_images)
        batch_data = {"data": images[start:end], "labels": list(range(start, end))}
        out_path = args.output_dir / f"smoke_train_batch_{b}.npz"
        np.savez_compressed(out_path, **batch_data)
        print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    print(f"DONE: {args.num_images} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
