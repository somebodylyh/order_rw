"""Download full ImageNet-64 from HF parquets, pre-select 800k subset, save as .npz.

Matches prepare.py's subset logic exactly: np.random.RandomState(42).permutation(N)[:800000].

Output: directory of .npz files in ImageNet-64 format, ready for prepare.py.

Features: resume from checkpoint, retry on corrupt parquets (auto-delete bad cache blobs).
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


TRAIN_PARQUETS = [
    f"data/train-{i:05d}-of-00075-{h}.parquet"
    for i, h in enumerate([
        "1a46269ad1efc5f4", "0b5cbc73b021eb23", "d44a64ce4a660538",
        "9f7e0c11338f33ca", "4445910cbff024d4", "bda2cc555487a257",
        "4784383bd5f4ea73", "8b3a9e9856b4dd40", "1718f3fe5dfe07e5",
        "5576897d5cec6f4c", "cb9357f4c9b72d14", "f45ef195c7c04d7c",
        "8dc480f70f0bd5d4", "e48ede52f81079ab", "d45b328f3ad16cc4",
        "57264740619e75a0", "4dbbaa27e29be178", "d2d2f09db2b4be06",
        "ed21bf309c58a50e", "8d102b17b4995b19", "525384685081e90a",
        "af3c5cfae64aca1e", "00b611878d2a364a", "46349dd48e375dac",
        "cbe2f504b6863be7", "ad4ccff796dc7e19", "c561084ef4157c4c",
        "7e4a60fd866b3c85", "ae39ec9c6f4313d0", "a918898f5539178a",
        "9a7ac63eb51d9117", "e8fc67dd7e78df1c", "c0a4e8e7369fb202",
        "d3908f6e963fc27a", "9281de7be1be1ac8", "970a7a6257a3eefc",
        "4c6b768eaf697cff", "baf458e81a3c15cc", "3b926d5d8f44105d",
        "d3e9f1f7cd12d5a7", "6dba596484de9b06", "4bb8b1df23026cc6",
        "8f354ff9c0630a8d", "0858748fb31959eb", "9a497f8007671c6a",
        "a8f4fd23fcc6c957", "10eed02e428271c4", "2aa0609de0f121eb",
        "9670354e8cb368ea", "e82a0ae5e3137791", "1da8f1ab5d8b422a",
        "472a5a66c4774d61", "a48381619e59e030", "1b866b7f284c0261",
        "843bf3f0d880e4eb", "5df1c2e6788d1da3", "36841e410ffce743",
        "d077c75752c9f9fb", "ea084dd634dbcaac", "10da20168ba430a9",
        "df8ada3329c9c24b", "1014ed37566bec14", "1a967952ffdb640c",
        "a7dde00282bfc930", "ad23d89e280a8b1f", "11af325689266929",
        "df445fe56e962464", "ef1e01bf85bb57b9", "432d8dd2cb981c17",
        "235db860aa45739b", "ec60db4d672b24da", "56e761193cdf09e5",
        "6d092bf40e424774", "fc2aac57cf1ec38e", "72dd1604b5473a36",
    ])
]

VAL_PARQUETS = [
    "data/validation-00000-of-00003-af9e438bb67851db.parquet",
    "data/validation-00001-of-00003-fa3327cf9fadd9ec.parquet",
    "data/validation-00002-of-00003-75f4f66ad60a8663.parquet",
]

TOTAL_TRAIN = 1_281_167
SUBSET_SIZE = 800_000
SUBSET_SEED = 42


def compute_subset_mask() -> np.ndarray:
    """Reproduce prepare.py's select_subset: permutation of [0, N), take first K."""
    rng = np.random.RandomState(SUBSET_SEED)
    order = rng.permutation(TOTAL_TRAIN)
    selected = order[:SUBSET_SIZE]
    mask = np.zeros(TOTAL_TRAIN, dtype=bool)
    mask[selected] = True
    return mask


def _read_parquet_with_retry(pq_name, max_retries=3):
    """Download and read parquet with retry on corruption."""
    for attempt in range(max_retries):
        pq_path = hf_hub_download(
            repo_id="sradc/imagenet_resized_64x64",
            filename=pq_name,
            repo_type="dataset",
        )
        try:
            table = pq.read_table(pq_path)
            return table
        except OSError as e:
            if "Corrupt" in str(e) or "snappy" in str(e).lower():
                print(f"    CORRUPT parquet (attempt {attempt+1}/{max_retries}), deleting cache blob and retrying...", flush=True)
                real_path = os.path.realpath(pq_path)
                if os.path.exists(real_path):
                    os.remove(real_path)
                if os.path.islink(pq_path) and os.path.exists(pq_path):
                    os.remove(pq_path)
            else:
                raise
    raise RuntimeError(f"Failed to download {pq_name} after {max_retries} retries")


def _load_checkpoint(ckpt_path):
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    return None


def _save_checkpoint(ckpt_path, pq_idx, global_offset, batch_idx, total_selected):
    with open(ckpt_path, "w") as f:
        json.dump({"pq_idx": pq_idx, "global_offset": global_offset,
                   "batch_idx": batch_idx, "total_selected": total_selected}, f)


def download_and_convert_parquets(parquet_list, mask, output_dir, split_name):
    """Download parquets, select images matching mask, save as .npz. Supports resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f".{split_name}_checkpoint.json"

    ckpt = _load_checkpoint(ckpt_path)
    if ckpt:
        start_pq = ckpt["pq_idx"]
        global_offset = ckpt["global_offset"]
        batch_idx = ckpt["batch_idx"]
        total_selected = ckpt["total_selected"]
        print(f"  [{split_name}] RESUMING from parquet {start_pq}, "
              f"global_offset={global_offset}, batch_idx={batch_idx}, "
              f"total_selected={total_selected}", flush=True)
    else:
        start_pq = 0
        global_offset = 0
        batch_idx = 0
        total_selected = 0

    batch_images = []
    BATCH_MAX = 1280

    for pq_i, pq_name in enumerate(parquet_list):
        if pq_i < start_pq:
            continue

        print(f"  [{split_name}] [{pq_i+1}/{len(parquet_list)}] downloading {pq_name} ...", flush=True)
        table = _read_parquet_with_retry(pq_name)
        n_rows = len(table)
        print(f"    {n_rows} images (global {global_offset}-{global_offset + n_rows - 1})", flush=True)

        for i in range(n_rows):
            global_idx = global_offset + i
            if global_idx < len(mask) and mask[global_idx]:
                row = table.column('image')[i].as_py()
                img = np.array(row, dtype=np.uint8).transpose(2, 0, 1).ravel()
                batch_images.append(img)
                total_selected += 1

                if len(batch_images) >= BATCH_MAX:
                    _save_batch(batch_images, output_dir, split_name, batch_idx)
                    batch_images = []
                    batch_idx += 1

            if (i + 1) % 5000 == 0:
                print(f"    processed {i+1}/{n_rows}, selected {total_selected} so far", flush=True)

            if total_selected >= SUBSET_SIZE and split_name == "train":
                break

        global_offset += n_rows
        _save_checkpoint(ckpt_path, pq_i + 1, global_offset, batch_idx, total_selected)

        if total_selected >= SUBSET_SIZE and split_name == "train":
            print(f"  [{split_name}] reached {SUBSET_SIZE} subset; stopping early", flush=True)
            break

    if batch_images:
        _save_batch(batch_images, output_dir, split_name, batch_idx)

    ckpt_path.unlink(missing_ok=True)
    print(f"  [{split_name}] DONE: {total_selected} images in {batch_idx + 1} .npz files", flush=True)


def _save_batch(images, output_dir, split_name, batch_idx):
    data = np.stack(images, axis=0)
    path = output_dir / f"{split_name}_batch_{batch_idx:04d}.npz"
    np.savez_compressed(path, data=data, labels=list(range(len(data))))
    size_kb = path.stat().st_size / 1024
    print(f"    saved {path.name} ({len(data)} imgs, {size_kb:.0f} KB)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    args = p.parse_args()

    print(f"Computing subset mask: seed={SUBSET_SEED}, total={TOTAL_TRAIN}, subset={args.subset_size}")
    mask = compute_subset_mask()
    print(f"  selected {mask.sum()} indices")

    print(f"\nDownloading & converting TRAIN parquets ({len(TRAIN_PARQUETS)} files)")
    train_dir = args.output_dir / "train_npz"
    download_and_convert_parquets(TRAIN_PARQUETS, mask, train_dir, "train")

    print(f"\nDownloading & converting VAL parquets ({len(VAL_PARQUETS)} files)")
    val_dir = args.output_dir / "val_npz"
    # For val, select ALL images (no subset)
    val_mask = np.ones(50_000, dtype=bool)
    download_and_convert_parquets(VAL_PARQUETS, val_mask, val_dir, "val")

    print(f"\nDONE: Files in {args.output_dir}")
    print(f"  train: {list(train_dir.glob('*.npz'))[:3]}...")
    print(f"  val:   {list(val_dir.glob('*.npz'))[:3]}...")


if __name__ == "__main__":
    main()
