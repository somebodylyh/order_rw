"""Fill missing 1,036 train images by re-scanning cached parquets for specific indices."""
import json, os, numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

TRAIN_PARQUETS = [
    f"data/train-{i:05d}-of-00075-{h}.parquet"
    for i, h in enumerate([
        "1a46269ad1efc5f4","0b5cbc73b021eb23","d44a64ce4a660538",
        "9f7e0c11338f33ca","4445910cbff024d4","bda2cc555487a257",
        "4784383bd5f4ea73","8b3a9e9856b4dd40","1718f3fe5dfe07e5",
        "5576897d5cec6f4c","cb9357f4c9b72d14","f45ef195c7c04d7c",
        "8dc480f70f0bd5d4","e48ede52f81079ab","d45b328f3ad16cc4",
        "57264740619e75a0","4dbbaa27e29be178","d2d2f09db2b4be06",
        "ed21bf309c58a50e","8d102b17b4995b19","525384685081e90a",
        "af3c5cfae64aca1e","00b611878d2a364a","46349dd48e375dac",
        "cbe2f504b6863be7","ad4ccff796dc7e19","c561084ef4157c4c",
        "7e4a60fd866b3c85","ae39ec9c6f4313d0","a918898f5539178a",
        "9a7ac63eb51d9117","e8fc67dd7e78df1c","c0a4e8e7369fb202",
        "d3908f6e963fc27a","9281de7be1be1ac8","970a7a6257a3eefc",
        "4c6b768eaf697cff","baf458e81a3c15cc","3b926d5d8f44105d",
        "d3e9f1f7cd12d5a7","6dba596484de9b06","4bb8b1df23026cc6",
        "8f354ff9c0630a8d","0858748fb31959eb","9a497f8007671c6a",
        "a8f4fd23fcc6c957","10eed02e428271c4","2aa0609de0f121eb",
        "9670354e8cb368ea","e82a0ae5e3137791","1da8f1ab5d8b422a",
        "472a5a66c4774d61","a48381619e59e030","1b866b7f284c0261",
        "843bf3f0d880e4eb","5df1c2e6788d1da3","36841e410ffce743",
        "d077c75752c9f9fb","ea084dd634dbcaac","10da20168ba430a9",
        "df8ada3329c9c24b","1014ed37566bec14","1a967952ffdb640c",
        "a7dde00282bfc930","ad23d89e280a8b1f","11af325689266929",
        "df445fe56e962464","ef1e01bf85bb57b9","432d8dd2cb981c17",
        "235db860aa45739b","ec60db4d672b24da","56e761193cdf09e5",
        "6d092bf40e424774","fc2aac57cf1ec38e","72dd1604b5473a36",
    ])
]
TOTAL_TRAIN = 1_281_167
SUBSET_SIZE = 800_000
SUBSET_SEED = 42
BATCH_MAX = 1280
REPO_ID = "sradc/imagenet_resized_64x64"


def get_parquet_path(pq_name):
    return hf_hub_download(repo_id=REPO_ID, filename=pq_name, repo_type="dataset")


def main():
    output_dir = Path("/tmp/imagenet64_800k/train_npz")

    # Load missing indices
    missing = np.load("/tmp/imagenet64_800k_missing_indices.npy")
    missing_set = set(missing.tolist())
    print(f"Need to find {len(missing_set)} missing images")

    # Find parquet row counts to map global index → (pq_idx, local_idx)
    global_offsets = []
    for pq_name in TRAIN_PARQUETS:
        pq_path = get_parquet_path(pq_name)
        n_rows = pq.read_metadata(pq_path).num_rows
        global_offsets.append(n_rows)
    print(f"Parquet row counts computed: {len(global_offsets)} files, {sum(global_offsets)} total rows")

    # Build parquet → missing row mapping
    pq_to_rows = {}
    cumsum = 0
    for pq_i, n_rows in enumerate(global_offsets):
        for local_i in range(n_rows):
            global_idx = cumsum + local_i
            if global_idx in missing_set:
                pq_to_rows.setdefault(pq_i, []).append(local_i)
        cumsum += n_rows
        if (pq_i + 1) % 10 == 0:
            print(f"  mapped {pq_i+1}/75 parquets, found {sum(len(v) for v in pq_to_rows.values())} rows so far", flush=True)

    print(f"Missing rows found in {len(pq_to_rows)} parquets: {sum(len(v) for v in pq_to_rows.values())} total")

    # Extract and save
    batch_idx = 625  # start after train_batch_0624
    batch_images = []
    saved = 0

    for pq_i in sorted(pq_to_rows.keys()):
        local_indices = pq_to_rows[pq_i]
        pq_name = TRAIN_PARQUETS[pq_i]
        pq_path = get_parquet_path(pq_name)
        table = pq.read_table(pq_path)
        print(f"  parquet {pq_i}: extracting {len(local_indices)} rows ...", flush=True)

        for local_i in local_indices:
            row = table.column('image')[local_i].as_py()
            img = np.array(row, dtype=np.uint8).transpose(2, 0, 1).ravel()
            batch_images.append(img)
            saved += 1

            if len(batch_images) >= BATCH_MAX:
                data = np.stack(batch_images, axis=0)
                path = output_dir / f"train_batch_{batch_idx:04d}.npz"
                np.savez_compressed(path, data=data, labels=list(range(len(data))))
                print(f"    saved {path.name} ({len(data)} imgs, {path.stat().st_size / 1024:.0f} KB)", flush=True)
                batch_images = []
                batch_idx += 1

    if batch_images:
        data = np.stack(batch_images, axis=0)
        path = output_dir / f"train_batch_{batch_idx:04d}.npz"
        np.savez_compressed(path, data=data, labels=list(range(len(data))))
        print(f"    saved {path.name} ({len(data)} imgs, {path.stat().st_size / 1024:.0f} KB)", flush=True)

    # Integrity report
    import glob
    train_files = sorted(glob.glob(str(output_dir / "train_batch_*.npz")))
    total = 0
    for f in train_files:
        d = np.load(f)
        total += d['data'].shape[0]
    print(f"\n=== INTEGRITY REPORT ===")
    print(f"Train files: {len(train_files)}")
    print(f"Train images: {total}")
    print(f"Val files: {len(list((Path('/tmp/imagenet64_800k') / 'val_npz').glob('*.npz')))}")
    print(f"Expected train: 800000")
    print(f"Status: {'OK' if total == 800000 else 'MISMATCH (' + str(800000 - total) + ' short)' }")


if __name__ == "__main__":
    main()
