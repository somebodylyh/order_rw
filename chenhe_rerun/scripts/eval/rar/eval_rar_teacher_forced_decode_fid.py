"""
Teacher-forced decoded FID for official RAR checkpoints.

This is a RAR-only adapter for comparing an official RAR checkpoint with this
repo's AO-GPT teacher-forced decode protocol. It is not the official RAR
generation benchmark:

  1. Load fixed-record MaskGIT-VQGAN token records from data/Imagenet256MaskGITVQ.
  2. Run the RAR checkpoint on true tokens in raster teacher-forced mode.
  3. Use either true ImageNet labels or RAR's null condition token.
  4. Take argmax logits as predicted VQ token ids.
  5. Decode predicted tokens with the MaskGIT-VQGAN f16 tokenizer.
  6. Compute torch-fidelity Inception-FID against ADM reference statistics.

The script is staged to keep peak VRAM low: the RAR model is released before the
VQGAN decoder and Inception feature extractor are loaded.
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import linalg
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
RAR_ROOT = REPO_ROOT / "external" / "1d-tokenizer"
sys.path.insert(0, str(RAR_ROOT))

from modeling.rar import RAR  # noqa: E402
from modeling.titok import PretrainedTokenizer  # noqa: E402


def load_token_records(dataset_dir, split, num_images, seed):
    meta_path = dataset_dir / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing dataset metadata: {meta_path}")
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)

    tokens_per_image = int(meta["tokens_per_image"])
    count = int(meta.get(f"{split}_records", meta.get(f"{split}_images", 0)))
    if count <= 0:
        raise ValueError(f"could not infer {split} record count from {meta_path}")

    limit = int(num_images)
    if limit <= 0 or limit >= count:
        record_ids = np.arange(count, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        record_ids = np.sort(rng.choice(count, size=limit, replace=False).astype(np.int64))

    bin_path = dataset_dir / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing split tokens: {bin_path}")
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    tokens = np.asarray(tokens.reshape(count, tokens_per_image)[record_ids], dtype=np.int64)

    label_path = dataset_dir / str(meta.get(f"{split}_labels", f"{split}_labels.npy"))
    labels = None
    if label_path.exists():
        labels = np.asarray(np.load(label_path)[record_ids], dtype=np.int64)

    return tokens, labels, record_ids, meta


def load_rar_config(args):
    config = OmegaConf.load(args.rar_config)
    config.experiment.generator_checkpoint = str(args.rar_ckpt)
    config.model.vq_model.pretrained_tokenizer_weight = str(args.vqgan_ckpt)
    config.model.generator.hidden_size = int(args.hidden_size)
    config.model.generator.num_hidden_layers = int(args.num_hidden_layers)
    config.model.generator.num_attention_heads = int(args.num_attention_heads)
    config.model.generator.intermediate_size = int(args.intermediate_size)
    return config


def load_rar_model(config, device):
    model = RAR(config)
    state = torch.load(str(config.experiment.generator_checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    model.set_random_ratio(0.0)
    return model


def make_condition(model, labels, condition_mode, device):
    if condition_mode == "label":
        if labels is None:
            raise ValueError("--condition_mode label requires split label file")
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
        return model.preprocess_condition(labels_t, cond_drop_prob=0.0)
    if condition_mode == "none":
        batch_size = int(labels.shape[0]) if labels is not None else 0
        return torch.full((batch_size,), int(model.none_condition_id), dtype=torch.long, device=device)
    raise ValueError(f"unsupported condition_mode={condition_mode!r}")


@torch.no_grad()
def predict_argmax_tokens(model, token_records, label_records, *, condition_mode, device, batch_size, dtype):
    ctx_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    use_amp = device.type == "cuda" and dtype != "float32"

    preds = []
    loss_sum = 0.0
    correct = 0
    total = 0
    vocab_size = int(model.target_codebook_size)
    progress = tqdm(total=int(token_records.shape[0]), desc=f"RAR argmax ({condition_mode})", unit="img")

    for start in range(0, int(token_records.shape[0]), int(batch_size)):
        batch_np = token_records[start : start + int(batch_size)]
        labels_np = None
        if label_records is not None:
            labels_np = label_records[start : start + int(batch_size)]
        else:
            labels_np = np.zeros((int(batch_np.shape[0]),), dtype=np.int64)

        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.long)
        condition = make_condition(model, labels_np, condition_mode, device)

        with torch.autocast(device_type=device.type, dtype=ctx_dtype, enabled=use_amp):
            logits = model.forward_fn(batch, condition, orders=None, is_sampling=False)
            token_logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(
                token_logits.view(-1, vocab_size),
                batch.contiguous().view(-1),
                reduction="sum",
            )
            pred = token_logits.argmax(dim=-1)

        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        preds.append(pred_np)
        loss_sum += float(loss.item())
        correct += int((pred_np == batch_np).sum())
        total += int(batch_np.size)
        progress.update(int(batch_np.shape[0]))

    progress.close()
    return np.concatenate(preds, axis=0), {
        "mean_ce_loss": float(loss_sum / max(1, total)),
        "token_accuracy": float(correct / max(1, total)),
    }


class FeatureStats:
    def __init__(self, feature_dim):
        self.n = 0
        self.sum = torch.zeros(int(feature_dim), dtype=torch.float64, device="cpu")
        self.sumsq = torch.zeros((int(feature_dim), int(feature_dim)), dtype=torch.float64, device="cpu")

    def update(self, features):
        features = features.detach().cpu().to(torch.float64)
        self.n += int(features.shape[0])
        self.sum += features.sum(dim=0)
        self.sumsq += features.t().matmul(features)

    def mean_cov(self):
        if self.n < 2:
            raise ValueError("at least two images are required to compute FID covariance")
        mu = self.sum / float(self.n)
        sigma = (self.sumsq - float(self.n) * torch.outer(mu, mu)) / float(self.n - 1)
        return mu.numpy(), sigma.numpy()


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape:
        raise ValueError(f"mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}")
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"sqrtm returned significant imaginary component: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def load_adm_reference_stats(path):
    obj = np.load(path)
    for key in ("mu", "sigma"):
        if key not in obj:
            raise ValueError(f"{path} is missing ADM reference key {key!r}")
    return obj["mu"], obj["sigma"]


def create_feature_extractor(device):
    try:
        from torch_fidelity.utils import create_feature_extractor
    except Exception as exc:
        raise RuntimeError("torch_fidelity is required for FID feature extraction") from exc
    return create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=bool(device.type == "cuda"),
        verbose=False,
    )


@torch.no_grad()
def accumulate_decoded_stats(tokenizer, extractor, token_records, *, device, batch_size, dtype, desc):
    ctx_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    use_amp = device.type == "cuda" and dtype != "float32"
    stats = FeatureStats(feature_dim=2048)
    saved_preview = []
    progress = tqdm(total=int(token_records.shape[0]), desc=desc, unit="img")

    for start in range(0, int(token_records.shape[0]), int(batch_size)):
        batch_np = token_records[start : start + int(batch_size)]
        codes = torch.from_numpy(batch_np).to(device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=ctx_dtype, enabled=use_amp):
            images = tokenizer.decode_tokens(codes)
        images_u8 = images.detach().float().clamp(0.0, 1.0).mul(255.0).round().clamp(0, 255).to(torch.uint8)
        features = extractor(images_u8)[0]
        stats.update(features)
        if len(saved_preview) < 16:
            preview = images_u8[: 16 - len(saved_preview)].permute(0, 2, 3, 1).cpu().numpy()
            saved_preview.extend([preview[idx] for idx in range(preview.shape[0])])
        progress.update(int(batch_np.shape[0]))

    progress.close()
    return stats, saved_preview


def save_preview_grid(images, out_path, *, columns=4):
    if not images:
        return None
    from PIL import Image

    columns = max(1, int(columns))
    rows = int(math.ceil(len(images) / columns))
    h, w = images[0].shape[:2]
    canvas = Image.new("RGB", (columns * w, rows * h), color=(255, 255, 255))
    for idx, arr in enumerate(images):
        row, col = divmod(idx, columns)
        canvas.paste(Image.fromarray(arr), (col * w, row * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rar_root", type=Path, default=RAR_ROOT)
    parser.add_argument("--rar_config", type=Path, default=RAR_ROOT / "configs/training/generator/rar.yaml")
    parser.add_argument("--rar_ckpt", type=Path, default=RAR_ROOT / "rar_b.bin")
    parser.add_argument("--dataset_dir", type=Path, default=REPO_ROOT / "data/Imagenet256MaskGITVQ")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vqgan_ckpt", type=Path, default=REPO_ROOT / "checkpoints/vq/maskgit-vqgan-imagenet-f16-256.bin")
    parser.add_argument("--adm_ref_npz", type=Path, default=REPO_ROOT / "data/imagenet256_adm_ref/VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--condition_mode", choices=("none", "label"), default="none")
    parser.add_argument("--num_images", type=int, default=50000, help="0 means all records in the split.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch_size", type=int, default=8, help="RAR teacher-forced argmax batch size.")
    parser.add_argument("--decode_batch_size", type=int, default=8, help="VQGAN decode/Inception batch size.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--decode_dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--compute_target_vq_fid", action="store_true")
    parser.add_argument("--save_pred_tokens", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=24)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    token_records, label_records, record_ids, meta = load_token_records(
        args.dataset_dir,
        args.split,
        args.num_images,
        args.seed,
    )
    tokens_per_image = int(meta["tokens_per_image"])
    if tokens_per_image != 256:
        raise ValueError(f"official RAR-B expects 256 tokens/image, got {tokens_per_image}")
    if int(meta["vocab_size"]) != 1024:
        raise ValueError(f"official RAR-B expects vocab_size=1024, got {meta['vocab_size']}")

    config = load_rar_config(args)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = load_rar_model(config, device)
    pred_tokens, token_metrics = predict_argmax_tokens(
        model,
        token_records,
        label_records,
        condition_mode=args.condition_mode,
        device=device,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    model_peak_gib = None
    if device.type == "cuda":
        model_peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if args.save_pred_tokens:
        np.save(args.out_dir / "pred_tokens.npy", pred_tokens.astype(np.uint16))

    tokenizer = PretrainedTokenizer(str(args.vqgan_ckpt)).eval().to(device)
    extractor = create_feature_extractor(device).eval()
    pred_stats, pred_preview = accumulate_decoded_stats(
        tokenizer,
        extractor,
        pred_tokens,
        device=device,
        batch_size=args.decode_batch_size,
        dtype=args.decode_dtype,
        desc="decode+fid RAR pred",
    )
    target_stats = None
    target_preview = []
    if args.compute_target_vq_fid:
        target_stats, target_preview = accumulate_decoded_stats(
            tokenizer,
            extractor,
            token_records,
            device=device,
            batch_size=args.decode_batch_size,
            dtype=args.decode_dtype,
            desc="decode+fid target-vq",
        )

    decode_peak_gib = None
    if device.type == "cuda":
        decode_peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3))

    adm_mu, adm_sigma = load_adm_reference_stats(args.adm_ref_npz)
    pred_mu, pred_sigma = pred_stats.mean_cov()
    fid = {
        "model_argmax_decode_vs_adm_ref": {
            "fid": frechet_distance(adm_mu, adm_sigma, pred_mu, pred_sigma),
            "num_images": int(pred_stats.n),
            "reference": str(args.adm_ref_npz),
            "feature_extractor": "torch_fidelity inception-v3-compat",
            "reference_stats_keys": ["mu", "sigma"],
        }
    }
    if target_stats is not None:
        target_mu, target_sigma = target_stats.mean_cov()
        fid["target_vq_reconstruction_vs_adm_ref"] = {
            "fid": frechet_distance(adm_mu, adm_sigma, target_mu, target_sigma),
            "num_images": int(target_stats.n),
            "reference": str(args.adm_ref_npz),
            "feature_extractor": "torch_fidelity inception-v3-compat",
            "reference_stats_keys": ["mu", "sigma"],
        }

    pred_grid = save_preview_grid(pred_preview, args.out_dir / "pred_preview_grid.png")
    target_grid = save_preview_grid(target_preview, args.out_dir / "target_vq_preview_grid.png")
    token_exact = np.all(pred_tokens == token_records, axis=1)
    summary = {
        "protocol": "rar_teacher_forced_argmax_maskgit_vqgan_raster",
        "rar_ckpt": str(args.rar_ckpt),
        "rar_config": str(args.rar_config),
        "condition_mode": args.condition_mode,
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "num_images": int(token_records.shape[0]),
        "seed": int(args.seed),
        "record_ids_first10": [int(v) for v in record_ids[:10].tolist()],
        "device": str(device),
        "dtype": args.dtype,
        "decode_dtype": args.decode_dtype,
        "batch_size": int(args.batch_size),
        "decode_batch_size": int(args.decode_batch_size),
        "model_token_metrics": {
            **token_metrics,
            "sequence_exact_match_rate": float(token_exact.mean()),
            "pred_unique_codes": int(np.unique(pred_tokens).shape[0]),
            "target_unique_codes": int(np.unique(token_records).shape[0]),
            "vocab_size": int(meta["vocab_size"]),
        },
        "fid": fid,
        "dataset_meta": {
            "vocab_size": int(meta["vocab_size"]),
            "tokens_per_image": int(tokens_per_image),
            "image_size": int(meta.get("image_size", 16)),
            "original_image_size": int(meta.get("original_image_size", 256)),
            "tokenizer": str(meta.get("tokenizer", "")),
            "has_labels": bool(label_records is not None),
        },
        "frame_mapping": {
            "input_tokens_frame": "original_raster_vq",
            "decoded_tokens_frame": "original_raster_vq",
            "permutation_applied": False,
        },
        "memory": {
            "model_phase_peak_allocated_gib": model_peak_gib,
            "decode_fid_phase_peak_allocated_gib": decode_peak_gib,
        },
        "outputs": {
            "pred_preview_grid": pred_grid,
            "target_vq_preview_grid": target_grid,
            "pred_tokens_npy": str(args.out_dir / "pred_tokens.npy") if args.save_pred_tokens else None,
            "summary_json": str(args.out_dir / "summary.json"),
        },
        "note": (
            "This is not official RAR generation FID. It is a teacher-forced "
            "argmax decoded-FID adapter for comparison with this repo's AO-GPT "
            "checkpoint decode protocol."
        ),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
