import math
from collections import OrderedDict

import torch
import torch.nn as nn


class FlatAttentionOrderMLP(nn.Module):
    """Flatten a block attention matrix and predict one priority logit per block."""

    def __init__(
        self,
        num_blocks,
        hidden_dims=None,
        dropout=0.0,
        activation="gelu",
        input_normalization="none",
    ):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.input_dim = self.num_blocks * self.num_blocks
        self.input_normalization = str(input_normalization or "none")
        hidden_dims = [int(value) for value in (hidden_dims or [1024, 1024])]
        dims = [self.input_dim] + hidden_dims + [self.num_blocks]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(_activation(activation))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def normalize_input(self, matrix):
        matrix = matrix.float()
        mode = self.input_normalization
        if mode in {"none", ""}:
            return matrix
        if mode == "zscore":
            mean = matrix.mean()
            std = matrix.std(unbiased=False).clamp_min(1e-6)
            return (matrix - mean) / std
        if mode == "robust_zscore":
            flat = matrix.reshape(-1)
            median = torch.median(flat)
            q25 = torch.quantile(flat, 0.25)
            q75 = torch.quantile(flat, 0.75)
            scale = (q75 - q25) / 1.349
            if (not torch.isfinite(scale)) or float(scale.item()) < 1e-6:
                scale = flat.std(unbiased=False)
            if (not torch.isfinite(scale)) or float(scale.item()) < 1e-6:
                scale = flat.new_tensor(1.0)
            return (matrix - median) / scale
        if mode == "row_normalize":
            denom = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return matrix / denom
        raise ValueError(f"Unsupported attn MLP input_normalization={mode!r}")

    def forward(self, matrix):
        if matrix.ndim == 2:
            if tuple(matrix.shape) != (self.num_blocks, self.num_blocks):
                raise ValueError(
                    f"Expected attention matrix shape {(self.num_blocks, self.num_blocks)}, "
                    f"got {tuple(matrix.shape)}."
                )
            matrix = self.normalize_input(matrix)
            return self.net(matrix.reshape(1, -1)).squeeze(0)
        if matrix.ndim == 3:
            if tuple(matrix.shape[1:]) != (self.num_blocks, self.num_blocks):
                raise ValueError(
                    f"Expected batched attention matrix shape (B, {self.num_blocks}, {self.num_blocks}), "
                    f"got {tuple(matrix.shape)}."
                )
            mode = self.input_normalization
            matrix = matrix.float()
            if mode in {"none", ""}:
                normalized = matrix
            elif mode == "zscore":
                mean = matrix.mean(dim=(1, 2), keepdim=True)
                std = matrix.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
                normalized = (matrix - mean) / std
            elif mode == "row_normalize":
                denom = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                normalized = matrix / denom
            else:
                normalized = torch.stack([self.normalize_input(row) for row in matrix], dim=0)
            return self.net(normalized.reshape(matrix.size(0), -1))
        raise ValueError(f"Expected 2D or 3D attention matrix, got ndim={matrix.ndim}.")


class FeatureAttentionOrderMLP(nn.Module):
    """Flatten block-level feature planes and predict one priority logit per block."""

    def __init__(
        self,
        num_blocks,
        input_channels,
        hidden_dims=None,
        dropout=0.0,
        activation="gelu",
        input_normalization="none",
    ):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.input_channels = int(input_channels)
        self.input_normalization = str(input_normalization or "none")
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}.")
        if self.input_normalization not in {"none", ""}:
            raise ValueError(
                "FeatureAttentionOrderMLP expects pre-normalized feature planes; "
                f"got input_normalization={self.input_normalization!r}."
            )
        hidden_dims = [int(value) for value in (hidden_dims or [1024, 1024])]
        input_dim = self.num_blocks * self.num_blocks * self.input_channels
        dims = [input_dim] + hidden_dims + [self.num_blocks]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(_activation(activation))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        if features.ndim == 3:
            expected = (self.input_channels, self.num_blocks, self.num_blocks)
            if tuple(features.shape) != expected:
                raise ValueError(f"Expected feature shape {expected}, got {tuple(features.shape)}.")
            return self.net(features.float().reshape(1, -1)).squeeze(0)
        if features.ndim == 4:
            expected = (self.input_channels, self.num_blocks, self.num_blocks)
            if tuple(features.shape[1:]) != expected:
                raise ValueError(f"Expected batched feature suffix {expected}, got {tuple(features.shape[1:])}.")
            return self.net(features.float().reshape(features.size(0), -1))
        raise ValueError(f"Expected 3D or 4D feature tensor, got ndim={features.ndim}.")


def load_frozen_attn_mlp_policy(path, num_blocks, device, input_normalization="none"):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    config = _extract_config(checkpoint)
    inferred = _infer_config_from_state_dict(state_dict, expected_num_blocks=int(num_blocks))
    if "num_blocks" not in config:
        config["num_blocks"] = inferred.get("num_blocks", int(num_blocks))
    if int(config["num_blocks"]) != int(num_blocks):
        raise ValueError(
            f"Attn MLP num_blocks={config['num_blocks']} does not match training num_blocks={num_blocks}."
        )
    if "input_channels" not in config and "input_channels" in inferred:
        config["input_channels"] = inferred["input_channels"]
    config.setdefault("hidden_dims", inferred.get("hidden_dims", [1024, 1024]))
    config.setdefault("dropout", 0.0)
    config.setdefault("activation", "gelu")
    config["input_normalization"] = str(
        config.get("input_normalization", input_normalization or "none")
    )
    input_channels = int(config.get("input_channels", 1))
    if input_channels == 1:
        model_config = dict(config)
        model_config.pop("input_channels", None)
        model = FlatAttentionOrderMLP(**model_config)
    else:
        model = FeatureAttentionOrderMLP(**config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config


def logits_to_order(logits, mode="argsort"):
    mode = str(mode)
    if mode == "argsort":
        return torch.argsort(logits.detach().float(), descending=False)
    if mode in {"argsort_desc", "argsort_descending"}:
        return torch.argsort(logits.detach().float(), descending=True)
    raise ValueError(f"Unsupported attn MLP order_mode={mode!r}")


def _activation(name):
    name = str(name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation={name!r}")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "policy_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _strip_common_prefixes(value)
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return _strip_common_prefixes(checkpoint)
    raise ValueError("Could not find a state_dict in the attn MLP checkpoint.")


def _extract_config(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}
    for key in ("config", "model_config", "policy_config", "mlp_config"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            out = dict(value)
            break
    else:
        out = {}
    aliases = {
        "n_blocks": "num_blocks",
        "N": "num_blocks",
        "hidden_dim": "hidden_dims",
        "hidden_size": "hidden_dims",
    }
    for old_key, new_key in aliases.items():
        if old_key in out and new_key not in out:
            out[new_key] = out[old_key]
    if isinstance(out.get("hidden_dims"), int):
        out["hidden_dims"] = [int(out["hidden_dims"])]
    allowed = {"num_blocks", "input_channels", "hidden_dims", "dropout", "activation", "input_normalization"}
    return {key: value for key, value in out.items() if key in allowed}


def _strip_common_prefixes(state_dict):
    cleaned = OrderedDict()
    prefixes = ("module.", "_orig_mod.", "policy.", "attn_mlp.", "model.")
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _infer_config_from_state_dict(state_dict, expected_num_blocks=None):
    linear_weights = []
    for key, value in state_dict.items():
        if key.endswith(".weight") and torch.is_tensor(value) and value.ndim == 2:
            linear_weights.append((key, tuple(value.shape)))
    linear_weights.sort(key=lambda item: _natural_key(item[0]))
    if not linear_weights:
        return {}
    input_dim = int(linear_weights[0][1][1])
    output_dim = int(linear_weights[-1][1][0])
    if expected_num_blocks is not None:
        root = int(expected_num_blocks)
        square = root * root
        if input_dim % square != 0:
            raise ValueError(
                f"Cannot infer input_channels from first linear input_dim={input_dim} "
                f"and num_blocks={root}."
            )
        input_channels = int(input_dim // square)
    else:
        root = int(math.isqrt(input_dim))
        if root * root != input_dim:
            raise ValueError(
                f"Cannot infer square num_blocks from first linear input_dim={input_dim}."
            )
        input_channels = 1
    hidden_dims = [int(shape[0]) for _, shape in linear_weights[:-1]]
    return {
        "num_blocks": int(output_dim if output_dim == root else root),
        "input_channels": int(input_channels),
        "hidden_dims": hidden_dims,
    }


def _natural_key(text):
    parts = []
    current = ""
    for char in str(text):
        if char.isdigit():
            current += char
        else:
            if current:
                parts.append(int(current))
                current = ""
            parts.append(char)
    if current:
        parts.append(int(current))
    return parts
