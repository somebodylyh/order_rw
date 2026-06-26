"""Attention trajectory logging: extract L0 all-head B maps at eval time.

Saves model-frame strict65 B maps, computes physical-frame remap (posthoc),
generates summary metrics / heatmaps, and optionally logs to W&B.

All extraction is label-free: no CDL teacher, no NLL, no training-time order
labels. Uses fixed eval samples so trajectories are comparable across runs.

Integration point: called from ``train_clean_aogpt.run_eval_and_save`` at
each eval step when ``--attn-trajectory`` is enabled.

Convention:
  B[source, target] — B is A^T (transposed attention), consistent with the
  project-wide convention used in ``l0_strict65.build_model_frame_strict65``
  and ``none_separated_block_graph``.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── project-local imports ────────────────────────────────────────────────
from batch_readout.l0_strict65 import build_model_frame_strict65
from per_head_order_scan import _batch_mean_B

# Optional: matplotlib for heatmap generation
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Optional: wandb for image logging
try:
    import wandb as _wandb_mod
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# ── constants ────────────────────────────────────────────────────────────
SEQ_LEN = 256
BLOCK_LEN = 4
N_BLOCKS = 64    # content blocks only (excludes None)
N_NODES = 65     # None + 64 content blocks


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_all_layer_B(attn_list, probe_orders) -> np.ndarray:
    """Build model-frame strict65 B for every layer.

    attn_list: list of (S, H, 257, 257) per layer (torch or numpy).
    probe_orders: (S, 256) int array of model-coordinate token reveal orders.
    Returns (L, S, H, 65, 65) float32.
    """
    out = []
    for attn_l in attn_list:
        arr = attn_l.cpu().numpy() if hasattr(attn_l, "cpu") else np.asarray(attn_l)
        out.append(build_model_frame_strict65(arr, probe_orders))
    return np.stack(out, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# Summary metrics
# ═══════════════════════════════════════════════════════════════════════════

def _row_entropy(B: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """Per-row entropy of abs(B), returns (mean, std) across rows."""
    abs_B = np.abs(B) + eps
    row_sum = abs_B.sum(axis=-1, keepdims=True)
    p = abs_B / row_sum
    H = -np.sum(p * np.log(p + eps), axis=-1)
    return float(H.mean()), float(H.std())


def _near_diag_mass(B: np.ndarray, d: int) -> float:
    """Fraction of total abs(B) mass within distance d of diagonal."""
    n = B.shape[0]
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= d
    return float(np.abs(B[mask]).sum() / (np.abs(B).sum() + 1e-12))


def _topk_mass(B: np.ndarray, k: int) -> float:
    """Mean fraction of row mass in top-k entries."""
    abs_B = np.abs(B) + 1e-12
    row_sum = abs_B.sum(axis=-1, keepdims=True)
    sorted_vals = np.sort(abs_B, axis=-1)[:, ::-1]
    return float((sorted_vals[:, :k].sum(axis=-1) / row_sum.squeeze(-1)).mean())


def _locality_score(B: np.ndarray) -> float:
    """Weighted-mean distance from diagonal (lower = more local)."""
    n = B.shape[0]
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(np.float32)
    w = np.abs(B) + 1e-12
    w /= w.sum()
    return float((w * d).sum())


def compute_head_summary(
    B: np.ndarray,
    head_idx: int,
    phys_perm: Optional[np.ndarray] = None,
) -> dict:
    """Compute per-head summary metrics on a (65, 65) or (64, 64) B matrix.

    Args:
        B: (n_nodes, n_nodes) B[source, target] matrix.
        head_idx: head index for the output label.
        phys_perm: optional physical permutation (64,) for content-block remap.
            If provided, also compute phys-frame metrics on content blocks.

    Returns:
        dict of scalar metrics.
    """
    n = B.shape[0]
    info = {
        "head": int(head_idx),
        "nodes": n,
        "mean": float(B.mean()),
        "std": float(B.std()),
        "max": float(B.max()),
        "min": float(B.min()),
        "sparsity": float((np.abs(B) < 1e-8).mean()),
    }

    # Row entropy on full B
    ent_mean, ent_std = _row_entropy(B)
    info["row_entropy_mean"] = ent_mean
    info["row_entropy_std"] = ent_std

    # Near-diag mass on full B
    for d in (1, 2, 4):
        info[f"near_diag_mass_d{d}"] = _near_diag_mass(B, d)

    info["diag_mass"] = _near_diag_mass(B, 0)
    info["offdiag_mass"] = 1.0 - info["diag_mass"]
    info["top1_mass_mean"] = _topk_mass(B, 1)
    info["top4_mass_mean"] = _topk_mass(B, 4)
    info["locality_score"] = _locality_score(B)

    # Content-block-only metrics (strip None node if 65×65)
    B_content = B[1:, 1:] if n == N_NODES else B
    if B_content.shape == (N_BLOCKS, N_BLOCKS):
        info["content_mean"] = float(B_content.mean())
        info["content_std"] = float(B_content.std())
        ent_cm, ent_cs = _row_entropy(B_content)
        info["content_row_entropy_mean"] = ent_cm
        info["content_row_entropy_std"] = ent_cs
        for d in (1, 2, 4):
            info[f"content_near_diag_mass_d{d}"] = _near_diag_mass(B_content, d)
        info["content_locality_score"] = _locality_score(B_content)

        # Physical-frame content-block metrics
        if phys_perm is not None and len(phys_perm) == N_BLOCKS:
            B_phys = B_content[np.ix_(phys_perm, phys_perm)]
            for d in (1, 2, 4):
                info[f"phys_near_diag_mass_d{d}"] = _near_diag_mass(B_phys, d)
            info["phys_locality_score"] = _locality_score(B_phys)
            ent_pm, ent_ps = _row_entropy(B_phys)
            info["phys_row_entropy_mean"] = ent_pm
            info["phys_row_entropy_std"] = ent_ps

    return info


def compute_sample_summary(
    B_sample: np.ndarray,
    phys_perm: Optional[np.ndarray] = None,
) -> dict:
    """Compute aggregated metrics across heads for one sample.

    Args:
        B_sample: (heads, 65, 65) model-frame strict65 B.
        phys_perm: optional physical permutation.

    Returns:
        dict with head-level list metrics and cross-head aggregates.
    """
    n_heads = B_sample.shape[0]
    per_head = []
    for h in range(n_heads):
        per_head.append(compute_head_summary(B_sample[h], h, phys_perm=phys_perm))

    # Cross-head aggregates
    keys = [k for k in per_head[0] if isinstance(per_head[0][k], (int, float))]
    agg = {"n_heads": n_heads}
    for k in keys:
        vals = [d[k] for d in per_head]
        agg[f"head_mean_{k}"] = float(np.mean(vals))
        agg[f"head_std_{k}"] = float(np.std(vals))
        agg[f"best_head_{k}"] = float(max(vals))

    agg["per_head"] = per_head
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# Heatmap generation
# ═══════════════════════════════════════════════════════════════════════════

def _make_all_head_grid(
    B_sample: np.ndarray,
    head_labels: Optional[List[str]] = None,
    title_prefix: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    row_norm: bool = False,
) -> plt.Figure:
    """Create an all-head grid figure from a single sample's B maps.

    Args:
        B_sample: (heads, n, n) B matrices.
        head_labels: optional per-head labels.
        title_prefix: prefix for the suptitle.
        vmin, vmax: colorbar range (auto if None).
        row_norm: if True, row-normalize before plotting (enhances structure).

    Returns:
        matplotlib Figure.
    """
    n_heads = B_sample.shape[0]
    cols = min(4, n_heads)
    rows = int(np.ceil(n_heads / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    axes = axes.flatten()

    if vmin is None:
        vmin = float(np.percentile(B_sample, 1))
    if vmax is None:
        vmax = float(np.percentile(B_sample, 99))

    for h in range(n_heads):
        ax = axes[h]
        B_plot = B_sample[h]
        if row_norm:
            row_sum = np.abs(B_plot).sum(axis=-1, keepdims=True) + 1e-12
            B_plot = B_plot / row_sum
        im = ax.imshow(B_plot, aspect="auto", cmap="RdBu_r",
                       vmin=vmin if not row_norm else None,
                       vmax=vmax if not row_norm else None,
                       interpolation="nearest")
        label = head_labels[h] if head_labels else f"H{h}"
        ax.set_title(f"L0 {label}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)

    suptitle = f"{title_prefix}L0 All-Head B (model-frame strict65)"
    if row_norm:
        suptitle += " [row-norm]"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main logger class
# ═══════════════════════════════════════════════════════════════════════════

class AttentionTrajectoryLogger:
    """Extract and save L0 all-head B maps at eval time for trajectory analysis.

    Usage::

        logger = AttentionTrajectoryLogger(
            output_root="outputs/attention_trajectory/my_run",
            run_name="my_run",
            n_attention_samples=8,
            wandb_run=wandb_run,
            phys_perm=inv_perm,  # 64-array mapping model→phys blocks
            heatmap_interval=5000,
            seed=123,
        )

        # In eval loop:
        logger.log_snapshot(
            model=model,
            eval_batch=idx_eval_model[:n_samples],
            global_step=5000,
            clean_perm=clean_perm,
            device=device,
        )
    """

    def __init__(
        self,
        output_root: str | Path,
        run_name: str,
        n_attention_samples: int = 8,
        wandb_run=None,
        phys_perm: Optional[np.ndarray] = None,
        heatmap_interval: int = 5000,
        heatmap_png_interval: int = 10000,
        seed: int = 123,
        extra_metadata: Optional[dict] = None,
    ):
        self.output_root = Path(output_root)
        self.run_name = run_name
        self.n_samples = n_attention_samples
        self.wandb_run = wandb_run
        self.heatmap_interval = int(heatmap_interval)
        self.heatmap_png_interval = int(heatmap_png_interval)
        self.seed = int(seed)

        # Physical permutation (model block → physical block), content only
        # phys_perm[i] = physical position of model-block i
        self.phys_perm = None
        if phys_perm is not None:
            self.phys_perm = np.asarray(phys_perm, dtype=np.int64)

        # Fixed probe samples (set once on first call)
        self._probe_indices: Optional[np.ndarray] = None
        self._probe_orders: Optional[np.ndarray] = None
        self._sample_ids_set: bool = False

        # Per-step summary log
        self._summary_log: List[dict] = []

        self._ensure_dirs()

        # Write global metadata
        meta = {
            "run_name": run_name,
            "n_attention_samples": n_attention_samples,
            "matrix_definition": "B = A^T",
            "matrix_convention": "B[source, target]",
            "frame": "model-frame strict65",
            "num_layers_extracted": "all",
            "all_layers": True,
            "num_heads": 8,
            "nodes": N_NODES,
            "none_node": 0,
            "actual_blocks": f"1..{N_BLOCKS}",
            "block_size": BLOCK_LEN,
            "seq_len": SEQ_LEN,
            "heatmap_interval": self.heatmap_interval,
            "heatmap_png_interval": self.heatmap_png_interval,
            "seed": seed,
            "phys_perm_provided": phys_perm is not None,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        (self.output_root / "metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

        self._log_fn = print

    def set_log_fn(self, fn):
        self._log_fn = fn

    def _ensure_dirs(self):
        _ensure_dir(self.output_root)
        _ensure_dir(self.output_root / "raw")
        _ensure_dir(self.output_root / "heatmaps")
        _ensure_dir(self.output_root / "summaries")

    def _set_fixed_samples(
        self,
        eval_model_tokens: torch.Tensor,
        device: torch.device,
    ):
        """Initialise fixed eval samples for attention extraction.

        Uses a seeded RNG so the same indices are picked across runs sharing
        the same seed and eval set size.  The probe orders are fixed at step 0
        (reveal order = identity) so differences across steps are purely model
        changes.
        """
        n_avail = eval_model_tokens.shape[0]
        if n_avail < self.n_samples:
            raise ValueError(
                f"Only {n_avail} eval samples available, need {self.n_samples}"
            )
        rng = np.random.default_rng(self.seed * 10000 + 7777)
        indices = rng.choice(n_avail, size=self.n_samples, replace=False)
        indices.sort()  # deterministic order
        self._probe_indices = indices.astype(np.int64)

        # Fixed probe order: identity (model-ascending order). This gives a
        # consistent view of attention structure — the model sees tokens in
        # increasing order, same every step.
        self._probe_orders = np.tile(
            np.arange(SEQ_LEN, dtype=np.int64), (self.n_samples, 1)
        )

        self._sample_ids_set = True

        sample_ids_hash = hex(hash(tuple(self._probe_indices.tolist())) & 0xFFFF_FFFF)
        self._log_fn(
            f"[attn_traj] fixed {self.n_samples} attention samples, "
            f"indices_hash={sample_ids_hash}"
        )
        # Persist sample indices
        np.savez(
            self.output_root / "attention_sample_indices.npz",
            indices=self._probe_indices,
            probe_orders=self._probe_orders,
            seed=self.seed,
            n_avail=n_avail,
        )

    @torch.no_grad()
    def log_snapshot(
        self,
        model,
        eval_model_tokens: torch.Tensor,
        global_step: int,
        clean_perm,
        device: torch.device,
    ) -> dict:
        """Extract and save L0 all-head B maps at the given step.

        Returns a dict of lightweight metrics suitable for W&B logging.
        """
        t0 = time.time()

        # Lazy init fixed samples
        if not self._sample_ids_set:
            self._set_fixed_samples(eval_model_tokens, device)

        was_training = model.training
        model.eval()

        # Forward pass on fixed samples
        probe_chunks = eval_model_tokens[self._probe_indices].to(device)
        probe_orders_t = torch.from_numpy(self._probe_orders).to(device)

        _, _, attn_list = model.forward_fn(
            probe_chunks, probe_orders_t, return_attentions=True
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # all-layer extraction (was L0-only)
        B_all = extract_all_layer_B(attn_list, self._probe_orders)  # (L,S,H,65,65)
        if was_training:
            model.train()
        B_model = B_all[0]  # keep existing L0 summaries/heatmaps working
        # B_model shape: (n_samples, n_heads, 65, 65)

        # Build physical-frame B maps if perm is available
        B_phys = None
        if self.phys_perm is not None:
            B_phys = np.empty_like(B_model)
            inv_phys = np.argsort(self.phys_perm)
            for s in range(self.n_samples):
                for h in range(B_model.shape[1]):
                    # Remap content blocks 1..64 (skip None node 0)
                    Bc = B_model[s, h, 1:, 1:]  # (64, 64) model-frame content
                    Bc_phys = Bc[np.ix_(inv_phys, inv_phys)]  # physical-frame
                    B_phys[s, h, 1:, 1:] = Bc_phys
                    # None edges: None→target stays model-frame (target remapped)
                    B_phys[s, h, 0, 1:] = B_model[s, h, 0, 1:][inv_phys]
                    B_phys[s, h, :, 0] = 0.0

        elapsed = time.time() - t0

        # ── Save raw tensors ───────────────────────────────────────────
        step_dir = _ensure_dir(self.output_root / "raw" / f"step_{int(global_step):06d}")
        torch.save(torch.from_numpy(B_model), step_dir / "B_model.pt")      # L0 alias
        if B_phys is not None:
            torch.save(torch.from_numpy(B_phys), step_dir / "B_phys.pt")

        # ── Per-sample summaries ───────────────────────────────────────
        sample_summaries = []
        for s in range(self.n_samples):
            summ = compute_sample_summary(B_model[s], phys_perm=self.phys_perm)
            summ["sample_idx"] = int(s)
            sample_summaries.append(summ)

        # ── Mean across samples (per-head) ────────────────────────────
        B_model_mean = B_model.mean(axis=0)  # (heads, 65, 65)
        per_head_mean = []
        for h in range(B_model_mean.shape[0]):
            per_head_mean.append(
                compute_head_summary(B_model_mean[h], h, phys_perm=self.phys_perm)
            )

        # ── Metadata for this step ────────────────────────────────────
        step_meta = {
            "run_name": self.run_name,
            "effective_step": int(global_step),
            "n_samples": self.n_samples,
            "n_heads": B_model.shape[1],
            "extraction_time_s": round(elapsed, 3),
            "sample_indices_hash": hex(
                hash(tuple(self._probe_indices.tolist())) & 0xFFFF_FFFF
            ),
            "phys_perm_available": self.phys_perm is not None,
        }
        (step_dir / "metadata.json").write_text(
            json.dumps(step_meta, indent=2) + "\n"
        )

        # ── Summary JSON (lightweight) ────────────────────────────────
        summary_payload = {
            "step": int(global_step),
            "extraction_time_s": round(elapsed, 3),
            "per_head_mean": per_head_mean,
            "sample_summaries": sample_summaries,
        }
        summary_path = (
            self.output_root / "summaries" / f"summary_step{int(global_step):06d}.json"
        )
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

        self._summary_log.append({
            "step": int(global_step),
            "extraction_time_s": round(elapsed, 3),
            "head_mean_entropy": float(np.mean([d["row_entropy_mean"] for d in per_head_mean])),
            "best_head_near_diag_d1": float(max(d.get("near_diag_mass_d1", 0) for d in per_head_mean)),
        })

        # ── Heatmap PNG (low frequency) ───────────────────────────────
        if _HAS_MPL and int(global_step) % self.heatmap_png_interval == 0:
            try:
                # Model-frame grid
                fig_model = _make_all_head_grid(
                    B_model_mean,
                    title_prefix=f"step {global_step} — ",
                )
                png_model = (
                    self.output_root / "heatmaps"
                    / f"l0_B_model_grid_step{int(global_step):06d}.png"
                )
                fig_model.savefig(png_model, dpi=120, bbox_inches="tight")
                plt.close(fig_model)

                # Row-norm version
                fig_row = _make_all_head_grid(
                    B_model_mean,
                    title_prefix=f"step {global_step} — ",
                    row_norm=True,
                )
                png_row = (
                    self.output_root / "heatmaps"
                    / f"l0_B_model_grid_rowprob_step{int(global_step):06d}.png"
                )
                fig_row.savefig(png_row, dpi=120, bbox_inches="tight")
                plt.close(fig_row)

                if B_phys is not None:
                    B_phys_mean = B_phys.mean(axis=0)
                    fig_phys = _make_all_head_grid(
                        B_phys_mean,
                        title_prefix=f"step {global_step} (phys) — ",
                    )
                    png_phys = (
                        self.output_root / "heatmaps"
                        / f"l0_B_phys_grid_step{int(global_step):06d}.png"
                    )
                    fig_phys.savefig(png_phys, dpi=120, bbox_inches="tight")
                    plt.close(fig_phys)

                    fig_phys_row = _make_all_head_grid(
                        B_phys_mean,
                        title_prefix=f"step {global_step} (phys) — ",
                        row_norm=True,
                    )
                    png_phys_row = (
                        self.output_root / "heatmaps"
                        / f"l0_B_phys_grid_rowprob_step{int(global_step):06d}.png"
                    )
                    fig_phys_row.savefig(png_phys_row, dpi=120, bbox_inches="tight")
                    plt.close(fig_phys_row)

                self._log_fn(
                    f"[attn_traj] heatmaps saved @ step {global_step} "
                    f"(extraction {elapsed:.1f}s)"
                )
            except Exception as exc:
                warnings.warn(f"[attn_traj] heatmap generation failed: {exc}")

        # ── W&B logging ──────────────────────────────────────────────
        wb_payload = self._wandb_payload(per_head_mean, elapsed, global_step)
        if self.wandb_run is not None and _HAS_WANDB:
            try:
                self.wandb_run.log(wb_payload, step=int(global_step))

                # Upload heatmap images at low frequency
                if int(global_step) % self.heatmap_png_interval == 0 and _HAS_MPL:
                    for key, png_path in [
                        ("attn/l0_B_model_grid", png_model),
                        ("attn/l0_B_model_grid_rowprob", png_row),
                    ]:
                        if png_path.exists():
                            self.wandb_run.log(
                                {key: _wandb_mod.Image(str(png_path))},
                                step=int(global_step),
                            )
                    if B_phys is not None:
                        for key, png_path in [
                            ("attn/l0_B_phys_grid", png_phys),
                            ("attn/l0_B_phys_grid_rowprob", png_phys_row),
                        ]:
                            if png_path.exists():
                                self.wandb_run.log(
                                    {key: _wandb_mod.Image(str(png_path))},
                                    step=int(global_step),
                                )
            except Exception as exc:
                warnings.warn(f"[attn_traj] W&B logging failed: {exc}")

        return wb_payload

    def _wandb_payload(
        self,
        per_head: List[dict],
        elapsed: float,
        global_step: int,
    ) -> dict:
        """Build a flat dict of W&B-safe attention metrics."""
        payload = {
            "attn/extraction_time": round(elapsed, 3),
            "attn/snapshot_saved": True,
            "attn/snapshot_step": int(global_step),
        }

        # Per-head metrics (selected keys)
        for d in per_head:
            h = d["head"]
            for key in (
                "row_entropy_mean",
                "near_diag_mass_d1",
                "near_diag_mass_d2",
                "top1_mass_mean",
                "top4_mass_mean",
                "locality_score",
            ):
                if key in d:
                    payload[f"attn/l0_h{h}_{key}"] = d[key]
            if "phys_near_diag_mass_d1" in d:
                payload[f"attn/l0_h{h}_phys_near_diag_d1"] = d["phys_near_diag_mass_d1"]

        # Cross-head aggregates
        row_ents = [d["row_entropy_mean"] for d in per_head if "row_entropy_mean" in d]
        nd1s = [d.get("near_diag_mass_d1", 0.0) for d in per_head]
        phys_nd1s = [d.get("phys_near_diag_mass_d1", 0.0) for d in per_head]
        top1s = [d.get("top1_mass_mean", 0.0) for d in per_head]

        if row_ents:
            payload["attn/l0_mean_entropy"] = float(np.mean(row_ents))
            payload["attn/l0_head_entropy_std"] = float(np.std(row_ents))
        if nd1s:
            payload["attn/l0_mean_near_diag_d1_model"] = float(np.mean(nd1s))
            payload["attn/l0_best_head_near_diag_d1_model"] = float(max(nd1s))
        if phys_nd1s:
            payload["attn/l0_mean_near_diag_d1_phys"] = float(np.mean(phys_nd1s))
            payload["attn/l0_best_head_near_diag_d1_phys"] = float(max(phys_nd1s))
        if top1s:
            payload["attn/l0_mean_top1_mass"] = float(np.mean(top1s))

        return payload

    def flush_summary_log(self):
        """Write the accumulated per-step summary log to disk."""
        path = self.output_root / "trajectory_summary.jsonl"
        with path.open("w") as f:
            for entry in self._summary_log:
                f.write(json.dumps(entry) + "\n")
        return path


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: extract phys_perm from CleanPermutation
# ═══════════════════════════════════════════════════════════════════════════

def phys_perm_from_clean_perm(clean_perm) -> np.ndarray:
    """Extract the physical-block permutation array from a CleanPermutation.

    Returns a (64,) int array where phys_perm[model_block] = physical_block.
    ``inv_perm_model_to_phys`` is already a block-level (num_blocks,) mapping,
    NOT token-level — do NOT subsample with ``[::BLOCK_LEN]``.
    """
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    if inv.ndim != 1 or inv.shape[0] not in (64, 65):
        raise ValueError(
            f"Expected inv_perm_model_to_phys to be block-level [64] or [65], "
            f"got {inv.shape}"
        )
    return inv.astype(np.int64)
