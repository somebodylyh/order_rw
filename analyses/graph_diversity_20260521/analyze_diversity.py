"""Step 2: Analyze per-sample g(B) diversity across setups.

Produces:
  a) within-setup spread (mean / std / IQR per metric per setup)
  b) between-setup separation (L2 & Mahalanobis between centroids)
  c) ratio = within_std / between_distance
  d) PCA(2D) and UMAP(2D) scatter plots colored by setup
  e) boundary case mining (samples closer to another setup's centroid)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

G_B_COLS = [
    "readiness_strength", "asymmetry", "row_entropy", "top1_mass", "top4_mass",
    "argmax_dist", "p_nbr_le1", "locality_score", "local_greedy_dist",
    "directionality", "out_degree_std", "in_degree_std",
]


def load_all():
    dfs = []
    for tsv in sorted(OUT_DIR.glob("*_per_sample_gB.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def within_setup_stats(df):
    """(a) Per-setup, per-metric: mean, std, IQR."""
    rows = []
    for setup, grp in df.groupby("setup"):
        for col in G_B_COLS:
            vals = grp[col].values
            rows.append({
                "setup": setup,
                "metric": col,
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "q25": np.percentile(vals, 25),
                "median": np.percentile(vals, 50),
                "q75": np.percentile(vals, 75),
                "max": vals.max(),
                "iqr": np.percentile(vals, 75) - np.percentile(vals, 25),
                "cv": vals.std() / (abs(vals.mean()) + 1e-12),
            })
    return pd.DataFrame(rows)


def between_setup_distances(df):
    """(b) L2 distance between setup centroids in g(B) space."""
    setups = sorted(df["setup"].unique())
    centroids = {}
    for s in setups:
        centroids[s] = df[df["setup"] == s][G_B_COLS].mean().values

    rows = []
    for i, s1 in enumerate(setups):
        for j, s2 in enumerate(setups):
            if j <= i:
                continue
            d_l2 = np.linalg.norm(centroids[s1] - centroids[s2])
            rows.append({"setup_i": s1, "setup_j": s2, "L2_distance": d_l2})
    return pd.DataFrame(rows), centroids


def ratio_analysis(df, centroids):
    """(c) For each metric, within_std / between_centroid_distance."""
    setups = sorted(df["setup"].unique())
    # real setups only (exclude synthetic controls for ratio)
    real_setups = [s for s in setups if s not in ("random_uniform", "shuffled_rows")]

    rows = []
    for col in G_B_COLS:
        # within_std: average std across real setups
        stds = []
        for s in real_setups:
            stds.append(df[df["setup"] == s][col].std())
        avg_within_std = np.mean(stds)

        # between_distance: mean pairwise |centroid_i - centroid_j| for this metric
        dists = []
        for i, s1 in enumerate(real_setups):
            for j, s2 in enumerate(real_setups):
                if j <= i:
                    continue
                d = abs(centroids[s1][G_B_COLS.index(col)] - centroids[s2][G_B_COLS.index(col)])
                dists.append(d)
        avg_between_dist = np.mean(dists) if dists else 1e-12

        ratio = avg_within_std / (avg_between_dist + 1e-12)
        rows.append({
            "metric": col,
            "avg_within_std": avg_within_std,
            "avg_between_dist": avg_between_dist,
            "ratio_within_over_between": ratio,
            "verdict": "A (setup dominates)" if ratio < 0.3 else
                       "B (within matters)" if ratio > 1.0 else "Mixed",
        })
    return pd.DataFrame(rows)


def pca_2d(df):
    """(d) PCA on standardized g(B)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = df[G_B_COLS].values
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_std)
    df_pca = df[["setup", "sample_idx"]].copy()
    df_pca["PC1"] = X_2d[:, 0]
    df_pca["PC2"] = X_2d[:, 1]
    return df_pca, pca.explained_variance_ratio_


def umap_2d(df):
    """(d) UMAP on standardized g(B)."""
    try:
        import umap
    except ImportError:
        print("  UMAP not available (pip install umap-learn); skipping.")
        return None
    from sklearn.preprocessing import StandardScaler

    X = df[G_B_COLS].values
    X_std = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    X_2d = reducer.fit_transform(X_std)
    df_umap = df[["setup", "sample_idx"]].copy()
    df_umap["UMAP1"] = X_2d[:, 0]
    df_umap["UMAP2"] = X_2d[:, 1]
    return df_umap


def boundary_cases(df, centroids):
    """(e) For each pair (setup_i, setup_j), find samples in setup_i closer to
    centroid_j than centroid_i."""
    from sklearn.preprocessing import StandardScaler
    setups = sorted(df["setup"].unique())
    real_setups = [s for s in setups if s not in ("random_uniform", "shuffled_rows")]

    scaler = StandardScaler().fit(df[G_B_COLS].values)
    centroids_std = {s: scaler.transform([centroids[s]])[0] for s in real_setups}

    results = []
    for s_i in real_setups:
        for s_j in real_setups:
            if s_i == s_j:
                continue
            grp = df[df["setup"] == s_i]
            X_std = scaler.transform(grp[G_B_COLS].values)
            dist_own = np.linalg.norm(X_std - centroids_std[s_i], axis=1)
            dist_other = np.linalg.norm(X_std - centroids_std[s_j], axis=1)
            cross_mask = dist_other < dist_own
            n_cross = cross_mask.sum()
            frac = n_cross / len(grp)

            # Top crossers (sorted by how much closer to other)
            margin = dist_own - dist_other
            top_idx = np.argsort(margin)[::-1][:20]
            top_samples = []
            for idx in top_idx:
                if margin[idx] > 0:
                    top_samples.append({
                        "sample_id": int(grp.iloc[idx]["sample_id"]),
                        "margin": float(margin[idx]),
                    })

            results.append({
                "from_setup": s_i,
                "to_setup": s_j,
                "n_cross": int(n_cross),
                "frac_cross": float(frac),
                "top_crossers": top_samples[:20],
            })
    return results


def make_plots(df_pca, df_umap, var_ratio):
    """Generate scatter plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_colors = {
        "e3_ctrl_small": "#1f77b4",
        "text": "#ff7f0e",
        "e2_small": "#2ca02c",
        "e2_large": "#d62728",
        "random_uniform": "#7f7f7f",
        "shuffled_rows": "#bcbd22",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA
    ax = axes[0]
    for setup in df_pca["setup"].unique():
        mask = df_pca["setup"] == setup
        ax.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                   c=setup_colors.get(setup, "black"), label=setup, alpha=0.6, s=20)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title("PCA of per-sample g(B)")
    ax.legend(fontsize=8)

    # UMAP
    ax = axes[1]
    if df_umap is not None:
        for setup in df_umap["setup"].unique():
            mask = df_umap["setup"] == setup
            ax.scatter(df_umap.loc[mask, "UMAP1"], df_umap.loc[mask, "UMAP2"],
                       c=setup_colors.get(setup, "black"), label=setup, alpha=0.6, s=20)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("UMAP of per-sample g(B)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "UMAP unavailable", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    fig_path = OUT_DIR / "scatter_pca_umap.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


def make_violin_plot(df):
    """Per-metric violin plot comparing setups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_setups = ["e3_ctrl_small", "text", "e2_small", "e2_large"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, col in enumerate(G_B_COLS):
        ax = axes[idx]
        data_by_setup = [df[df["setup"] == s][col].values for s in real_setups]
        parts = ax.violinplot(data_by_setup, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(real_setups) + 1))
        ax.set_xticklabels(real_setups, rotation=45, fontsize=7)
        ax.set_title(col, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "violin_per_metric.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


def main():
    print("Loading all per-sample g(B) data...")
    df = load_all()
    print(f"  Total: {len(df)} rows, setups: {df['setup'].unique().tolist()}")

    # Merge into single TSV
    all_tsv = OUT_DIR / "all_setups_combined.tsv"
    df.to_csv(all_tsv, sep="\t", index=False)
    print(f"  Saved combined: {all_tsv}")

    # (a) Within-setup spread
    print("\n(a) Within-setup spread...")
    ws = within_setup_stats(df)
    ws.to_csv(OUT_DIR / "within_setup_stats.tsv", sep="\t", index=False, float_format="%.6f")
    print(f"  Saved: within_setup_stats.tsv")

    # (b) Between-setup distances
    print("\n(b) Between-setup distances...")
    bd, centroids = between_setup_distances(df)
    bd.to_csv(OUT_DIR / "between_setup_distances.tsv", sep="\t", index=False, float_format="%.6f")
    print(f"  Saved: between_setup_distances.tsv")

    # (c) Ratio analysis
    print("\n(c) Ratio = within_std / between_distance...")
    ra = ratio_analysis(df, centroids)
    ra.to_csv(OUT_DIR / "ratio_within_over_between.tsv", sep="\t", index=False, float_format="%.6f")
    print(ra.to_string(index=False))

    # (d) PCA + UMAP
    print("\n(d) PCA...")
    df_pca, var_ratio = pca_2d(df)
    df_pca.to_csv(OUT_DIR / "pca_2d.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"  Variance explained: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%")

    print("  UMAP...")
    df_umap = umap_2d(df)
    if df_umap is not None:
        df_umap.to_csv(OUT_DIR / "umap_2d.tsv", sep="\t", index=False, float_format="%.4f")

    # (e) Boundary cases
    print("\n(e) Boundary case mining...")
    bc = boundary_cases(df, centroids)
    with open(OUT_DIR / "boundary_cases.json", "w") as f:
        json.dump(bc, f, indent=2)
    # Print summary
    for item in bc:
        if item["n_cross"] > 0:
            print(f"  {item['from_setup']:15s} -> {item['to_setup']:15s}: "
                  f"{item['n_cross']:3d} / 300 ({item['frac_cross']*100:.1f}%) cross boundary")

    # Plots
    print("\n  Generating plots...")
    make_plots(df_pca, df_umap, var_ratio)
    make_violin_plot(df)

    print("\n  All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
