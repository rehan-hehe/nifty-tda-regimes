"""
Phase 3 -- Regime Detection and Labelling.

Input:  topo_features.csv  (from phase2_topology.py)
Output: regime_labels.csv       (rebal_date, umap_x, umap_y, [umap_z], regime)
        regime_validation.png   (regimes overlaid on Nifty price history + known events)
        regime_clustering_metrics.txt

    pip install umap-learn hdbscan scikit-learn --break-system-packages
    python phase3_regime_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import hdbscan

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_COMPONENTS = 2          # 2D embedding for viz; proposal allows 2D or 3D
UMAP_N_NEIGHBORS = 15     # umap default; lower = more local structure preserved
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER_SIZE = 20   # smallest group of rebalance dates to call a "regime"
RANDOM_STATE = 42

KNOWN_EVENTS = {
    "2008 GFC": "2008-09-15",
    "2013 Taper Tantrum": "2013-05-22",
    "2016 Demonetisation": "2016-11-08",
    "2020 COVID Crash": "2020-03-23",
    "2022 Russia-Ukraine": "2022-02-24",
    "2023 Adani Crisis": "2023-01-24",
}


def load_features(path: str = "topo_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "rebal_date"
    return df


def reduce_dimensionality(features: pd.DataFrame, n_components: int = N_COMPONENTS):
    """Standardize then UMAP. Scaling matters a lot here -- entropy scalars,
    Betti-curve bin counts, and landscape bin values live on very different
    numeric scales, and UMAP (like most distance-based methods) will let the
    largest-magnitude features dominate the embedding if you skip this."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
    )
    embedding = reducer.fit_transform(X_scaled)
    return embedding, scaler, reducer


def cluster_regimes(embedding: np.ndarray, min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE):
    """HDBSCAN on the UMAP embedding, per the proposal's spec (cluster the
    *embedded* features, not the raw 1216-dim space -- HDBSCAN's density
    notion degrades in high dimensions, which is exactly why the proposal
    puts UMAP before it)."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def validate_clustering(embedding: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette + Davies-Bouldin, per the proposal's evaluation criteria.
    HDBSCAN's -1 (noise) points are excluded from both -- neither metric is
    defined for a 'non-cluster' label, and including them would corrupt
    the score."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = int((labels == -1).sum())

    metrics = {"n_clusters": n_clusters, "n_noise_points": n_noise,
               "noise_fraction": n_noise / len(labels)}

    if n_clusters >= 2:
        metrics["silhouette_score"] = silhouette_score(embedding[mask], labels[mask])
        metrics["davies_bouldin_score"] = davies_bouldin_score(embedding[mask], labels[mask])
    else:
        metrics["silhouette_score"] = None
        metrics["davies_bouldin_score"] = None
        print("WARNING: fewer than 2 clusters found -- metrics undefined. "
              "Consider lowering HDBSCAN_MIN_CLUSTER_SIZE.")

    return metrics


def plot_validation(dates: pd.DatetimeIndex, embedding: np.ndarray, labels: np.ndarray,
                     out_path: str = "regime_validation.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- panel 1: regime label over time, with known events marked ---
    ax = axes[0]
    scatter = ax.scatter(dates, labels, c=labels, cmap="tab10", s=8)
    for name, d in KNOWN_EVENTS.items():
        ax.axvline(pd.Timestamp(d), color="red", linestyle="--", alpha=0.4)
        ax.text(pd.Timestamp(d), ax.get_ylim()[1], name, rotation=90,
                 fontsize=7, va="top", ha="right", color="red")
    ax.set_title("Detected regime over time (label -1 = HDBSCAN noise, unassigned)")
    ax.set_ylabel("Regime label")

    # --- panel 2: UMAP embedding, colored by regime ---
    ax2 = axes[1]
    sc = ax2.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=8)
    ax2.set_title("UMAP embedding, colored by detected regime")
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")
    plt.colorbar(sc, ax=ax2, label="regime")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main(features_path: str = "topo_features.csv"):
    print("Loading topological features...")
    features = load_features(features_path)
    print(f"  {features.shape[0]} dates x {features.shape[1]} features")

    print("\nReducing dimensionality (StandardScaler -> UMAP)...")
    embedding, scaler, reducer = reduce_dimensionality(features)
    print(f"  embedding shape: {embedding.shape}")

    print("\nClustering regimes (HDBSCAN on embedding)...")
    labels, clusterer = cluster_regimes(embedding)
    n_regimes = len(set(labels[labels != -1]))
    print(f"  found {n_regimes} regimes, "
          f"{(labels == -1).sum()} unassigned/noise dates out of {len(labels)}")

    print("\nValidating clustering quality...")
    metrics = validate_clustering(embedding, labels)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- save labels ---
    out = pd.DataFrame({
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1],
        "regime": labels,
    }, index=features.index)
    out.to_csv("regime_labels.csv")
    print("\nSaved regime_labels.csv")

    # --- save metrics ---
    with open("regime_clustering_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nRegime sizes:\n")
        f.write(pd.Series(labels).value_counts().sort_index().to_string())
    print("Saved regime_clustering_metrics.txt")

    # --- plot ---
    plot_validation(features.index, embedding, labels)

    return out, metrics


if __name__ == "__main__":
    main()
