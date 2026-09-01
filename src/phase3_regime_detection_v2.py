"""
Phase 3 -- Regime Detection and Labelling.

Input:
    topo_features.csv
        884 dates x 1,216 topological features

Output:
    phase3_iterations/
        iteration_01/
            config.txt
            regime_labels.csv
            regime_clustering_metrics.txt
            regime_validation.png

        iteration_02/
            ...

        ...

        combined_iteration_metrics.csv
        combined_regime_labels.csv
        best_iteration.txt

The purpose of this version is to systematically test the Phase 3
configurations recommended by the analysis before proceeding to Phase 4.

Recommended experiments:
    - 3D UMAP
    - lower n_neighbors
    - lower min_dist
    - smaller HDBSCAN min_cluster_size
    - HDBSCAN min_samples
    - PCA before UMAP

    pip install umap-learn hdbscan scikit-learn --break-system-packages
    python phase3_regime_detection.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

import umap
import hdbscan


# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------

ITERATIONS_DIR = Path("phase3_iterations")

# All individual experiments are stored below this directory.
ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# PCA
USE_PCA = True
PCA_VARIANCE = 0.95

# ---------------------------------------------------------------------------
# Iterations to test
#
# The first iteration is the concrete configuration recommended by the
# Phase 3 analysis:
#
#   N_COMPONENTS = 3
#   UMAP_N_NEIGHBORS = 10
#   UMAP_MIN_DIST = 0.05
#   HDBSCAN_MIN_CLUSTER_SIZE = 10
#   min_samples = 5
#
# Additional iterations vary the parameters identified for tuning.
# ---------------------------------------------------------------------------

ITERATIONS = [

    {
        "name": "recommended_3d",
        "n_components": 3,
        "n_neighbors": 10,
        "min_dist": 0.05,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

    {
        "name": "neighbors_15",
        "n_components": 3,
        "n_neighbors": 15,
        "min_dist": 0.05,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

    {
        "name": "neighbors_20",
        "n_components": 3,
        "n_neighbors": 20,
        "min_dist": 0.05,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

    {
        "name": "neighbors_30",
        "n_components": 3,
        "n_neighbors": 30,
        "min_dist": 0.05,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

    {
        "name": "min_dist_0",
        "n_components": 3,
        "n_neighbors": 10,
        "min_dist": 0.0,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

    {
        "name": "min_dist_01",
        "n_components": 3,
        "n_neighbors": 10,
        "min_dist": 0.1,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

    {
        "name": "cluster_size_15",
        "n_components": 3,
        "n_neighbors": 10,
        "min_dist": 0.05,
        "min_cluster_size": 15,
        "min_samples": 5,
    },

    {
        "name": "cluster_size_20",
        "n_components": 3,
        "n_neighbors": 10,
        "min_dist": 0.05,
        "min_cluster_size": 20,
        "min_samples": 5,
    },

    {
        "name": "no_pca",
        "n_components": 3,
        "n_neighbors": 10,
        "min_dist": 0.05,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

    {
        "name": "2d_comparison",
        "n_components": 2,
        "n_neighbors": 10,
        "min_dist": 0.05,
        "min_cluster_size": 10,
        "min_samples": 5,
    },

]


KNOWN_EVENTS = {
    "2008 GFC": "2008-09-15",
    "2013 Taper Tantrum": "2013-05-22",
    "2016 Demonetisation": "2016-11-08",
    "2020 COVID Crash": "2020-03-23",
    "2022 Russia-Ukraine": "2022-02-24",
    "2023 Adani Crisis": "2023-01-24",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_features(path: str = "topo_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "rebal_date"
    return df


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensionality(
    features: pd.DataFrame,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    use_pca: bool = USE_PCA,
):
    """
    StandardScaler -> optional PCA -> UMAP.

    PCA is applied after standardisation and before UMAP when enabled.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    pca = None

    if use_pca:
        pca = PCA(
            n_components=PCA_VARIANCE,
            random_state=RANDOM_STATE,
        )

        X_reduced = pca.fit_transform(X_scaled)

        print(
            f"  PCA: {X_scaled.shape[1]} -> "
            f"{X_reduced.shape[1]} components "
            f"({PCA_VARIANCE:.0%} variance)"
        )

    else:
        X_reduced = X_scaled
        print("  PCA: disabled")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=RANDOM_STATE,
    )

    embedding = reducer.fit_transform(X_reduced)

    return embedding, scaler, pca, reducer


# ---------------------------------------------------------------------------
# HDBSCAN clustering
# ---------------------------------------------------------------------------

def cluster_regimes(
    embedding: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
):
    """
    HDBSCAN on the UMAP embedding.
    """

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )

    labels = clusterer.fit_predict(embedding)

    return labels, clusterer


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_clustering(
    embedding: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Calculate clustering metrics.

    HDBSCAN label -1 represents noise/unassigned observations and is
    excluded from silhouette and Davies-Bouldin calculations.
    """

    mask = labels != -1

    n_clusters = len(set(labels[mask]))
    n_noise = int((labels == -1).sum())

    metrics = {
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "noise_fraction": n_noise / len(labels),
    }

    if n_clusters >= 2:

        metrics["silhouette_score"] = silhouette_score(
            embedding[mask],
            labels[mask],
        )

        metrics["davies_bouldin_score"] = davies_bouldin_score(
            embedding[mask],
            labels[mask],
        )

    else:

        metrics["silhouette_score"] = None
        metrics["davies_bouldin_score"] = None

        print(
            "WARNING: fewer than 2 clusters found -- "
            "clustering metrics undefined."
        )

    return metrics


# ---------------------------------------------------------------------------
# Plot validation
# ---------------------------------------------------------------------------

def plot_validation(
    dates: pd.DatetimeIndex,
    embedding: np.ndarray,
    labels: np.ndarray,
    out_path: str,
):
    """
    Validation plot.

    Panel 1:
        Regime labels over time.

    Panel 2:
        UMAP embedding.

    For 3D UMAP, only the first two coordinates are plotted because
    the validation figure remains a 2D visualization.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
    )

    # -----------------------------------------------------------------------
    # Panel 1: regimes over time
    # -----------------------------------------------------------------------

    ax = axes[0]

    scatter = ax.scatter(
        dates,
        labels,
        c=labels,
        cmap="tab10",
        s=8,
    )

    for name, d in KNOWN_EVENTS.items():

        ax.axvline(
            pd.Timestamp(d),
            color="red",
            linestyle="--",
            alpha=0.4,
        )

        ax.text(
            pd.Timestamp(d),
            ax.get_ylim()[1],
            name,
            rotation=90,
            fontsize=7,
            va="top",
            ha="right",
            color="red",
        )

    ax.set_title(
        "Detected regime over time "
        "(label -1 = HDBSCAN noise, unassigned)"
    )

    ax.set_ylabel("Regime label")

    # -----------------------------------------------------------------------
    # Panel 2: UMAP embedding
    # -----------------------------------------------------------------------

    ax2 = axes[1]

    sc = ax2.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
    )

    ax2.set_title(
        "UMAP embedding, colored by detected regime"
    )

    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")

    plt.colorbar(
        sc,
        ax=ax2,
        label="regime",
    )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Save iteration configuration
# ---------------------------------------------------------------------------

def save_config(
    iteration_dir: Path,
    config: dict,
    features: pd.DataFrame,
):
    """
    Save all parameters associated with one experiment.
    """

    with open(iteration_dir / "config.txt", "w") as f:

        f.write("PHASE 3 ITERATION CONFIGURATION\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Input observations : {features.shape[0]}\n")
        f.write(f"Input features     : {features.shape[1]}\n\n")

        f.write(f"PCA enabled        : {USE_PCA}\n")

        if USE_PCA:
            f.write(f"PCA variance       : {PCA_VARIANCE}\n")

        f.write("\nUMAP / HDBSCAN parameters\n")
        f.write("-" * 60 + "\n")

        for key, value in config.items():
            f.write(f"{key}: {value}\n")


# ---------------------------------------------------------------------------
# Run one iteration
# ---------------------------------------------------------------------------

def run_iteration(
    iteration_number: int,
    config: dict,
    features: pd.DataFrame,
):
    """
    Execute and save one complete Phase 3 experiment.
    """

    iteration_name = (
        f"iteration_{iteration_number:02d}_{config['name']}"
    )

    iteration_dir = ITERATIONS_DIR / iteration_name
    iteration_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print(f"ITERATION {iteration_number}: {config['name']}")
    print("=" * 70)

    save_config(
        iteration_dir,
        config,
        features,
    )

    # -----------------------------------------------------------------------
    # UMAP
    # -----------------------------------------------------------------------

    print("\nReducing dimensionality...")

    embedding, scaler, pca, reducer = reduce_dimensionality(
        features,
        n_components=config["n_components"],
        n_neighbors=config["n_neighbors"],
        min_dist=config["min_dist"],
        use_pca=(
            False
            if config["name"] == "no_pca"
            else USE_PCA
        ),
    )

    print(f"  embedding shape: {embedding.shape}")

    # -----------------------------------------------------------------------
    # HDBSCAN
    # -----------------------------------------------------------------------

    print("\nClustering regimes...")

    labels, clusterer = cluster_regimes(
        embedding,
        min_cluster_size=config["min_cluster_size"],
        min_samples=config["min_samples"],
    )

    n_regimes = len(
        set(labels[labels != -1])
    )

    n_noise = int(
        (labels == -1).sum()
    )

    print(
        f"  found {n_regimes} regimes, "
        f"{n_noise} unassigned/noise dates "
        f"out of {len(labels)}"
    )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    print("\nValidating clustering quality...")

    metrics = validate_clustering(
        embedding,
        labels,
    )

    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # -----------------------------------------------------------------------
    # Save regime labels
    # -----------------------------------------------------------------------

    out = pd.DataFrame(
        {
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
            "regime": labels,
        },
        index=features.index,
    )

    # Save UMAP-3 if this is a 3D embedding.
    if embedding.shape[1] >= 3:
        out["umap_z"] = embedding[:, 2]

    out.index.name = "rebal_date"

    labels_path = iteration_dir / "regime_labels.csv"

    out.to_csv(labels_path)

    print(f"Saved {labels_path}")

    # -----------------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------------

    metrics["iteration"] = iteration_number
    metrics["iteration_name"] = config["name"]

    metrics["n_components"] = config["n_components"]
    metrics["n_neighbors"] = config["n_neighbors"]
    metrics["min_dist"] = config["min_dist"]
    metrics["min_cluster_size"] = config["min_cluster_size"]
    metrics["min_samples"] = config["min_samples"]

    iteration_metrics_path = (
        iteration_dir /
        "regime_clustering_metrics.txt"
    )

    with open(iteration_metrics_path, "w") as f:

        f.write("PHASE 3 CLUSTERING METRICS\n")
        f.write("=" * 60 + "\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nRegime sizes:\n")

        regime_sizes = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        f.write(
            regime_sizes.to_string()
        )

    print(
        f"Saved {iteration_metrics_path}"
    )

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    plot_path = (
        iteration_dir /
        "regime_validation.png"
    )

    plot_validation(
        features.index,
        embedding,
        labels,
        str(plot_path),
    )

    return {
        "metrics": metrics,
        "labels": out,
        "iteration_name": iteration_name,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    features_path: str = "topo_features.csv",
):
    print("=" * 70)
    print("PHASE 3 -- REGIME DETECTION")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load features
    # -----------------------------------------------------------------------

    print("\nLoading topological features...")

    features = load_features(
        features_path
    )

    print(
        f"  {features.shape[0]} dates x "
        f"{features.shape[1]} features"
    )

    # -----------------------------------------------------------------------
    # Run all iterations
    # -----------------------------------------------------------------------

    all_metrics = []
    all_labels = []

    for i, config in enumerate(
        ITERATIONS,
        start=1,
    ):

        result = run_iteration(
            i,
            config,
            features,
        )

        all_metrics.append(
            result["metrics"]
        )

        # Rename regime column so that results from different iterations
        # can coexist in one combined table.
        labels = result["labels"].copy()

        labels = labels.rename(
            columns={
                "umap_x":
                    f"iter_{i:02d}_umap_x",

                "umap_y":
                    f"iter_{i:02d}_umap_y",

                "umap_z":
                    f"iter_{i:02d}_umap_z",

                "regime":
                    f"iter_{i:02d}_regime",
            }
        )

        all_labels.append(
            labels
        )

    # -----------------------------------------------------------------------
    # Combine metrics from every iteration
    # -----------------------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_metrics
    )

    # Put important comparison columns first.
    preferred_columns = [
        "iteration",
        "iteration_name",
        "n_components",
        "n_neighbors",
        "min_dist",
        "min_cluster_size",
        "min_samples",
        "n_clusters",
        "n_noise_points",
        "noise_fraction",
        "silhouette_score",
        "davies_bouldin_score",
    ]

    metrics_df = metrics_df[
        [
            c
            for c in preferred_columns
            if c in metrics_df.columns
        ]
    ]

    metrics_path = (
        ITERATIONS_DIR /
        "combined_iteration_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"\nSaved combined metrics: "
        f"{metrics_path}"
    )

    # -----------------------------------------------------------------------
    # Combine regime labels from every iteration
    # -----------------------------------------------------------------------

    combined_labels = pd.concat(
        all_labels,
        axis=1,
    )

    combined_labels.index.name = "rebal_date"

    labels_path = (
        ITERATIONS_DIR /
        "combined_regime_labels.csv"
    )

    combined_labels.to_csv(
        labels_path
    )

    print(
        f"Saved combined regime labels: "
        f"{labels_path}"
    )

    # -----------------------------------------------------------------------
    # Identify best iteration by silhouette score
    # -----------------------------------------------------------------------

    valid_metrics = metrics_df.dropna(
        subset=["silhouette_score"]
    )

    if len(valid_metrics) > 0:

        best_row = valid_metrics.loc[
            valid_metrics["silhouette_score"].idxmax()
        ]

        best_iteration = int(
            best_row["iteration"]
        )

        best_name = best_row[
            "iteration_name"
        ]

        best_silhouette = best_row[
            "silhouette_score"
        ]

        best_path = (
            ITERATIONS_DIR /
            "best_iteration.txt"
        )

        with open(best_path, "w") as f:

            f.write(
                "BEST PHASE 3 ITERATION\n"
            )
            f.write(
                "=" * 60 + "\n\n"
            )

            f.write(
                f"Iteration       : "
                f"{best_iteration}\n"
            )

            f.write(
                f"Name            : "
                f"{best_name}\n"
            )

            f.write(
                f"Silhouette      : "
                f"{best_silhouette}\n"
            )

            f.write(
                f"Clusters        : "
                f"{best_row['n_clusters']}\n"
            )

            f.write(
                f"Noise points    : "
                f"{best_row['n_noise_points']}\n"
            )

            f.write(
                f"Noise fraction  : "
                f"{best_row['noise_fraction']}\n"
            )

            f.write(
                f"Davies-Bouldin  : "
                f"{best_row['davies_bouldin_score']}\n"
            )

        print(
            "\nBest iteration according to "
            f"silhouette score: "
            f"{best_iteration} ({best_name})"
        )

        print(
            f"  silhouette = "
            f"{best_silhouette:.4f}"
        )

        print(
            f"Saved {best_path}"
        )

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ITERATION COMPARISON")
    print("=" * 70)

    print(
        metrics_df[
            [
                "iteration",
                "iteration_name",
                "n_clusters",
                "n_noise_points",
                "noise_fraction",
                "silhouette_score",
                "davies_bouldin_score",
            ]
        ].to_string(index=False)
    )

    print("\nPhase 3 iteration analysis complete.")
    print(
        f"All results stored in: "
        f"{ITERATIONS_DIR}/"
    )

    return metrics_df, combined_labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()