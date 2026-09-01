"""
Phase 2 — Topological Feature Extraction via Persistent Homology.

Input:  distance_matrices.pkl  (from Phase 1's build_phase1_pipeline.py)
        { rebal_date: {"correlation": DataFrame, "distance": DataFrame, "n_stocks": int} }

Output: topo_features.csv      (rows = rebalance dates, columns = topological features)
        persistence_diagrams.pkl  (raw diagrams, kept for Phase 3 / debugging / plots)

    pip install giotto-tda --break-system-packages
    python phase2_topology.py
"""

import pickle
import numpy as np
import pandas as pd

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, BettiCurve, PersistenceLandscape, Amplitude

# ---------------------------------------------------------------------------
# Config — matches the proposal's spec (Section on Phase 2)
# ---------------------------------------------------------------------------
HOMOLOGY_DIMS = [0, 1]          # H0 (components), H1 (loops)
EPS_MAX = np.sqrt(2)            # distance metric d_ij = sqrt(2(1-rho)) maxes out here;
                                 # capping the filtration there avoids wasted computation
                                 # on distances that can never occur
N_LANDSCAPE_LAYERS = 5          # K = 5, per proposal
N_BETTI_BINS = 100              # resolution of the Betti curve discretisation
LP_NORMS = [1, 2]               # L1 and L2 norms of persistence landscapes, per proposal


def load_distance_stack(pkl_path: str):
    """Load Phase 1's distance matrices and stack into the (n_dates, n, n)
    array giotto-tda expects. Requires every date to have the same number
    of stocks (should always be 50 -- Phase 1's verify_50x50_guarantee.py
    is the check for this)."""
    with open(pkl_path, "rb") as f:
        matrices = pickle.load(f)

    dates = sorted(matrices.keys())
    sizes = {matrices[d]["n_stocks"] for d in dates}
    if len(sizes) != 1:
        raise ValueError(
            f"Inconsistent matrix sizes across dates: {sizes}. "
            "Every rebalance date must have exactly 50 stocks -- re-run "
            "Phase 1's verify_50x50_guarantee.py before proceeding."
        )

    D_stack = np.stack([matrices[d]["distance"].values for d in dates])
    tickers_per_date = [list(matrices[d]["distance"].columns) for d in dates]
    return dates, D_stack, tickers_per_date


def compute_persistence_diagrams(D_stack: np.ndarray):
    """Vietoris-Rips persistence on precomputed distance matrices."""
    VR = VietorisRipsPersistence(
        metric="precomputed",
        homology_dimensions=HOMOLOGY_DIMS,
        max_edge_length=EPS_MAX,
        n_jobs=-1,
    )
    diagrams = VR.fit_transform(D_stack)  # shape (n_dates, n_features, 3): (birth, death, dim)
    return diagrams


def vectorize_diagrams(diagrams: np.ndarray) -> pd.DataFrame:
    """Turn raw persistence diagrams into the feature set the proposal specifies:
    persistence landscapes (K=5), persistent entropy, Lp-norms, Betti curves."""
    n_dates = diagrams.shape[0]
    feature_blocks = {}

    # --- Persistent entropy: one scalar per homology dimension ---
    entropy = PersistenceEntropy().fit_transform(diagrams)  # (n_dates, len(HOMOLOGY_DIMS))
    for i, dim in enumerate(HOMOLOGY_DIMS):
        feature_blocks[f"entropy_H{dim}"] = entropy[:, i]

    # --- Betti curves: rank(H_k) as a function of epsilon, discretised ---
    betti = BettiCurve(n_bins=N_BETTI_BINS).fit_transform(diagrams)
    # betti shape: (n_dates, len(HOMOLOGY_DIMS), n_bins)
    for i, dim in enumerate(HOMOLOGY_DIMS):
        for b in range(betti.shape[2]):
            feature_blocks[f"betti_H{dim}_bin{b}"] = betti[:, i, b]

    # --- Persistence landscapes: K=5 layers, discretised ---
    # Output shape is (n_dates, n_homology_dims * n_layers, n_bins): landscapes
    # from different homology dimensions are stacked, dimension-major then
    # layer -- i.e. layer k of homology dim j lives at row index j*n_layers+k
    # (confirmed against this giotto-tda version's docstring, not assumed).
    landscape = PersistenceLandscape(n_layers=N_LANDSCAPE_LAYERS, n_bins=N_BETTI_BINS).fit_transform(diagrams)
    for i, dim in enumerate(HOMOLOGY_DIMS):
        for layer in range(N_LANDSCAPE_LAYERS):
            row_idx = i * N_LANDSCAPE_LAYERS + layer
            curve = landscape[:, row_idx, :]
            # keep both a cheap scalar summary (mean) and the full discretised
            # curve -- the full curve is what Phase 3's clustering should use,
            # the scalar is handy for quick exploratory plots
            feature_blocks[f"landscape_H{dim}_layer{layer}_mean"] = curve.mean(axis=1)
            for b in range(curve.shape[1]):
                feature_blocks[f"landscape_H{dim}_layer{layer}_bin{b}"] = curve[:, b]

    # --- Lp-norms of persistence landscapes (p=1,2), per Gidea & Katz [3] ---
    # order=None returns one amplitude value per homology dimension directly,
    # rather than a single value aggregated across dimensions.
    for p in LP_NORMS:
        amp = Amplitude(
            metric="landscape",
            metric_params={"p": p, "n_layers": N_LANDSCAPE_LAYERS},
            order=None,
            n_jobs=-1,
        ).fit_transform(diagrams)  # shape (n_dates, len(HOMOLOGY_DIMS))
        for i, dim in enumerate(HOMOLOGY_DIMS):
            feature_blocks[f"L{p}norm_H{dim}"] = amp[:, i]

    return pd.DataFrame(feature_blocks)


def extract_topo_features(pkl_path: str = "distance_matrices.pkl"):
    print("Loading distance matrices...")
    dates, D_stack, tickers_per_date = load_distance_stack(pkl_path)
    print(f"  {len(dates)} rebalance dates, {D_stack.shape[1]}x{D_stack.shape[2]} matrices")

    print("Computing Vietoris-Rips persistence (this is the slow step)...")
    diagrams = compute_persistence_diagrams(D_stack)
    print(f"  diagrams shape: {diagrams.shape}")

    print("Vectorizing diagrams into feature set...")
    features = vectorize_diagrams(diagrams)
    features.index = dates
    features.index.name = "rebal_date"

    features.to_csv("topo_features.csv")
    with open("persistence_diagrams.pkl", "wb") as f:
        pickle.dump({"dates": dates, "diagrams": diagrams}, f)

    print(f"\nDone. topo_features.csv: {features.shape[0]} dates x {features.shape[1]} features")
    print(f"Column groups: entropy (2), betti (2 x {N_BETTI_BINS}), "
          f"landscape (2 x {N_LANDSCAPE_LAYERS} x [{N_BETTI_BINS} bins + 1 mean]), "
          f"Lp-norms (2 dims x {len(LP_NORMS)} p-values)")
    return features


if __name__ == "__main__":
    extract_topo_features()
