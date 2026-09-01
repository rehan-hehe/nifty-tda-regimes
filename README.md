# Topological Market Regime Detection — NSE Nifty-50

Detects market regimes in Indian equities using **Topological Data Analysis (TDA)** applied to rolling stock correlation networks, covering all 100 historical Nifty-50 constituents from **2008 to 2025**.

> **Status**: Phases 1–3 complete. Phase 4 (regime-conditional TCN forecasting + Adaptive Conformal Intervals) and Phase 5 (portfolio backtest) in progress.

---

## What This Does

| Phase | Input | Method | Output |
|---|---|---|---|
| 1 | 100 price CSVs | Rolling 120-day 50×50 correlation matrices → distance matrices | `distance_matrices.pkl` (884 snapshots) |
| 2 | Distance matrices | Vietoris-Rips persistent homology (giotto-tda) | `topo_features.csv` (884 × 1,216) |
| 3 | Topological features | PCA → UMAP → K-Means | `regime_labels_final.csv` (4 regimes) |

### The 4 Regimes

| Label | Name | Dates (%) | Key periods |
|---|---|---|---|
| 0 | **Crisis** | 15.7% | GFC 2008, COVID crash 2020, post-Russia rate shock 2022 |
| 1 | **Recovery** | 35.4% | Post-GFC recovery, COVID liquidity rally, current 2024–25 |
| 2 | **Bull** | 41.4% | Modi rally 2014–15, 2016–2019 3-year continuous bull |
| 3 | **Structural Shift** | 7.5% | Post-rate-hike 2022–23, Adani crisis, FII exodus 2024 |

Regimes exhibit **95.8% 5-day persistence** and a coherent Markov transition structure.

---

## Repository Structure

```
tda-nifty50-regime-detection/
├── data/
│   ├── prices/          # 100 NSE price CSVs (Date, Close), 2008–2025
│   └── weights.csv      # Monthly Nifty-50 constituent weights (figshare dataset)
│
├── src/
│   ├── phase1_build_matrices.py        # Builds rolling distance matrices
│   ├── phase2_topology.py              # Vietoris-Rips + topological feature extraction
│   ├── phase3_regime_detection_v1.py   # Initial regime detection (baseline)
│   └── phase3_regime_detection_v2.py   # Round 2: HDBSCAN sweep + K-Means baseline
│
├── outputs/
│   ├── topo_features.csv              # 884 × 1,216 topological feature matrix
│   ├── persistence_diagrams.pkl       # Raw H0/H1 persistence diagrams
│   ├── regime_labels_final.csv        # Final K-Means K=4 regime labels
│   └── figures/
│       ├── phase3_v1_original_4cluster.png
│       ├── phase3_v2_hdbscan_2cluster.png
│       └── phase3_v2_kmeans4_final.png
│
└── reports/
    ├── phase1_phase2_technical_report.md
    └── guide_to_use_data.md
```

> `distance_matrices.pkl` (34 MB) is not tracked in git. Regenerate with `src/phase1_build_matrices.py`.

---

## Reproducing Results

### Environment
```bash
pip install pandas numpy scikit-learn umap-learn hdbscan giotto-tda matplotlib
```

### Step-by-step
```bash
python src/phase1_build_matrices.py    # ~2 min CPU
python src/phase2_topology.py          # ~15-20 min CPU  (faster on Colab GPU)
python src/phase3_regime_detection_v2.py
```

---

## Key Results

### Regime detection metrics
- Round 1 (HDBSCAN mcs=10): 32 micro-clusters, silhouette 0.61 — over-segmented
- Round 2 sweep: HDBSCAN collapses to **2 macro density clusters** at mcs ≥ 50
- **Final: K-Means K=4** — silhouette 0.512, 0% noise, min cluster 66 windows

### Markov transition matrix
```
from\to    Crisis  Recovery   Bull  Str.Shift
Crisis      0.957    0.029   0.007     0.007
Recovery    0.013    0.949   0.038     0.000
Bull        0.003    0.030   0.964     0.003
Str.Shift   0.000    0.015   0.015     0.970
```
Overall regime persistence: **95.8%**

---

## Data Notes

- **90 tickers**: `Date, Close` format (yfinance)
- **10 tickers** (delisted/merged — HDFC, RANBAXY, CAIRN, RPL, SATYAMCOMP, STER, RELCAPITAL, IDFC, JPASSOCIAT, INFRATEL): pre-window data from NSE Gadiyar Bhavcopy archives
- See `reports/guide_to_use_data.md` for full data quirks and sourcing details

---

## Dependencies

| Package | Purpose |
|---|---|
| `giotto-tda` | Vietoris-Rips filtration, persistence diagrams |
| `umap-learn` | UMAP dimensionality reduction |
| `hdbscan` | Density-based clustering |
| `scikit-learn` | PCA, K-Means, metrics |
| `pandas`, `numpy` | Data handling |
| `matplotlib` | Validation plots |
