# Phase 1 and Phase 2 Technical Report  
## Construction of Financial Distance Networks and Topological Feature Extraction

---

## 1. Overview

The objective of the first two phases of the project is to transform historical stock-price data into a time series of topological descriptors representing the evolving structure of the stock market.

The complete transformation performed so far is:

\[
\boxed{
\text{Historical Prices}
\rightarrow
\text{Log Returns}
\rightarrow
\text{Correlation Matrices}
\rightarrow
\text{Distance Matrices}
\rightarrow
\text{Vietoris--Rips Complexes}
\rightarrow
\text{Persistence Diagrams}
\rightarrow
\text{Topological Features}
}
\]

The project currently contains:

- **100 stock price datasets**
- **216 constituent-weight snapshots**
- **884 rebalance dates**
- **50 stocks per rebalance date**
- **884 × 50 × 50 distance matrices**
- **884 persistence-diagram observations**
- **884 × 1216 final topological feature matrix**

The two completed phases are:

### Phase 1 — Financial Network Construction

Historical prices are converted into rolling return correlations and subsequently into metric distance matrices.

### Phase 2 — Topological Feature Extraction

The distance matrices are treated as metric spaces and analyzed using Vietoris–Rips persistent homology. The resulting persistence diagrams are converted into fixed-dimensional numerical features.

---

# 2. Project Data Structure

The project uses the following directory structure:

```text
PROJECT/
│
├── build_distance_matrices.py
├── phase2_topology.py
├── weights.csv
│
├── data/
│   ├── RELIANCE.csv
│   ├── TCS.csv
│   ├── INFY.csv
│   ├── HDFCBANK.csv
│   ├── ...
│   └── <100 ticker CSV files>
│
├── distance_matrices.pkl
├── persistence_diagrams.pkl
└── topo_features.csv
```

The files are divided into three categories.

### Input files

```text
weights.csv
data/*.csv
```

### Phase 1 output

```text
distance_matrices.pkl
```

### Phase 2 outputs

```text
persistence_diagrams.pkl
topo_features.csv
```

---

# 3. Input Price Data

The `data/` directory contains one CSV file per stock.

For example:

```text
data/
├── RELIANCE.csv
├── TCS.csv
├── INFY.csv
├── TATAMOTORS.csv
└── ...
```

The code supports two price-data formats.

### Format 1 — Simple price format

```csv
Date,Close
2008-01-01,123.45
2008-01-02,125.10
2008-01-03,124.80
```

### Format 2 — Bhavcopy-style format

```csv
date,open,high,low,close,volume,source
2008-01-01,120,126,119,123.45,123456,NSE
2008-01-02,124,127,123,125.10,145678,NSE
```

Only the date and closing-price columns are required by Phase 1.

The loader normalizes column names to lowercase and searches for columns containing:

```python
"date"
"close"
```

This allows both formats to be processed.

---

# 4. Constituent Data — `weights.csv`

The second major input is:

```text
weights.csv
```

The current dataset contains:

```text
216 snapshots
100 ticker columns
```

with snapshots ranging from:

```text
2008-01-31
to
2025-08-31
```

The weights table determines which stocks belong to the relevant index at each point in time.

The structure is conceptually:

```text
date,RELIANCE,TCS,INFY,HDFCBANK,...
2008-01-31,...
2008-02-29,...
...
```

The first column is interpreted as the date:

```python
pd.read_csv(
    path,
    index_col=0,
    parse_dates=True
)
```

For a given rebalance date \(t\), the code selects the most recent constituent snapshot satisfying:

\[
\text{snapshot date}\leq t
\]

Then all stocks whose weight is positive are treated as constituents.

---

# 5. Phase 1 — Financial Network Construction

## 5.1 Objective

The purpose of Phase 1 is to construct a financial network for every rebalance date.

Each network contains:

- 50 stocks as nodes
- pairwise return correlations
- pairwise correlation-derived distances

The resulting distance matrix becomes the input to Phase 2.

---

# 6. Configuration Used in Phase 1

The important parameters are:

```python
WINDOW = 120
STEP = 5
REFERENCE_TICKER = "RELIANCE"
```

Therefore:

### Rolling window

\[
W=120
\]

meaning that each network is constructed using the previous **120 trading days** of returns.

### Rebalance frequency

\[
S=5
\]

meaning that rebalance dates are sampled every five trading days from the reference trading calendar.

### Reference ticker

```text
RELIANCE
```

is used to construct the global trading-date calendar.

---

# 7. Rebalance Calendar

The code obtains the complete date index of:

```text
RELIANCE.csv
```

and restricts it to:

```text
2008-01-31
through
2025-12-31
```

The dates are then sampled every five trading days:

```python
return study_days[::step]
```

This resulted in:

\[
\boxed{884\text{ rebalance dates}}
\]

with the reported range:

```text
2008-01-31 → 2025-12-31
```

Thus, Phase 1 ultimately attempts to construct a 50-stock network at each of 884 dates.

---

# 8. Historical Return Construction

For every constituent stock and every rebalance date, only historical information strictly before the rebalance date is used.

The relevant code is:

```python
pre = s[s.index < rebal_date].tail(window + 1)
```

The strict inequality:

```python
s.index < rebal_date
```

is important because it prevents the current rebalance-date price from entering the estimation window.

This avoids look-ahead bias.

The code then calculates logarithmic returns:

```python
log_ret = np.log(pre / pre.shift(1)).dropna()
```

Mathematically:

\[
\boxed{
r_t=\log\left(\frac{P_t}{P_{t-1}}\right)
}
\]

where:

- \(P_t\) = closing price at time \(t\)
- \(r_t\) = logarithmic return

For each stock, the resulting return vector contains 120 observations.

Therefore, at each rebalance date the underlying data matrix has the conceptual dimension:

\[
\boxed{120\times50}
\]

where:

- 120 = trading-day observations
- 50 = stocks

---

# 9. Complete-Case Return Matrix

The individual return series are combined:

```python
ret_df = pd.DataFrame(returns).dropna()
```

`dropna()` ensures that the correlation calculation uses dates for which all included stocks have valid returns.

The project verified that all resulting networks contain exactly 50 stocks.

The final Phase 1 output was:

```text
Matrix size distribution:
min = 50
max = 50
dates with exactly 50 = 884/884
```

Therefore:

\[
\boxed{100\% \text{ of the 884 matrices contain exactly 50 stocks}}
\]

This is an important data-quality check.

---

# 10. Correlation Matrix

For every rebalance date, Pearson correlations are calculated:

```python
corr = ret_df.corr(method="pearson")
```

For stocks \(i\) and \(j\):

\[
\rho_{ij}
=
\operatorname{corr}(r_i,r_j)
\]

This produces:

\[
\boxed{50\times50}
\]

correlation matrix.

The diagonal is:

\[
\rho_{ii}=1
\]

and the matrix is symmetric:

\[
\rho_{ij}=\rho_{ji}
\]

---

# 11. Correlation-to-Distance Transformation

The correlation matrix is transformed into a distance matrix using:

\[
\boxed{
d_{ij}=\sqrt{2(1-\rho_{ij})}
}
\]

This is a standard transformation used to convert correlations into Euclidean-style distances.

Its interpretation is:

### Perfect positive correlation

If:

\[
\rho_{ij}=1
\]

then:

\[
d_{ij}=0
\]

The two stocks behave identically according to the measured return correlation.

### Zero correlation

If:

\[
\rho_{ij}=0
\]

then:

\[
d_{ij}=\sqrt{2}
\]

### Perfect negative correlation

If:

\[
\rho_{ij}=-1
\]

then:

\[
d_{ij}=2
\]

Thus:

\[
0\leq d_{ij}\leq2
\]

The resulting matrix is again:

\[
\boxed{50\times50}
\]

---

# 12. Phase 1 Output — `distance_matrices.pkl`

The complete Phase 1 result is stored in:

```text
distance_matrices.pkl
```

The object has the conceptual structure:

```python
{
    rebal_date_1: {
        "correlation": DataFrame(50, 50),
        "distance": DataFrame(50, 50),
        "n_stocks": 50
    },

    rebal_date_2: {
        ...
    },

    ...
}
```

There are:

\[
\boxed{884}
\]

date entries.

Thus Phase 1 produces:

\[
884\times50\times50
\]

distance values before considering symmetry and the diagonal.

---

# 13. Phase 1 Data Flow

The complete Phase 1 transformation is:

```text
Historical prices
      │
      ▼
Select 50 constituents
      │
      ▼
Previous 120 trading days
      │
      ▼
Log returns
      │
      ▼
120 × 50 return matrix
      │
      ▼
Pearson correlation
      │
      ▼
50 × 50 correlation matrix
      │
      ▼
dij = sqrt(2(1-rhoij))
      │
      ▼
50 × 50 distance matrix
      │
      ▼
distance_matrices.pkl
```

---

# 14. Data-Quality Issue Encountered During Phase 1

During implementation, a duplicate-date problem was detected in:

```text
data/TATAMOTORS.csv
```

Initially, this caused:

```text
ValueError:
cannot reindex on an axis with duplicate labels
```

because Pandas could not safely align multiple return series containing duplicate dates.

The file contained multiple observations for the same trading dates, with values corresponding to different price records.

The problematic records were investigated and the duplicate records were resolved so that the final price series had one observation per date.

After correction, Phase 1 successfully completed.

The final validation showed:

```text
Loaded 100 tickers
884 rebalance dates
min matrix size = 50
max matrix size = 50
884/884 matrices have exactly 50 stocks
```

---

# 15. Phase 2 — Persistent Homology

Phase 2 takes:

```text
distance_matrices.pkl
```

as its input.

The objective is no longer simply to measure pairwise similarity.

Instead, each 50-stock distance matrix is treated as a metric space and its **topological structure** is analyzed.

The implementation uses:

```python
GUDHI
```

with:

```python
from gudhi.representations import Landscape, BettiCurve, Entropy
```

GUDHI provides the computational machinery for constructing Vietoris–Rips complexes and persistence diagrams.

---

# 16. Phase 2 Configuration

The main parameters are:

```python
HOMOLOGY_DIMS = [0, 1]

EPS_MAX = sqrt(2)

N_LANDSCAPE_LAYERS = 5

RESOLUTION = 100

LP_NORMS = [1, 2]

SAMPLE_RANGE = [0, sqrt(2)]
```

These parameters define the topological information retained from each network.

---

# 17. Vietoris–Rips Complex

For each date, the 50 × 50 distance matrix is supplied to:

```python
gudhi.RipsComplex(
    distance_matrix=dist_matrix,
    max_edge_length=EPS_MAX
)
```

A Vietoris–Rips filtration examines the network as the distance threshold \(\epsilon\) increases.

At small \(\epsilon\):

- only very close stocks are connected.

As \(\epsilon\) increases:

- more edges appear,
- disconnected components merge,
- cycles appear and disappear,
- higher-dimensional simplices are formed.

Conceptually:

```text
epsilon increases
       │
       ▼

isolated stocks
       ↓
clusters
       ↓
connected network
       ↓
cycles / holes
       ↓
eventual topological simplification
```

---

# 18. Homology Dimensions

Only two homology dimensions are retained:

```python
HOMOLOGY_DIMS = [0, 1]
```

## H0 — Connected Components

H0 measures the number of connected components.

At the beginning of the filtration, the 50 stocks may be completely disconnected:

\[
\beta_0\approx50
\]

As \(\epsilon\) increases, components merge.

Therefore H0 captures information about:

- clustering
- connectivity
- separation between groups of stocks

---

## H1 — Loops

H1 captures one-dimensional holes or loops.

These represent nontrivial cycles in the evolving Vietoris–Rips complex.

H1 therefore provides information that cannot be represented simply by looking at the number of connected components.

---

# 19. Persistence Diagrams

Persistent homology records each topological feature using its:

\[
(b,d)
\]

pair, where:

- \(b\) = birth scale
- \(d\) = death scale

The persistence of a feature is:

\[
\boxed{
p=d-b
}
\]

A feature with large \(d-b\) persists over a wider range of distance scales.

A persistence diagram therefore summarizes the entire evolution of the topology rather than selecting one arbitrary threshold.

---

# 20. Infinite H0 Features

In persistent homology, the final connected component may never disappear within the filtration.

This results in:

\[
d=\infty
\]

The implementation converts such infinite death times to:

```python
EPS_MAX
```

so that the downstream numerical representations can operate on finite values.

The code therefore applies:

```python
death = EPS_MAX if np.isinf(death) else death
```

---

# 21. Persistence Diagram Storage

The raw diagrams are stored in:

```text
persistence_diagrams.pkl
```

The structure is:

```python
{
    "dates": [...],

    "diagrams": {
        0: [
            diagram_for_date_1,
            diagram_for_date_2,
            ...
        ],

        1: [
            diagram_for_date_1,
            diagram_for_date_2,
            ...
        ]
    }
}
```

The number of persistence points can vary from date to date.

This is why the diagrams are stored as lists of variable-length arrays rather than as a fixed rectangular matrix.

---

# 22. Why Vectorization Is Necessary

A persistence diagram is not a fixed-dimensional vector.

For example:

```text
Date 1 → 37 persistence points
Date 2 → 44 persistence points
Date 3 → 29 persistence points
```

Machine-learning algorithms generally require a fixed number of features.

Therefore, Phase 2 converts every persistence diagram into a fixed-dimensional feature vector.

Four major representations are used:

1. Persistent entropy
2. Betti curves
3. Persistence landscapes
4. Landscape L1/L2 norms

---

# 23. Persistent Entropy

For each homology dimension, persistence values are used to calculate an entropy-like measure.

For persistence values:

\[
p_1,p_2,\ldots,p_n
\]

normalized weights are:

\[
q_i=
\frac{p_i}{\sum_j p_j}
\]

and persistent entropy is conceptually:

\[
H=-\sum_iq_i\log q_i
\]

The implementation generates one scalar per date per homology dimension.

Therefore:

```text
entropy_H0
entropy_H1
```

give:

\[
\boxed{2\text{ features}}
\]

---

# 24. Betti Curves

The Betti number is a function of the filtration scale:

\[
\beta_k(\epsilon)
\]

For H0:

\[
\beta_0(\epsilon)
=
\text{number of connected components}
\]

For H1:

\[
\beta_1(\epsilon)
=
\text{number of loops}
\]

The curves are discretized using:

```python
RESOLUTION = 100
```

over:

\[
[0,\sqrt2]
\]

Therefore:

### H0

\[
100\text{ features}
\]

### H1

\[
100\text{ features}
\]

Total:

\[
\boxed{200\text{ Betti features}}
\]

---

# 25. Persistence Landscapes

A persistence landscape converts the persistence diagram into a collection of functions.

The project retains:

```python
N_LANDSCAPE_LAYERS = 5
```

layers for each homology dimension.

Each landscape is discretized at:

```python
RESOLUTION = 100
```

points.

Thus, for one homology dimension:

\[
5\times100=500
\]

landscape values are retained.

For both H0 and H1:

\[
2\times5\times100
=
1000
\]

landscape-bin features.

---

# 26. Landscape Mean Features

In addition to the complete discretized landscapes, the code calculates the mean value of each landscape layer.

For each layer:

```python
curve.mean(axis=1)
```

There are:

\[
2\text{ homology dimensions}
\times
5\text{ layers}
=
10
\]

mean features.

Therefore landscape representations contribute:

\[
1000+10
=
\boxed{1010}
\]

features.

---

# 27. L1 and L2 Landscape Norms

The project also calculates numerical norms of the persistence landscapes.

The configured norms are:

```python
LP_NORMS = [1, 2]
```

giving:

\[
\|\lambda\|_1
\]

and:

\[
\|\lambda\|_2
\]

for each homology dimension.

This produces:

```text
L1norm_H0
L2norm_H0
L1norm_H1
L2norm_H1
```

and therefore:

\[
\boxed{4\text{ features}}
\]

---

# 28. Final Feature Dimension

The complete feature count is:

### Persistent entropy

\[
2
\]

### Betti curves

\[
2\times100=200
\]

### Persistence landscapes

\[
2\times5\times100=1000
\]

### Landscape means

\[
2\times5=10
\]

### L1/L2 norms

\[
2\times2=4
\]

Therefore:

\[
2+200+1000+10+4
=
\boxed{1216}
\]

features per rebalance date.

---

# 29. Final Dataset Dimensions

There are:

\[
884
\]

rebalance dates and:

\[
1216
\]

topological features per date.

Therefore:

\[
\boxed{
884\times1216
}
\]

is the final feature matrix.

The resulting file is:

```text
topo_features.csv
```

with:

```text
Rows    = 884
Columns = 1216
```

plus the `rebal_date` index when represented in the CSV.

The program reported:

```text
Done.
topo_features.csv:
884 dates × 1216 features
```

---

# 30. Complete Feature Structure

Each row of `topo_features.csv` represents one rebalance date.

The structure is:

```text
rebal_date
│
├── entropy_H0
├── entropy_H1
│
├── betti_H0_bin0
├── betti_H0_bin1
├── ...
├── betti_H0_bin99
│
├── betti_H1_bin0
├── betti_H1_bin1
├── ...
├── betti_H1_bin99
│
├── landscape_H0_layer0_mean
├── landscape_H0_layer0_bin0
├── ...
├── landscape_H0_layer0_bin99
│
├── landscape_H0_layer1_mean
├── ...
├── landscape_H0_layer4_bin99
│
├── landscape_H1_layer0_mean
├── ...
├── landscape_H1_layer4_bin99
│
├── L1norm_H0
├── L2norm_H0
├── L1norm_H1
└── L2norm_H1
```

Thus each date is transformed into a fixed 1216-dimensional topological state vector.

---

# 31. Interpretation of One Row

Suppose:

```text
rebal_date = 2018-07-16
```

The corresponding row in `topo_features.csv` describes the topology of the 50-stock network constructed using the previous 120 trading days.

It therefore represents:

\[
\boxed{
\text{Market structure around 2018-07-16}
}
\]

in terms of:

- connectivity structure
- persistence of connected components
- persistence of loops
- Betti-number evolution
- persistence landscape shape
- landscape magnitude
- persistent entropy

The 1216 values collectively form a numerical representation of the market's topological state.

---

# 32. Complete Phase 1 → Phase 2 Pipeline

The entire implemented system can be represented as follows:

```text
                     INPUT DATA
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Historical prices         weights.csv
       100 CSV files             216 snapshots
             │                       │
             └───────────┬───────────┘
                         ▼
                  ┌──────────────┐
                  │   PHASE 1    │
                  └──────┬───────┘
                         │
                  Select 50 stocks
                         │
                         ▼
                 Previous 120 days
                         │
                         ▼
                   Log returns
                         │
                         ▼
                 120 × 50 matrix
                         │
                         ▼
               Pearson correlation
                         │
                         ▼
                  50 × 50 matrix
                         │
                         ▼
             d = sqrt(2(1 - rho))
                         │
                         ▼
                  50 × 50 distance
                         │
                         ▼
              distance_matrices.pkl
                         │
                         ▼
                  ┌──────────────┐
                  │   PHASE 2    │
                  └──────┬───────┘
                         │
                         ▼
                Vietoris-Rips
                   filtration
                         │
                         ▼
                Persistent homology
                         │
                    ┌────┴────┐
                    │         │
                    ▼         ▼
                   H0        H1
                    │         │
                    └────┬────┘
                         ▼
                Persistence diagrams
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
          Entropy      Betti     Landscape
                                     │
                                     ▼
                                 L1 / L2
                                     │
             └───────────┬───────────┘
                         ▼
                 1216 features/date
                         │
                         ▼
                  topo_features.csv
                    884 × 1216
```

---

# 33. Current Project State

The following components have been successfully completed.

| Component | Status | Result |
|---|---|---:|
| Historical price loading | Complete | 100 tickers |
| Constituent snapshots | Complete | 216 snapshots |
| Rebalance calendar | Complete | 884 dates |
| Constituent selection | Complete | 50 stocks/date |
| 120-day return windows | Complete | 120 returns/date/stock |
| Log returns | Complete | Calculated |
| Pearson correlations | Complete | 50 × 50/date |
| Correlation-distance conversion | Complete | 50 × 50/date |
| Phase 1 persistence file | Complete | `distance_matrices.pkl` |
| Vietoris-Rips complexes | Complete | 884 dates |
| H0 persistence | Complete | 884 diagrams |
| H1 persistence | Complete | 884 diagrams |
| Persistence diagrams | Complete | `persistence_diagrams.pkl` |
| Persistent entropy | Complete | 2/date |
| Betti curves | Complete | 200/date |
| Persistence landscapes | Complete | 1010/date |
| L1/L2 landscape norms | Complete | 4/date |
| Final feature matrix | Complete | 884 × 1216 |
| Phase 2 output | Complete | `topo_features.csv` |

---

# 34. Important Methodological Note — Filtration Cutoff

One methodological issue should be explicitly reviewed before treating the results as final.

The distance transformation is:

\[
d_{ij}=\sqrt{2(1-\rho_{ij})}
\]

and its theoretical range is:

\[
0\leq d_{ij}\leq2.
\]

However, Phase 2 uses:

```python
EPS_MAX = sqrt(2)
```

Therefore, the filtration only extends to:

\[
\epsilon_{\max}=\sqrt2.
\]

Since:

\[
d_{ij}>\sqrt2
\]

corresponds to:

\[
\rho_{ij}<0,
\]

the current filtration does not include edges corresponding to negative correlations.

This may be intentional, but it should be explicitly justified in the methodology before final BTP analysis.

It is therefore important to distinguish:

\[
\boxed{
\text{pipeline successfully implemented}
}
\]

from:

\[
\boxed{
\text{all methodological choices finally validated}
}
\]

The former has been achieved; the latter should be checked before Phase 3.

---

# 35. Final Summary

Phase 1 converts financial time-series information into a sequence of metric networks.

For every rebalance date:

\[
\boxed{
50\text{ stocks}
\times
120\text{ historical returns}
\rightarrow
50\times50\text{ correlation matrix}
\rightarrow
50\times50\text{ distance matrix}
}
\]

Across 884 dates, these matrices form the input to Phase 2.

Phase 2 interprets each distance matrix as a metric point cloud and studies its evolving topology using Vietoris–Rips persistent homology.

The two retained homology dimensions are:

\[
H_0=\text{connected components}
\]

and:

\[
H_1=\text{loops}
\]

The persistence diagrams are then vectorized using:

- persistent entropy,
- 100-point Betti curves,
- five-layer persistence landscapes,
- landscape means,
- L1 landscape norms,
- L2 landscape norms.

The final representation is:

\[
\boxed{
884\text{ dates}\times1216\text{ topological features}
}
\]

stored in:

```text
topo_features.csv
```

while the raw persistent-homology information is retained in:

```text
persistence_diagrams.pkl
```

The project has therefore successfully completed the transformation from **raw financial price data to a time-indexed high-dimensional topological representation of the stock market**.

This 884 × 1216 dataset is now the principal input that can be used for the next stage of the project, such as dimensionality reduction, clustering, market-regime identification, temporal analysis, or downstream predictive modelling.