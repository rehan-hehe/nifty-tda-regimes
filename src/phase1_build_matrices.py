"""
Phase 1 -- builds distance_matrices.pkl from the local data/ folder + weights.csv.
This is the missing link between what you already have (data/, weights.csv) and
what Phase 2 (phase2_topology.py) expects as input.

Directory layout expected (matches guide_to_use_data.md):
    ./weights.csv
    ./data/<TICKER>.csv    (one file per ticker, either yfinance "Date,Close"
                             format or Gadiyar bhavcopy "date,open,high,low,
                             close,volume,source" format -- both handled)

Output:
    distance_matrices.pkl
        { rebal_date: {"correlation": DataFrame, "distance": DataFrame, "n_stocks": int} }
    (this is exactly what phase2_topology.py's load_distance_stack() expects)

    python build_distance_matrices.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
WEIGHTS_PATH = Path("weights.csv")
WINDOW = 120          # W, trading days
STEP = 5              # S, trading days
REFERENCE_TICKER = "RELIANCE"   # long, gap-free, always-in-index calendar reference


# ---------------------------------------------------------------------------
# Loading (this is your guide's load_close(), unchanged)
# ---------------------------------------------------------------------------
def load_close(ticker: str, data_dir: Path = DATA_DIR) -> pd.Series:
    """Load close price as a DatetimeIndex Series. Handles both CSV formats."""
    path = data_dir / f"{ticker}.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c)
    close_col = next(c for c in df.columns if "close" in c)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    s = pd.to_numeric(df[close_col], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(df[date_col])
    s.name = ticker
    return s


def load_all_prices(data_dir: Path = DATA_DIR) -> dict:
    """Load every CSV in data/ into a {ticker: Series} dict."""
    prices = {}
    failed = []
    for path in sorted(data_dir.glob("*.csv")):
        ticker = path.stem
        try:
            prices[ticker] = load_close(ticker, data_dir)
        except Exception as e:
            failed.append((ticker, str(e)))
    if failed:
        print(f"WARNING: {len(failed)} tickers failed to load:")
        for t, err in failed:
            print(f"  {t}: {err}")
    print(f"Loaded {len(prices)} tickers from {data_dir}/")
    return prices


def load_weights(path: Path = WEIGHTS_PATH) -> pd.DataFrame:
    weights = pd.read_csv(path, index_col=0, parse_dates=True)
    return weights.sort_index()


def make_get_constituents(weights: pd.DataFrame):
    """Returns a get_constituents(date) closure over a fixed weights table
    (your guide's version, wrapped so it doesn't rely on a module-level global)."""
    def get_constituents(as_of_date: pd.Timestamp) -> list:
        past = weights.index[weights.index <= as_of_date]
        if len(past) == 0:
            raise ValueError(f"{as_of_date.date()} is before the first weights.csv snapshot")
        row = weights.loc[past[-1]]
        return row[row > 0].index.tolist()
    return get_constituents


# ---------------------------------------------------------------------------
# Rebalance calendar
# ---------------------------------------------------------------------------
def build_rebalance_dates(prices: dict, study_start: str, study_end: str,
                           step: int = STEP, reference_ticker: str = REFERENCE_TICKER) -> pd.DatetimeIndex:
    if reference_ticker not in prices:
        raise ValueError(
            f"Reference ticker '{reference_ticker}' not found in loaded prices. "
            f"Pick another long, gap-free, always-in-index stock as reference."
        )
    all_trading = pd.DatetimeIndex(sorted(prices[reference_ticker].index))
    study_start, study_end = pd.Timestamp(study_start), pd.Timestamp(study_end)
    study_days = all_trading[(all_trading >= study_start) & (all_trading <= study_end)]
    return study_days[::step]


# ---------------------------------------------------------------------------
# Rolling correlation / distance matrices
# ---------------------------------------------------------------------------
def build_distance_matrices(prices: dict, weights: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
                             window: int = WINDOW) -> dict:
    get_constituents = make_get_constituents(weights)
    results = {}
    skipped = []

    for i, rebal_date in enumerate(rebal_dates):
        constituents = get_constituents(rebal_date)
        returns = {}
        for t in constituents:
            if t not in prices:
                continue
            s = prices[t]
            pre = s[s.index < rebal_date].tail(window + 1)   # strictly < rebal_date -- no lookahead
            log_ret = np.log(pre / pre.shift(1)).dropna()
            if len(log_ret) >= window:
                returns[t] = log_ret.tail(window)

        if len(returns) < len(constituents):
            missing = set(constituents) - set(returns.keys())
            skipped.append((rebal_date, missing))

        ret_df = pd.DataFrame(returns).dropna()
        if ret_df.shape[1] < 2:
            continue  # can't build a correlation matrix from <2 stocks

        corr = ret_df.corr(method="pearson")
        dist = np.sqrt(2 * (1 - corr))
        results[rebal_date] = {
            "correlation": corr,
            "distance": dist,
            "n_stocks": corr.shape[0],
        }

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(rebal_dates)} rebalance dates processed")

    if skipped:
        print(f"\nWARNING: {len(skipped)} rebalance dates had missing constituents "
              f"(insufficient pre-window data or missing ticker file):")
        for d, missing in skipped[:10]:
            print(f"  {d.date()}: missing {sorted(missing)}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more. Full list not printed -- "
                  f"check matrix sizes below before trusting downstream results.")

    sizes = [v["n_stocks"] for v in results.values()]
    if sizes:
        print(f"\nMatrix size distribution: min={min(sizes)}, max={max(sizes)}, "
              f"dates with exactly 50: {sum(1 for s in sizes if s == 50)}/{len(sizes)}")
        if min(sizes) < 50:
            print("NOTE: some matrices have <50 stocks. This is expected if you're "
                  "still filling in gaps (e.g. the 10 manually-sourced tickers). "
                  "Re-run once all pre-window data is confirmed complete, and treat "
                  "any date below 50 as unverified until then.")

    return results


def main(study_start="2008-01-31", study_end="2025-12-31"):
    print("Loading weights.csv...")
    weights = load_weights()
    print(f"  {weights.shape[0]} snapshots, {weights.shape[1]} tickers, "
          f"{weights.index.min().date()} to {weights.index.max().date()}")

    print("\nLoading price data...")
    prices = load_all_prices()

    print("\nBuilding rebalance calendar...")
    rebal_dates = build_rebalance_dates(prices, study_start, study_end)
    print(f"  {len(rebal_dates)} rebalance dates, "
          f"{rebal_dates[0].date()} to {rebal_dates[-1].date()}")

    print("\nBuilding rolling correlation/distance matrices...")
    matrices = build_distance_matrices(prices, weights, rebal_dates)

    with open("distance_matrices.pkl", "wb") as f:
        pickle.dump(matrices, f)
    print(f"\nSaved distance_matrices.pkl -- {len(matrices)} dates, "
          f"ready for phase2_topology.py")


if __name__ == "__main__":
    main()
