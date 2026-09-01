# Phase 1 Data Reference — Nifty-50 Price Data
**BTP: Topology-Guided Conformal Prediction for Adaptive Portfolio Allocation**

---

## 1. What Is in `data/`

The [`data/`](file:///r:/BTP/data) folder contains **100 CSV files** — one per ticker — covering every company that was ever a constituent of the Nifty-50 between **January 2008 and August 2025**, as tracked by [`dataset_figshare/weights.csv`](file:///r:/BTP/dataset_figshare/weights.csv).

### Two CSV formats coexist

| Format | Columns | Used by |
|---|---|---|
| **yfinance** | `Date, Close` | 90 stocks sourced cleanly from Yahoo Finance |
| **Gadiyar Bhavcopy** | `date, open, high, low, close, volume, source` | 10 stocks that were delisted/merged and needed manual sourcing |

> [!IMPORTANT]
> Your pipeline must handle both formats. Always normalise by lowercasing column names, then find the date column with `"date" in col_name` and the close column with `"close" in col_name`. Never hardcode column positions.

### Pipeline-verified status
- **884 rebalance dates** (5-day step, Jan 31 2008 → Dec 31 2025)
- **Every date**: exactly **50 active constituents**, every constituent has **≥ 120 trading days** of prior data
- **Zero violations** confirmed by [`verify_50x50_guarantee.py`](file:///r:/BTP/verify_50x50_guarantee.py)

---

## 2. The 10 Problematic Companies — What Went Wrong and How It Was Fixed

### Why they were problematic

All 10 are companies that **delisted, merged, or were acquired** during the study period. yfinance either has no data for them or returns incomplete historical series. The Gadiyar Bhavcopy data you originally sourced for them started **on the exact date they first entered the index** — giving zero pre-window days when the pipeline needs 120.

### The fix — two rounds of Gadiyar Bhavcopy collection

**Round 1 — 4 companies** whose entry date was mid-study (entering after Jan 2008):

| Ticker | Company | Problem | Pre-window fetched | Source folder |
|---|---|---|---|---|
| `RELCAPITAL` | Reliance Capital | Data started Jan 12, 2009 (index entry date) | Jul 1, 2008 → Jan 11, 2009 | `4_comps/` |
| `IDFC` | IDFC Ltd | Data started Oct 1, 2009 (index entry date) | Apr 1, 2009 → Sep 30, 2009 | `4_comps/` |
| `JPASSOCIAT` | Jaiprakash Associates | Data started Oct 1, 2009 (index entry date) | Apr 1, 2009 → Sep 30, 2009 | `4_comps/` |
| `INFRATEL` | Bharti Infratel | Data started Apr 1, 2016 (index entry date) | Oct 1, 2015 → Mar 31, 2016 | `4_comps/` |

**Round 2 — 6 companies** present in the index from the very first snapshot (Jan 31, 2008):

| Ticker | Company | Why problematic | Pre-window fetched | Source folder |
|---|---|---|---|---|
| `HDFC` | HDFC Ltd | Merged into HDFCBANK Jul 2023; original data started Jan 31 2008 | Jul 2, 2007 → Jan 30, 2008 | `6_comps/` |
| `RANBAXY` | Ranbaxy Laboratories | Acquired by Sun Pharma 2014; data started Jan 31 2008 | Jul 2, 2007 → Jan 30, 2008 | `6_comps/` |
| `CAIRN` | Cairn India | IPO Jul 2007; data started Jan 31 2008 | Jul 2, 2007 → Jan 30, 2008 | `6_comps/` |
| `RPL` | Reliance Petroleum | Merged into RELIANCE May 2009; data started Jan 31 2008 | Jul 2, 2007 → Jan 30, 2008 | `6_comps/` |
| `SATYAMCOMP` | Satyam Computer | Accounting scandal Jan 2009; data started Jan 31 2008 | Jul 2, 2007 → Jan 30, 2008 | `6_comps/` |
| `STER` | Sterlite Industries | Became VEDL; data started Jan 31 2008 | Jul 2, 2007 → Jan 30, 2008 | `6_comps/` |

### Final state of all 10 after merging

| Ticker | Full data range in `data/` | Days before first index use | Active in index |
|---|---|---|---|
| `HDFC` | 2007-07-02 → 2023-05-31 | **149** | Jan 2008 → May 2023 |
| `RANBAXY` | 2007-07-02 → 2014-02-26 | **149** | Jan 2008 → Feb 2014 |
| `CAIRN` | 2007-07-02 → 2016-02-29 | **149** | Jan 2008 → Feb 2016 |
| `RPL` | 2007-07-02 → 2009-05-29 | **149** | Jan 2008 → May 2009 |
| `SATYAMCOMP` | 2007-07-02 → 2008-12-31 | **149** | Jan 2008 → Dec 2008 |
| `STER` | 2007-07-02 → 2012-09-28 | **149** | Jan 2008 → Sep 2012 |
| `RELCAPITAL` | 2008-07-01 → 2011-08-30 | **129** | Jan 2009 → Aug 2011 |
| `IDFC` | 2009-04-01 → 2015-04-30 | **123** | Oct 2009 → Apr 2015 |
| `JPASSOCIAT` | 2009-04-01 → 2014-02-26 | **123** | Oct 2009 → Feb 2014 |
| `INFRATEL` | 2015-10-01 → 2020-08-28 | **122** | Apr 2016 → Aug 2020 |

---

## 3. Complete Stock Inventory — All 100 Tickers

### Group A — Continuously listed, full data (yfinance, 2007-01-02 → 2025-12-31, ~4,686 rows each)

These 68 stocks are the clean core. Data starts Jan 2, 2007 and runs to Dec 31, 2025.

`ABB · ACC · AMBUJACEM · ASIANPAINT · AUROPHARMA · AXISBANK · BAJAJAUTO · BAJAJFINSV · BAJFINANCE · BANKBARODA · BEL · BHARTIARTL · BHEL · BPCL · BRITANNIA · CIPLA · DIVISLAB · DRREDDY · EICHERMOT · GAIL · GLAXO · GRASIM · HCLTECH · HDFCBANK · HEROMOTOCO · HINDALCO · HINDUNILVR · ICICIBANK · INFY · ITC · JSWSTEEL · KOTAKBANK · LT · M&M · MARUTI · MCDOWELL-N · NATIONALUM · NESTLEIND · NTPC · ONGC · PNB · RCOM · RELIANCE · RELINFRA · SAIL · SBIN · SESAGOA · SHREECEM · SHRIRAMFIN · SIEMENS · SUNPHARMA · SUZLON · TATACOMM · TATACONSUM · TATAMOTORS · TATAPOWER · TATASTEEL · TCS · TECHM · TITAN · TRENT · ULTRACEMCO · UNITECH · UPL · VEDL · VSNL · WIPRO · YESBANK · ZEEL`

### Group B — Listed mid-study, sufficient pre-window (yfinance)

| Ticker | Data range | Rows | Notes |
|---|---|---|---|
| `ADANIPORTS` | 2007-11-27 → 2025-12-31 | 4,461 | IPO Nov 2007, 1,928 days before first use |
| `COALINDIA` | 2010-11-04 → 2025-12-31 | 3,738 | IPO Nov 2010, 226 days before first use |
| `DLF` | 2007-07-05 → 2025-12-31 | 4,561 | IPO Jul 2007, 185 days before first use |
| `ETERNAL` | 2021-07-23 → 2025-12-31 | 1,099 | (formerly Zomato), 912 days before first use |
| `HDFCLIFE` | 2017-11-17 → 2025-12-31 | 2,006 | IPO Nov 2017, 663 days before first use |
| `HINDPETRO` | 2007-01-02 → 2025-12-31 | 4,686 | In index 2017–2019 only |
| `IBULHSGFIN` | 2013-07-23 → 2025-12-31 | 3,068 | 906 days before first use |
| `IDEA` | 2007-03-09 → 2025-12-31 | 4,641 | 225 days before first use |
| `INDUSINDBK` | 2007-01-02 → 2025-12-31 | 4,686 | 1,540 days before first use |
| `IOC` | 2007-01-02 → 2025-12-31 | 4,686 | In index 2017–2022 only |
| `JINDALSTEL` | 2007-01-02 → 2025-12-31 | 4,686 | 602 days before first use |
| `JIOFIN` | 2023-08-21 → 2025-12-31 | 584 | 397 days before first use |
| `LTIM` | 2016-07-21 → 2025-12-31 | 2,334 | 1,716 days before first use |
| `LUPIN` | 2007-01-02 → 2025-12-31 | 4,686 | In index 2012–2018 only |
| `NMDC` | 2008-03-03 → 2025-12-31 | 4,394 | 1,248 days before first use |
| `POWERGRID` | 2007-10-05 → 2025-12-31 | 4,497 | **121 days** before first use (tightest OK stock) |
| `RPOWER` | 2008-02-11 → 2025-12-31 | 4,409 | 158 days before first use |
| `SBILIFE` | 2017-10-03 → 2025-12-31 | 2,038 | IPO Oct 2017, 735 days before first use |

### Group C — The 10 manually-sourced stocks (Gadiyar Bhavcopy, `Date,Close` format after merge)

See the full detail in Section 2 above. These end when their companies left the market (delisting, merger, acquisition) — that is correct and expected.

---

## 4. The "Missing End" Non-Issue

Six stocks show a `MISSING_END` flag in the audit because their data ends **1–2 calendar days before the last date they appear in weights.csv**:

| Ticker | Data ends | Last in weights.csv | Gap | Explanation |
|---|---|---|---|---|
| `RPL` | May 29, 2009 | May 31, 2009 | 2 days | May 31 was a Sunday — last trading day was May 29 |
| `RELCAPITAL` | Aug 30, 2011 | Aug 31, 2011 | 1 day | Aug 31 was a Wednesday but NSE holiday |
| `STER` | Sep 28, 2012 | Sep 30, 2012 | 2 days | Sep 30 was a Sunday |
| `JPASSOCIAT` | Feb 26, 2014 | Feb 27, 2014 | 1 day | Feb 27 was a Thursday but NSE holiday |
| `RANBAXY` | Feb 26, 2014 | Feb 27, 2014 | 1 day | Same holiday as JPASSOCIAT |
| `INFRATEL` | Aug 28, 2020 | Aug 30, 2020 | 2 days | Aug 30 was a Sunday |

**This is not a data problem.** The weights.csv records calendar month-end dates, but these stocks genuinely stopped trading on the last business day. Your forward-fill step will carry the last known price for 1–2 days — which is the methodologically correct thing to do.

---

## 5. How to Use This Data in Your Pipeline

### Loading a single stock

```python
import pandas as pd
from pathlib import Path

def load_close(ticker: str, data_dir: Path) -> pd.Series:
    """Load close price as a DatetimeIndex Series. Handles both CSV formats."""
    df = pd.read_csv(data_dir / f"{ticker}.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    date_col  = next(c for c in df.columns if "date"  in c)
    close_col = next(c for c in df.columns if "close" in c)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    s = pd.to_numeric(df[close_col], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(df[date_col])
    s.name  = ticker
    return s
```

### Getting constituents at any date (from weights.csv)

```python
weights = pd.read_csv("dataset_figshare/weights.csv", index_col=0, parse_dates=True)
weights = weights.sort_index()

def get_constituents(as_of_date: pd.Timestamp) -> list[str]:
    """Returns the 50 active tickers on or before as_of_date (forward-fill)."""
    past = weights.index[weights.index <= as_of_date]
    row  = weights.loc[past[-1]]
    return row[row > 0].index.tolist()
```

### Building the 120-day correlation matrix at a rebalance date

```python
import numpy as np

def build_correlation_matrix(
    rebal_date: pd.Timestamp,
    price_data: dict,          # {ticker: pd.Series}
    window: int = 120,
) -> pd.DataFrame:
    constituents = get_constituents(rebal_date)
    returns = {}
    for t in constituents:
        s   = price_data[t]
        pre = s[s.index < rebal_date].tail(window + 1)
        log_ret = np.log(pre / pre.shift(1)).dropna()
        if len(log_ret) >= window:
            returns[t] = log_ret.tail(window)
    ret_df = pd.DataFrame(returns).dropna()
    return ret_df.corr(method="pearson")   # always 50×50
```

### Rebalance schedule

```python
# Reference trading calendar from any continuously-listed stock
ref = load_close("RELIANCE", data_dir)
all_trading = pd.DatetimeIndex(sorted(ref.index))

STUDY_START = pd.Timestamp("2008-01-31")
STUDY_END   = pd.Timestamp("2025-12-31")
STEP        = 5   # every 5 trading days

study_days  = all_trading[(all_trading >= STUDY_START) & (all_trading <= STUDY_END)]
rebal_dates = study_days[::STEP]   # 884 dates
```

---

## 6. Precautions — What to Never Do

> [!CAUTION]
> **Never hardcode column indices.** Both CSV formats exist in `data/`. Always find columns by name (see `load_close()` above), never by position (`df.iloc[:, 1]`).

> [!CAUTION]
> **Never use future data in the rolling window.** The window for rebalance date `t` must use only `price < t` (strictly less than). The `pre = s[s.index < rebal_date]` idiom in the example above is correct. Using `<=` would leak the current day's return into the correlation matrix.

> [!WARNING]
> **Do not re-download data for the 10 manually-sourced stocks.** yfinance will not return correct pre-merge/pre-delisting history for `HDFC, RANBAXY, CAIRN, RPL, SATYAMCOMP, STER, RELCAPITAL, IDFC, JPASSOCIAT, INFRATEL`. The current files in `data/` are authoritative for these tickers. If you ever re-run a data download script, exclude these 10.

> [!WARNING]
> **POWERGRID has exactly 121 pre-window days** — it is the tightest stock in the dataset. If you ever increase `WINDOW` beyond 120, or change the first rebalance date to anything earlier than 2008-04-01, POWERGRID will fall below the threshold. Rerun `verify_50x50_guarantee.py` any time you change `WINDOW` or `STUDY_START`.

> [!NOTE]
> **The 10 manually-sourced stocks have data only for their active period.** For example, `SATYAMCOMP` ends Dec 31, 2008. For any date after that, `get_constituents()` will correctly not include it (weight = 0 in weights.csv). Do not attempt to extrapolate or extend their data.

> [!NOTE]
> **The 1–2 day end gaps in 6 stocks are correct.** Do not try to fill them with synthetic data. Forward-fill from the rolling correlation pipeline handles them automatically.

> [!TIP]
> **Use RELIANCE as your trading calendar reference.** It is one of the longest continuously-listed stocks in the dataset (Jan 2, 2007 → Dec 31, 2025, 4,686 rows, zero gaps) and is always in the index. Derive `all_trading` from it as shown in the rebalance schedule code above.

---

## 7. Key Numbers at a Glance

| Parameter | Value |
|---|---|
| Study period | Jan 31, 2008 → Dec 31, 2025 (~18 years) |
| Total rebalance dates | **884** (every 5 trading days) |
| Matrix size at every date | **50 × 50** |
| Minimum pre-window days | **121** (POWERGRID) |
| Maximum pre-window days | **4,375** (BEL, TRENT — in index very late) |
| Tickers in `data/` | **100** |
| Tickers with pipeline violations | **0** |
| Total rows across all CSVs | ~**440,000** |
| Verified by | [`verify_50x50_guarantee.py`](file:///r:/BTP/verify_50x50_guarantee.py) |

---

## 8. Script Inventory

| Script | Purpose |
|---|---|
| [`analyze_data_coverage.py`](file:///r:/BTP/analyze_data_coverage.py) | Original audit: checks every ticker against 120-day requirement |
| [`fetch_prewindow_data.py`](file:///r:/BTP/fetch_prewindow_data.py) | Attempted yfinance/Stooq fetch (blocked by rate limits — kept for reference) |
| [`check_4comps.py`](file:///r:/BTP/check_4comps.py) | Verified the 4_comps files before merging |
| [`merge_4comps.py`](file:///r:/BTP/merge_4comps.py) | Merged RELCAPITAL, IDFC, JPASSOCIAT, INFRATEL pre-window data |
| [`merge_and_verify_6comps.py`](file:///r:/BTP/merge_and_verify_6comps.py) | Verified + merged all 6 remaining stocks, then ran full simulation |
| [`verify_50x50_guarantee.py`](file:///r:/BTP/verify_50x50_guarantee.py) | **Master verification** — simulates all 884 rebalance dates end-to-end |
| [`final_data_audit.py`](file:///r:/BTP/final_data_audit.py) | Per-ticker summary of current data/ state |
| [`nifty_membership.py`](file:///r:/BTP/nifty_membership.py) | Utility: constituent lookup and yfinance ticker map |
