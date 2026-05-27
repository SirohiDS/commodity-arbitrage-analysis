"""
============================================================
data_loader.py — Unified data acquisition layer
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
============================================================

Sources
-------
* Yahoo Finance  — price/volume for commodities, ETFs, indices, FX
* FRED           — macro series (CPI, PPI, rates, money supply)

Analogy: Think of this module as the "Bloomberg terminal" of the project —
         it normalises every data feed into a common daily-returns format
         so every downstream analysis speaks the same language.
"""

import os
import sys
import time
import warnings
import logging
from typing import Dict, List, Optional, Tuple

import numpy  as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── append project root so config is importable from any cwd ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COMMODITY_TICKERS, CROSS_ASSET_TICKERS, FRED_SERIES,
    FRED_API_KEY, START_DATE, END_DATE, DATA_DIR,
)


# ═══════════════════════════════════════════════════════════════════════
# 1.  Yahoo Finance helpers
# ═══════════════════════════════════════════════════════════════════════

def fetch_yahoo(tickers: Dict[str, str],
                start: str = START_DATE,
                end:   str = END_DATE,
                retries: int = 3,
                delay: float = 2.0) -> pd.DataFrame:
    """
    Download adjusted-close prices for a dict of {label: ticker}.

    Parameters
    ----------
    tickers : dict  label → Yahoo Finance ticker symbol
    start, end : str  YYYY-MM-DD
    retries : int   retry count on transient failures
    delay   : float seconds between retries

    Returns
    -------
    DataFrame indexed by date, columns = labels, values = Adj Close prices.

    Analogy: Like placing multiple Reuters feeds side-by-side on one table —
             every column is a different instrument, every row a trading day.
    """
    all_frames: Dict[str, pd.Series] = {}

    for label, ticker in tickers.items():
        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(ticker, start=start, end=end,
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    log.warning("  ⚠  %s (%s) returned empty data", label, ticker)
                    break
                series = raw["Close"].squeeze()
                series.name = label
                series.index = pd.to_datetime(series.index).tz_localize(None)
                all_frames[label] = series
                log.info("  ✓  %-20s  %s  rows=%d", label, ticker, len(series))
                break
            except Exception as exc:
                log.warning("  ! Attempt %d/%d for %s failed: %s",
                            attempt, retries, ticker, exc)
                time.sleep(delay)

    if not all_frames:
        raise RuntimeError("No data fetched — check ticker list and internet connection.")

    df = pd.DataFrame(all_frames)
    df.sort_index(inplace=True)
    return df


def fetch_all_prices(start: str = START_DATE,
                     end:   str = END_DATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper — fetches both commodity and cross-asset price tables.

    Returns
    -------
    (commodity_prices, cross_asset_prices)  — two DataFrames of Adj Close
    """
    log.info("📥 Fetching commodity prices from Yahoo Finance …")
    comm  = fetch_yahoo(COMMODITY_TICKERS,  start, end)

    log.info("📥 Fetching cross-asset prices from Yahoo Finance …")
    cross = fetch_yahoo(CROSS_ASSET_TICKERS, start, end)

    return comm, cross


# ═══════════════════════════════════════════════════════════════════════
# 2.  FRED macro data
# ═══════════════════════════════════════════════════════════════════════

def fetch_fred(series_ids: Optional[Dict[str, str]] = None,
               start: str = START_DATE,
               end:   str = END_DATE) -> pd.DataFrame:
    """
    Download macro series from the St. Louis Federal Reserve FRED API.

    Parameters
    ----------
    series_ids : dict  label → FRED series ID  (defaults to FRED_SERIES from config)
    start, end : str   YYYY-MM-DD

    Returns
    -------
    DataFrame (monthly or mixed freq, forward-filled to daily).

    Analogy: FRED is the government data warehouse — it stores inflation,
             interest rates, and money supply figures compiled from official
             sources like the BLS and Treasury.
    """
    if series_ids is None:
        series_ids = FRED_SERIES

    try:
        from fredapi import Fred
        fred_client = Fred(api_key=FRED_API_KEY)
    except ImportError:
        log.error("fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()
    except Exception as exc:
        log.error("FRED client init failed: %s", exc)
        return pd.DataFrame()

    frames: Dict[str, pd.Series] = {}
    for label, sid in series_ids.items():
        try:
            s = fred_client.get_series(sid, observation_start=start,
                                        observation_end=end)
            s.name = label
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames[label] = s
            log.info("  ✓  FRED %-20s  %s  rows=%d", label, sid, len(s))
        except Exception as exc:
            log.warning("  ⚠  FRED %s (%s) failed: %s", label, sid, exc)

    if not frames:
        log.warning("No FRED data fetched. Continuing without macro data.")
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.sort_index(inplace=True)
    # Forward-fill to convert monthly → daily
    df = df.resample("D").last().ffill()
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3.  Return / transformation utilities
# ═══════════════════════════════════════════════════════════════════════

def compute_returns(prices: pd.DataFrame,
                    method: str = "log") -> pd.DataFrame:
    """
    Compute price returns.

    Parameters
    ----------
    prices : DataFrame of prices (Adj Close)
    method : "log"  → log returns  (preferred for statistics)
             "pct"  → simple percentage returns

    Returns
    -------
    DataFrame of returns with same columns, NaN first row dropped.

    Note: Log returns are additive over time — essential for regression
          and portfolio attribution. Think of them as the GPS coordinates
          of price movement vs simple % which are like odometer readings.
    """
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")


def align_data(*frames: pd.DataFrame,
               fill_method: str = "ffill") -> Tuple[pd.DataFrame, ...]:
    """
    Align multiple DataFrames to a common date index (intersection).

    Parameters
    ----------
    *frames : variable number of DataFrames
    fill_method : "ffill" forward-fill, "bfill" back-fill, or None to drop NaN

    Returns
    -------
    Tuple of aligned DataFrames in same order as input.
    """
    combined = pd.concat(frames, axis=1, join="outer")
    if fill_method:
        combined = combined.fillna(method=fill_method)  # type: ignore[arg-type]
    # Drop rows where ALL values are NaN
    combined = combined.dropna(how="all")

    # Split back into original DataFrames
    results = []
    idx = 0
    for f in frames:
        results.append(combined.iloc[:, idx : idx + f.shape[1]])
        idx += f.shape[1]
    return tuple(results)


def resample_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end for lower-frequency analysis."""
    return prices.resample("ME").last()


def compute_rolling_vol(returns: pd.DataFrame,
                        window: int = 21,
                        annualize: bool = True) -> pd.DataFrame:
    """
    Rolling historical volatility (annualised by default, 252 trading days).

    Analogy: Like measuring the width of a river at different points —
             wider = more volatile, narrower = calmer market conditions.
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def compute_rolling_corr(s1: pd.Series, s2: pd.Series,
                         window: int = 63) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    return s1.rolling(window=window).corr(s2)


def build_spread(price_a: pd.Series, price_b: pd.Series,
                 hedge_ratio: float = 1.0) -> pd.Series:
    """
    Compute a price spread  S = A − β·B.

    Used for:  Brent−WTI, Gold−GDX, NatGas−UNG, etc.
    The hedge_ratio β is typically estimated by OLS regression.
    """
    spread = price_a - hedge_ratio * price_b
    spread.name = f"{price_a.name} - {hedge_ratio:.3f}×{price_b.name}"
    return spread


def zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Compute rolling (or full-sample) z-score of a series.

    z = (x − μ) / σ

    Analogy: z-score tells you how many standard deviations the current
             spread is from its historical mean — the core signal for
             mean-reversion (statistical arbitrage) strategies.
    """
    if window:
        mu  = series.rolling(window).mean()
        sig = series.rolling(window).std()
    else:
        mu  = series.mean()
        sig = series.std()
    return (series - mu) / sig


# ═══════════════════════════════════════════════════════════════════════
# 4.  Cache helpers  (optional local CSV cache)
# ═══════════════════════════════════════════════════════════════════════

def save_to_cache(df: pd.DataFrame, name: str) -> None:
    """Persist DataFrame to CSV in data/ directory."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path)
    log.info("  💾 Saved cache: %s", path)


def load_from_cache(name: str) -> Optional[pd.DataFrame]:
    """Load cached CSV if it exists (returns None otherwise)."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(path):
        log.info("  📂 Loading cache: %s", path)
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def get_commodity_data(use_cache: bool = True,
                       start: str = START_DATE,
                       end:   str = END_DATE) -> Dict[str, pd.DataFrame]:
    """
    Master function — returns a dictionary with all price and return frames.

    Keys
    ----
    "commodity_prices"   : raw Adj Close for commodity instruments
    "cross_asset_prices" : raw Adj Close for cross-asset instruments
    "commodity_returns"  : log returns — commodities
    "cross_asset_returns": log returns — cross-assets
    "fred_macro"         : FRED macro series (forward-filled daily)
    "all_prices"         : merged price table (commodities + cross-assets)
    "all_returns"        : merged returns table
    """
    # ── try cache ──
    if use_cache:
        c_prices = load_from_cache("commodity_prices")
        x_prices = load_from_cache("cross_asset_prices")
        fred_df  = load_from_cache("fred_macro")
    else:
        c_prices = x_prices = fred_df = None

    # ── fetch if missing ──
    if c_prices is None or x_prices is None:
        c_prices, x_prices = fetch_all_prices(start, end)
        save_to_cache(c_prices, "commodity_prices")
        save_to_cache(x_prices, "cross_asset_prices")

    if fred_df is None:
        fred_df = fetch_fred(start=start, end=end)
        if not fred_df.empty:
            save_to_cache(fred_df, "fred_macro")

    # ── compute returns ──
    c_rets = compute_returns(c_prices,  method="log")
    x_rets = compute_returns(x_prices,  method="log")

    # ── merged tables ──
    all_prices  = pd.concat([c_prices,  x_prices],  axis=1, join="outer").ffill()
    all_returns = pd.concat([c_rets,    x_rets],    axis=1, join="outer").dropna(how="all")

    return {
        "commodity_prices":    c_prices,
        "cross_asset_prices":  x_prices,
        "commodity_returns":   c_rets,
        "cross_asset_returns": x_rets,
        "fred_macro":          fred_df,
        "all_prices":          all_prices,
        "all_returns":         all_returns,
    }


# ── Quick sanity check ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATA LOADER — Quick Fetch Test")
    print("="*60)
    data = get_commodity_data(use_cache=False,
                              start="2020-01-01",
                              end="2024-12-31")
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and not v.empty:
            print(f"\n{k}: shape={v.shape}")
            print(v.tail(3).to_string())
    print("\n✅ Data loader working correctly.")
