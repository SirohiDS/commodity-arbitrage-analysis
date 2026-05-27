"""
============================================================
05_geo_spreads.py
Geographical Commodity Spread Analysis: Brent vs WTI
============================================================

Objective
---------
Analyse the Brent-WTI spread — the most liquid and traded commodity
spread in global energy markets. Evaluate drivers, trading signals,
and opportunities in regional crude oil pricing differentials.

Background
----------
Brent Crude:    North Sea blend, global benchmark for 2/3 of world's oil
WTI Crude:      West Texas Intermediate, US benchmark (delivery at Cushing, OK)

Normal spread:  Brent at $1-3 premium over WTI (historical)
Divergence causes:
  • Cushing, Oklahoma storage gluts (WTI discount)
  • Middle East/Libya/Nigeria supply disruptions (Brent premium spike)
  • US shale boom (WTI flood at landlocked Cushing)
  • Export restrictions / pipeline capacity constraints

Analogy: Brent and WTI are the same commodity — crude oil — but from
different postcodes. The spread is like the price difference between
the same car model at two dealerships 200 miles apart. If the gap
gets too wide, someone ships oil (or buys at one, sells at the other).

Run
---
    python analysis/05_geo_spreads.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates   as mdates
from scipy import stats
from tabulate import tabulate

from config import CHARTS_DIR, COLOR_PALETTE
from src.data_loader         import get_commodity_data
from src.statistical_analysis import (
    adf_test, engle_granger_coint,
    compute_spread_zscore, rolling_correlation,
    ols_regression,
)
from src.backtesting  import run_zscore_strategy, BacktestEngine
from src.risk_metrics import risk_report
from src.visualization import plot_geo_spread, plot_spread_zscore

COLORS = COLOR_PALETTE

print("\n" + "="*70)
print("  ANALYSIS 05 — Geographical Commodity Spreads")
print("="*70)
print("  Focus: Brent vs WTI Crude Oil — Global Benchmark Arbitrage")

# ── 1. Load Data ──────────────────────────────────────────────────────
print("\n[1/6] Loading energy market data …")
data = get_commodity_data(use_cache=True)
prices = data["all_prices"]
rets   = data["all_returns"]

brent_col = "Brent_Oil"
wti_col   = "WTI_Oil"

if brent_col not in prices.columns or wti_col not in prices.columns:
    print("  ⚠  Brent or WTI data not available. Check tickers in config.py")
    sys.exit(0)

aligned = pd.concat([prices[brent_col], prices[wti_col]], axis=1).dropna()
brent = aligned[brent_col]
wti   = aligned[wti_col]
spread = brent - wti

print(f"  ✓ Data: {aligned.index[0].date()} → {aligned.index[-1].date()}")
print(f"  ✓ Brent range: ${brent.min():.2f} – ${brent.max():.2f}")
print(f"  ✓ WTI range:   ${wti.min():.2f} – ${wti.max():.2f}")
print(f"  ✓ Spread range: ${spread.min():.2f} – ${spread.max():.2f}")

# ── 2. Spread Statistics ──────────────────────────────────────────────
print("\n[2/6] Spread statistics & stationarity …")

spread_stats = {
    "Mean Spread ($/bbl)":    spread.mean(),
    "Median Spread ($/bbl)":  spread.median(),
    "Std Dev ($/bbl)":        spread.std(),
    "Min Spread ($/bbl)":     spread.min(),
    "Max Spread ($/bbl)":     spread.max(),
    "Brent Premium % of time": (spread > 0).mean() * 100,
    "WTI Premium % of time":   (spread < 0).mean() * 100,
    "% time spread > $5":      (spread >  5).mean() * 100,
    "% time spread < −$3":     (spread < -3).mean() * 100,
}

print("\n  📊 Brent-WTI Spread Statistics:")
for k, v in spread_stats.items():
    print(f"    {k:35s}: {v:8.3f}")

# ADF test on spread
adf_r = adf_test(spread.dropna())
print(f"\n  ADF Stationarity Test on Spread:")
print(f"    ADF Statistic: {adf_r['adf_statistic']:.4f}  p-value: {adf_r['p_value']:.4f}")
print(f"    Conclusion: {adf_r['integration_order']}")
print(f"    → {'Mean-reverting spread — tradable!' if adf_r['stationary_at_5pct'] else 'Trend-following spread — not stationary'}")

# ── 3. Regime Analysis ────────────────────────────────────────────────
print("\n[3/6] Historical regime analysis …")

# Define key spread regimes
regimes = [
    ("2008 Financial Crisis",    "2007-01-01", "2009-12-31"),
    ("US Shale Boom",            "2011-01-01", "2014-12-31"),
    ("Oil Price Collapse",       "2015-01-01", "2016-12-31"),
    ("COVID-19 Shock",           "2020-01-01", "2020-12-31"),
    ("Post-COVID Energy Crisis", "2021-01-01", "2022-12-31"),
]

regime_data = []
for name, start, end in regimes:
    sub = spread.loc[start:end].dropna()
    if len(sub) < 10:
        continue
    regime_data.append({
        "Period":              name,
        "Avg Spread $/bbl":    round(sub.mean(), 2),
        "Std Dev $/bbl":       round(sub.std(), 2),
        "Max Spread $/bbl":    round(sub.max(), 2),
        "Min Spread $/bbl":    round(sub.min(), 2),
        "Brent Premium %":     round((sub > 0).mean() * 100, 1),
    })

print("\n  📊 Spread by Market Regime:")
print(tabulate(regime_data, headers="keys", tablefmt="fancy_grid",
               showindex=False, floatfmt=".2f"))

# ── 4. Cointegration & Hedge Ratio ────────────────────────────────────
print("\n[4/6] Cointegration test — Brent vs WTI …")
coint_r = engle_granger_coint(brent, wti)
print(f"  Engle-Granger: stat={coint_r['eg_statistic']:.4f}  "
      f"p={coint_r['p_value']:.4f}  β={coint_r['beta']:.6f}")
print(f"  R² = {coint_r['r_squared']:.4f}")
print(f"  {coint_r['interpretation']}")

# Z-score spread for trading
spread_df = compute_spread_zscore(brent, wti, coint_r, zscore_window=63)

# Main geo-spread dashboard chart
plot_geo_spread(brent, wti, filename="05a_brent_wti_dashboard")

# Z-score spread chart
plot_spread_zscore(spread_df,
                    pair_name="Brent − WTI (Geographical Spread)",
                    filename="05b_brent_wti_zscore")

# ── 5. Backtest: Brent-WTI Spread Strategy ───────────────────────────
print("\n[5/6] Backtesting Brent-WTI spread mean-reversion strategy …")

brent_rets = rets[brent_col] if brent_col in rets.columns else None
wti_rets   = rets[wti_col]   if wti_col   in rets.columns else None

if brent_rets is not None and wti_rets is not None:
    try:
        bt_res = run_zscore_strategy(
            brent, wti,
            zscore_window=63,
            entry_z=1.5,     # tighter entry for liquid market
            exit_z=0.3,
            notional=5_000_000,  # $5M notional — liquid market supports this
            tc_bps=3,            # tight bid-ask for Brent/WTI
            name="Brent-WTI Spread Arb",
        )

        s = bt_res["summary"]
        print("\n  📊 Backtest Results:")
        metrics_show = ["Ann. Return %", "Ann. Volatility %", "Sharpe Ratio",
                         "Sortino Ratio", "Calmar Ratio", "Max Drawdown %",
                         "Total Trades", "Win Rate %", "Profit Factor"]
        for m in metrics_show:
            if m in s.index:
                print(f"    {m:25s}: {s[m]}")

        # Plot tearsheet
        bt_res["engine"].plot_results(show=False,
            save_path=os.path.join(CHARTS_DIR, "05c_brent_wti_backtest.png"))

        # Risk report
        rpt = risk_report(bt_res["net_returns"])
        print(f"\n  VaR 95% (Hist): {rpt['var_95_hist_%']:.3f}%  "
              f"CVaR 95%: {rpt['cvar_95_hist_%']:.3f}%")
    except Exception as e:
        print(f"  ⚠  Backtest failed: {e}")

# ── 6. Macro Driver Analysis ──────────────────────────────────────────
print("\n[6/6] Macro driver regression — what drives the spread? …")

fred_df = data.get("fred_macro", pd.DataFrame())
driver_cols = []
if not fred_df.empty:
    # Check available FRED columns
    for col in ["CPI", "10Y_Treasury", "VIX", "M2_Money"]:
        if col in fred_df.columns:
            driver_cols.append(col)

if driver_cols:
    macro_rets = fred_df[driver_cols].pct_change().dropna()
    spread_pct_chg = spread.pct_change().dropna()
    combined = pd.concat([spread_pct_chg, macro_rets], axis=1).dropna()
    if len(combined) > 50:
        reg_y = combined.iloc[:, 0]
        reg_X = combined.iloc[:, 1:]
        ols_r = ols_regression(reg_y, reg_X)
        print(f"\n  OLS: Spread % Change ~ Macro Variables")
        print(f"  R² = {ols_r['r_squared']:.4f}  |  Adj.R² = {ols_r['adj_r_squared']:.4f}")
        print(f"  Significant drivers (p < 0.10):")
        for var, pval in ols_r["p_values"].items():
            if pval < 0.10 and var != "const":
                coef = ols_r["coefficients"].get(var, 0)
                print(f"    {var}: coef={coef:.4f}  p={pval:.4f}")
else:
    print("  ℹ  FRED data not available — configure FRED_API_KEY in .env")

# Rolling correlation: spread vs USD
if "DXY" in rets.columns:
    roll_c = rolling_correlation(spread.pct_change().dropna(), rets["DXY"],
                                  windows=[21, 63, 252])
    print(f"\n  Full-sample Brent-WTI Spread vs USD correlation: "
          f"{spread.pct_change().dropna().corr(rets['DXY']):.4f}")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  TRADING INSIGHTS — Brent vs WTI                                │
│  ─────────────────────────────────────────────────────────────  │
│  1. Normal premium: Brent $1–3/bbl over WTI                    │
│     → When spread > $6/bbl → SELL Brent, BUY WTI              │
│     → When spread < $0/bbl → BUY Brent, SELL WTI              │
│  2. Contango structure difference: When Cushing storage fills,  │
│     WTI contango steepens while Brent stays flatter — this      │
│     creates a time-spread within the geographical spread.        │
│  3. Key catalysts to monitor:                                   │
│     • EIA Weekly Petroleum Status Report (every Wed.)           │
│     • OPEC+ production quota decisions                          │
│     • Keystone XL / pipeline capacity headlines                 │
│     • Libyan / Nigerian supply disruptions                      │
│  4. Risk management: The spread can gap $5-10/bbl overnight     │
│     on a major geopolitical shock. Always use stop-losses.      │
│  5. Carry trade: In deep backwardation, long WTI futures earn   │
│     positive roll yield while hedging with short Brent limits   │
│     the oil-price directional risk.                             │
└─────────────────────────────────────────────────────────────────┘
""")

plt.close("all")
print(f"\n✅  Analysis 05 complete. Charts saved to: {CHARTS_DIR}")
