"""
============================================================
01_correlation_analysis.py
Cross-Asset Correlation Dashboard for Commodity Markets
============================================================

Objective
---------
Map the full correlation landscape across commodities and cross-asset
instruments using Pearson, Spearman, and rolling correlations.
Identify natural hedges, diversification opportunities, and
regime shifts in correlations.

Key Questions Answered
----------------------
1. Which commodities move together (substitutes / correlated production)?
2. Which assets provide genuine diversification for a commodity portfolio?
3. Do correlations change during crises (correlation contagion)?
4. Which cross-asset relationships are most stable over time?

Run
---
    python analysis/01_correlation_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from config import START_DATE, END_DATE, CHARTS_DIR
from src.data_loader        import get_commodity_data
from src.statistical_analysis import (
    pearson_corr_matrix, spearman_corr_matrix,
    rolling_correlation, correlation_change_test,
    full_statistical_summary,
)
from src.visualization import (
    plot_correlation_heatmap, plot_price_series,
    plot_rolling_correlations, plot_rolling_volatility,
)

print("\n" + "="*70)
print("  ANALYSIS 01 — Cross-Asset Correlation Dashboard")
print("="*70)

# ── 1. Load Data ──────────────────────────────────────────────────────
print("\n[1/6] Loading market data …")
data = get_commodity_data(use_cache=True)
comm_rets  = data["commodity_returns"]
cross_rets = data["cross_asset_returns"]
all_rets   = data["all_returns"]
all_prices = data["all_prices"]

print(f"  ✓ Commodity instruments:  {comm_rets.shape[1]}")
print(f"  ✓ Cross-asset instruments: {cross_rets.shape[1]}")
print(f"  ✓ Date range: {all_rets.index[0].date()} → {all_rets.index[-1].date()}")

# ── 2. Descriptive Statistics ─────────────────────────────────────────
print("\n[2/6] Computing descriptive statistics …")
stats_summary = full_statistical_summary(
    all_prices[[c for c in all_prices.columns if c in all_rets.columns]],
    all_rets
)
summary_df = stats_summary["summary"].round(4)

# Select key columns for display
display_cols = ["ann_return_%", "ann_vol_%", "sharpe_ratio",
                "max_drawdown_%", "skewness", "kurtosis"]
display_cols = [c for c in display_cols if c in summary_df.columns]
print("\n  📊 Summary Statistics:")
print(tabulate(summary_df[display_cols].head(20), headers="keys",
               tablefmt="fancy_grid", floatfmt=".3f"))

# ── 3. Commodity-Commodity Correlation ───────────────────────────────
print("\n[3/6] Computing commodity correlation matrix …")

# Select core commodity columns
comm_cols = [c for c in ["Gold_Spot", "WTI_Oil", "Brent_Oil", "NatGas",
                           "Wheat", "Copper"] if c in all_rets.columns]
comm_corr  = pearson_corr_matrix(all_rets[comm_cols])
comm_spear = spearman_corr_matrix(all_rets[comm_cols])

print("\n  Pearson Correlations — Commodities:")
print(tabulate(comm_corr.round(3), headers="keys",
               tablefmt="fancy_grid", floatfmt=".3f"))

fig_comm = plot_correlation_heatmap(
    comm_corr,
    title="Commodity-Commodity Pearson Correlation Matrix",
    filename="01a_commodity_correlation",
)

# ── 4. Commodity vs Cross-Asset Correlation ───────────────────────────
print("\n[4/6] Computing cross-asset correlation matrix …")

cross_cols = [c for c in ["SP500_ETF", "TLT", "HYG", "DXY", "TIP",
                            "MSCI_EM", "Gold_ETF"] if c in all_rets.columns]
# Merge commodity + cross-asset
merge_cols = comm_cols + cross_cols
merge_cols = [c for c in merge_cols if c in all_rets.columns]
full_corr  = pearson_corr_matrix(all_rets[merge_cols])

fig_full = plot_correlation_heatmap(
    full_corr,
    title="Commodity & Cross-Asset Full Correlation Matrix",
    filename="01b_full_crossasset_correlation",
)

# ── 5. Rolling Correlations — Key Pairs ──────────────────────────────
print("\n[5/6] Computing rolling correlations …")

key_pairs = []
if all(c in all_rets.columns for c in ["Gold_Spot", "SP500_ETF"]):
    key_pairs.append(("Gold_Spot", "SP500_ETF", "Gold vs S&P 500 (Safe-Haven Test)"))
if all(c in all_rets.columns for c in ["WTI_Oil", "Copper"]):
    key_pairs.append(("WTI_Oil", "Copper", "WTI Oil vs Copper (Growth Proxy)"))
if all(c in all_rets.columns for c in ["Gold_Spot", "DXY"]):
    key_pairs.append(("Gold_Spot", "DXY", "Gold vs USD Index (Inverse Relationship)"))
if all(c in all_rets.columns for c in ["WTI_Oil", "Brent_Oil"]):
    key_pairs.append(("WTI_Oil", "Brent_Oil", "WTI vs Brent (Geographical Spread)"))
if all(c in all_rets.columns for c in ["Copper", "SP500_ETF"]):
    key_pairs.append(("Copper", "SP500_ETF", "Copper vs S&P 500 (Dr. Copper)"))

for a, b, label in key_pairs:
    roll_c = rolling_correlation(all_rets[a], all_rets[b],
                                  windows=[21, 63, 126, 252])
    fname  = f"01c_rolling_{a.lower()}_{b.lower()}"
    fig_rc = plot_rolling_correlations(roll_c, pair_label=label, filename=fname)
    print(f"  ✓ Rolling correlation chart: {label}")

    # Structural break test (pre/post COVID)
    try:
        change_test = correlation_change_test(all_rets[a], all_rets[b], "2020-03-01")
        print(f"    Correlation change test (pre/post COVID-19):")
        print(f"      r_pre={change_test['r_pre_split']:.3f}  "
              f"r_post={change_test['r_post_split']:.3f}  "
              f"p={change_test['p_value']:.4f}  "
              f"→ {change_test['interpretation']}")
    except Exception as e:
        print(f"    Change test skipped: {e}")

# ── 6. Price Normalised Chart ─────────────────────────────────────────
print("\n[6/6] Generating price charts …")

# Commodity prices
comm_price_cols = [c for c in ["Gold_Spot", "WTI_Oil", "NatGas", "Wheat", "Copper"]
                   if c in all_prices.columns]
if comm_price_cols:
    plot_price_series(all_prices[comm_price_cols].dropna(),
                      title="Commodity Prices — Normalised to 100",
                      filename="01d_commodity_prices_normalised")

# Rolling volatility
plot_rolling_volatility(all_rets[comm_cols].dropna(),
                        window=21,
                        filename="01e_commodity_rolling_vol")

# ── Print Summary of Key Findings ─────────────────────────────────────
print("\n" + "="*70)
print("  KEY FINDINGS — Correlation Analysis")
print("="*70)

findings = []
for a, b, label in key_pairs:
    if a in all_rets.columns and b in all_rets.columns:
        r = all_rets[a].corr(all_rets[b])
        findings.append([label, f"{r:.3f}",
                         "Strong Positive" if r > 0.5 else
                         "Moderate Positive" if r > 0.2 else
                         "Neutral" if abs(r) < 0.2 else
                         "Moderate Negative" if r > -0.5 else
                         "Strong Negative"])

print(tabulate(findings, headers=["Pair", "Pearson r", "Relationship"],
               tablefmt="fancy_grid"))

print("""
┌─────────────────────────────────────────────────────────────────┐
│  TRADING INSIGHTS                                               │
│  ─────────────────────────────────────────────────────────────  │
│  1. Gold historically shows negative correlation to USD —       │
│     gold rallies when dollar weakens (effective FX hedge)       │
│  2. Copper-Equity correlation signals growth/recession cycles   │
│     ("Dr. Copper" as a leading economic indicator)             │
│  3. Oil-Brent correlation > 0.9 enables basis trading when     │
│     the spread deviates from historical mean (~$2-5/bbl)       │
│  4. Commodity correlations spike during crises — portfolio      │
│     diversification benefits diminish when most needed          │
│  5. Gold maintains low correlation to equities most of the      │
│     time — effective portfolio risk-reducer                     │
└─────────────────────────────────────────────────────────────────┘
""")

plt.close("all")
print(f"\n✅  Analysis 01 complete. Charts saved to: {CHARTS_DIR}")
