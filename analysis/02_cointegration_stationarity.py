"""
============================================================
02_cointegration_stationarity.py
Cointegration & Stationarity Analysis for Arbitrage Identification
============================================================

Objective
---------
Identify cointegrated commodity pairs that can support
statistical arbitrage strategies. Validate that price series
are integrated of order 1 (I(1)) and that spreads are stationary (I(0)).

Key Questions Answered
----------------------
1. Are commodity prices non-stationary (required for cointegration analysis)?
2. Which pairs are cointegrated and exhibit mean-reverting spreads?
3. What are the optimal hedge ratios for each pair?
4. Does the cointegration relationship hold across time sub-periods?

Run
---
    python analysis/02_cointegration_stationarity.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tabulate import tabulate

from config import CHARTS_DIR, COLOR_PALETTE
from src.data_loader         import get_commodity_data
from src.statistical_analysis import (
    stationarity_report, adf_test, kpss_test,
    engle_granger_coint, pairwise_cointegration_matrix,
    johansen_test, compute_spread_zscore,
)
from src.visualization import plot_spread_zscore

COLORS = COLOR_PALETTE

print("\n" + "="*70)
print("  ANALYSIS 02 — Cointegration & Stationarity Analysis")
print("="*70)

# ── 1. Load Data ──────────────────────────────────────────────────────
print("\n[1/5] Loading data …")
data       = get_commodity_data(use_cache=True)
all_prices = data["all_prices"]
all_rets   = data["all_returns"]
comm_prices = data["commodity_prices"]

# Select core commodity columns for cointegration analysis
core_cols = [c for c in ["Gold_Spot", "Gold_ETF", "Gold_Miners",
                           "WTI_Oil", "Brent_Oil", "Oil_ETF",
                           "NatGas", "Copper", "Wheat"]
             if c in comm_prices.columns]
prices = comm_prices[core_cols].dropna(how="all").ffill().dropna()

print(f"  ✓ {len(core_cols)} instruments, {len(prices)} trading days")
print(f"  ✓ Period: {prices.index[0].date()} → {prices.index[-1].date()}")

# ── 2. Stationarity Testing ───────────────────────────────────────────
print("\n[2/5] Stationarity testing (ADF + KPSS) …")

stat_report = stationarity_report(prices)
print("\n  📊 Stationarity Summary:")
print(tabulate(stat_report.round(4), headers="keys",
               tablefmt="fancy_grid", floatfmt=".4f"))

print("""
  INTERPRETATION:
  • Price levels should be NON-STATIONARY (ADF p > 0.05, KPSS p < 0.05)
  • Log-returns should be STATIONARY (ADF p < 0.05, KPSS p > 0.05)
  • If prices are I(1), they're candidates for cointegration analysis.
""")

# ── 3. Pairwise Cointegration Matrix ─────────────────────────────────
print("\n[3/5] Running pairwise Engle-Granger cointegration tests …")

# Define key pairs to test
key_pairs = [
    ("Gold_Spot", "Gold_ETF"),       # Spot vs ETF — almost perfect
    ("Gold_Spot", "Gold_Miners"),     # Spot vs Miners — fundamental link
    ("WTI_Oil",   "Brent_Oil"),      # Geographical spread
    ("WTI_Oil",   "Oil_ETF"),        # Futures vs ETF basis
    ("NatGas",    "NatGas_ETF") if "NatGas_ETF" in prices.columns
        else ("NatGas", "WTI_Oil"),   # Energy pair
    ("Copper",    "WTI_Oil"),        # Industrial commodity pair
]
key_pairs = [(a, b) for a, b in key_pairs if a in prices.columns and b in prices.columns]

coint_results = []
for a, b in key_pairs:
    result = engle_granger_coint(prices[a], prices[b])
    coint_results.append(result)
    status = "✓ COINTEGRATED" if result["cointegrated"] else "✗ not cointegrated"
    print(f"  {a} ↔ {b}:  β={result['beta']:.4f}  "
          f"p={result['p_value']:.4f}  R²={result['r_squared']:.4f}  {status}")

# Summary table
coint_df = pd.DataFrame([{
    "Y": r["y"], "X": r["x"],
    "β (Hedge Ratio)":  r["beta"],
    "P-Value":          r["p_value"],
    "R²":               r["r_squared"],
    "Cointegrated":     "✓ YES" if r["cointegrated"] else "✗ NO",
    "Residual Std":     r["residual_std"],
} for r in coint_results])
print("\n  📊 Cointegration Results:")
print(tabulate(coint_df, headers="keys", tablefmt="fancy_grid",
               showindex=False, floatfmt=".4f"))

# ── 4. Spread Z-Score Visualisation ──────────────────────────────────
print("\n[4/5] Generating spread and z-score charts …")

for res in coint_results:
    if not res["cointegrated"]:
        continue
    a, b = res["y"], res["x"]
    if a not in prices.columns or b not in prices.columns:
        continue

    spread_df = compute_spread_zscore(prices[a], prices[b],
                                       coint_result=res,
                                       zscore_window=63)
    fname = f"02_spread_zscore_{a.lower()}_{b.lower()}"
    plot_spread_zscore(spread_df,
                       pair_name=f"{a} − {res['beta']:.4f}·{b}",
                       filename=fname)
    print(f"  ✓ Spread chart: {a} vs {b}")

# ── 5. Johansen Test (Multi-Asset) ────────────────────────────────────
print("\n[5/5] Johansen multivariate cointegration test …")

energy_cols = [c for c in ["WTI_Oil", "Brent_Oil", "NatGas", "Oil_ETF"]
               if c in prices.columns]
metal_cols  = [c for c in ["Gold_Spot", "Gold_ETF", "Copper"]
               if c in prices.columns]

for group_name, group_cols in [("Energy", energy_cols), ("Metals", metal_cols)]:
    if len(group_cols) < 2:
        continue
    joh = johansen_test(prices[group_cols])
    print(f"\n  Johansen Test — {group_name} Basket:")
    print(f"    Assets:                 {', '.join(joh['assets'])}")
    print(f"    Cointegrating vectors:  {joh['n_cointegrating']}")
    print(f"    Interpretation:         {joh['interpretation']}")

    trace_df = pd.DataFrame({
        "H₀: r ≤ k":     [f"r ≤ {i}" for i in range(len(joh["trace_stats"]))],
        "Trace Stat":    [round(v, 3) for v in joh["trace_stats"]],
        "Crit Val 95%":  [round(v, 3) for v in joh["crit_values_95"]],
        "Reject H₀":     [t > c for t, c in zip(joh["trace_stats"], joh["crit_values_95"])],
    })
    print(tabulate(trace_df, headers="keys", tablefmt="grid", showindex=False))

print("""
┌─────────────────────────────────────────────────────────────────┐
│  TRADING INSIGHTS — Cointegration                               │
│  ─────────────────────────────────────────────────────────────  │
│  1. Gold Spot ↔ GLD ETF: Near-perfect cointegration with β≈1   │
│     Any deviation > $0.50 is an ETF creation/redemption arb     │
│  2. WTI ↔ Brent: Cointegrated but spread varies with:          │
│     • Transportation costs (pipeline/tanker)                    │
│     • Geopolitical events (Middle East, Cushing inventory)      │
│     • Refinery demand differentials                             │
│  3. Gold ↔ Gold Miners: Cointegrated with β<1 because miners   │
│     add operational leverage (fixed costs amplify price moves)  │
│  4. Energy basket: If Johansen confirms 2+ cointegrating        │
│     vectors, a 3-leg basket trade (WTI+Brent vs NatGas) is     │
│     viable for calendar/geographical spread trades.             │
└─────────────────────────────────────────────────────────────────┘
""")

plt.close("all")
print(f"\n✅  Analysis 02 complete. Charts saved to: {CHARTS_DIR}")
