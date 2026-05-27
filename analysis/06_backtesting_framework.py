"""
============================================================
06_backtesting_framework.py
Multi-Strategy Backtesting, Comparison & Risk Attribution
============================================================

Objective
---------
Run the full backtesting suite across all strategies, produce
performance tearsheets, and compare against buy-and-hold benchmarks.

Strategies Tested
-----------------
1. Z-Score Statistical Arb — WTI vs Brent (geo spread)
2. Z-Score Statistical Arb — Gold vs GLD (basis arb)
3. Z-Score Statistical Arb — Gold vs Gold Miners
4. Momentum — Gold (trend-following CTA-style)
5. Momentum — WTI Oil (trend-following CTA-style)
6. Volatility-Scaled Portfolio (risk parity)
7. Commodity Basket (equal-weight long)

Run
---
    python analysis/06_backtesting_framework.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

from config import CHARTS_DIR, COLOR_PALETTE, STRATEGY_PARAMS
from src.data_loader    import get_commodity_data
from src.backtesting    import (
    BacktestEngine, run_zscore_strategy, run_momentum_strategy,
    compare_strategies, benchmark_comparison,
)
from src.risk_metrics   import (
    risk_report, var_comparison, kelly_criterion,
    vol_target_sizing, drawdown_series,
)
from src.visualization  import plot_strategy_comparison, plot_var_distribution

COLORS = COLOR_PALETTE

print("\n" + "="*70)
print("  ANALYSIS 06 — Multi-Strategy Backtesting Framework")
print("="*70)

# ── 1. Load Data ──────────────────────────────────────────────────────
print("\n[1/4] Loading data …")
data   = get_commodity_data(use_cache=True)
prices = data["all_prices"]
rets   = data["all_returns"]

print(f"  ✓ {rets.shape[1]} instruments, {len(rets)} daily observations")

# ── 2. Run All Strategies ─────────────────────────────────────────────
print("\n[2/4] Running strategy backtests …")

all_results   = {}
equity_curves = {}

# ── Strategy 1: WTI-Brent Geo Spread ────────────────────────────────
if "WTI_Oil" in prices.columns and "Brent_Oil" in prices.columns:
    print("  Running: WTI-Brent Geo Spread Arb …", end=" ")
    try:
        r = run_zscore_strategy(prices["WTI_Oil"], prices["Brent_Oil"],
                                 zscore_window=63, entry_z=1.5, exit_z=0.3,
                                 notional=1_000_000, tc_bps=3,
                                 name="WTI-Brent Spread")
        all_results["WTI-Brent Spread"] = r
        equity_curves["WTI-Brent Spread"] = r["equity_curve"]
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

# ── Strategy 2: Gold-GLD Basis Arb ───────────────────────────────────
if "Gold_Spot" in prices.columns and "Gold_ETF" in prices.columns:
    print("  Running: Gold-GLD Basis Arb …", end=" ")
    try:
        r = run_zscore_strategy(prices["Gold_Spot"], prices["Gold_ETF"],
                                 zscore_window=63, entry_z=2.0, exit_z=0.5,
                                 notional=1_000_000, tc_bps=5,
                                 name="Gold-GLD Basis")
        all_results["Gold-GLD Basis"] = r
        equity_curves["Gold-GLD Basis"] = r["equity_curve"]
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

# ── Strategy 3: Gold-Miners Arb ───────────────────────────────────────
if "Gold_Spot" in prices.columns and "Gold_Miners" in prices.columns:
    print("  Running: Gold-GDX Miners Arb …", end=" ")
    try:
        r = run_zscore_strategy(prices["Gold_Spot"], prices["Gold_Miners"],
                                 zscore_window=63, entry_z=2.0, exit_z=0.5,
                                 notional=1_000_000, tc_bps=8,
                                 name="Gold-GDX Arb")
        all_results["Gold-GDX Arb"] = r
        equity_curves["Gold-GDX Arb"] = r["equity_curve"]
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

# ── Strategy 4: Gold Momentum ─────────────────────────────────────────
if "Gold_Spot" in rets.columns:
    print("  Running: Gold Momentum (21-day) …", end=" ")
    try:
        r = run_momentum_strategy(rets["Gold_Spot"], lookback=21,
                                   notional=1_000_000, tc_bps=3,
                                   name="Gold Momentum")
        all_results["Gold Momentum"] = r
        equity_curves["Gold Momentum"] = r["equity_curve"]
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

# ── Strategy 5: WTI Momentum ──────────────────────────────────────────
if "WTI_Oil" in rets.columns:
    print("  Running: WTI Momentum (21-day) …", end=" ")
    try:
        r = run_momentum_strategy(rets["WTI_Oil"], lookback=21,
                                   notional=1_000_000, tc_bps=3,
                                   name="WTI Momentum")
        all_results["WTI Momentum"] = r
        equity_curves["WTI Momentum"] = r["equity_curve"]
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

# ── Strategy 6: Equal-Weight Commodity Long (Benchmark) ───────────────
print("  Running: Equal-Weight Commodity Benchmark …", end=" ")
try:
    bench_cols = [c for c in ["Gold_Spot", "WTI_Oil", "NatGas", "Wheat", "Copper"]
                  if c in rets.columns]
    if bench_cols:
        bench_rets = rets[bench_cols].mean(axis=1).dropna()
        signal_all_long = pd.Series(1.0, index=bench_rets.index)
        eng = BacktestEngine(bench_rets, signal_all_long,
                              notional=1_000_000, tc_bps=2,
                              name="EW Commodity Long")
        r = eng.run()
        r["engine"] = eng
        all_results["EW Commodity Long"] = r
        equity_curves["EW Commodity Long"] = r["equity_curve"]
        print("✓")
except Exception as e:
    print(f"✗ ({e})")

print(f"\n  ✓ {len(all_results)} strategies successfully backtested")

# ── 3. Performance Comparison ─────────────────────────────────────────
print("\n[3/4] Performance comparison …")

comparison = compare_strategies(all_results)
if not comparison.empty:
    key_metrics = ["Ann. Return %", "Ann. Volatility %", "Sharpe Ratio",
                   "Sortino Ratio", "Max Drawdown %", "Win Rate %",
                   "Total Trades", "Profit Factor"]
    disp_metrics = [m for m in key_metrics if m in comparison.index]
    print("\n  📊 Strategy Performance Summary:")
    print(tabulate(comparison.loc[disp_metrics].T, headers="keys",
                   tablefmt="fancy_grid", floatfmt=".3f"))

# ── Strategy Equity Curves ────────────────────────────────────────────
if equity_curves:
    plot_strategy_comparison(equity_curves, filename="06a_strategy_comparison")
    print(f"  💾 Strategy comparison chart saved")

# ── Detailed Tearsheets ───────────────────────────────────────────────
print("\n  Generating individual tearsheets …")
for name, res in all_results.items():
    if "engine" in res:
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        try:
            res["engine"].plot_results(
                show=False,
                save_path=os.path.join(CHARTS_DIR, f"06b_tearsheet_{safe_name}.png")
            )
            print(f"  💾 Tearsheet: {name}")
        except Exception as e:
            print(f"  ⚠  Tearsheet {name}: {e}")

# ── 4. Risk Attribution & VaR ─────────────────────────────────────────
print("\n[4/4] Risk attribution across strategies …")

risk_rows = []
for name, res in all_results.items():
    net_rets = res["net_returns"]
    rpt = risk_report(net_rets, position=1_000_000)

    # Kelly criterion
    trades = res.get("trade_log", pd.DataFrame())
    if len(trades) > 0 and "pnl_%" in trades.columns:
        wins   = trades[trades["pnl_%"] > 0]["pnl_%"] / 100
        losses = trades[trades["pnl_%"] < 0]["pnl_%"].abs() / 100
        win_r  = len(wins) / len(trades)
        avg_w  = wins.mean() if len(wins) > 0 else 0
        avg_l  = losses.mean() if len(losses) > 0 else 0.001
        kc     = kelly_criterion(win_r, avg_w, avg_l)
        kelly_f = kc["kelly_fraction"]
    else:
        kelly_f = np.nan

    risk_rows.append({
        "Strategy":        name,
        "Sharpe":          rpt["sharpe_ratio"],
        "Sortino":         rpt["sortino_ratio"],
        "Calmar":          rpt["calmar_ratio"],
        "MaxDD %":         rpt["max_drawdown_%"],
        "VaR 95%":         rpt["var_95_hist_%"],
        "CVaR 95%":        rpt["cvar_95_hist_%"],
        "VaR 99%":         rpt["var_99_hist_%"],
        "Skewness":        rpt["skewness"],
        "Kurtosis":        rpt["kurtosis"],
        "Kelly f%":        round(kelly_f * 100, 1) if not np.isnan(kelly_f) else "—",
    })

risk_df = pd.DataFrame(risk_rows).set_index("Strategy")
print("\n  📊 Risk Attribution:")
print(tabulate(risk_df, headers="keys", tablefmt="fancy_grid", floatfmt=".3f"))

# VaR distribution for best strategy
if all_results:
    best_strat = max(all_results.items(),
                     key=lambda x: x[1]["summary"].get("Sharpe Ratio", -999)
                     if "summary" in x[1] else -999)
    best_name, best_res = best_strat
    plot_var_distribution(best_res["net_returns"],
                           filename=f"06c_var_distribution_best")
    print(f"\n  VaR chart: {best_name}")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  BACKTESTING INSIGHTS & KEY METRICS INTERPRETATION              │
│  ─────────────────────────────────────────────────────────────  │
│  Sharpe Ratio:                                                  │
│    > 2.0 = Excellent  |  1.0–2.0 = Good  |  < 0.5 = Poor      │
│                                                                 │
│  Max Drawdown:                                                  │
│    < 10% = Low  |  10–25% = Moderate  |  > 30% = High          │
│                                                                 │
│  Win Rate by Strategy Type:                                     │
│    Mean-reversion (z-score): Target 55–65% hit rate            │
│    Momentum: Target 40–50% (wins are larger than losses)        │
│    Basis arb: Target 70%+ (low per-trade risk)                 │
│                                                                 │
│  Kelly Fraction:                                                │
│    Full Kelly = theoretically optimal but high variance         │
│    Half Kelly (recommended) = safer, 75% of max growth rate    │
│    Use Kelly < 10% for conservative institutional sizing        │
│                                                                 │
│  Risk-Adjusted Ranking (typical):                               │
│    1. Basis Arb (high Sharpe, low vol)                         │
│    2. Geo Spread Arb (liquid, tight spreads)                   │
│    3. Gold-Miners Arb (more idiosyncratic risk)                │
│    4. Momentum (trend-dependent, higher drawdowns)              │
└─────────────────────────────────────────────────────────────────┘
""")

plt.close("all")
print(f"\n✅  Analysis 06 complete. Charts saved to: {CHARTS_DIR}")
