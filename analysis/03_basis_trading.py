"""
============================================================
03_basis_trading.py
Spot vs Futures Basis Trading Analysis
============================================================

Concept
-------
BASIS = Futures Price − Spot Price

In commodity markets, the basis reflects:
  • Storage costs (carry)
  • Convenience yield (benefit of holding physical commodity)
  • Transportation costs
  • Supply/demand imbalances

Contango:     Futures > Spot → Negative roll yield for long holders
Backwardation: Spot > Futures → Positive roll yield (rare, signals supply tightness)

"Basis trading" = exploiting predictable convergence of futures to spot at expiry.
Analogy: Buying a concert ticket at face value (spot) when the resale (futures)
         price is $50 higher — you profit as the premium erodes before the show.

Run
---
    python analysis/03_basis_trading.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
from tabulate import tabulate

from config import CHARTS_DIR, COLOR_PALETTE, STRATEGY_PARAMS
from src.data_loader         import get_commodity_data, compute_returns
from src.statistical_analysis import (
    engle_granger_coint, compute_spread_zscore,
    adf_test, rolling_correlation,
)
from src.backtesting  import BacktestEngine, run_basis_trade
from src.risk_metrics import risk_report, var_comparison
from src.visualization import plot_spread_zscore, plot_var_distribution

COLORS = COLOR_PALETTE

print("\n" + "="*70)
print("  ANALYSIS 03 — Spot vs Futures Basis Trading")
print("="*70)

# ── 1. Load Data ──────────────────────────────────────────────────────
print("\n[1/6] Loading data …")
data = get_commodity_data(use_cache=True)
all_prices  = data["all_prices"]
all_rets    = data["all_returns"]

# Key basis pairs: Futures ticker vs ETF (closest available proxy for spot)
basis_pairs = [
    ("Gold_Spot",  "Gold_ETF",   "Gold: GC=F vs GLD"),
    ("WTI_Oil",    "Oil_ETF",    "WTI Oil: CL=F vs USO"),
    ("NatGas",     "NatGas_ETF", "NatGas: NG=F vs UNG"),
    ("Copper",     "Copper_ETF", "Copper: HG=F vs CPER"),
]
basis_pairs = [(a, b, lbl) for a, b, lbl in basis_pairs
               if a in all_prices.columns and b in all_prices.columns]

print(f"  ✓ Analysing {len(basis_pairs)} basis pairs")

# ── 2. Basis Level Analysis ───────────────────────────────────────────
print("\n[2/6] Computing basis levels and statistics …")

basis_stats = []
for futures_col, spot_col, label in basis_pairs:
    # Align prices
    aligned = pd.concat([all_prices[futures_col], all_prices[spot_col]],
                         axis=1).dropna()
    fut_price  = aligned.iloc[:, 0]
    spot_price = aligned.iloc[:, 1]

    # Compute basis (raw and normalised)
    basis     = fut_price - spot_price
    basis_pct = basis / spot_price * 100

    # ADF on basis — should be stationary if basis is mean-reverting
    adf_r = adf_test(basis.dropna())

    basis_stats.append({
        "Pair":              label,
        "Mean Basis":        round(basis.mean(), 4),
        "Std Basis":         round(basis.std(),  4),
        "Mean Basis %":      round(basis_pct.mean(), 4),
        "Std Basis %":       round(basis_pct.std(),  4),
        "Min Basis":         round(basis.min(),  4),
        "Max Basis":         round(basis.max(),  4),
        "ADF p-value":       adf_r["p_value"],
        "Basis Stationary":  adf_r["stationary_at_5pct"],
        "Contango %":        round((basis > 0).mean() * 100, 1),
        "Backwardation %":   round((basis < 0).mean() * 100, 1),
    })

print("\n  📊 Basis Statistics:")
basis_df = pd.DataFrame(basis_stats)
print(tabulate(basis_df, headers="keys", tablefmt="fancy_grid",
               showindex=False, floatfmt=".4f"))

# ── 3. Basis Structure Charts ─────────────────────────────────────────
print("\n[3/6] Plotting basis structure …")

fig, axes = plt.subplots(len(basis_pairs), 2,
                          figsize=(16, 4 * len(basis_pairs)))
if len(basis_pairs) == 1:
    axes = axes.reshape(1, -1)

for i, (fut_col, spot_col, label) in enumerate(basis_pairs):
    aligned = pd.concat([all_prices[fut_col], all_prices[spot_col]], axis=1).dropna()
    fut_p   = aligned.iloc[:, 0]
    spot_p  = aligned.iloc[:, 1]
    basis   = fut_p - spot_p
    basis_z = (basis - basis.rolling(63).mean()) / basis.rolling(63).std()

    # Left: Basis level
    ax_l = axes[i, 0]
    ax_l.plot(basis.index, basis, color=COLORS[i % len(COLORS)], lw=1.3)
    ax_l.axhline(basis.mean(), color="gray", ls="--", lw=1, label=f"Mean: {basis.mean():.2f}")
    ax_l.fill_between(basis.index, 0, basis,
                       where=basis > 0, alpha=0.2, color=COLORS[2], label="Contango")
    ax_l.fill_between(basis.index, 0, basis,
                       where=basis < 0, alpha=0.2, color=COLORS[3], label="Backwardation")
    ax_l.set_title(f"{label} — Basis Level", fontsize=10, fontweight="bold")
    ax_l.legend(fontsize=8); ax_l.grid(alpha=0.3)

    # Right: Basis Z-Score
    ax_r = axes[i, 1]
    ax_r.plot(basis_z.index, basis_z, color=COLORS[(i+2) % len(COLORS)], lw=1.3)
    ax_r.axhline(2, color=COLORS[3], ls="--", lw=1)
    ax_r.axhline(-2, color=COLORS[2], ls="--", lw=1)
    ax_r.fill_between(basis_z.index, 2, basis_z.clip(lower=2),
                       where=basis_z > 2, alpha=0.2, color=COLORS[3])
    ax_r.fill_between(basis_z.index, basis_z.clip(upper=-2), -2,
                       where=basis_z < -2, alpha=0.2, color=COLORS[2])
    ax_r.set_title(f"{label} — Basis Z-Score", fontsize=10, fontweight="bold")
    ax_r.set_ylim(-4, 4); ax_r.grid(alpha=0.3)

plt.suptitle("Commodity Basis Structure Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_path = os.path.join(CHARTS_DIR, "03a_basis_structure.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"  💾 {save_path}")

# ── 4. Backtest: Basis Mean-Reversion Strategy ────────────────────────
print("\n[4/6] Backtesting basis mean-reversion strategy …")

backtest_results = {}
for fut_col, spot_col, label in basis_pairs:
    fut_rets  = all_rets[fut_col] if fut_col  in all_rets.columns else None
    spot_rets = all_rets[spot_col] if spot_col in all_rets.columns else None
    if fut_rets is None or spot_rets is None:
        continue

    try:
        res = run_basis_trade(spot_rets, fut_rets,
                               basis_window=21,
                               notional=1_000_000,
                               tc_bps=5,
                               name=f"Basis: {label}")
        backtest_results[label] = res
        s = res["summary"]
        print(f"\n  {label}:")
        print(f"    Sharpe: {s['Sharpe Ratio']:.3f}  |  Ann.Ret: {s['Ann. Return %']:.2f}%"
              f"  |  MaxDD: {s['Max Drawdown %']:.2f}%"
              f"  |  WinRate: {s['Win Rate %']:.1f}%")
    except Exception as e:
        print(f"  ⚠  {label}: {e}")

# ── 5. Risk Report ────────────────────────────────────────────────────
print("\n[5/6] Risk analysis — basis strategies …")

for label, res in backtest_results.items():
    net_rets = res["net_returns"]
    rpt = risk_report(net_rets)
    print(f"\n  Risk Report: {label}")
    risk_display = {k: v for k, v in rpt.items()
                    if k in ["annualised_return_%", "annualised_vol_%",
                              "sharpe_ratio", "max_drawdown_%",
                              "var_95_hist_%", "cvar_95_hist_%"]}
    for k, v in risk_display.items():
        print(f"    {k}: {v}")

# ── 6. Roll Yield Analysis ────────────────────────────────────────────
print("\n[6/6] Roll yield analysis …")

print("""
  ROLL YIELD EXPLAINED:
  ─────────────────────
  When a futures contract is held and "rolled" forward (sold before expiry,
  new contract bought), the roll yield is the profit/loss from this process:

  • Contango  (F > S): Roll yield is NEGATIVE — you sell cheap (near), buy expensive (far)
    Example: WTI often in contango during storage gluts (2020 COVID crash)
    → Long futures underperform spot by 3-8% p.a. in deep contango

  • Backwardation (S > F): Roll yield is POSITIVE — you sell expensive (near), buy cheap (far)
    Example: WTI in backwardation during supply shocks (2021-2022)
    → Long futures OUTPERFORM spot — time is on your side

  IMPLICATION FOR ETFs:
  Many commodity ETFs (USO, UNG) suffer persistent negative roll yield in contango.
  Sophisticated investors use ETFs like PDBC or CPER that use optimised roll strategies.
""")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  TRADING INSIGHTS — Basis Trading                               │
│  ─────────────────────────────────────────────────────────────  │
│  1. Gold basis is usually tight (< 0.1%) — dominated by        │
│     financing cost (LIBOR/SOFR + 0.05%). Any widening          │
│     above 0.5% signals market stress or ETF demand surge.       │
│  2. WTI basis varies widely — driven by Cushing storage levels  │
│     and pipeline capacity. Peak contango in April 2020:         │
│     negative $37/bbl on the May contract (historic).            │
│  3. NatGas basis is highly seasonal — Henry Hub spikes in       │
│     winter; mean-reversion from summer to winter spread is      │
│     a well-known seasonal trade in commodity CTAs.              │
│  4. Copper basis is geopolitically sensitive — LME vs COMEX     │
│     spreads widen during Chinese inventory drawdowns.           │
└─────────────────────────────────────────────────────────────────┘
""")

plt.close("all")
print(f"\n✅  Analysis 03 complete. Charts saved to: {CHARTS_DIR}")
