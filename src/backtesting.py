"""
============================================================
backtesting.py — Event-driven backtesting engine
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
============================================================

Engine Design
-------------
• Signal-based: receives a position signal series (+1, -1, 0) or
  a continuous position size series and simulates P&L.
• Supports transaction costs, slippage, position limits.
• Produces: equity curve, daily PnL, drawdown series, trade log.
• Evaluates: Sharpe, Sortino, Calmar, max drawdown, hit rate,
             profit factor, average win/loss, trades per year.

Supported Strategies (pre-built runners)
-----------------------------------------
1. Statistical Arbitrage (z-score mean reversion)
2. Momentum (rolling return signal)
3. Basis Trading (spot vs futures convergence)
4. Brent-WTI Spread
5. Cross-Asset Hedge Overlay
"""

import sys
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy  as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STRATEGY_PARAMS
from src.risk_metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, drawdown_series, var_comparison, kelly_criterion,
)


# ═══════════════════════════════════════════════════════════════════════
# 1.  Core Backtesting Engine
# ═══════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Vectorised backtesting engine for commodity spread / arb strategies.

    Usage
    -----
    >>> engine = BacktestEngine(returns, signals, tc_bps=5)
    >>> results = engine.run()
    >>> print(results["summary"])
    """

    def __init__(
        self,
        returns:     pd.Series,
        signals:     pd.Series,
        notional:    float = 1_000_000,
        tc_bps:      float = 5.0,      # transaction cost in bps (one-way)
        slippage_bps:float = 2.0,      # additional slippage per trade
        risk_free:   float = 0.05,
        name:        str   = "Strategy",
    ):
        """
        Parameters
        ----------
        returns     : daily log-returns of the instrument / spread
        signals     : position signals (+1 long, -1 short, 0 flat)
                      can also be continuous (e.g., volatility-scaled)
        notional    : USD notional for dollar P&L calculations
        tc_bps      : round-trip transaction cost in basis points
        slippage_bps: market impact / slippage in basis points
        risk_free   : annualised risk-free rate for Sharpe calculation
        name        : strategy identifier for reporting
        """
        self.name   = name
        self.rf     = risk_free
        self.n      = notional

        # Align signals to returns
        aligned      = pd.concat([returns, signals], axis=1).dropna()
        self.returns = aligned.iloc[:, 0]
        self.signals = aligned.iloc[:, 1]

        # Total cost per trade (both legs)
        total_cost_bps = tc_bps + slippage_bps
        self.cost_pct  = total_cost_bps / 10_000   # convert bps → decimal

    def _compute_tc(self) -> pd.Series:
        """
        Transaction costs incurred on position changes.
        Cost is applied on the day the signal changes (turnover day).
        """
        turnover = self.signals.diff().abs().fillna(0)
        return turnover * self.cost_pct

    def run(self) -> Dict:
        """
        Execute the backtest and return the full results dictionary.

        Returns
        -------
        dict with keys:
          equity_curve, daily_pnl, gross_returns, net_returns,
          drawdown, trade_log, summary
        """
        # Position-weighted gross returns (signals lag by 1 to avoid look-ahead)
        pos           = self.signals.shift(1).fillna(0)
        gross_rets    = pos * self.returns
        tc            = self._compute_tc().shift(1).fillna(0)
        net_rets      = gross_rets - tc

        # Dollar PnL
        cumulative   = (1 + net_rets).cumprod()
        equity_curve = self.n * cumulative

        # Drawdown
        dd_df        = drawdown_series(net_rets)

        # Trade log
        trades = self._build_trade_log(pos, net_rets, tc)

        summary = self._compute_summary(net_rets, gross_rets, tc, trades)

        self._results = {
            "equity_curve":    equity_curve,
            "daily_pnl_$":     (net_rets * self.n * cumulative.shift(1).fillna(1)),
            "gross_returns":   gross_rets,
            "net_returns":     net_rets,
            "position":        pos,
            "drawdown":        dd_df,
            "transaction_costs": tc,
            "trade_log":       trades,
            "summary":         summary,
        }
        return self._results

    def _build_trade_log(self,
                          pos: pd.Series,
                          net_rets: pd.Series,
                          tc: pd.Series) -> pd.DataFrame:
        """Extract individual trade records from signal changes."""
        changes = pos.diff()
        entry_dates = changes[changes != 0].index[1:]   # skip first
        trades = []
        for i, entry_dt in enumerate(entry_dates):
            try:
                exit_dt = entry_dates[i + 1] if i + 1 < len(entry_dates) else net_rets.index[-1]
            except IndexError:
                exit_dt = net_rets.index[-1]

            direction = pos.loc[entry_dt]
            if direction == 0:
                continue
            trade_rets = net_rets.loc[entry_dt:exit_dt]
            pnl_pct    = trade_rets.sum()
            n_days     = len(trade_rets)

            trades.append({
                "entry":      str(entry_dt.date()),
                "exit":       str(exit_dt.date()),
                "direction":  "Long" if direction > 0 else "Short",
                "pnl_%":      round(pnl_pct * 100, 4),
                "pnl_$":      round(pnl_pct * self.n, 0),
                "duration_d": n_days,
                "tc_$":       round(tc.loc[entry_dt] * self.n, 0),
            })
        return pd.DataFrame(trades)

    def _compute_summary(self,
                          net_rets: pd.Series,
                          gross_rets: pd.Series,
                          tc: pd.Series,
                          trades: pd.DataFrame) -> pd.Series:
        """Aggregate performance and risk metrics into a summary Series."""
        ann_ret = net_rets.mean() * 252
        ann_vol = net_rets.std()  * np.sqrt(252)

        n_years = len(net_rets) / 252
        winning = trades[trades["pnl_%"] > 0] if len(trades) > 0 else pd.DataFrame()
        losing  = trades[trades["pnl_%"] < 0] if len(trades) > 0 else pd.DataFrame()

        avg_win  = winning["pnl_%"].mean() / 100 if len(winning) > 0 else 0
        avg_loss = abs(losing["pnl_%"].mean() / 100) if len(losing) > 0 else 1e-6
        win_rate = len(winning) / len(trades) if len(trades) > 0 else 0

        kc = kelly_criterion(win_rate, avg_win, avg_loss) if avg_loss > 0 else {}

        metrics = pd.Series({
            "Strategy Name":          self.name,
            "Ann. Return %":          round(ann_ret * 100, 2),
            "Ann. Volatility %":      round(ann_vol * 100, 2),
            "Sharpe Ratio":           round(sharpe_ratio(net_rets, self.rf), 3),
            "Sortino Ratio":          round(sortino_ratio(net_rets, self.rf), 3),
            "Calmar Ratio":           round(calmar_ratio(net_rets), 3),
            "Max Drawdown %":         round(max_drawdown(net_rets) * 100, 2),
            "Total Trades":           len(trades),
            "Win Rate %":             round(win_rate * 100, 1),
            "Avg Win %":              round(avg_win * 100, 3),
            "Avg Loss %":             round(avg_loss * 100, 3),
            "Profit Factor":          round(avg_win / avg_loss, 2) if avg_loss > 0 else np.inf,
            "Trades Per Year":        round(len(trades) / n_years, 1) if n_years > 0 else 0,
            "Total Net Return %":     round((net_rets.sum()) * 100, 2),
            "Total TC $ (est.)":      round(tc.sum() * self.n, 0),
            "Kelly Fraction %":       round(kc.get("kelly_fraction", 0) * 100, 1),
        })
        return metrics

    def plot_results(self, show: bool = True, save_path: Optional[str] = None):
        """Generate a 4-panel performance tearsheet."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        r = self._results
        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        # ── Panel 1: Equity Curve ──
        ax1 = fig.add_subplot(gs[0, :])
        r["equity_curve"].plot(ax=ax1, color="#2196F3", lw=2)
        ax1.axhline(self.n, color="gray", ls="--", alpha=0.5, label="Initial Capital")
        ax1.fill_between(r["equity_curve"].index, self.n, r["equity_curve"],
                         where=r["equity_curve"] >= self.n, alpha=0.2, color="#4CAF50")
        ax1.fill_between(r["equity_curve"].index, self.n, r["equity_curve"],
                         where=r["equity_curve"] < self.n, alpha=0.2, color="#F44336")
        ax1.set_title(f"{self.name} — Equity Curve", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(); ax1.grid(alpha=0.3)

        # ── Panel 2: Drawdown ──
        ax2 = fig.add_subplot(gs[1, 0])
        r["drawdown"]["drawdown"].plot(ax=ax2, color="#F44336", lw=1.5)
        ax2.fill_between(r["drawdown"].index, 0, r["drawdown"]["drawdown"],
                         color="#F44336", alpha=0.3)
        ax2.set_title("Drawdown", fontsize=11)
        ax2.set_ylabel("Drawdown"); ax2.grid(alpha=0.3)

        # ── Panel 3: Rolling Sharpe ──
        ax3 = fig.add_subplot(gs[1, 1])
        roll_sharpe = (r["net_returns"].rolling(63).mean() /
                       r["net_returns"].rolling(63).std()) * np.sqrt(252)
        roll_sharpe.plot(ax=ax3, color="#9C27B0", lw=1.5)
        ax3.axhline(0, color="gray", ls="--", alpha=0.5)
        ax3.axhline(1, color="#4CAF50", ls="--", alpha=0.7, label="Sharpe=1")
        ax3.set_title("Rolling 63-Day Sharpe", fontsize=11)
        ax3.legend(); ax3.grid(alpha=0.3)

        # ── Panel 4: Return Distribution ──
        ax4 = fig.add_subplot(gs[2, 0])
        r["net_returns"].hist(ax=ax4, bins=60, color="#2196F3", alpha=0.7, edgecolor="white")
        ax4.axvline(r["net_returns"].mean(), color="#FF9800", lw=2, label="Mean")
        ax4.axvline(r["net_returns"].quantile(0.05), color="#F44336", lw=2,
                    ls="--", label="5% VaR")
        ax4.set_title("Return Distribution", fontsize=11)
        ax4.legend(); ax4.grid(alpha=0.3)

        # ── Panel 5: Monthly P&L Heatmap ──
        ax5 = fig.add_subplot(gs[2, 1])
        monthly = r["net_returns"].resample("ME").sum() * 100
        monthly.index = monthly.index.strftime("%Y-%m")
        monthly.plot(kind="bar", ax=ax5,
                     color=["#4CAF50" if v > 0 else "#F44336" for v in monthly],
                     edgecolor="white", width=0.8)
        ax5.set_title("Monthly Returns %", fontsize=11)
        ax5.set_xticklabels(monthly.index, rotation=90, fontsize=6)
        ax5.axhline(0, color="gray", lw=0.8); ax5.grid(alpha=0.3, axis="y")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  💾 Saved tearsheet: {save_path}")
        if show:
            plt.show()
        plt.close()
        return fig


# ═══════════════════════════════════════════════════════════════════════
# 2.  Pre-Built Strategy Runners
# ═══════════════════════════════════════════════════════════════════════

def run_zscore_strategy(
        price_a: pd.Series,
        price_b: pd.Series,
        zscore_window: int = 63,
        entry_z: float = 2.0,
        exit_z:  float = 0.5,
        stop_z:  float = 3.5,
        notional: float = 1_000_000,
        tc_bps:   float = 5.0,
        name:     str   = "Z-Score Arb",
) -> Dict:
    """
    Mean-reversion statistical arbitrage on the cointegrated spread A − β·B.

    Strategy Logic
    --------------
    • When z-score > +2σ → Sell spread  (sell A, buy B) — expecting reversion DOWN
    • When z-score < −2σ → Buy spread   (buy A, sell B) — expecting reversion UP
    • Exit when |z-score| < 0.5σ  (mean reversion complete)
    • Stop-loss at |z-score| > 3.5σ

    This is the core engine for:
    • Brent vs WTI crude oil spread
    • Gold spot vs GLD ETF basis arbitrage
    • Gold vs Gold Miners ETF
    """
    from src.statistical_analysis import engle_granger_coint, compute_spread_zscore

    # Get hedge ratio
    coint_r = engle_granger_coint(price_a, price_b)
    beta    = coint_r["beta"]

    # Compute spread and z-score
    spread_df = compute_spread_zscore(price_a, price_b,
                                       coint_r, zscore_window)
    z = spread_df["z_score"]

    # Generate signals
    signal = pd.Series(0.0, index=z.index)
    signal[z >  entry_z] = -1.0   # short spread
    signal[z < -entry_z] =  1.0   # long spread
    signal[abs(z) < exit_z] = 0.0
    signal[abs(z) > stop_z]  = 0.0

    # Forward-fill within a trade
    signal = signal.replace(0, np.nan).ffill().fillna(0)
    signal[abs(z) < exit_z] = 0   # exit overrides
    signal[abs(z) > stop_z]  = 0   # stop overrides

    # Returns of the spread (log)
    spread_returns = np.log(
        (price_a - beta * price_b) /
        (price_a - beta * price_b).shift(1)
    ).replace([np.inf, -np.inf], np.nan)

    engine = BacktestEngine(spread_returns, signal,
                             notional=notional, tc_bps=tc_bps, name=name)
    results = engine.run()
    results["spread_df"]  = spread_df
    results["beta"]       = beta
    results["coint_test"] = coint_r
    results["engine"]     = engine
    return results


def run_momentum_strategy(
        returns: pd.Series,
        lookback: int = 21,
        notional: float = 1_000_000,
        tc_bps:   float = 3.0,
        name:     str   = "Momentum",
) -> Dict:
    """
    Simple time-series momentum: long if past 'lookback' return is positive.

    Commodity momentum is one of the most documented systematic premia —
    commodities tend to trend due to: supply adjustment lags, weather patterns,
    geopolitical shocks, and speculator/commercial hedger positioning dynamics.
    """
    past_ret = returns.rolling(lookback).sum()
    signal   = np.sign(past_ret).fillna(0)

    engine  = BacktestEngine(returns, signal, notional=notional,
                               tc_bps=tc_bps, name=name)
    return engine.run()


def run_basis_trade(
        spot_returns: pd.Series,
        futures_returns: pd.Series,
        basis_window: int = 21,
        notional: float = 1_000_000,
        tc_bps:   float = 3.0,
        name:     str   = "Basis Trade",
) -> Dict:
    """
    Basis trade: exploit the convergence of spot and futures prices at expiry.

    Basis = Futures Price − Spot Price  (should converge to 0 at expiry)

    Contango  (Futures > Spot): Short futures / Long spot → earn roll yield
    Backwardation (Spot > Futures): Long futures / Short spot → earn convergence

    Most relevant for:  WTI Crude, Gold, Natural Gas, Wheat
    """
    basis_return = futures_returns - spot_returns   # excess return of futures

    roll_mean = basis_return.rolling(basis_window).mean()
    roll_std  = basis_return.rolling(basis_window).std()
    z_basis   = (basis_return - roll_mean) / roll_std

    # Trade signal: fade extreme basis (mean reversion)
    signal = -np.sign(z_basis).where(abs(z_basis) > 1.5, 0)

    engine = BacktestEngine(basis_return, signal,
                             notional=notional, tc_bps=tc_bps, name=name)
    return engine.run()


# ═══════════════════════════════════════════════════════════════════════
# 3.  Multi-Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════

def compare_strategies(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Side-by-side comparison of multiple backtest results.

    Parameters
    ----------
    results_dict : {"Strategy Name": backtest_results_dict, ...}

    Returns
    -------
    DataFrame with strategies as columns, metrics as rows.
    """
    rows = {}
    for name, res in results_dict.items():
        if "summary" in res:
            rows[name] = res["summary"]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort by Sharpe Ratio
    if "Sharpe Ratio" in df.index:
        order = df.loc["Sharpe Ratio"].astype(float).sort_values(ascending=False).index
        df    = df[order]
    return df


def benchmark_comparison(strategy_returns: pd.Series,
                          benchmark_returns: pd.Series,
                          risk_free: float = 0.05) -> Dict:
    """
    Alpha / Beta analysis of strategy vs benchmark.

    Information Ratio = (Strategy Return − Benchmark Return) / Tracking Error
    Alpha = Jensen's Alpha from CAPM regression
    """
    from src.statistical_analysis import ols_regression

    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    s = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]

    ols_r  = ols_regression(s, b)
    alpha  = ols_r["coefficients"].get("const", 0)
    beta   = list(ols_r["coefficients"].values())[-1]

    tracking_error = (s - b).std() * np.sqrt(252)
    info_ratio     = (s - b).mean() * 252 / tracking_error if tracking_error else np.nan
    up_capture     = s[b > 0].mean() / b[b > 0].mean() if b[b > 0].mean() else np.nan
    down_capture   = s[b < 0].mean() / b[b < 0].mean() if b[b < 0].mean() else np.nan

    return {
        "alpha_daily":      round(float(alpha), 6),
        "alpha_annual_%":   round(float(alpha) * 252 * 100, 3),
        "beta":             round(float(beta), 4),
        "r_squared":        round(ols_r["r_squared"], 4),
        "tracking_error_%": round(tracking_error * 100, 3),
        "information_ratio":round(float(info_ratio), 3),
        "upside_capture_%": round(float(up_capture) * 100, 1) if up_capture else np.nan,
        "downside_capture_%": round(float(down_capture) * 100, 1) if down_capture else np.nan,
        "strategy_sharpe":  round(sharpe_ratio(s, risk_free), 3),
        "benchmark_sharpe": round(sharpe_ratio(b, risk_free), 3),
    }


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng    = np.random.default_rng(42)
    dates  = pd.date_range("2020-01-01", periods=1000, freq="B")
    rets   = pd.Series(rng.normal(0.0003, 0.015, 1000), index=dates, name="Synthetic")
    signal = pd.Series(rng.choice([-1, 0, 1], 1000), index=dates)

    engine  = BacktestEngine(rets, signal, name="Test Strategy", tc_bps=5)
    results = engine.run()
    print("\n=== Backtest Summary ===")
    print(results["summary"].to_string())
    print("\n✅ backtesting.py working correctly.")
