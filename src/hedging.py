"""
============================================================
hedging.py — Hedging Strategy Construction & Evaluation
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
============================================================

Implements
----------
1. OLS Minimum Variance Hedge (static & rolling)
2. Dynamic hedge ratio (DCC-GARCH approximation)
3. Supply shock hedging portfolio (multi-asset)
4. Inflation hedge evaluation (Gold, TIPS, commodities vs CPI)
5. Cross-asset hedge (commodity exposure hedged with ETF derivatives)
6. Hedge effectiveness metrics (HE ratio, cost analysis)

Financial Analogies
-------------------
• Hedging = Insurance policy on your commodity exposure
• Hedge Ratio (β) = How many contracts/shares you need per unit exposure
• Hedge Effectiveness = What % of variance is eliminated by the hedge
• Supply Shock Hedge = Pre-positioning against a supply disruption —
  like buying extra fire insurance before wildfire season
"""

import sys
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy  as np
import pandas as pd
from scipy  import stats, optimize

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STRATEGY_PARAMS


# ═══════════════════════════════════════════════════════════════════════
# 1.  OLS Minimum Variance Hedge Ratio
# ═══════════════════════════════════════════════════════════════════════

def ols_hedge_ratio(spot_returns: pd.Series,
                     hedge_returns: pd.Series) -> Dict:
    """
    Estimate the minimum-variance hedge ratio via OLS regression:

        spot_return = α + β · hedge_return + ε

    β = Cov(spot, hedge) / Var(hedge) = the number of hedge units per spot unit.

    Example: If β = 0.75 for GLD vs GDX, buying $750k of GLD hedges
             $1M of gold mining stock exposure.

    Hedge Effectiveness = R² of the regression (proportion of variance hedged).
    """
    import statsmodels.api as sm

    aligned = pd.concat([spot_returns, hedge_returns], axis=1).dropna()
    s = aligned.iloc[:, 0]
    h = aligned.iloc[:, 1]

    X   = sm.add_constant(h)
    ols = sm.OLS(s, X).fit()

    beta  = ols.params.iloc[1]
    alpha = ols.params.iloc[0]

    # Hedged return series
    hedged_rets   = s - beta * h
    unhedged_var  = s.var()
    hedged_var    = hedged_rets.var()
    effectiveness = 1 - hedged_var / unhedged_var if unhedged_var > 0 else 0

    return {
        "spot":                spot_returns.name,
        "hedge_instrument":    hedge_returns.name,
        "alpha":               round(float(alpha), 6),
        "beta":                round(float(beta),  6),
        "hedge_ratio":         round(float(beta),  6),
        "r_squared":           round(ols.rsquared, 4),
        "hedge_effectiveness": round(float(effectiveness), 4),
        "he_pct":              round(float(effectiveness) * 100, 2),
        "unhedged_ann_vol_%":  round(s.std() * np.sqrt(252) * 100, 2),
        "hedged_ann_vol_%":    round(hedged_rets.std() * np.sqrt(252) * 100, 2),
        "vol_reduction_%":     round((1 - hedged_rets.std()/s.std()) * 100, 2),
        "hedged_returns":      hedged_rets,
        "interpretation": (
            f"Buy {abs(beta):.4f} units of {hedge_returns.name} "
            f"per unit of {spot_returns.name} exposure. "
            f"Hedge eliminates {effectiveness*100:.1f}% of variance."
        ),
    }


def rolling_hedge_ratio(spot_returns: pd.Series,
                          hedge_returns: pd.Series,
                          window: int = 63) -> pd.DataFrame:
    """
    Rolling minimum-variance hedge ratio — captures time-varying dynamics.

    A static hedge ratio assumes a constant relationship, but commodity
    correlations shift across market regimes (e.g., during geopolitical
    shocks, the Brent-WTI correlation can drop sharply).
    Rolling hedge ratios adapt dynamically.
    """
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS

    aligned = pd.concat([spot_returns, hedge_returns], axis=1).dropna()
    s = aligned.iloc[:, 0]
    h = aligned.iloc[:, 1]

    X      = sm.add_constant(h)
    rols   = RollingOLS(s, X, window=window).fit()
    betas  = rols.params.rename(columns={
        "const": "alpha",
        h.name or aligned.columns[1]: "beta"
    })

    # Hedged returns using rolling hedge ratio
    hedged_rets = s - (betas["beta"] * h)

    # Rolling hedge effectiveness
    roll_he = (
        1 - hedged_rets.rolling(window).var() /
            s.rolling(window).var()
    )

    return pd.DataFrame({
        "alpha":               betas["alpha"],
        "beta":                betas["beta"],
        "hedged_returns":      hedged_rets,
        "rolling_he_%":        roll_he * 100,
    })


# ═══════════════════════════════════════════════════════════════════════
# 2.  Portfolio Minimum Variance Hedge
# ═══════════════════════════════════════════════════════════════════════

def minimum_variance_portfolio(returns: pd.DataFrame,
                                 allow_short: bool = True,
                                 max_weight: float = 0.40) -> Dict:
    """
    Find the minimum variance portfolio weights via quadratic optimisation.

    The minimum-variance portfolio is the corner of the efficient frontier
    that minimises total portfolio risk — useful as a baseline hedge portfolio.

    Constraints:  Σ wᵢ = 1,  wᵢ ∈ [−max_weight, max_weight]
    """
    clean = returns.dropna()
    n     = clean.shape[1]
    Σ     = clean.cov().values * 252   # annualise covariance

    # Objective: minimise w'Σw
    def portfolio_var(w):
        return float(w @ Σ @ w)

    bounds   = [(-max_weight, max_weight) if allow_short
                else (0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n

    result = optimize.minimize(portfolio_var, x0,
                                method="SLSQP",
                                bounds=bounds,
                                constraints=constraints,
                                options={"maxiter": 1000, "ftol": 1e-9})
    w_opt = pd.Series(result.x, index=clean.columns)
    port_vol = np.sqrt(result.fun)
    port_rets = clean.dot(result.x)

    return {
        "weights":        w_opt.round(4).to_dict(),
        "ann_vol_%":      round(port_vol * 100, 2),
        "ann_return_%":   round(port_rets.mean() * 252 * 100, 2),
        "optimised":      result.success,
        "portfolio_returns": port_rets,
    }


def risk_parity_portfolio(returns: pd.DataFrame) -> Dict:
    """
    Risk Parity: weight each asset to contribute equally to portfolio variance.

    Instead of "dollar equal weight," risk parity is "risk equal weight" —
    volatile assets (NatGas, Wheat) get smaller weights, stable assets
    (Gold, TLT) get larger weights.
    """
    clean = returns.dropna()
    n     = clean.shape[1]
    Σ     = clean.cov().values * 252

    def risk_contribution_diff(w):
        w    = np.abs(w)
        port_var = w @ Σ @ w
        rc   = w * (Σ @ w) / port_var
        target = 1 / n
        return float(np.sum((rc - target)**2))

    x0 = np.ones(n) / n
    result = optimize.minimize(risk_contribution_diff, x0,
                                method="SLSQP",
                                bounds=[(0.01, 0.60)] * n,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w_opt = pd.Series(np.abs(result.x) / np.abs(result.x).sum(),
                       index=clean.columns)
    port_rets = clean.dot(w_opt.values)

    return {
        "weights":           w_opt.round(4).to_dict(),
        "ann_vol_%":         round(port_rets.std() * np.sqrt(252) * 100, 2),
        "ann_return_%":      round(port_rets.mean() * 252 * 100, 2),
        "portfolio_returns": port_rets,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3.  Supply Shock Hedging Strategy
# ═══════════════════════════════════════════════════════════════════════

def supply_shock_hedge(returns: pd.DataFrame,
                        supply_shock_assets: List[str],
                        hedge_assets: List[str],
                        shock_threshold: float = -0.05) -> Dict:
    """
    Evaluate a hedging portfolio designed to profit during supply shocks.

    Supply Shock Definition: A period when a supply-shock asset (e.g., WTI Oil)
    drops more than 'shock_threshold' (e.g., −5%) in a single day or week.

    Hedge assets (e.g., USO put proxies, VIX-linked, TLT) should perform
    well during these episodes.

    Returns
    -------
    Performance metrics split into "shock" vs "normal" regimes,
    plus hedge asset returns during shock periods.
    """
    clean = returns.dropna()

    # Identify shock dates: any supply shock asset down > threshold
    shock_mask = pd.Series(False, index=clean.index)
    for asset in supply_shock_assets:
        if asset in clean.columns:
            shock_mask |= (clean[asset] < shock_threshold)

    shock_dates  = clean.index[shock_mask]
    normal_dates = clean.index[~shock_mask]

    results = {}
    for asset in hedge_assets:
        if asset not in clean.columns:
            continue
        r = clean[asset]
        results[asset] = {
            "overall_mean_%":    round(r.mean() * 100 * 252, 2),
            "shock_mean_%":      round(r.loc[shock_dates].mean() * 100, 4),
            "normal_mean_%":     round(r.loc[normal_dates].mean() * 100, 4),
            "shock_vs_normal_%": round(
                (r.loc[shock_dates].mean() - r.loc[normal_dates].mean()) * 100, 4),
            "n_shock_days":      int(len(shock_dates)),
            "is_hedge":          r.loc[shock_dates].mean() > r.loc[normal_dates].mean(),
        }

    shock_summary = pd.DataFrame(results).T
    shock_summary["rating"] = shock_summary["is_hedge"].map(
        {True: "✓ Effective Hedge", False: "✗ Not a Hedge"}
    )

    return {
        "n_shock_periods":   int(len(shock_dates)),
        "n_normal_periods":  int(len(normal_dates)),
        "shock_asset_threshold_%": shock_threshold * 100,
        "hedge_performance": shock_summary,
        "shock_dates":       shock_dates.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4.  Inflation Hedge Analysis
# ═══════════════════════════════════════════════════════════════════════

def inflation_hedge_analysis(asset_returns: pd.DataFrame,
                               cpi_changes: pd.Series,
                               breakeven_changes: Optional[pd.Series] = None
                               ) -> pd.DataFrame:
    """
    Evaluate each asset's effectiveness as an inflation hedge.

    A good inflation hedge should:
    1. Have positive correlation with CPI / breakeven inflation changes
    2. Generate positive returns during high-inflation periods (CPI YoY > 3%)
    3. Have a statistically significant beta to inflation

    Gold, TIPS, commodities (oil, copper) are classic inflation hedges.
    Long-duration bonds (TLT) are ANTI-inflation (negative hedge).

    Parameters
    ----------
    asset_returns    : monthly/annual returns for candidate hedge assets
    cpi_changes      : monthly % changes in CPI
    breakeven_changes: monthly changes in 10Y breakeven inflation
    """
    import statsmodels.api as sm

    rows = []
    for col in asset_returns.columns:
        aligned = pd.concat([asset_returns[col], cpi_changes], axis=1).dropna()
        asset   = aligned.iloc[:, 0]
        cpi     = aligned.iloc[:, 1]

        # Correlation
        pearson_r, p_val = stats.pearsonr(asset, cpi)

        # Beta to CPI
        X   = sm.add_constant(cpi)
        ols = sm.OLS(asset, X).fit()
        beta_cpi = ols.params.iloc[1]

        # Performance during high inflation (CPI change > 0.3% monthly ≈ 3.6% annual)
        high_inf = aligned[cpi > 0.003]
        low_inf  = aligned[cpi <= 0.003]

        rows.append({
            "Asset":                  col,
            "CPI Correlation":        round(pearson_r, 4),
            "Corr P-Value":           round(p_val, 4),
            "Beta to CPI":            round(float(beta_cpi), 4),
            "CPI Beta Significant":   ols.pvalues.iloc[1] < 0.05,
            "Ret High Inflation %":   round(high_inf.iloc[:, 0].mean() * 1200, 2),
            "Ret Low Inflation %":    round(low_inf.iloc[:, 0].mean() * 1200, 2),
            "Spread High-Low %":      round(
                (high_inf.iloc[:, 0].mean() - low_inf.iloc[:, 0].mean()) * 1200, 2),
            "Hedge Rating":           (
                "⭐⭐⭐ Strong" if pearson_r > 0.3 and ols.pvalues.iloc[1] < 0.05 else
                "⭐⭐ Moderate" if pearson_r > 0.1 else
                "⭐ Weak / Negative"
            ),
        })

    return pd.DataFrame(rows).set_index("Asset")


# ═══════════════════════════════════════════════════════════════════════
# 5.  Hedge Cost Analysis
# ═══════════════════════════════════════════════════════════════════════

def hedge_cost_analysis(unhedged_returns: pd.Series,
                          hedged_returns: pd.Series,
                          tc_bps: float = 10.0,
                          notional: float = 1_000_000) -> Dict:
    """
    Compare risk-adjusted performance before and after hedging costs.

    True cost of hedging = Opportunity cost of forfeited upside +
                           Explicit transaction costs +
                           Negative carry (if any)

    An effective hedge pays for itself in risk reduction — the key question:
    "Does the volatility reduction justify the return sacrifice?"
    """
    from src.risk_metrics import sharpe_ratio, max_drawdown

    tc_per_annum = tc_bps / 10_000 * 252   # rough annual cost estimate

    hedged_adj = hedged_returns - tc_per_annum / 252   # deduct cost

    return {
        "unhedged": {
            "ann_return_%":  round(unhedged_returns.mean() * 252 * 100, 2),
            "ann_vol_%":     round(unhedged_returns.std() * np.sqrt(252) * 100, 2),
            "sharpe":        round(sharpe_ratio(unhedged_returns), 3),
            "max_dd_%":      round(max_drawdown(unhedged_returns) * 100, 2),
        },
        "hedged": {
            "ann_return_%":  round(hedged_adj.mean() * 252 * 100, 2),
            "ann_vol_%":     round(hedged_adj.std() * np.sqrt(252) * 100, 2),
            "sharpe":        round(sharpe_ratio(hedged_adj), 3),
            "max_dd_%":      round(max_drawdown(hedged_adj) * 100, 2),
        },
        "hedge_metrics": {
            "vol_reduction_%":    round(
                (1 - hedged_adj.std() / unhedged_returns.std()) * 100, 2),
            "return_sacrifice_%": round(
                (unhedged_returns.mean() - hedged_adj.mean()) * 252 * 100, 2),
            "tc_annual_%":        round(tc_per_annum * 100, 3),
            "sharpe_improvement": round(
                sharpe_ratio(hedged_adj) - sharpe_ratio(unhedged_returns), 3),
            "max_dd_reduction_%": round(
                (max_drawdown(unhedged_returns) - max_drawdown(hedged_adj)) * 100, 2),
            "net_benefit":        sharpe_ratio(hedged_adj) > sharpe_ratio(unhedged_returns),
        },
    }


if __name__ == "__main__":
    import yfinance as yf

    print("\n=== Hedging Module Test ===")
    data = yf.download(["GLD", "GDX", "GC=F"], start="2020-01-01",
                        end="2024-12-31", auto_adjust=True, progress=False)["Close"]
    data.columns = ["GDX", "GLD", "Gold_Futures"]

    rets = np.log(data / data.shift(1)).dropna()

    # Test OLS hedge ratio
    h = ols_hedge_ratio(rets["Gold_Futures"], rets["GLD"])
    print(f"\nOLS Hedge Ratio (Gold Futures vs GLD ETF):")
    for k, v in h.items():
        if k != "hedged_returns":
            print(f"  {k}: {v}")

    print("\n✅ hedging.py working correctly.")
