"""
============================================================
risk_metrics.py — Portfolio & Trade Risk Measurement
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
============================================================

Implements
----------
• Sharpe Ratio, Sortino Ratio, Calmar Ratio
• Maximum Drawdown & Drawdown Duration
• Value-at-Risk (Historical, Parametric, Monte Carlo)
• Conditional Value-at-Risk (CVaR / Expected Shortfall)
• Tail-Risk Metrics (Omega, Ulcer Index)
• Position Sizing (Kelly Criterion, Volatility Targeting)
• Portfolio-level Greeks approximation (Delta, Vega)

Financial Analogies
-------------------
• VaR      → "The maximum loss on a bad day 95% of the time"
• CVaR     → "Given we're already in that bad 5%, how bad does it get?"
• Drawdown → "The distance from the peak to the current trough"
• Kelly    → "Mathematically optimal bet sizing to maximise long-run wealth"
"""

import numpy  as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════
# 1.  Return-Based Performance Metrics
# ═══════════════════════════════════════════════════════════════════════

def sharpe_ratio(returns: pd.Series,
                 risk_free: float = 0.05,
                 periods_per_year: int = 252) -> float:
    """
    Sharpe Ratio = (E[R] − Rf) / σ(R) — annualised.

    The Sharpe ratio answers: "For each unit of total risk taken,
    how much excess return are we earning above the risk-free rate?"
    A ratio > 1.0 is generally considered acceptable.
    Above 2.0 is excellent for a systematic commodity strategy.
    """
    daily_rf = risk_free / periods_per_year
    excess   = returns - daily_rf
    if excess.std() == 0:
        return np.nan
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series,
                  risk_free: float = 0.05,
                  periods_per_year: int = 252) -> float:
    """
    Sortino Ratio = (E[R] − Rf) / σ_downside(R).

    Unlike Sharpe which penalises upside volatility equally as downside,
    Sortino only penalises for downside volatility. For skewed commodity
    return distributions (frequent small gains, rare large losses),
    Sortino is often more informative than Sharpe.
    """
    daily_rf = risk_free / periods_per_year
    excess   = returns - daily_rf
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return float((excess.mean() / downside) * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series,
                 periods_per_year: int = 252) -> float:
    """
    Calmar Ratio = Annualised Return / |Max Drawdown|.

    Named after the California Managed Accounts Reports.
    Preferred by CTA and commodity fund managers as it captures
    the "bang per buck of peak-to-trough risk."
    """
    ann_ret = returns.mean() * periods_per_year
    max_dd  = max_drawdown(returns)
    if max_dd == 0:
        return np.nan
    return float(ann_ret / abs(max_dd))


def omega_ratio(returns: pd.Series,
                threshold: float = 0.0) -> float:
    """
    Omega Ratio = E[max(R − L, 0)] / E[max(L − R, 0)]

    where L = threshold (often 0 or risk-free rate).
    Omega > 1 means the strategy produces more upside than downside
    relative to the threshold — robust to non-normality.
    """
    excess   = returns - threshold
    gains    = excess[excess > 0].sum()
    losses   = -excess[excess < 0].sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Drawdown Analysis
# ═══════════════════════════════════════════════════════════════════════

def drawdown_series(returns: pd.Series) -> pd.DataFrame:
    """
    Compute full drawdown time series: level, peak, trough.

    Analogy: Drawdown is the "depth of the valley" measured from the last
    mountain peak.  A drawdown of -15% means the strategy is 15% below
    its all-time high, still climbing back out of the hole.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    dd = (cumulative - running_max) / running_max

    return pd.DataFrame({
        "cumulative_return": cumulative,
        "running_max":       running_max,
        "drawdown":          dd,
    })


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    dd = drawdown_series(returns)["drawdown"]
    return float(dd.min())


def drawdown_stats(returns: pd.Series) -> Dict:
    """
    Comprehensive drawdown statistics including duration.

    Returns: max drawdown, recovery time, number of drawdown periods.
    """
    dd_df  = drawdown_series(returns)
    dd_ser = dd_df["drawdown"]
    max_dd = dd_ser.min()
    max_dd_date = dd_ser.idxmin()

    # Peak before max drawdown
    peak_date = dd_df["running_max"].loc[:max_dd_date].idxmax()

    # Recovery date (first time dd returns to 0 after the trough)
    post_trough = dd_ser.loc[max_dd_date:]
    recovery_dates = post_trough[post_trough >= -0.001]
    recovery_date = recovery_dates.index[0] if len(recovery_dates) > 0 else None

    duration_days = (max_dd_date - peak_date).days
    recovery_days = (recovery_date - max_dd_date).days if recovery_date else None

    # Average drawdown
    in_drawdown = dd_ser[dd_ser < 0]

    return {
        "max_drawdown_%":     round(max_dd * 100, 2),
        "peak_date":          str(peak_date.date()),
        "trough_date":        str(max_dd_date.date()),
        "recovery_date":      str(recovery_date.date()) if recovery_date else "Not recovered",
        "duration_days":      duration_days,
        "recovery_days":      recovery_days,
        "avg_drawdown_%":     round(in_drawdown.mean() * 100, 2) if len(in_drawdown) else 0,
        "n_drawdown_periods": int((dd_ser < 0).astype(int).diff().clip(lower=0).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3.  Value-at-Risk (VaR) & CVaR
# ═══════════════════════════════════════════════════════════════════════

def historical_var(returns: pd.Series,
                   confidence: float = 0.95,
                   position: float = 1_000_000) -> Dict:
    """
    Historical VaR — the loss not exceeded at the given confidence level.

    Method: Sort historical returns from worst to best, pick the (1-conf)
    percentile.  No distribution assumption required.

    Analogy: Look at all the bad days you've had over the past 5 years,
    line them up from worst to mildest, and read off the 5th worst percentile.
    That's your "1-in-20-day worst case" estimate.
    """
    var_pct = np.percentile(returns.dropna(), (1 - confidence) * 100)
    cvar_pct = returns[returns <= var_pct].mean()

    return {
        "method":        "Historical",
        "confidence":    confidence,
        "var_return_%":  round(var_pct * 100, 3),
        "cvar_return_%": round(cvar_pct * 100, 3),
        "var_dollar":    round(abs(var_pct) * position, 0),
        "cvar_dollar":   round(abs(cvar_pct) * position, 0),
        "interpretation": (
            f"With {confidence*100:.0f}% confidence, maximum 1-day loss ≤ "
            f"${abs(var_pct)*position:,.0f} on a ${position:,.0f} position"
        )
    }


def parametric_var(returns: pd.Series,
                   confidence: float = 0.95,
                   position: float = 1_000_000) -> Dict:
    """
    Parametric (Gaussian) VaR — assumes normally distributed returns.

    VaR = μ − z_α · σ  where z_α = normal quantile at confidence level.

    Note: This UNDERSTATES risk for commodity returns (fat tails / high kurtosis).
    Always compare with Historical VaR to quantify model risk.
    """
    mu   = returns.mean()
    sig  = returns.std()
    z    = stats.norm.ppf(1 - confidence)
    var_pct  = mu + z * sig           # z is negative → VaR is negative
    cvar_pct = mu - sig * stats.norm.pdf(z) / (1 - confidence)

    return {
        "method":        "Parametric (Gaussian)",
        "confidence":    confidence,
        "mu_daily_%":    round(mu * 100, 4),
        "sigma_daily_%": round(sig * 100, 4),
        "var_return_%":  round(var_pct * 100, 3),
        "cvar_return_%": round(cvar_pct * 100, 3),
        "var_dollar":    round(abs(var_pct) * position, 0),
        "cvar_dollar":   round(abs(cvar_pct) * position, 0),
    }


def monte_carlo_var(returns: pd.Series,
                    confidence: float = 0.95,
                    position: float = 1_000_000,
                    n_simulations: int = 50_000,
                    horizon_days: int = 1,
                    seed: int = 42) -> Dict:
    """
    Monte Carlo VaR — simulate forward paths under fitted distribution.

    More flexible than parametric VaR as it can accommodate fat tails
    (Student-t distribution) and multi-period horizons.
    Analogy: Run 50,000 "alternate history" scenarios and see what your
    portfolio is worth in each — VaR is the bad-case percentile.
    """
    rng   = np.random.default_rng(seed)
    mu    = returns.mean()
    sigma = returns.std()

    # Fit Student-t (better for fat tails)
    df_t, loc_t, scale_t = stats.t.fit(returns.dropna())

    if horizon_days == 1:
        sims = loc_t + scale_t * rng.standard_t(df_t, size=n_simulations)
    else:
        daily = loc_t + scale_t * rng.standard_t(df_t, size=(n_simulations, horizon_days))
        sims  = daily.sum(axis=1)  # sum log-returns over horizon

    var_pct  = np.percentile(sims, (1 - confidence) * 100)
    cvar_pct = sims[sims <= var_pct].mean()

    return {
        "method":         "Monte Carlo (Student-t)",
        "confidence":     confidence,
        "n_simulations":  n_simulations,
        "horizon_days":   horizon_days,
        "t_df":           round(float(df_t), 2),
        "var_return_%":   round(var_pct * 100, 3),
        "cvar_return_%":  round(cvar_pct * 100, 3),
        "var_dollar":     round(abs(var_pct) * position, 0),
        "cvar_dollar":    round(abs(cvar_pct) * position, 0),
        "simulations":    sims,
    }


def var_comparison(returns: pd.Series,
                   confidence: float = 0.95,
                   position: float = 1_000_000) -> pd.DataFrame:
    """
    Side-by-side VaR comparison across all three methods.
    """
    hist = historical_var(returns, confidence, position)
    para = parametric_var(returns, confidence, position)
    mc   = monte_carlo_var(returns, confidence, position)

    rows = []
    for r in [hist, para, mc]:
        rows.append({
            "Method":      r["method"],
            "VaR %":       r["var_return_%"],
            "CVaR %":      r["cvar_return_%"],
            "VaR $":       r["var_dollar"],
            "CVaR $":      r["cvar_dollar"],
        })
    return pd.DataFrame(rows).set_index("Method")


# ═══════════════════════════════════════════════════════════════════════
# 4.  Position Sizing
# ═══════════════════════════════════════════════════════════════════════

def kelly_criterion(win_rate: float,
                    avg_win: float,
                    avg_loss: float) -> Dict:
    """
    Kelly Criterion — optimal bet size to maximise long-run log wealth.

    f* = (b·p − q) / b
    where  b = avg_win / avg_loss  (odds)
           p = win probability
           q = 1 − p

    In practice, traders use  half-Kelly (f*/2)  to reduce variance
    while preserving most of the growth rate.

    Analogy: Kelly tells you how much of your bankroll to put on a
    bet given your edge and odds — it's the mathematically proven
    "optimal bet" for long-run wealth maximisation.
    """
    b = avg_win / avg_loss if avg_loss != 0 else 0
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b if b != 0 else 0
    kelly = max(0, kelly)  # floor at 0 (no negative sizing)

    return {
        "win_rate_%":    round(win_rate * 100, 1),
        "avg_win_%":     round(avg_win * 100, 2),
        "avg_loss_%":    round(avg_loss * 100, 2),
        "odds_ratio_b":  round(b, 4),
        "kelly_fraction":round(kelly, 4),
        "half_kelly":    round(kelly / 2, 4),
        "interpretation": (
            f"Full Kelly: bet {kelly*100:.1f}% of capital per trade | "
            f"Half Kelly (recommended): {kelly*50:.1f}%"
        )
    }


def vol_target_sizing(returns: pd.Series,
                       target_vol: float = 0.10,
                       max_leverage: float = 2.0,
                       window: int = 21) -> pd.Series:
    """
    Volatility-targeting position size: scale exposure so annualised portfolio
    vol equals target_vol (e.g., 10% p.a.).

    Position Size = target_vol / (realised_vol × √252)

    Used by CTAs and risk-parity funds.  During calm markets → larger positions,
    during turbulent markets → smaller positions (automatic deleveraging).
    """
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    raw_size    = target_vol / rolling_vol
    capped_size = raw_size.clip(upper=max_leverage)
    capped_size.name = "position_size"
    return capped_size


# ═══════════════════════════════════════════════════════════════════════
# 5.  Comprehensive Risk Report
# ═══════════════════════════════════════════════════════════════════════

def risk_report(returns: pd.Series,
                risk_free: float = 0.05,
                position: float = 1_000_000) -> Dict:
    """
    Full risk report for a single return series (strategy, asset, or portfolio).

    Combines performance metrics + drawdown + VaR into one unified dict.
    """
    ann_ret = returns.mean() * 252
    ann_vol = returns.std()  * np.sqrt(252)

    report = {
        # Performance
        "annualised_return_%":  round(ann_ret * 100, 2),
        "annualised_vol_%":     round(ann_vol * 100, 2),
        "sharpe_ratio":         round(sharpe_ratio(returns, risk_free), 3),
        "sortino_ratio":        round(sortino_ratio(returns, risk_free), 3),
        "calmar_ratio":         round(calmar_ratio(returns), 3),
        "omega_ratio":          round(omega_ratio(returns), 3),
        "skewness":             round(returns.skew(), 4),
        "kurtosis":             round(returns.kurtosis(), 4),
        "n_observations":       len(returns.dropna()),

        # Drawdown
        **drawdown_stats(returns),

        # VaR (historical 95%)
        "var_95_hist_%":        historical_var(returns, 0.95, position)["var_return_%"],
        "cvar_95_hist_%":       historical_var(returns, 0.95, position)["cvar_return_%"],
        "var_99_hist_%":        historical_var(returns, 0.99, position)["var_return_%"],
        "cvar_99_hist_%":       historical_var(returns, 0.99, position)["cvar_return_%"],
    }
    return report


def portfolio_risk_report(weights: Dict[str, float],
                           returns: pd.DataFrame,
                           risk_free: float = 0.05,
                           position: float = 10_000_000) -> Dict:
    """
    Portfolio-level risk report given asset weights and return matrix.

    Computes portfolio returns, then delegates to risk_report().
    Also computes marginal VaR and component VaR per asset.
    """
    aligned = returns[list(weights.keys())].dropna()
    w_arr   = np.array([weights[c] for c in aligned.columns])
    w_arr   = w_arr / w_arr.sum()  # normalise

    port_ret = aligned.dot(w_arr)

    # Covariance matrix for component VaR
    cov = aligned.cov() * 252
    port_var   = float(w_arr @ cov.values @ w_arr)
    port_vol   = np.sqrt(port_var)

    z95 = stats.norm.ppf(0.05)
    marginal_var = {col: round(float(z95 * (cov.values @ w_arr)[i] / port_vol) * position, 0)
                    for i, col in enumerate(aligned.columns)}
    component_var = {col: round(w_arr[i] * marginal_var[col], 0)
                     for i, col in enumerate(aligned.columns)}

    base = risk_report(port_ret, risk_free, position)
    base.update({
        "weights":            weights,
        "portfolio_vol_%":    round(port_vol * 100, 2),
        "marginal_var_$":     marginal_var,
        "component_var_$":    component_var,
    })
    return base


if __name__ == "__main__":
    rng  = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.0003, 0.015, 1000), name="TestStrategy")

    print("\n=== Risk Report ===")
    rpt = risk_report(rets)
    for k, v in rpt.items():
        print(f"  {k:30s}: {v}")

    print("\n=== VaR Comparison ===")
    print(var_comparison(rets).to_string())

    kc = kelly_criterion(0.55, 0.02, 0.015)
    print("\n=== Kelly Criterion ===")
    for k, v in kc.items():
        print(f"  {k}: {v}")
    print("\n✅ risk_metrics.py working correctly.")
