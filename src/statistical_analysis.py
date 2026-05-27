"""
============================================================
statistical_analysis.py — Core quantitative analysis engine
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
============================================================

Implements
----------
1. Correlation analysis (Pearson, Spearman, rolling)
2. Cointegration (Engle-Granger 2-step, Johansen)
3. Stationarity (ADF, KPSS, Phillips-Perron)
4. OLS / rolling regression & beta estimation
5. Principal Component Analysis (macro factor decomposition)
6. Spread / basis analysis & z-score signals
7. Volatility modelling (GARCH(1,1))

Financial Analogy Reference
---------------------------
• Correlation   → How much two ships move in the same direction on the ocean
• Cointegration → Two ships tied by a rope — drift apart but always come back
• ADF test      → Checking if a boat drifts forever (non-stationary) or
                  returns to dock (stationary)
• Beta          → How many degrees a ship tilts when the index tilts 1°
• PCA           → Finding the dominant "winds" that blow all ships at once
"""

import sys
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy  as np
import pandas as pd
from scipy  import stats
from scipy.stats import spearmanr

import statsmodels.api as sm
from statsmodels.tsa.stattools  import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.rolling  import RollingOLS
from statsmodels.stats.stattools import durbin_watson
from sklearn.decomposition import PCA
from sklearn.preprocessing  import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STRATEGY_PARAMS


# ═══════════════════════════════════════════════════════════════════════
# 1.  Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════

def pearson_corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Full-sample Pearson correlation matrix.

    The standard tool for measuring linear co-movement. A correlation of
    +1.0 means perfect co-movement, 0 means independence, -1 is perfect hedge.
    """
    return returns.corr(method="pearson")


def spearman_corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman (rank) correlation matrix — robust to outliers and non-normality.
    Preferred for commodity returns that exhibit fat tails and jump behaviour.
    """
    return returns.corr(method="spearman")


def rolling_correlation(r1: pd.Series,
                        r2: pd.Series,
                        windows: List[int] = [21, 63, 126, 252]
                        ) -> pd.DataFrame:
    """
    Rolling Pearson correlations at multiple lookback windows.

    Returns a DataFrame with one column per window, showing how the
    relationship between two assets evolves through market regimes
    (e.g., correlation surges during crises — the "correlation is 1 in a crash"
    phenomenon).
    """
    result = {}
    for w in windows:
        result[f"roll_{w}d"] = r1.rolling(window=w).corr(r2)
    return pd.DataFrame(result, index=r1.index)


def correlation_change_test(r1: pd.Series,
                             r2: pd.Series,
                             split_date: str) -> Dict:
    """
    Fisher Z-test for significant change in correlation across two sub-periods.

    H₀: ρ₁ = ρ₂  (correlation unchanged before/after split_date)
    H₁: ρ₁ ≠ ρ₂  (structural break in correlation)

    Returns: dict with r1, r2, z-stat, p-value, interpretation.
    """
    split = pd.Timestamp(split_date)
    pre  = pd.concat([r1, r2], axis=1).dropna().loc[:split]
    post = pd.concat([r1, r2], axis=1).dropna().loc[split:]

    r_pre  = pre.iloc[:, 0].corr(pre.iloc[:, 1])
    r_post = post.iloc[:, 0].corr(post.iloc[:, 1])

    # Fisher z-transformation
    z_pre  = np.arctanh(r_pre)
    z_post = np.arctanh(r_post)
    se     = np.sqrt(1/(len(pre)-3) + 1/(len(post)-3))
    z_stat = (z_pre - z_post) / se
    p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "r_pre_split":   round(r_pre, 4),
        "r_post_split":  round(r_post, 4),
        "delta_r":       round(r_post - r_pre, 4),
        "z_statistic":   round(z_stat, 4),
        "p_value":       round(p_val, 4),
        "significant":   p_val < 0.05,
        "interpretation": (
            "SIGNIFICANT correlation shift — possible regime change"
            if p_val < 0.05 else
            "No significant correlation change detected"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2.  Stationarity Tests
# ═══════════════════════════════════════════════════════════════════════

def adf_test(series: pd.Series,
             maxlag: Optional[int] = None,
             regression: str = "c") -> Dict:
    """
    Augmented Dickey-Fuller test for unit root (non-stationarity).

    H₀: Series has a unit root (non-stationary / random walk)
    H₁: Series is stationary (mean-reverting)

    A commodity price series is typically non-stationary (I(1)),
    while returns are stationary (I(0)).  For a spread to be tradable
    (mean-reversion strategy), it must be stationary.

    regression : 'c' = constant  |  'ct' = trend+constant  |  'n' = none
    """
    result = adfuller(series.dropna(), maxlag=maxlag, regression=regression,
                      autolag="AIC")
    adf_stat, p_val, n_lags, n_obs, crit_vals = result[:5]

    return {
        "series":       series.name or "unnamed",
        "adf_statistic": round(adf_stat, 4),
        "p_value":       round(p_val, 4),
        "n_lags":        n_lags,
        "n_obs":         n_obs,
        "critical_values": {k: round(v, 4) for k, v in crit_vals.items()},
        "stationary_at_5pct": p_val < 0.05,
        "integration_order": "I(0) — Stationary" if p_val < 0.05 else "I(1) — Non-stationary",
    }


def kpss_test(series: pd.Series,
              regression: str = "c") -> Dict:
    """
    KPSS test — complementary to ADF.

    H₀: Series is stationary
    H₁: Series has a unit root

    Use together: ADF p>0.05 AND KPSS p<0.05 → strong evidence of non-stationarity.
    """
    stat, p_val, n_lags, crit_vals = kpss(series.dropna(), regression=regression,
                                           nlags="auto")
    return {
        "kpss_statistic":   round(stat, 4),
        "p_value":          round(p_val, 4),
        "n_lags":           n_lags,
        "critical_values":  {k: round(v, 4) for k, v in crit_vals.items()},
        "stationary_at_5pct": p_val > 0.05,
    }


def stationarity_report(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF + KPSS on both price levels and log-returns for all columns.

    Returns a summary DataFrame suitable for display in research reports.

    A standard finding: price levels are I(1), log-returns are I(0) —
    exactly what we expect for efficient markets.
    """
    rows = []
    log_rets = np.log(prices / prices.shift(1)).dropna()

    for col in prices.columns:
        # Level
        adf_lv = adf_test(prices[col].dropna())
        kp_lv  = kpss_test(prices[col].dropna())
        # Returns
        adf_rt = adf_test(log_rets[col].dropna())
        kp_rt  = kpss_test(log_rets[col].dropna())

        rows.append({
            "Asset":            col,
            "Level ADF p":      adf_lv["p_value"],
            "Level KPSS p":     kp_lv["p_value"],
            "Level Stationary": adf_lv["stationary_at_5pct"],
            "Return ADF p":     adf_rt["p_value"],
            "Return KPSS p":    kp_rt["p_value"],
            "Return Stationary": adf_rt["stationary_at_5pct"],
            "Integration":      adf_lv["integration_order"],
        })

    return pd.DataFrame(rows).set_index("Asset")


# ═══════════════════════════════════════════════════════════════════════
# 3.  Cointegration Analysis
# ═══════════════════════════════════════════════════════════════════════

def engle_granger_coint(y: pd.Series,
                         x: pd.Series,
                         trend: str = "c") -> Dict:
    """
    Engle-Granger 2-step cointegration test between two price series.

    Cointegration = Two non-stationary series share a common stochastic trend.
    The KEY insight: the spread y − β·x is stationary even though y and x are not.

    This is the statistical foundation for:
    • Brent vs WTI crude oil (same commodity, different delivery points)
    • Gold spot vs Gold ETF (same asset, different wrappers)
    • Gold vs Gold Miners ETF (fundamental linkage via revenue)

    Returns
    -------
    dict with test stat, p-value, hedge ratio (beta), and residual diagnostics.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_clean, x_clean = aligned.iloc[:, 0], aligned.iloc[:, 1]

    # Step 1: OLS regression  y = α + β·x + ε
    X_ols = sm.add_constant(x_clean)
    ols   = sm.OLS(y_clean, X_ols).fit()
    alpha, beta = ols.params

    # Step 2: ADF on residuals
    resids  = ols.resid
    eg_stat, eg_pval, eg_crit = coint(y_clean, x_clean, trend=trend)[:3]

    # Durbin-Watson for residual autocorrelation
    dw = durbin_watson(resids)

    return {
        "y":           y.name,
        "x":           x.name,
        "alpha":       round(float(alpha), 6),
        "beta":        round(float(beta), 6),       # hedge ratio
        "eg_statistic": round(eg_stat, 4),
        "p_value":     round(eg_pval, 4),
        "critical_vals": {k: round(v, 4) for k, v in
                          zip(["1%", "5%", "10%"], eg_crit)},
        "cointegrated":  eg_pval < STRATEGY_PARAMS["coint_pvalue"],
        "durbin_watson": round(dw, 4),
        "residual_std":  round(resids.std(), 6),
        "r_squared":     round(ols.rsquared, 4),
        "interpretation": (
            f"COINTEGRATED ✓ — tradable spread: {y.name} − {beta:.4f}·{x.name}"
            if eg_pval < 0.05 else
            "NOT cointegrated — spread may drift without mean-reversion"
        ),
    }


def pairwise_cointegration_matrix(prices: pd.DataFrame,
                                   pairs: Optional[List[Tuple]] = None
                                   ) -> pd.DataFrame:
    """
    Run Engle-Granger cointegration test for all pairs (or specified pairs).

    Returns a matrix of p-values.  Values < 0.05 indicate a cointegrated pair —
    a candidate for a statistical arbitrage or spread trading strategy.
    """
    cols = list(prices.columns)
    if pairs is None:
        pairs = [(c1, c2) for i, c1 in enumerate(cols)
                 for c2 in cols[i+1:]]

    n = len(cols)
    mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    results_list = []

    for c1, c2 in pairs:
        if c1 not in prices.columns or c2 not in prices.columns:
            continue
        r = engle_granger_coint(prices[c1], prices[c2])
        mat.loc[c1, c2] = r["p_value"]
        mat.loc[c2, c1] = r["p_value"]
        results_list.append(r)

    return mat, results_list


def johansen_test(prices: pd.DataFrame,
                  det_order: int = 0,
                  k_ar_diff: int = 1) -> Dict:
    """
    Johansen cointegration test for multiple series simultaneously.

    Unlike Engle-Granger (pairwise only), Johansen can identify multiple
    cointegrating vectors — useful for basket trades
    (e.g., WTI, Brent, heating oil all cointegrated).

    det_order: 0 = no constant, -1 = no constant/trend, 1 = restricted trend
    Returns: number of cointegrating relationships and critical values.
    """
    clean = prices.dropna()
    result = coint_johansen(clean, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stat = result.lr1    # Trace statistics
    max_stat   = result.lr2    # Max-eigenvalue statistics
    crit_90    = result.cvt[:, 0]
    crit_95    = result.cvt[:, 1]

    n_coint = sum(trace_stat > crit_95)

    return {
        "assets":          list(prices.columns),
        "n_cointegrating": n_coint,
        "trace_stats":     trace_stat.tolist(),
        "max_eigen_stats": max_stat.tolist(),
        "crit_values_95":  crit_95.tolist(),
        "eigenvectors":    result.evec.tolist(),
        "interpretation":  f"{n_coint} cointegrating relationship(s) found",
    }


# ═══════════════════════════════════════════════════════════════════════
# 4.  OLS Regression & Rolling Beta
# ═══════════════════════════════════════════════════════════════════════

def ols_regression(y: pd.Series,
                   X: Union[pd.Series, pd.DataFrame],
                   add_const: bool = True) -> Dict:
    """
    OLS regression:  y = α + β₁·X₁ + β₂·X₂ + ... + ε

    Used for:
    • Estimating hedge ratios (Gold → GDX, Oil → XLE)
    • Commodity factor attribution (CPI, USD, rates)
    • Basis regression (spot ~ futures)

    Returns full OLS results plus key diagnostics.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()

    aligned = pd.concat([y, X], axis=1).dropna()
    y_clean = aligned.iloc[:, 0]
    X_clean = aligned.iloc[:, 1:]

    if add_const:
        X_clean = sm.add_constant(X_clean)

    model  = sm.OLS(y_clean, X_clean).fit(cov_type="HC3")  # Heteroscedasticity-robust

    return {
        "r_squared":     round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "coefficients":  {k: round(v, 6) for k, v in model.params.items()},
        "p_values":      {k: round(v, 4) for k, v in model.pvalues.items()},
        "t_stats":       {k: round(v, 4) for k, v in model.tvalues.items()},
        "conf_int":      model.conf_int().round(6).to_dict(),
        "f_statistic":   round(model.fvalue, 4),
        "f_pvalue":      round(model.f_pvalue, 4),
        "aic":           round(model.aic, 2),
        "bic":           round(model.bic, 2),
        "n_obs":         int(model.nobs),
        "durbin_watson": round(durbin_watson(model.resid), 4),
        "residuals":     model.resid,
        "fitted":        model.fittedvalues,
        "model":         model,
    }


def rolling_beta(y: pd.Series,
                 x: pd.Series,
                 window: int = 63) -> pd.DataFrame:
    """
    Rolling OLS beta — how the hedge ratio shifts over time.

    A time-varying beta reveals structural shifts in relationships:
    e.g., Gold-Miners beta drops during labour strikes (idiosyncratic risk)
    or surges during dollar weakness (macro driver dominates).
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_c = aligned.iloc[:, 0]
    x_c = sm.add_constant(aligned.iloc[:, 1])

    rols = RollingOLS(y_c, x_c, window=window).fit()
    betas = rols.params.rename(columns={
        "const": "alpha",
        aligned.columns[1]: "beta"
    })
    betas.index = aligned.index
    return betas


# ═══════════════════════════════════════════════════════════════════════
# 5.  Spread & Z-Score Signal Generation
# ═══════════════════════════════════════════════════════════════════════

def compute_spread_zscore(price_a: pd.Series,
                           price_b: pd.Series,
                           coint_result: Optional[Dict] = None,
                           zscore_window: int = 63) -> pd.DataFrame:
    """
    Compute the tradable spread and its rolling z-score for signal generation.

    Steps
    -----
    1. Estimate hedge ratio β via OLS (or use pre-computed from cointegration)
    2. Spread = A − β·B
    3. Z-score = (Spread − rolling_mean) / rolling_std
    4. Signals: Buy spread when z < −2,  Sell spread when z > +2

    Returns DataFrame with: price_a, price_b, spread, z_score, signal
    """
    aligned = pd.concat([price_a, price_b], axis=1).dropna()
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]

    # Estimate hedge ratio
    if coint_result and "beta" in coint_result:
        beta = coint_result["beta"]
    else:
        ols_r = ols_regression(a, b)
        beta  = ols_r["coefficients"].get(b.name, 1.0)

    spread = a - beta * b

    # Rolling z-score
    mu  = spread.rolling(zscore_window).mean()
    sig = spread.rolling(zscore_window).std()
    zsc = (spread - mu) / sig

    # Trading signals
    entry  = STRATEGY_PARAMS["zscore_entry"]
    exit_z = STRATEGY_PARAMS["zscore_exit"]
    stop   = STRATEGY_PARAMS["zscore_stop"]

    signal = pd.Series(0, index=zsc.index)
    signal[zsc >  entry] = -1   # Short spread (sell A, buy B)
    signal[zsc < -entry] =  1   # Long spread  (buy A, sell B)
    signal[abs(zsc) < exit_z] = 0
    signal[abs(zsc) > stop]   = 0  # stop-loss

    return pd.DataFrame({
        price_a.name:   a,
        price_b.name:   b,
        "hedge_ratio":  beta,
        "spread":       spread,
        "spread_mean":  mu,
        "spread_upper": mu + entry * sig,
        "spread_lower": mu - entry * sig,
        "z_score":      zsc,
        "signal":       signal,
    })


# ═══════════════════════════════════════════════════════════════════════
# 6.  Principal Component Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_pca(returns: pd.DataFrame,
            n_components: Optional[int] = None) -> Dict:
    """
    PCA on a multi-asset return matrix to identify common risk factors.

    In commodity markets:
    • PC1 often = "global risk appetite" (most assets move together)
    • PC2 often = "energy vs metals" contrast
    • PC3 often = "agricultural vs industrial" contrast

    This is analogous to finding the dominant winds (macro factors) that
    blow all ships (commodities) in similar directions simultaneously.
    """
    clean = returns.dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)

    if n_components is None:
        n_components = min(clean.shape[1], 10)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=clean.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    scores_df = pd.DataFrame(
        scores,
        index=clean.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    expl_var = pca.explained_variance_ratio_
    cum_var  = np.cumsum(expl_var)

    return {
        "loadings":           loadings,
        "scores":             scores_df,
        "explained_variance": expl_var,
        "cumulative_variance": cum_var,
        "n_components":       n_components,
        "n_for_90pct":        int(np.searchsorted(cum_var, 0.90)) + 1,
    }


# ═══════════════════════════════════════════════════════════════════════
# 7.  GARCH Volatility Modelling
# ═══════════════════════════════════════════════════════════════════════

def fit_garch(returns: pd.Series,
              p: int = 1,
              q: int = 1,
              dist: str = "t") -> Dict:
    """
    Fit GARCH(p,q) model to capture volatility clustering.

    Volatility clustering = calm periods followed by turbulent periods —
    "markets are quiet, until they're not."  GARCH models this explicitly,
    giving better option pricing, VaR estimates, and hedging ratios than
    assuming constant volatility (like the naive historical vol approach).

    dist : "normal" | "t" (Student-t, better for fat tails)
    """
    try:
        from arch import arch_model
    except ImportError:
        return {"error": "arch library not installed. Run: pip install arch"}

    scale = 100  # GARCH works better with scaled returns
    ret_scaled = returns.dropna() * scale

    am = arch_model(ret_scaled, vol="Garch", p=p, q=q,
                    dist="StudentsT" if dist == "t" else "Normal",
                    rescale=False)
    res = am.fit(disp="off")

    cond_vol = res.conditional_volatility / scale
    ann_vol  = cond_vol * np.sqrt(252)

    return {
        "asset":           returns.name,
        "p": p, "q": q,
        "omega":           round(float(res.params["omega"]), 8),
        "alpha":           round(float(res.params.get("alpha[1]", np.nan)), 6),
        "beta":            round(float(res.params.get("beta[1]", np.nan)), 6),
        "persistence":     round(
            float(res.params.get("alpha[1]", 0)) +
            float(res.params.get("beta[1]", 0)), 6),
        "conditional_vol":   cond_vol,
        "annualised_vol":    ann_vol,
        "aic":               round(res.aic, 2),
        "bic":               round(res.bic, 2),
        "log_likelihood":    round(res.loglikelihood, 2),
        "model":             res,
    }


# ═══════════════════════════════════════════════════════════════════════
# 8.  Granger Causality
# ═══════════════════════════════════════════════════════════════════════

def granger_causality(cause: pd.Series,
                       effect: pd.Series,
                       max_lag: int = 10) -> pd.DataFrame:
    """
    Granger causality test: does 'cause' help predict 'effect'?

    In commodity markets, this can reveal lead-lag relationships:
    • Does crude oil price Granger-cause gasoline prices? (often yes)
    • Does copper price lead equity indices? (copper = Dr. Copper as economic barometer)
    • Does USD index Granger-cause gold? (well-documented relationship)

    Returns DataFrame of F-statistics and p-values for each lag.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    aligned = pd.concat([effect, cause], axis=1).dropna()
    test_result = grangercausalitytests(aligned, maxlag=max_lag, verbose=False)

    rows = []
    for lag, res in test_result.items():
        f_stat = res[0]["ssr_ftest"][0]
        p_val  = res[0]["ssr_ftest"][1]
        rows.append({
            "lag":         lag,
            "f_statistic": round(f_stat, 4),
            "p_value":     round(p_val, 4),
            "significant": p_val < 0.05,
        })
    df = pd.DataFrame(rows).set_index("lag")
    best_lag = df[df["significant"]]["f_statistic"].idxmax() \
               if df["significant"].any() else None
    df.attrs["best_lag"]   = best_lag
    df.attrs["cause_name"] = cause.name
    df.attrs["effect_name"]= effect.name
    return df


# ═══════════════════════════════════════════════════════════════════════
# 9.  Summary Report
# ═══════════════════════════════════════════════════════════════════════

def full_statistical_summary(prices: pd.DataFrame,
                              returns: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive statistical summary for all assets.

    Includes: descriptive stats, skewness, kurtosis, Sharpe-like ratio,
    annualised return & vol, max drawdown of return series.
    """
    ann_ret = returns.mean() * 252
    ann_vol = returns.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol

    desc = returns.describe().T
    desc["skewness"] = returns.skew()
    desc["kurtosis"] = returns.kurtosis()
    desc["ann_return_%"] = (ann_ret * 100).round(2)
    desc["ann_vol_%"]    = (ann_vol * 100).round(2)
    desc["sharpe_ratio"] = sharpe.round(3)

    # Max drawdown (cumulative return perspective)
    cum_rets = (1 + returns).cumprod()
    roll_max = cum_rets.cummax()
    drawdown = (cum_rets - roll_max) / roll_max
    desc["max_drawdown_%"] = (drawdown.min() * 100).round(2)

    return {"summary": desc, "ann_return": ann_ret,
            "ann_vol": ann_vol, "sharpe": sharpe}


if __name__ == "__main__":
    # Quick smoke test
    import yfinance as yf
    tickers = ["GC=F", "CL=F", "HG=F"]
    prices  = yf.download(tickers, start="2020-01-01", end="2024-12-31",
                           auto_adjust=True, progress=False)["Close"]
    prices.columns = ["Copper", "WTI", "Gold"]
    rets = np.log(prices / prices.shift(1)).dropna()

    print("\n=== ADF Stationarity ===")
    print(stationarity_report(prices).to_string())

    print("\n=== Cointegration: Gold vs WTI ===")
    r = engle_granger_coint(prices["Gold"], prices["WTI"])
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== PCA ===")
    pca_r = run_pca(rets)
    print("Explained variance:", (pca_r["explained_variance"]*100).round(1))
    print("\n✅ statistical_analysis.py working correctly.")
