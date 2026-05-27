"""
============================================================
visualization.py — Chart generation for research report
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
============================================================

All charts are designed for portfolio-grade research report presentation.
Each function saves a PNG and optionally returns the Figure object.
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy  as np
import pandas as pd
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.dates     as mdates
import matplotlib.ticker    as mticker
import seaborn              as sns
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHARTS_DIR, COLOR_PALETTE, PLOT_STYLE, FIGURE_DPI, COLORMAP

try:
    plt.style.use(PLOT_STYLE)
except:
    plt.style.use("seaborn-v0_8-darkgrid")

COLORS = COLOR_PALETTE


# ── Helper ────────────────────────────────────────────────────────────
def _save(fig, name: str, save: bool = True) -> str:
    path = os.path.join(CHARTS_DIR, f"{name}.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  💾 {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# 1.  Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                              title: str = "Correlation Matrix",
                              filename: str = "correlation_heatmap",
                              save: bool = True) -> plt.Figure:
    """
    Publication-quality annotated correlation heatmap.
    Colour range: Red (−1) → White (0) → Green (+1)
    """
    n   = corr_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, n*0.8), max(8, n*0.7)))

    mask  = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # upper triangle
    cmap  = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(
        corr_matrix, ax=ax, mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2.  Price Series Dashboard
# ═══════════════════════════════════════════════════════════════════════

def plot_price_series(prices: pd.DataFrame,
                       title:  str = "Commodity Prices (Normalised)",
                       filename: str = "price_series",
                       normalise: bool = True,
                       save: bool = True) -> plt.Figure:
    """
    Multi-series normalised price chart (rebased to 100 at start).
    Normalising allows apples-to-apples comparison across assets
    with very different price levels (e.g., Gold ~$2000 vs Natural Gas ~$3).
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    data = prices.copy()
    if normalise:
        data = data / data.iloc[0] * 100

    for i, col in enumerate(data.columns):
        ax.plot(data.index, data[col], label=col,
                color=COLORS[i % len(COLORS)], lw=1.5, alpha=0.9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Indexed Price (Base = 100)" if normalise else "Price")
    ax.axhline(100 if normalise else data.iloc[0].mean(),
               color="gray", ls="--", alpha=0.4, lw=0.8)

    ax.legend(fontsize=9, ncol=min(4, len(data.columns)),
              loc="upper left", framealpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3.  Spread & Z-Score Chart
# ═══════════════════════════════════════════════════════════════════════

def plot_spread_zscore(spread_df: pd.DataFrame,
                        pair_name: str = "Pair",
                        filename:  str = "spread_zscore",
                        save: bool = True) -> plt.Figure:
    """
    3-panel chart: prices | spread | z-score with entry/exit bands.
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35)

    cols = list(spread_df.columns)
    asset_a = cols[0]
    asset_b = cols[1] if len(cols) > 1 else cols[0]

    # ── Panel 1: Prices ──
    ax1 = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()
    if asset_a in spread_df.columns:
        ax1.plot(spread_df.index, spread_df[asset_a],
                 color=COLORS[0], lw=1.5, label=asset_a)
        ax1.set_ylabel(asset_a, color=COLORS[0])
    if asset_b in spread_df.columns and asset_b != asset_a:
        ax1b.plot(spread_df.index, spread_df[asset_b],
                  color=COLORS[1], lw=1.5, label=asset_b, ls="--")
        ax1b.set_ylabel(asset_b, color=COLORS[1])
    ax1.set_title(f"{pair_name} — Spread Arbitrage Analysis", fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Spread ──
    ax2 = fig.add_subplot(gs[1])
    if "spread" in spread_df.columns:
        ax2.plot(spread_df.index, spread_df["spread"],
                 color=COLORS[4], lw=1.3, label="Spread")
    if "spread_mean" in spread_df.columns:
        ax2.plot(spread_df.index, spread_df["spread_mean"],
                 color="gray", lw=1, ls="--", label="Rolling Mean")
    if all(c in spread_df.columns for c in ["spread_upper", "spread_lower"]):
        ax2.fill_between(spread_df.index,
                         spread_df["spread_lower"], spread_df["spread_upper"],
                         alpha=0.15, color=COLORS[4], label="±2σ Band")
    ax2.set_title("Spread Level", fontsize=11)
    ax2.set_ylabel("Spread")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # ── Panel 3: Z-Score + Signals ──
    ax3 = fig.add_subplot(gs[2])
    if "z_score" in spread_df.columns:
        z = spread_df["z_score"]
        ax3.plot(spread_df.index, z, color=COLORS[0], lw=1.3, label="Z-Score")
        ax3.axhline(2.0,  color=COLORS[3], ls="--", lw=1.2, label="+2σ (Short)")
        ax3.axhline(-2.0, color=COLORS[2], ls="--", lw=1.2, label="−2σ (Long)")
        ax3.axhline(0.5,  color="gray",    ls=":",  lw=0.8, label="±0.5σ Exit")
        ax3.axhline(-0.5, color="gray",    ls=":",  lw=0.8)
        ax3.fill_between(spread_df.index, 2, z.clip(upper=4),
                         where=z > 2, alpha=0.2, color=COLORS[3], label="Short Zone")
        ax3.fill_between(spread_df.index, z.clip(lower=-4), -2,
                         where=z < -2, alpha=0.2, color=COLORS[2], label="Long Zone")
        ax3.set_ylim(-4.5, 4.5)

    if "signal" in spread_df.columns:
        sig = spread_df["signal"]
        buy_dates  = spread_df.index[sig > 0]
        sell_dates = spread_df.index[sig < 0]
        if "z_score" in spread_df.columns:
            ax3.scatter(buy_dates,  z.loc[buy_dates],  marker="^",
                        color=COLORS[2], s=30, zorder=5, label="Long Signal")
            ax3.scatter(sell_dates, z.loc[sell_dates], marker="v",
                        color=COLORS[3], s=30, zorder=5, label="Short Signal")

    ax3.set_title("Z-Score & Trading Signals", fontsize=11)
    ax3.set_ylabel("Z-Score (σ)")
    ax3.legend(fontsize=8, ncol=3); ax3.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 4.  Rolling Correlation Chart
# ═══════════════════════════════════════════════════════════════════════

def plot_rolling_correlations(roll_corr: pd.DataFrame,
                               pair_label: str = "Asset Pair",
                               filename: str = "rolling_correlation",
                               save: bool = True) -> plt.Figure:
    """
    Multi-window rolling correlation with regime shading.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    line_styles = ["-", "--", "-.", ":"]
    for i, col in enumerate(roll_corr.columns):
        ax.plot(roll_corr.index, roll_corr[col], label=col,
                lw=1.4, ls=line_styles[i % 4],
                color=COLORS[i % len(COLORS)])

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axhline(0.7, color=COLORS[2], ls=":", alpha=0.7, lw=1, label="High Corr (+0.7)")
    ax.axhline(-0.7, color=COLORS[3], ls=":", alpha=0.7, lw=1, label="High Neg. Corr (−0.7)")

    ax.fill_between(roll_corr.index, 0.7, roll_corr.iloc[:, -1].clip(lower=0.7),
                    alpha=0.1, color=COLORS[2])
    ax.fill_between(roll_corr.index, roll_corr.iloc[:, -1].clip(upper=-0.7), -0.7,
                    alpha=0.1, color=COLORS[3])

    ax.set_title(f"Rolling Correlation — {pair_label}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5.  PCA Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_pca_analysis(pca_results: Dict,
                       filename: str = "pca_analysis",
                       save: bool = True) -> plt.Figure:
    """
    3-panel PCA chart: scree plot | PC1 loadings | cumulative variance.
    """
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    expl_var   = pca_results["explained_variance"] * 100
    cum_var    = pca_results["cumulative_variance"] * 100
    n          = len(expl_var)
    loadings   = pca_results["loadings"]

    # ── Panel 1: Scree Plot ──
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(range(1, n+1), expl_var, color=COLORS[0], alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Principal Component"); ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Scree Plot", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, expl_var):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    # ── Panel 2: PC1 & PC2 Loadings ──
    ax2 = fig.add_subplot(gs[1])
    n_show = min(10, len(loadings))
    pc1  = loadings["PC1"].nlargest(n_show // 2).append(
           loadings["PC1"].nsmallest(n_show // 2)).sort_values()
    colors2 = [COLORS[3] if v < 0 else COLORS[2] for v in pc1.values]
    pc1.plot(kind="barh", ax=ax2, color=colors2, edgecolor="white")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_title("PC1 Loadings (Risk Factor)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Loading"); ax2.grid(alpha=0.3, axis="x")

    # ── Panel 3: Cumulative Variance ──
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(range(1, n+1), cum_var, marker="o", color=COLORS[0], lw=2)
    ax3.axhline(90, color=COLORS[3], ls="--", alpha=0.7, label="90% threshold")
    ax3.fill_between(range(1, n+1), 0, cum_var, alpha=0.1, color=COLORS[0])
    ax3.set_xlabel("Number of Components"); ax3.set_ylabel("Cumulative Variance (%)")
    ax3.set_title("Cumulative Explained Variance", fontsize=11, fontweight="bold")
    ax3.legend(); ax3.grid(alpha=0.3)

    fig.suptitle("Principal Component Analysis — Commodity Risk Factors",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6.  Rolling Volatility Chart
# ═══════════════════════════════════════════════════════════════════════

def plot_rolling_volatility(returns: pd.DataFrame,
                              window: int = 21,
                              filename: str = "rolling_volatility",
                              save: bool = True) -> plt.Figure:
    """
    Annualised rolling volatility for all assets — regime detection.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    roll_vol = returns.rolling(window).std() * np.sqrt(252) * 100

    for i, col in enumerate(roll_vol.columns):
        ax.plot(roll_vol.index, roll_vol[col],
                label=col, color=COLORS[i % len(COLORS)], lw=1.3, alpha=0.9)

    ax.set_title(f"Rolling {window}-Day Annualised Volatility",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Annualised Volatility (%)")
    ax.legend(fontsize=9, ncol=min(4, len(returns.columns)))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 7.  VaR & Return Distribution
# ═══════════════════════════════════════════════════════════════════════

def plot_var_distribution(returns: pd.Series,
                           var_levels: List[float] = [0.95, 0.99],
                           filename: str = "var_distribution",
                           save: bool = True) -> plt.Figure:
    """
    Return distribution with VaR and CVaR overlays.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r = returns.dropna()

    # ── Left: Histogram + Density ──
    ax = axes[0]
    ax.hist(r * 100, bins=80, density=True, color=COLORS[0],
            alpha=0.6, edgecolor="white", label="Empirical")

    x_range = np.linspace(r.min(), r.max(), 300)
    kde = stats.gaussian_kde(r)
    ax.plot(x_range * 100, kde(x_range), color=COLORS[0], lw=2)

    # Normal fit
    mu, sigma = r.mean(), r.std()
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range * 100, normal_pdf, color=COLORS[1], lw=1.5,
            ls="--", label="Normal Fit")

    # VaR lines
    for level, color in zip(var_levels, [COLORS[3], COLORS[5]]):
        var_val = np.percentile(r, (1-level)*100)
        cvar    = r[r <= var_val].mean()
        ax.axvline(var_val * 100, color=color, lw=2,
                   label=f"VaR {level*100:.0f}%: {var_val*100:.2f}%")
        ax.axvline(cvar * 100, color=color, lw=1.5, ls="--",
                   label=f"CVaR {level*100:.0f}%: {cvar*100:.2f}%")

    ax.set_title(f"{returns.name or 'Strategy'} — Return Distribution",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Daily Return (%)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Right: QQ Plot ──
    ax2 = axes[1]
    (osm, osr), (slope, intercept, r_val) = stats.probplot(r, dist="norm")
    ax2.scatter(osm, osr, color=COLORS[0], s=10, alpha=0.5, label="Empirical")
    line_x = np.array([min(osm), max(osm)])
    ax2.plot(line_x, slope * line_x + intercept,
             color=COLORS[3], lw=2, label="Normal Ref.")
    ax2.set_title("Q-Q Plot (vs Normal)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Theoretical Quantiles"); ax2.set_ylabel("Sample Quantiles")
    ax2.legend(); ax2.grid(alpha=0.3)
    skew = r.skew()
    kurt = r.kurtosis()
    ax2.text(0.05, 0.95, f"Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}",
             transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.1))

    plt.suptitle("Risk Distribution Analysis", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8.  Brent vs WTI Spread Dashboard
# ═══════════════════════════════════════════════════════════════════════

def plot_geo_spread(brent: pd.Series,
                    wti:   pd.Series,
                    filename: str = "brent_wti_spread",
                    save: bool = True) -> plt.Figure:
    """
    4-panel Brent-WTI spread dashboard.
    """
    spread = brent - wti
    spread_z = (spread - spread.rolling(63).mean()) / spread.rolling(63).std()

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Panel 1: Prices ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(brent.index, brent, color=COLORS[0], lw=1.5, label="Brent")
    ax1.plot(wti.index,   wti,   color=COLORS[1], lw=1.5, label="WTI", ls="--")
    ax1.set_title("Brent vs WTI Prices ($/bbl)", fontsize=11, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    # ── Panel 2: Spread ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(spread.index, spread, color=COLORS[4], lw=1.5)
    ax2.fill_between(spread.index, 0, spread,
                     where=spread > 0, alpha=0.25, color=COLORS[2], label="Brent Premium")
    ax2.fill_between(spread.index, 0, spread,
                     where=spread < 0, alpha=0.25, color=COLORS[3], label="WTI Premium")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Brent-WTI Spread ($/bbl)", fontsize=11, fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    # ── Panel 3: Z-Score ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(spread_z.index, spread_z, color=COLORS[0], lw=1.2)
    ax3.axhline(2, color=COLORS[3], ls="--", label="+2σ")
    ax3.axhline(-2, color=COLORS[2], ls="--", label="−2σ")
    ax3.fill_between(spread_z.index, 2, spread_z.clip(lower=2),
                     where=spread_z > 2, alpha=0.2, color=COLORS[3])
    ax3.fill_between(spread_z.index, spread_z.clip(upper=-2), -2,
                     where=spread_z < -2, alpha=0.2, color=COLORS[2])
    ax3.set_title("Spread Z-Score (63-Day Rolling)", fontsize=11, fontweight="bold")
    ax3.legend(); ax3.grid(alpha=0.3); ax3.set_ylim(-4, 4)

    # ── Panel 4: Spread Distribution ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(spread.dropna(), bins=60, color=COLORS[4], alpha=0.7,
             edgecolor="white", label="Spread Distribution")
    ax4.axvline(spread.mean(), color=COLORS[3], lw=2, label=f"Mean: ${spread.mean():.2f}")
    ax4.axvline(spread.median(), color=COLORS[2], lw=2, ls="--",
                label=f"Median: ${spread.median():.2f}")
    ax4.set_title("Spread Distribution", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Spread ($/bbl)"); ax4.legend(); ax4.grid(alpha=0.3)

    fig.suptitle("🛢  Brent vs WTI Geographical Spread Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, filename, save)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 9.  Strategy Performance Tearsheet
# ═══════════════════════════════════════════════════════════════════════

def plot_strategy_comparison(equity_curves: Dict[str, pd.Series],
                              filename: str = "strategy_comparison",
                              save: bool = True) -> plt.Figure:
    """
    Multi-strategy equity curve comparison on one chart.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # ── Panel 1: Equity Curves ──
    ax1 = axes[0]
    for i, (name, eq) in enumerate(equity_curves.items()):
        norm_eq = eq / eq.iloc[0] * 100
        ax1.plot(norm_eq.index, norm_eq, label=name,
                 color=COLORS[i % len(COLORS)], lw=2)
    ax1.axhline(100, color="gray", ls="--", alpha=0.5)
    ax1.set_title("Strategy Performance Comparison (Base = 100)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Indexed Value"); ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Underwater Plots (drawdowns) ──
    ax2 = axes[1]
    for i, (name, eq) in enumerate(equity_curves.items()):
        rets = eq.pct_change().dropna()
        cum  = (1 + rets).cumprod()
        roll_max = cum.cummax()
        dd   = (cum - roll_max) / roll_max * 100
        ax2.plot(dd.index, dd, label=name,
                 color=COLORS[i % len(COLORS)], lw=1.3, alpha=0.8)
        ax2.fill_between(dd.index, 0, dd, alpha=0.08,
                         color=COLORS[i % len(COLORS)])

    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Drawdown Comparison", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)"); ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, filename, save)
    return fig


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    df_prices = pd.DataFrame({
        "Gold": np.exp(np.cumsum(rng.normal(0.0003, 0.01, 500))),
        "WTI":  np.exp(np.cumsum(rng.normal(0.0002, 0.02, 500))),
        "Copper": np.exp(np.cumsum(rng.normal(0.0004, 0.015, 500))),
    }, index=idx) * 100

    rets = np.log(df_prices / df_prices.shift(1)).dropna()
    corr = rets.corr()

    fig1 = plot_correlation_heatmap(corr, save=False)
    fig2 = plot_rolling_volatility(rets, save=False)
    print("✅ visualization.py working correctly.")
    plt.close("all")
