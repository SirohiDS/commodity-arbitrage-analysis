# 📊 [Commodity Arbitrage & Cross-Asset Correlation Analysis](https://github.com/SirohiDS/commodity-arbitrage-analysis)

**Portfolio-grade quantitative research project** — by [Arjun Sirohi](https://github.com/SirohiDS)

[![View on GitHub](https://img.shields.io/badge/🔗%20View%20on%20GitHub-commodity--arbitrage--analysis-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SirohiDS/commodity-arbitrage-analysis)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-SirohiDS-black?logo=github)](https://github.com/SirohiDS)

---

## 🎯 Objective

Analyse statistical relationships between **5 core commodities** (Gold, WTI Oil, Brent, Natural Gas, Wheat, Copper) and **cross-assets** (equities, bonds, currencies, commodity ETFs) to:

- Identify **profitable statistical arbitrage** opportunities
- Build **robust hedging strategies** against supply shocks and inflation
- Perform **portfolio risk management** using derivatives and ETFs
- Deliver **backtested, Sharpe-ratio-validated trading strategies**

---

## 🏗️ Project Structure

```
commodity-arbitrage-analysis/
│
├── src/                            # Core quantitative modules
│   ├── data_loader.py              # Yahoo Finance + FRED data ingestion
│   ├── statistical_analysis.py     # Correlation, cointegration, ADF, KPSS, GARCH
│   ├── backtesting.py              # Event-driven backtesting engine
│   ├── hedging.py                  # OLS hedge ratios, min-var, risk parity
│   ├── risk_metrics.py             # VaR, CVaR, Sharpe, Sortino, Kelly
│   └── visualization.py            # Publication-quality charts
│
├── analysis/                       # 7 standalone analysis scripts
│   ├── 01_correlation_analysis.py  # Full cross-asset correlation dashboard
│   ├── 02_cointegration_stationarity.py  # ADF/KPSS, Engle-Granger, Johansen
│   ├── 03_basis_trading.py         # Spot vs futures basis + contango/backwardation
│   ├── 04_cross_asset_hedging.py   # Gold-GDX, rolling hedge ratios, PCA
│   ├── 05_geo_spreads.py           # Brent-WTI spread dashboard
│   ├── 06_backtesting_framework.py # Multi-strategy performance comparison
│   └── 07_portfolio_hedging.py     # Supply shock + inflation hedge case study
│
├── docs/
│   └── executive_summary.md        # Full research findings + trading insights
│
├── outputs/
│   └── charts/                     # All generated PNG charts (auto-created)
│
├── data/                           # CSV cache (auto-created after first run)
├── main.py                         # Full pipeline orchestrator
├── config.py                       # All tickers, parameters, paths
├── requirements.txt                # Python dependencies
├── push_to_github.sh               # One-click GitHub publisher
└── .env.example                    # API key template
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key (optional but recommended)
```bash
cp .env.example .env
# Edit .env and add your free FRED API key
# Get key: https://fred.stlouisfed.org/docs/api/api_key.html
```

### 3. Run full pipeline
```bash
python main.py
```

### 4. Run individual analyses
```bash
python main.py --run 01   # Correlation analysis
python main.py --run 02   # Cointegration & stationarity
python main.py --run 03   # Basis trading
python main.py --run 04   # Cross-asset hedging
python main.py --run 05   # Brent-WTI geo spread
python main.py --run 06   # Backtesting framework
python main.py --run 07   # Portfolio hedging case study
```

### 5. Quick summary (no charts)
```bash
python main.py --summary
```

---

## 📈 Instruments Covered

### Commodities (Spot/Futures)
| Instrument | Yahoo Ticker | Description |
|------------|-------------|-------------|
| Gold | GC=F | COMEX Gold Futures (front month) |
| WTI Crude Oil | CL=F | NYMEX WTI Crude Futures |
| Brent Crude | BZ=F | ICE Brent Crude Futures |
| Natural Gas | NG=F | NYMEX Henry Hub NatGas |
| Wheat | ZW=F | CBOT Wheat Futures |
| Copper | HG=F | COMEX Copper Futures |

### Cross-Asset Instruments
| Category | Instruments |
|----------|------------|
| Commodity ETFs | GLD, USO, UNG, DBA, CPER, GSG, DBC |
| Miners/Sector | GDX, GDXJ, XLE |
| Equities | SPY, QQQ, EEM, ^GSPC |
| Bonds | TLT, IEF, HYG, TIP |
| Currencies | DX-Y.NYB, EURUSD=X, AUDUSD=X, USDCAD=X |

### FRED Macro Series
CPI, PPI, Federal Funds Rate, 10Y Treasury, 2Y Treasury, 10Y Breakeven Inflation, VIX, M2 Money Supply

---

## 🔬 Statistical Methods

| Method | Implementation | Purpose |
|--------|---------------|---------|
| Pearson Correlation | `scipy.stats` | Linear co-movement |
| Spearman Correlation | `pandas` | Rank-based (robust to outliers) |
| Rolling Correlation | `pandas.rolling` | Regime-aware correlations |
| Fisher Z-test | `scipy.stats` | Correlation change detection |
| ADF Test | `statsmodels.tsa` | Stationarity testing |
| KPSS Test | `statsmodels.tsa` | Complementary stationarity |
| Engle-Granger | `statsmodels.tsa` | Pairwise cointegration |
| Johansen Test | `statsmodels.tsa` | Multivariate cointegration |
| OLS Regression | `statsmodels.api` | Hedge ratio estimation |
| Rolling OLS | `statsmodels.regression` | Time-varying hedge ratios |
| GARCH(1,1) | `arch` | Volatility clustering |
| Granger Causality | `statsmodels.tsa` | Lead-lag relationships |
| PCA | `sklearn.decomposition` | Factor decomposition |

---

## 💹 Trading Strategies (Backtested)

### 1. Z-Score Statistical Arbitrage
```
Entry:  |z-score| > 2.0σ  (spread is abnormally wide)
Exit:   |z-score| < 0.5σ  (spread reverts to mean)
Stop:   |z-score| > 3.5σ  (relationship has broken)
```
**Applied to:** WTI-Brent Spread, Gold-GLD Basis, Gold-Miners Spread

### 2. Momentum (Time-Series)
```
Signal: Sign of past 21-day cumulative return
Long if past return > 0, Short if past return < 0
```
**Applied to:** Gold, WTI Oil

### 3. Basis Mean-Reversion
```
Fade extreme basis deviations between futures and spot
Contango → Short futures (roll yield decay)
Backwardation → Long futures (positive carry)
```

---

## 📊 Key Performance Metrics

All strategies are evaluated on:

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | (Ann. Return − Risk-Free) / Ann. Vol |
| Sortino Ratio | Penalises only downside volatility |
| Calmar Ratio | Ann. Return / \|Max Drawdown\| |
| Max Drawdown | Peak-to-trough decline |
| Win Rate | % of trades profitable |
| Profit Factor | Total wins / Total losses |
| VaR 95% & 99% | Value-at-Risk (Historical, Parametric, MC) |
| CVaR 95% | Expected loss given VaR breach |
| Kelly Fraction | Optimal bet sizing |

---

## 📉 Risk Management Framework

### Value-at-Risk Methods
1. **Historical VaR** — Uses actual historical returns, no distribution assumption
2. **Parametric VaR** — Assumes normal distribution (underestimates tail risk)
3. **Monte Carlo VaR** — Simulates paths under Student-t (fat tails)

### Hedging Approach
```python
from src.hedging import ols_hedge_ratio, rolling_hedge_ratio

# Static hedge ratio
hr = ols_hedge_ratio(spot_returns, hedge_returns)
print(f"Hedge ratio: {hr['beta']:.4f}")
print(f"Effectiveness: {hr['he_pct']:.1f}%")

# Time-varying hedge ratio (recommended)
roll_hr = rolling_hedge_ratio(spot_returns, hedge_returns, window=63)
```

### Portfolio Construction
```python
from src.hedging import minimum_variance_portfolio, risk_parity_portfolio

# Minimum variance weights
mv = minimum_variance_portfolio(returns_df, allow_short=False)

# Risk parity weights
rp = risk_parity_portfolio(returns_df)
```

---

## 🗃️ Data Sources

| Source | Access | Data Type |
|--------|--------|-----------|
| Yahoo Finance | Free (via `yfinance`) | Prices, volumes, adjusted close |
| FRED (St. Louis Fed) | Free API key required | CPI, rates, money supply |
| CME Group | Free historical (via yfinance futures tickers) | Futures OHLCV |

---

## 📋 Dependencies

```
yfinance >= 0.2.28    # Market data
pandas >= 2.0.0       # Data manipulation
numpy >= 1.24.0       # Numerical computing
scipy >= 1.11.0       # Statistical tests
statsmodels >= 0.14   # Econometrics (ADF, cointegration, OLS)
scikit-learn >= 1.3   # PCA, preprocessing
arch >= 6.2.0         # GARCH volatility models
matplotlib >= 3.7     # Visualisation
seaborn >= 0.12       # Statistical charts
fredapi >= 0.5.0      # FRED macro data
plotly >= 5.15.0      # Interactive charts
```

---

## 🗝️ Key Financial Insights

### The Cointegration Opportunity
> Gold spot and GLD ETF have a 97%+ cointegrating relationship. Any deviation > 0.5% is a near risk-free arbitrage — exploited by ETF market makers daily.

### The Brent-WTI Convergence
> Normally ~$2.50/bbl Brent premium. When z-score exceeds ±1.5σ, mean reversion occurs within 3–15 days with ~60% win rate and Sharpe >1.2.

### The Inflation Hedge Pyramid
> TIPS > Energy Commodities > Gold > Copper > Equities > Long Bonds (negative hedge)

### The Supply Shock Protocol
> When oil/gas drops >4% in a day: buy Gold (+0.4% avg) + TIPS (+0.2% avg) + DXY (+0.3% avg)

---

## 📬 Contact & Attribution

**Author:** Arjun Sirohi  
**Email:** sirohi96@outlook.com  
**GitHub:** [github.com/SirohiDS](https://github.com/SirohiDS)  
**Reference:** [supply-chain-shock-predictor](https://github.com/SirohiDS/supply-chain-shock-predictor)

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute investment advice. Past backtest performance does not guarantee future results. Commodity markets carry significant financial risk.

---

*Generated with Python | Data: Yahoo Finance + FRED | Role: Quantitative Commodity Analyst & Risk Strategist*
