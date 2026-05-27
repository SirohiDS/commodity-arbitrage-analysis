"""
============================================================
config.py — Central configuration for the research project
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
============================================================
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()  # reads .env for FRED_API_KEY

# ── API Keys ──────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")
# Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html

# ── Date Range ────────────────────────────────────────────
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")  # 10 years
SHORT_START = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")  # 5 years

# ── Commodity Tickers (Yahoo Finance) ─────────────────────
COMMODITY_TICKERS = {
    # Gold
    "Gold_Spot":     "GC=F",       # Gold Futures (front month ≈ spot)
    "Gold_ETF":      "GLD",        # SPDR Gold Trust ETF
    "Gold_Miners":   "GDX",        # VanEck Gold Miners ETF
    "Gold_Jr":       "GDXJ",       # Junior Gold Miners

    # Oil
    "WTI_Oil":       "CL=F",       # WTI Crude Oil Futures
    "Brent_Oil":     "BZ=F",       # Brent Crude Oil Futures
    "Oil_ETF":       "USO",        # United States Oil Fund ETF
    "Energy_ETF":    "XLE",        # Energy Select Sector ETF

    # Natural Gas
    "NatGas":        "NG=F",       # Natural Gas Futures
    "NatGas_ETF":    "UNG",        # United States Natural Gas Fund

    # Wheat
    "Wheat":         "ZW=F",       # Wheat Futures (CBOT)
    "Ag_ETF":        "DBA",        # Invesco DB Agriculture Fund

    # Copper
    "Copper":        "HG=F",       # Copper Futures
    "Copper_ETF":    "CPER",       # United States Copper Index Fund
}

# ── Cross-Asset Tickers ───────────────────────────────────
CROSS_ASSET_TICKERS = {
    # Equities
    "SP500":         "^GSPC",      # S&P 500 Index
    "SP500_ETF":     "SPY",        # SPDR S&P 500 ETF
    "NASDAQ":        "^IXIC",      # NASDAQ Composite
    "Russell2000":   "^RUT",       # Russell 2000 (Small Cap)
    "MSCI_EM":       "EEM",        # Emerging Markets ETF

    # Bonds
    "TLT":           "TLT",        # iShares 20+ Year Treasury Bond ETF
    "IEF":           "IEF",        # iShares 7-10 Year Treasury ETF
    "HYG":           "HYG",        # iShares High Yield Corporate Bond ETF
    "TIP":           "TIP",        # iShares TIPS ETF (inflation-linked)

    # Currencies
    "DXY":           "DX-Y.NYB",   # US Dollar Index
    "EURUSD":        "EURUSD=X",   # EUR/USD
    "USDJPY":        "USDJPY=X",   # USD/JPY (safe haven)
    "AUDUSD":        "AUDUSD=X",   # AUD/USD (commodity currency)
    "USDCAD":        "USDCAD=X",   # USD/CAD (oil-linked)

    # Multi-Asset
    "GSCI":          "GSG",        # iShares S&P GSCI Commodity ETF
    "DBC":           "DBC",        # Invesco DB Commodity Index ETF
}

# ── FRED Series IDs ───────────────────────────────────────
FRED_SERIES = {
    "CPI":           "CPIAUCSL",   # Consumer Price Index (All Urban)
    "PPI":           "PPIACO",     # Producer Price Index (All Commodities)
    "Fed_Funds":     "FEDFUNDS",   # Federal Funds Rate
    "10Y_Treasury":  "GS10",       # 10-Year Treasury Constant Maturity
    "2Y_Treasury":   "GS2",        # 2-Year Treasury Constant Maturity
    "Breakeven_10Y": "T10YIE",     # 10-Year Breakeven Inflation Rate
    "VIX":           "VIXCLS",     # CBOE Volatility Index
    "WTI_FRED":      "DCOILWTICO", # WTI Crude Oil Spot (FRED)
    "NatGas_FRED":   "DHHNGSP",    # Henry Hub Natural Gas Spot
    "Gold_FRED":     "GOLDAMGBD228NLBM",  # Gold Spot London Bullion
    "Wheat_FRED":    "PWHEAMTUSDM",# IMF Wheat Price
    "Copper_FRED":   "PCOPPUSDM",  # IMF Copper Price
    "USD_Index":     "DTWEXBGS",   # USD Broad Goods & Services Index
    "M2_Money":      "M2SL",       # M2 Money Supply
}

# ── Trading Strategy Parameters ───────────────────────────
STRATEGY_PARAMS = {
    # Statistical Arbitrage
    "zscore_entry":   2.0,         # Enter trade when z-score > 2σ
    "zscore_exit":    0.5,         # Exit when z-score < 0.5σ
    "zscore_stop":    3.5,         # Stop-loss at 3.5σ
    "lookback_corr":  252,         # Rolling correlation window (1 year)
    "lookback_vol":   21,          # Rolling volatility window (1 month)

    # Cointegration
    "coint_pvalue":   0.05,        # Significance threshold for cointegration
    "adf_pvalue":     0.05,        # ADF stationarity threshold

    # Portfolio
    "risk_free_rate": 0.05,        # Annual risk-free rate (5%)
    "position_size":  100_000,     # Notional position size in USD
    "max_leverage":   2.0,         # Maximum portfolio leverage
}

# ── Output Paths ──────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR  = os.path.join(OUTPUT_DIR, "charts")
DATA_DIR    = os.path.join(BASE_DIR, "data")

for _dir in [OUTPUT_DIR, CHARTS_DIR, DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Plotting Style ────────────────────────────────────────
PLOT_STYLE     = "seaborn-v0_8-darkgrid"
FIGURE_DPI     = 150
COLORMAP       = "RdYlGn"       # Red-Yellow-Green for correlation heatmaps
COLOR_PALETTE  = [
    "#2196F3",   # Blue   — primary
    "#FF9800",   # Orange — secondary
    "#4CAF50",   # Green  — positive
    "#F44336",   # Red    — negative
    "#9C27B0",   # Purple — neutral
    "#00BCD4",   # Cyan   — accent
]

print("✅  Config loaded — project root:", BASE_DIR)
