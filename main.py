"""
============================================================
main.py — Full Pipeline Orchestrator
Commodity Arbitrage & Cross-Asset Correlation Analysis
Author: Arjun Sirohi | SirohiDS
GitHub: https://github.com/SirohiDS/commodity-arbitrage-analysis
============================================================

Usage
-----
    # Full pipeline (all 7 analyses)
    python main.py

    # Single analysis
    python main.py --run 01

    # Skip data download (use cache)
    python main.py --cache

    # Generate summary report only
    python main.py --summary

Options
-------
    --run N      Run only analysis N (01–07)
    --cache      Use cached data (skip download)
    --no-charts  Skip chart generation
    --summary    Print executive summary only
    --help       Show this message
"""

import sys
import os
import argparse
import time
import subprocess
from datetime import datetime

# ── Add project root to path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy  as np

from config import (
    CHARTS_DIR, OUTPUT_DIR, DATA_DIR,
    COMMODITY_TICKERS, CROSS_ASSET_TICKERS, FRED_SERIES,
    START_DATE, END_DATE,
)


# ═══════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║   Commodity Arbitrage & Cross-Asset Correlation Analysis            ║
║   Author   : Arjun Sirohi  |  SirohiDS                             ║
║   GitHub   : github.com/SirohiDS/commodity-arbitrage-analysis       ║
║   Version  : 1.0.0                                                  ║
║                                                                      ║
║   Instruments: Gold · WTI Oil · Brent · NatGas · Wheat · Copper    ║
║   Cross-Assets: S&P500 · TLT · HYG · DXY · TIPS · GLD · GDX       ║
║   Analyses   : Correlation · Cointegration · Basis · Hedging        ║
║                Geo-Spreads · Backtesting · Portfolio Risk            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

ANALYSES = {
    "01": ("Correlation Analysis",              "analysis/01_correlation_analysis.py"),
    "02": ("Cointegration & Stationarity",      "analysis/02_cointegration_stationarity.py"),
    "03": ("Basis Trading (Spot vs Futures)",   "analysis/03_basis_trading.py"),
    "04": ("Cross-Asset Hedging",               "analysis/04_cross_asset_hedging.py"),
    "05": ("Geographical Spreads (Brent/WTI)", "analysis/05_geo_spreads.py"),
    "06": ("Backtesting Framework",             "analysis/06_backtesting_framework.py"),
    "07": ("Portfolio Hedging Case Study",      "analysis/07_portfolio_hedging.py"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commodity Arbitrage & Cross-Asset Correlation Analysis"
    )
    parser.add_argument("--run",       type=str, default=None,
                        help="Run only analysis N (01-07)")
    parser.add_argument("--cache",     action="store_true",
                        help="Use cached data (skip download)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--summary",   action="store_true",
                        help="Print executive summary only")
    return parser.parse_args()


def print_env_check():
    """Verify environment and configuration."""
    print("\n[ENV CHECK]")

    # Python packages
    required = ["yfinance", "pandas", "numpy", "scipy", "statsmodels",
                 "sklearn", "matplotlib", "seaborn", "tabulate"]
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — run: pip install -r requirements.txt")

    # FRED API key
    from config import FRED_API_KEY
    if FRED_API_KEY and FRED_API_KEY != "YOUR_FRED_API_KEY_HERE":
        print("  ✓ FRED_API_KEY configured")
    else:
        print("  ⚠  FRED_API_KEY not set — macro data will be unavailable")
        print("     Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("     Add to .env: FRED_API_KEY=your_key_here")


def run_analysis(num: str, script: str) -> bool:
    """Execute an analysis script as a subprocess."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    if not os.path.exists(script_path):
        print(f"  ✗ Script not found: {script_path}")
        return False

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ✓ Analysis {num} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ Analysis {num} failed (exit code {result.returncode})")
        return False


def run_quick_summary():
    """Run an in-process quick summary without subprocess calls."""
    print("\n" + "="*70)
    print("  QUICK SUMMARY — Loading data & computing key metrics")
    print("="*70)

    try:
        from src.data_loader import get_commodity_data
        from src.statistical_analysis import (
            pearson_corr_matrix, full_statistical_summary,
            adf_test, engle_granger_coint,
        )
        from src.risk_metrics import risk_report
        from tabulate import tabulate

        data = get_commodity_data(use_cache=True)
        prices = data["all_prices"]
        rets   = data["all_returns"]

        print(f"\n  Data range: {rets.index[0].date()} → {rets.index[-1].date()}")
        print(f"  Instruments: {rets.shape[1]} | Observations: {len(rets)}")

        # Key stats
        stats_r = full_statistical_summary(
            prices[[c for c in prices.columns if c in rets.columns]], rets)
        s = stats_r["summary"]
        disp_cols = ["ann_return_%", "ann_vol_%", "sharpe_ratio", "max_drawdown_%"]
        disp_cols = [c for c in disp_cols if c in s.columns]

        core = [c for c in ["Gold_Spot", "WTI_Oil", "Brent_Oil", "NatGas",
                              "Wheat", "Copper"] if c in s.index]
        if core and disp_cols:
            print("\n  📊 Core Commodity Statistics:")
            print(tabulate(s.loc[core, disp_cols].round(3),
                           headers="keys", tablefmt="fancy_grid", floatfmt=".3f"))

        # Key correlations
        print("\n  📊 Key Correlations:")
        pairs = [("Gold_Spot", "SP500_ETF"),
                 ("WTI_Oil",   "Brent_Oil"),
                 ("Gold_Spot", "DXY"),
                 ("Copper",    "SP500_ETF")]
        for a, b in pairs:
            if a in rets.columns and b in rets.columns:
                r = rets[a].corr(rets[b])
                print(f"    {a:20s} ↔ {b:20s}: r = {r:.4f}")

        # Cointegration quick check
        print("\n  📊 Key Cointegration Tests:")
        coint_pairs = [("Gold_Spot", "Gold_ETF"), ("WTI_Oil", "Brent_Oil")]
        for a, b in coint_pairs:
            if a in prices.columns and b in prices.columns:
                cr = engle_granger_coint(prices[a], prices[b])
                status = "✓ Cointegrated" if cr["cointegrated"] else "✗ Not cointegrated"
                print(f"    {a:20s} ↔ {b:20s}: "
                      f"β={cr['beta']:.4f}  p={cr['p_value']:.4f}  {status}")

    except Exception as e:
        print(f"  ⚠  Quick summary failed: {e}")
        import traceback
        traceback.print_exc()


def generate_html_report(results_dir: str):
    """Generate a simple HTML summary report linking all charts."""
    charts = [f for f in os.listdir(CHARTS_DIR)
              if f.endswith(".png")]
    charts.sort()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Commodity Arbitrage & Cross-Asset Correlation Analysis</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 2px solid #21262d; padding-bottom: 10px; }}
  h2 {{ color: #79c0ff; margin-top: 40px; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 20px; margin: 20px 0; }}
  .chart-card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 12px; }}
  .chart-card img {{ width: 100%; border-radius: 4px; }}
  .chart-card p {{ font-size: 12px; color: #8b949e; margin: 8px 0 0; text-align: center; }}
  .meta {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; margin: 20px 0; }}
  .tag {{ background: #21262d; color: #58a6ff; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin: 3px; display: inline-block; }}
</style>
</head>
<body>
<h1>📊 Commodity Arbitrage & Cross-Asset Correlation Analysis</h1>
<div class="meta">
  <strong>Author:</strong> Arjun Sirohi | SirohiDS &nbsp;|&nbsp;
  <strong>GitHub:</strong> <a href="https://github.com/SirohiDS" style="color:#58a6ff">github.com/SirohiDS</a> &nbsp;|&nbsp;
  <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}
  <br><br>
  <span class="tag">Gold</span><span class="tag">WTI Oil</span><span class="tag">Brent</span>
  <span class="tag">Natural Gas</span><span class="tag">Wheat</span><span class="tag">Copper</span>
  <span class="tag">Statistical Arbitrage</span><span class="tag">Cointegration</span>
  <span class="tag">Hedging</span><span class="tag">Backtesting</span><span class="tag">VaR</span>
</div>
"""
    sections = {
        "01": "Analysis 01 — Cross-Asset Correlation",
        "02": "Analysis 02 — Cointegration & Stationarity",
        "03": "Analysis 03 — Basis Trading",
        "04": "Analysis 04 — Cross-Asset Hedging",
        "05": "Analysis 05 — Geographical Spreads",
        "06": "Analysis 06 — Backtesting Framework",
        "07": "Analysis 07 — Portfolio Hedging",
    }

    current_section = None
    for chart in charts:
        prefix = chart[:2]
        if prefix in sections and prefix != current_section:
            current_section = prefix
            html += f'<h2>{sections[prefix]}</h2>\n<div class="chart-grid">\n'

        rel_path = os.path.join("charts", chart)
        html += f'''  <div class="chart-card">
    <img src="{rel_path}" alt="{chart}" loading="lazy">
    <p>{chart.replace("_", " ").replace(".png", "")}</p>
  </div>\n'''

    if current_section:
        html += '</div>\n'

    html += "</body></html>"

    report_path = os.path.join(OUTPUT_DIR, "research_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"  💾 HTML report: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)
    args = parse_args()

    if args.summary:
        run_quick_summary()
        return

    print_env_check()
    print(f"\n  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data range: {START_DATE} → {END_DATE}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Charts dir: {CHARTS_DIR}")

    # Pre-cache data if --cache not set
    if not args.cache:
        print("\n[DATA] Downloading market data (this may take 1–2 minutes) …")
        try:
            from src.data_loader import get_commodity_data
            data = get_commodity_data(use_cache=False)
            print(f"  ✓ Data downloaded and cached")
            print(f"  ✓ Commodities: {data['commodity_prices'].shape}")
            print(f"  ✓ Cross-assets: {data['cross_asset_prices'].shape}")
        except Exception as e:
            print(f"  ⚠  Data download failed: {e}")
            print("  Using cached data if available …")

    # Run analyses
    total_start = time.time()
    results = {}

    if args.run:
        # Single analysis
        num = args.run.zfill(2)
        if num in ANALYSES:
            name, script = ANALYSES[num]
            print(f"\n[{num}] Running: {name}")
            results[num] = run_analysis(num, script)
        else:
            print(f"  ✗ Unknown analysis: {args.run}. Valid: 01–07")
    else:
        # All analyses
        print("\n[PIPELINE] Running all 7 analyses …")
        for num, (name, script) in ANALYSES.items():
            print(f"\n{'='*70}")
            print(f"  [{num}/07] {name}")
            print("="*70)
            results[num] = run_analysis(num, script)

    # Generate HTML report
    print("\n[REPORT] Generating HTML research report …")
    try:
        html_path = generate_html_report(OUTPUT_DIR)
        print(f"  Open: file://{html_path}")
    except Exception as e:
        print(f"  ⚠  HTML report generation failed: {e}")

    # Final summary
    elapsed = time.time() - total_start
    passed  = sum(1 for v in results.values() if v)
    failed  = len(results) - passed

    print(f"""
{'='*70}
  PIPELINE COMPLETE
  ─────────────────
  Analyses run:    {len(results)}/7
  Passed:          {passed}
  Failed:          {failed}
  Total time:      {elapsed:.1f}s
  Charts saved to: {CHARTS_DIR}
  HTML report:     {os.path.join(OUTPUT_DIR, "research_report.html")}
{'='*70}
""")

    if failed > 0:
        print("  ⚠  Some analyses failed. Common fixes:")
        print("     1. Install missing packages: pip install -r requirements.txt")
        print("     2. Set FRED_API_KEY in .env for macro data")
        print("     3. Check internet connection for Yahoo Finance")


if __name__ == "__main__":
    main()
