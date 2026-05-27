#!/bin/bash
# ============================================================
# push_to_github.sh — Publish project to GitHub
# Usage: bash push_to_github.sh [github-username] [repo-name] [pat-token]
# ============================================================

set -e

GITHUB_USER="${1:-SirohiDS}"
REPO_NAME="${2:-commodity-arbitrage-analysis}"
PAT_TOKEN="${3:-}"   # Pass as arg or set GITHUB_PAT env var
PAT_TOKEN="${PAT_TOKEN:-$GITHUB_PAT}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Commodity Arbitrage — GitHub Publisher                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  GitHub User : $GITHUB_USER"
echo "  Repository  : $REPO_NAME"
echo ""

if [ -z "$PAT_TOKEN" ]; then
  echo "  ✗ No PAT token found."
  echo "  Provide it as:"
  echo "    bash push_to_github.sh SirohiDS commodity-arbitrage-analysis YOUR_PAT"
  echo "  OR:"
  echo "    export GITHUB_PAT=your_token && bash push_to_github.sh"
  exit 1
fi

# ── Step 1: Create repo via GitHub API ─────────────────────────────
echo "[1/4] Creating GitHub repository …"
REPO_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: token $PAT_TOKEN" \
  "https://api.github.com/repos/$GITHUB_USER/$REPO_NAME")

if [ "$REPO_EXISTS" = "200" ]; then
  echo "  ✓ Repository already exists — will push to it"
else
  curl -s -X POST \
    -H "Authorization: token $PAT_TOKEN" \
    -H "Content-Type: application/json" \
    "https://api.github.com/user/repos" \
    -d "{
      \"name\": \"$REPO_NAME\",
      \"description\": \"Commodity Arbitrage & Cross-Asset Correlation Analysis — Quant Research by Arjun Sirohi\",
      \"private\": false,
      \"has_issues\": true,
      \"has_wiki\": false,
      \"auto_init\": false
    }" > /dev/null
  echo "  ✓ Repository created: https://github.com/$GITHUB_USER/$REPO_NAME"
fi

# ── Step 2: Initialise git ──────────────────────────────────────────
echo "[2/4] Initialising git repository …"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d .git ]; then
  git init
  echo "  ✓ Git repository initialised"
fi

# Configure git user
git config user.email "sirohi96@outlook.com" 2>/dev/null || true
git config user.name  "Arjun Sirohi"         2>/dev/null || true

# Add all files
git add -A

# ── Step 3: Commit ─────────────────────────────────────────────────
echo "[3/4] Committing files …"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git commit -m "🚀 Initial release: Commodity Arbitrage & Cross-Asset Correlation Analysis

Portfolio-grade quant research project by Arjun Sirohi (SirohiDS)

Analysis Modules:
- 01_correlation_analysis.py       — Pearson/Spearman/Rolling correlations
- 02_cointegration_stationarity.py — ADF/KPSS, Engle-Granger, Johansen
- 03_basis_trading.py              — Spot vs Futures basis / contango analysis
- 04_cross_asset_hedging.py        — Gold-GDX, OLS hedge ratios, PCA
- 05_geo_spreads.py                — Brent-WTI spread dashboard
- 06_backtesting_framework.py      — Multi-strategy backtest + risk metrics
- 07_portfolio_hedging.py          — Supply shock + inflation hedge case study

Tools: pandas, numpy, scipy, statsmodels, sklearn, arch, matplotlib
Data: Yahoo Finance, FRED Economic Data

Reference: https://github.com/SirohiDS/supply-chain-shock-predictor
Generated: $TIMESTAMP" 2>/dev/null || \
git commit --allow-empty -m "Commodity Arbitrage Analysis — $TIMESTAMP"

echo "  ✓ Committed"

# ── Step 4: Push ────────────────────────────────────────────────────
echo "[4/4] Pushing to GitHub …"

REMOTE_URL="https://$PAT_TOKEN@github.com/$GITHUB_USER/$REPO_NAME.git"

# Set or update remote
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

# Push (force to handle re-initialised repos)
git branch -M main
git push -u origin main --force

echo ""
echo "  ✅ Successfully published!"
echo "  🔗 https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
