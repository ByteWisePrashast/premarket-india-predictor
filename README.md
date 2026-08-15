# Pre-Market India & Multi-Asset Investment Engine

An institutional-grade quantitative prediction and multi-asset intelligence system for Indian markets.

Provides an 08:45 IST Nifty 50 opening-gap forecast alongside comprehensive **"Can I Buy?"** evaluations across **Stocks**, **ETFs**, and **Mutual Funds** for both **Short-Term** swing trading and **Long-Term** wealth compounding.

---

## Key Features

### 1. Pre-Market Nifty 50 Gap Forecast
- Quantitative probability split for **Gap Up**, **Gap Down**, and **Flat**.
- Composite scoring blending GIFT Nifty, Global Markets (S&P 500, Nasdaq, Nikkei, Hang Seng), Macro & Volatility (USD/INR, Brent Crude, US 10Y Yield, India VIX), Institutional Cash Flows (FII & DII), and News Sentiment.

### 2. Multi-Asset "Can I Buy?" Master Decision Matrix
- Unified search for any:
  - **NSE / BSE Stocks** (e.g. `RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `TATAMOTORS`)
  - **Indian ETFs** (e.g. `NIFTYBEES`, `BANKBEES`, `GOLDBEES`, `SILVERBEES`, `ITBEES`, `MON100`, `CPSEETF`)
  - **Mutual Funds** (e.g. `Parag Parikh Flexi Cap`, `Quant Small Cap`, `HDFC Balanced Advantage`, `Mirae Asset Large Cap`, `UTI Nifty 50 Index`)
- Instant 0–100 Decision Score with clear verdict badges (`STRONG BUY`, `BUY ON DIPS`, `HOLD / WATCHLIST`, `AVOID / HIGH RISK`).

### 3. Dual-Horizon Investment Guidance
- **⚡ Short-Term Trading Analysis (Intraday to 4-Week Horizon)**:
  - Actionable Trade Levels: Suggested Entry Zone, Target 1 (Conservative), Target 2 (Extended), Stop Loss / Invalidation, and Risk-to-Reward ratio.
  - Technical Matrix: RSI(14) with momentum zones, MACD bullish/bearish crossovers, EMA 9/21/50/200 structure, ATR daily volatility, and volume breakout detection.
- **🏛️ Long-Term Investment Analysis (1 to 5+ Year Horizon)**:
  - Multi-Year Compounding: 1-Month, 6-Month, 1-Year, 3-Year CAGR, and 5-Year CAGR.
  - Valuation & Quality: Trailing P/E, Forward P/E, Price-to-Book (P/B), Return on Equity (ROE %), Dividend Yield, Market Cap, and 200-day EMA trend alignment.
  - Risk & Volatility: Maximum Drawdown from peak, Volatility ratings, and Systematic SIP suitability.

### 4. Mutual Fund Specific Intelligence (via AMFI/MFAPI)
- Direct & Regular scheme historical NAV analysis across 40,000+ Indian mutual funds.
- **Monthly SIP Returns Simulator**: Calculates total invested vs accumulated value over 1Y, 3Y, and 5Y (based on ₹10,000/month).
- **Risk-Adjusted Alpha Metrics**: Annualized Volatility, Sharpe Ratio (vs 6.5% Indian G-Sec), Sortino Ratio (downside risk protection), and Max Drawdown.
- **Exit Load & STCG Advisory**: Clear notices on exit load periods and taxation for short-term vs long-term holding.

### 5. Multi-Asset Watchlists
- **📈 Stocks Watchlist**: Top intraday & swing momentum setups with targets and stop-losses.
- **🪙 ETFs Watchlist**: High-liquidity Index, Gold, Silver, Sector, and International tech ETFs.
- **🏦 Mutual Funds Watchlist**: Top-performing direct growth schemes across Flexi Cap, Small Cap, Hybrid, and Index categories.

---

## Setup & Running

```bash
cd /Users/prxst/premarket_india_predictor
source .venv/bin/activate
pip install -r requirements.txt
```

### Start Web Dashboard
```bash
python app.py
```
Open in browser:
```text
http://127.0.0.1:8000
```

### Run Terminal Pre-Market CLI Report
```bash
python premarket_predictor.py
```
To run early and wait until 08:45 IST:
```bash
python premarket_predictor.py --wait-until-845
```

---

## REST API Endpoints

- `GET /api/asset/<query>` — Evaluates any Stock, ETF, or Mutual Fund.
- `GET /api/search-suggest?q=<query>` — Instant search autocomplete for stocks, ETFs, and mutual funds.
- `GET /api/report` — JSON pre-market market cues and gap predictions.
- `GET /api/top-stocks` — Top watchlist trade candidates.

---

## Disclaimer
This is a quantitative research and analytics system for market analysis and asset evaluation. Not investment or financial advisory.
