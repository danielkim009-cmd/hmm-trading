# 📡 HMM Regime Terminal

A professional **regime-based algorithmic trading system** that uses a **7-state Gaussian Mixture Model Hidden Markov Model (GMM-HMM)** to detect market states and layers technical confirmation strategies with strict risk management on top.

Works with any ticker recognised by Yahoo Finance — stocks, ETFs, crypto, futures, and indices.

> *Inspired by the pioneering work of **Jim Simons** and [**Renaissance Technologies**](https://www.rentec.com/), whose Medallion Fund demonstrated that quantitative, model-driven strategies — including Hidden Markov Models — could systematically extract signal from market noise. This project applies those same principles at an accessible scale.*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

🚀 **Streamlit App** [gmm-hmm-trading-and-backtest.streamlit.app](https://gmm-hmm-trading-and-backtest.streamlit.app/)

🎬 **Video Overview** [Watch on YouTube](https://youtu.be/KOgGnvhgtFA?si=0prCtgkSfel8AzKv) — generated with Google NotebookLM

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/danielkim009-cmd/hmm-trading.git
cd hmm-trading

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. On first load it downloads ~365 days of daily OHLCV data and fits the GMM-HMM (~10–30 seconds). Subsequent loads use a 5-minute cache.

---

## ✨ Features

### 📡 Backtester Page
- **Detected Regime** — Bull Run / Bear/Crash / Neutral/Transition with sub-state name (e.g. "State 6 · Steady Uptrend")
- **Live Quote** — real-time price via `yf.fast_info` (60-second refresh) with PST timestamp
- **Company Profile** — sector, industry, market cap, employee count, business summary
- **Daily or Monthly Timeframe** — daily bars (1–5 years) or monthly bars (3–20 years) for long-term regime analysis
- **TradingView-style Price Chart** — candlestick with HMM regime background bands, EMA-21 / EMA-50 / EMA-100 overlays, volume pane, inline OHLC + volume legend chip with prev-close day change, buy/sell markers (sell shows P&L %) with an entry/exit-rule explainer, and a **10-day kernel regression forecast** with ±1σ confidence band
- **Technical Confirmation Scorecard** — 10 configurable checks (momentum, ATR, volume, ADX, MACD, Stochastic, SMAs, RSI)
- **Equity Curve** — strategy vs. Buy & Hold with Sharpe Ratio comparison
- **Performance Metrics** — Total Return, Alpha, Max Drawdown, Win Rate, B&H Sharpe delta
- **Trade Log** — full table with CSV export
- **Parameter Optimizer** — two-phase search: greedy confirmation-filter selection, then a grid over `bear_confirm_days`, `min_confirms`, `min_hold_days` — every combo benchmarked against Buy & Hold
- **Watchlist** — saved tickers with dropdown, auto-add on type, persisted in browser localStorage

### 🔍 Stock Screener Page
- Scans **S&P 500**, **Nasdaq 100**, or **Russell 2000** for Bull Run entries in parallel
- Ranks results by composite entry-quality score (HMM Bull probability + confirmations + freshness)
- Configurable min confirmations, max bars in bull run, thread count

---

## 🧠 How It Works

### GMM-HMM Engine
A **7-state GMMHMM** (`hmmlearn`) is fit on 4 engineered daily features:

| Feature | Formula | Captures |
|---|---|---|
| `returns` | log(Closeₜ / Closeₜ₋₁) | Daily directional momentum |
| `range` | (High − Low) / Close | Intraday volatility |
| `vol_change` | StdDev(Volume,20) / Mean(Volume,20) | Abnormal volume |
| `trend_return` | log(Closeₜ / Closeₜ₋₂₀) | 20-day trend |

States are auto-labelled by mean return:
- **Top 3** → 🟢 Bull Run (sub-states 5–7)
- **Bottom 2** → 🔴 Bear/Crash (sub-states 1–2)
- **Middle 2** → 🟡 Neutral/Transition (sub-states 3–4)

### Strategy Logic
- **Entry** — any Bull Run bar while flat with ≥ N confirmations met, or a Bear→Neutral transition with ≥ N confirmations (early-recovery entry)
- **Exit** — after `bear_confirm_days` consecutive Bear bars, once the position has been held ≥ `min_hold_days`; every exit starts a 2-day re-entry cooldown
- **Regime-Only Mode** — bypass confirmations entirely: enter on Bear→Bull or Bear→Neutral transitions, exit on the first Bear bar after the minimum hold

### 10-Day Kernel Regression Forecast
For today's feature vector (`vol_surge`, `atr_ratio`, `pct_from_high`, `momentum_5d`, `regime_num`), a Gaussian kernel finds the weighted distribution of similar historical 10-day forward paths — yielding an expected price path and ±1σ confidence band displayed as dashed lines on the chart.

---

## ⚙️ Configuration

All controls are in the sidebar:

| Control | Default | Description |
|---|---|---|
| Ticker Symbol | SPY | Any ticker recognised by Yahoo Finance (see examples below) |
| Timeframe | Daily | Daily bars (up to 5 years) or Monthly bars (up to 20 years) |
| Look-back Period | 365 days / 10 years | HMM training window — days for Daily, years for Monthly |
| Leverage | 1× | 1×, 2×, or 4× position sizing |
| Bear Confirm Days | 5 | Consecutive Bear bars required to exit |
| Min Hold Days | 7 | Minimum calendar days a position must be held before a regime exit can fire (0 = no minimum) |
| Starting Capital | $10,000 | Backtest starting equity ($1,000 – $10,000,000) |
| Regime-Only Mode | Off | Trade on regime transitions only |
| Min Confirmations | 3/10 | Minimum technical checks for entry |

### 🔖 Ticker Examples

Ticker symbols must be recognised by Yahoo Finance. The default watchlist includes:

| Category | Tickers |
|---|---|
| **Stocks** | AAPL, ALAB, AMD, AMZN, AVGO, CIEN, COHR, CRDO, GEV, GLW, GOOG, INTC, LITE, MU, NBIS, NTAP, NVDA, QCOM, SNDK, TSLA, VRT, WDC |
| **Crypto** | BTC-USD, ETH-USD, SOL-USD |
| **Futures** | CL=F (Crude Oil), ES=F (S&P 500 Futures), GC=F (Gold), HG=F (Copper), SI=F (Silver) |
| **ETFs** | EWY, GDX, GLD, IWM, QQQ, SLV, SPY, TLT, USO |

Any other valid Yahoo Finance symbol can be entered manually in the Ticker Symbol field.

---

## 📁 Project Structure

```
hmm-trading/
├── dashboard.py          # Streamlit UI — two pages: Backtester + Screener
├── backtester.py         # GMM-HMM engine, confirmations, strategy, metrics, optimizer
├── data_loader.py        # yfinance download + feature engineering
├── scanner.py            # Multi-index parallel Bull Run screener
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python 3.12 pin for Streamlit Community Cloud
├── watchlist.json        # Legacy saved-tickers file (gitignored — watchlist now lives in browser localStorage)
└── project_overview.html # Detailed technical documentation (open in browser)
```

---

## 📦 Dependencies

```
yfinance >= 0.2.36
hmmlearn >= 0.3.2
pandas   >= 2.0.0
numpy    >= 1.26.4
scikit-learn >= 1.3.0
streamlit >= 1.32.0
plotly   >= 5.18.0
streamlit-javascript >= 0.1.5
```

The price chart uses **lightweight-charts v4.2.1** loaded from the unpkg CDN — no additional install required. `streamlit-javascript` powers the browser-localStorage watchlist persistence.

---

## 📖 Full Documentation

Open the [full technical documentation](https://danielkim009-cmd.github.io/hmm-trading/project_overview.html) for a complete reference covering the GMM-HMM architecture, all 7 sub-states, strategy logic, confirmation rules, kernel forecast algorithm, and more.

---

## ⚠️ Disclaimer

For educational and research purposes only. Not financial advice. HMM regime labels are in-sample statistical artefacts and do not predict future returns. Past performance does not guarantee future results.
