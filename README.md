# 📡 HMM Regime Terminal

A professional **regime-based algorithmic trading system** that uses a **7-state Gaussian Mixture Model Hidden Markov Model (GMM-HMM)** to detect market states and layers technical confirmation strategies with strict risk management on top.

Works with any Yahoo Finance ticker — stocks, ETFs, crypto, indices.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/hmm-trading.git
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
- **TradingView-style Price Chart** — candlestick with HMM regime background bands, SMA-50, volume histogram, buy/sell markers (sell shows P&L %), and a **10-day kernel regression forecast** with ±1σ confidence band
- **Technical Confirmation Scorecard** — 10 configurable checks (momentum, ATR, volume, ADX, MACD, Stochastic, SMAs, RSI)
- **Equity Curve** — strategy vs. Buy & Hold with Sharpe Ratio comparison
- **Performance Metrics** — Total Return, Alpha, Max Drawdown, Win Rate, B&H Sharpe delta
- **Trade Log** — full table with CSV export
- **Parameter Optimizer** — grid-searches `bear_confirm_days`, `min_confirms`, `min_hold_days` to maximise return
- **Watchlist** — saved tickers with dropdown, auto-add on type, persistent across sessions

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
- **Entry** — triggers on Bull Run transition when ≥ N confirmations are met
- **Exit** — triggers after `bear_confirm_days` consecutive Bear bars
- **Regime-Only Mode** — bypass confirmations entirely; trade on regime transitions alone

### 10-Day Kernel Regression Forecast
For today's feature vector (`vol_surge`, `atr_ratio`, `pct_from_high`, `momentum_5d`, `regime_num`), a Gaussian kernel finds the weighted distribution of similar historical 10-day forward paths — yielding an expected price path and ±1σ confidence band displayed as dashed lines on the chart.

---

## ⚙️ Configuration

All controls are in the sidebar:

| Control | Default | Description |
|---|---|---|
| Ticker Symbol | BTC-USD | Any Yahoo Finance symbol |
| Look-back Period | 365 days | Historical data window for HMM training |
| Leverage | 1× | 1×, 2×, or 4× position sizing |
| Bear Confirm Days | 5 | Consecutive Bear bars required to exit |
| Min Hold Days | 7 | Minimum hold before trailing stop fires |
| Trailing Stop | Off | 2% trailing stop-loss |
| Regime-Only Mode | Off | Trade on regime transitions only |
| Min Confirmations | 3/10 | Minimum technical checks for entry |

---

## 📁 Project Structure

```
hmm-trading/
├── dashboard.py          # Streamlit UI — two pages: Backtester + Screener
├── backtester.py         # GMM-HMM engine, confirmations, strategy, metrics
├── data_loader.py        # yfinance download + feature engineering
├── scanner.py            # Multi-index parallel Bull Run screener
├── requirements.txt      # Python dependencies
├── watchlist.json        # Saved tickers (auto-created, gitignored)
└── project_overview.html # Detailed technical documentation (open in browser)
```

---

## 📦 Dependencies

```
yfinance >= 0.2.36
hmmlearn >= 0.3.2
pandas   >= 2.0.0
numpy    >= 1.24.0
scikit-learn >= 1.3.0
streamlit >= 1.32.0
plotly   >= 5.18.0
```

The price chart uses **lightweight-charts v4.2.1** loaded from the unpkg CDN — no additional install required.

---

## 📖 Full Documentation

Open the [full technical documentation](https://danielkim009-cmd.github.io/hmm-trading/project_overview.html) for a complete reference covering the GMM-HMM architecture, all 7 sub-states, strategy logic, confirmation rules, kernel forecast algorithm, and more.

---

## ⚠️ Disclaimer

For educational and research purposes only. Not financial advice. HMM regime labels are in-sample statistical artefacts and do not predict future returns. Past performance does not guarantee future results.
