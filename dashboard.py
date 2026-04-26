"""
dashboard.py
============
HMM Regime Terminal – Streamlit Dashboard

Launches a professional trading dashboard that:
  • Fetches & processes live BTC-USD hourly data
  • Fits the 7-state Gaussian HMM and decodes market regimes
  • Evaluates all 8 technical confirmations
  • Runs the regime-based backtester simulation
  • Displays:
      - Current Signal (regime, action, state probabilities)
      - Technical Confirmation Scorecard (8 checks)
      - Annotated Price Chart with colour-coded HMM regime bands
      - Equity Curve vs Buy & Hold
      - Backtest Performance Metrics
      - Full Trade Log DataFrame
  • Stock Screener page: scans S&P 500 for Bull Regime entries

Run with:
    streamlit run dashboard.py
"""

import html as html_lib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript
import plotly.graph_objects as go

# Internal modules
import data_loader as dl
import backtester  as bt_module
from backtester import (
    LABEL_BULL, LABEL_BEAR, LABEL_NEUTRAL,
    INITIAL_CAPITAL, build_and_run,
)
from scanner import (
    get_sp500_tickers, get_nasdaq100_tickers,
    get_russell2000_tickers,
    run_scanner,
)

warnings.filterwarnings("ignore")

# ===========================================================================
#  PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title  = "HMM Regime Terminal | BTC-USD Daily",
    page_icon   = "📡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Colour Palette – one colour per regime for all charts / UI elements
# ---------------------------------------------------------------------------
REGIME_COLORS = {
    LABEL_BULL    : "#00e676",   # vivid green
    LABEL_BEAR    : "#ff1744",   # vivid red
    LABEL_NEUTRAL : "#ffd740",   # amber/yellow
}

# Softer pastel colours used exclusively for chart background bands
REGIME_BAND_COLORS = {
    LABEL_BULL    : "#00e676",   # vivid green
    LABEL_BEAR    : "#ff1744",   # vivid red
    LABEL_NEUTRAL : "#ffd740",   # amber/yellow
}

# Map regime to a Streamlit "delta_color" compatible string
REGIME_DELTA = {
    LABEL_BULL    : "normal",
    LABEL_BEAR    : "inverse",
    LABEL_NEUTRAL : "off",
}

# ===========================================================================
#  WATCHLIST  (persisted to browser localStorage — survives tab/browser close)
# ===========================================================================
_DEFAULT_WATCHLIST = [
    "AAPL", "AMD", "AMZN", "AVGO", "BTC-USD", "BWXT", "CIEN", "CL=F",
    "COHR", "ES=F", "ETH-USD", "EWY", "GC=F", "GDX", "GEV", "GLD", "GLW",
    "GOOG", "HG=F", "INTC", "IWM", "KR", "LITE", "MU", "NVDA", "OKLO",
    "QCOM", "QQQ", "SI=F", "SLV", "SMR", "SNDK", "SOL-USD", "SPY",
    "TLT", "TSLA", "USO", "VRT", "WDC",
]
_LS_KEY = "hmm_watchlist"

def _save_watchlist(tickers: list[str]) -> None:
    """Write watchlist to browser localStorage."""
    st_javascript(f"localStorage.setItem('{_LS_KEY}', JSON.stringify({json.dumps(tickers)}))")

# Read current value from localStorage.
# Returns 0 on the first render (JS pending), then the stored string on rerun.
_ls_value = st_javascript(f"localStorage.getItem('{_LS_KEY}')")

if "watchlist" not in st.session_state:
    # First render: JS not resolved yet → start with default; will be
    # overwritten on the immediate rerun once _ls_value resolves.
    if isinstance(_ls_value, str) and _ls_value not in ("null", ""):
        try:
            st.session_state["watchlist"] = sorted(json.loads(_ls_value))
            st.session_state["_wl_synced"] = True
        except Exception:
            st.session_state["watchlist"] = sorted(_DEFAULT_WATCHLIST)
    else:
        st.session_state["watchlist"] = sorted(_DEFAULT_WATCHLIST)
elif not st.session_state.get("_wl_synced") and isinstance(_ls_value, str) and _ls_value not in ("null", ""):
    # JS just resolved on the second render — load the stored list once.
    try:
        st.session_state["watchlist"] = sorted(json.loads(_ls_value))
    except Exception:
        pass
    st.session_state["_wl_synced"] = True


# ===========================================================================
#  SIDEBAR
# ===========================================================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg",
        width=60,
    )
    st.title("⚙️ Configuration")
    st.markdown("---")

    # ── Page Navigation ───────────────────────────────────────────────────
    page = st.radio(
        "View",
        ["📡 Backtester", "🔍 Stock Screener"],
        label_visibility="collapsed",
        horizontal=True,
    )
    st.markdown("---")

    # =========================================================================
    #  BACKTESTER SIDEBAR
    # =========================================================================
    if page == "📡 Backtester":

        # ── Ticker / Symbol ──────────────────────────────────────────────────
        st.markdown("### 🔍 Asset")
        st.markdown(
            "<style>div[data-testid='stTextInput'] input { text-transform: uppercase; }</style>",
            unsafe_allow_html=True,
        )

        # Apply a pending ticker selected from the watchlist dropdown
        if st.session_state.get("_apply_ticker"):
            st.session_state["ticker_sym"] = st.session_state.pop("_apply_ticker")
            st.session_state["watchlist_select"] = "── Saved Tickers ──"

        st.session_state.setdefault("ticker_sym", "SPY")

        ticker_symbol = st.text_input(
            label       = "Ticker Symbol",
            key         = "ticker_sym",
            placeholder = "e.g. BTC-USD, ETH-USD, AAPL, SPY…",
            help        = (
                "Enter any valid Yahoo Finance ticker symbol.\n"
                "Crypto: BTC-USD, ETH-USD, SOL-USD\n"
                "Stocks: AAPL, TSLA, NVDA, SPY, QQQ"
            ),
        ).strip().upper()

        # ── Auto-add typed ticker to watchlist ───────────────────────────
        # Clear "manually removed" flag whenever user switches to a new ticker
        if "manually_removed" not in st.session_state:
            st.session_state["manually_removed"] = set()
        if ticker_symbol != st.session_state.get("_prev_ticker_sym", ""):
            st.session_state["manually_removed"].discard(
                st.session_state.get("_prev_ticker_sym", "")
            )
            st.session_state["_prev_ticker_sym"] = ticker_symbol

        if (ticker_symbol
                and ticker_symbol not in st.session_state["watchlist"]
                and ticker_symbol not in st.session_state["manually_removed"]):
            st.session_state["watchlist"].append(ticker_symbol)
            st.session_state["watchlist"].sort()
            _save_watchlist(st.session_state["watchlist"])

        # ── Watchlist dropdown ────────────────────────────────────────────
        _wl = st.session_state["watchlist"]
        st.session_state.setdefault("watchlist_select", "── Saved Tickers ──")
        _selected = st.selectbox(
            label             = "Watchlist",
            options           = ["── Saved Tickers ──"] + _wl,
            label_visibility  = "collapsed",
            key               = "watchlist_select",
        )
        if _selected != "── Saved Tickers ──":
            st.session_state["_apply_ticker"] = _selected
            st.rerun()

        # ── Remove button – always removes the current ticker_symbol ──────
        if st.button("➖ Remove from List", use_container_width=True,
                     help="Remove current ticker from saved list"):
            if ticker_symbol in st.session_state["watchlist"]:
                st.session_state["manually_removed"].add(ticker_symbol)
                st.session_state["watchlist"].remove(ticker_symbol)
                _save_watchlist(st.session_state["watchlist"])
                st.rerun()

        refresh = st.button("🔄 Refresh Data & Re-run", use_container_width=True,
                            help="Clear cache and re-download data")

        lookback_days = st.slider(
            label     = "Look-back Period (days)",
            min_value = 365,
            max_value = 1825,
            value     = 365,
            step      = 30,
            help      = (
                "Number of calendar days of daily OHLCV data to download.\n"
                "730 ≈ 2 years  |  1095 ≈ 3 years  |  1825 ≈ 5 years\n"
                "More history = better HMM training, especially for identifying "
                "Bear/Crash states which are rare events."
            ),
        )

        st.markdown("---")

        # ── Strategy ─────────────────────────────────────────────────────────
        st.markdown("### ⚡ Strategy")

        LEVERAGE_OPTIONS = {
            "1× — No Leverage"  : 1.0,
            "2× — Standard"     : 2.0,
            "4× — Aggressive"   : 4.0,
        }
        leverage_label = st.radio(
            label   = "Leverage Factor",
            options = list(LEVERAGE_OPTIONS.keys()),
            index   = 0,   # default: 1× — No Leverage
            help    = (
                "Multiplier applied to every position.\n"
                "1× = spot / no leverage\n"
                "2× = 2× position size\n"
                "4× = 4× position size (high risk)"
            ),
        )
        selected_leverage = LEVERAGE_OPTIONS[leverage_label]

        regime_only = st.toggle(
            label = "⚡ Regime-Only Mode",
            value = False,
            help  = (
                "Trade purely on HMM regime transitions — no technical confirmations required.\n\n"
                "• ENTER when regime changes Bear → Bull\n"
                "• ENTER when regime changes Bear → Neutral\n"
                "• EXIT immediately on the first Bear regime bar\n\n"
                "All confirmation filters and entry thresholds are ignored in this mode."
            ),
        )

        if regime_only:
            st.info(
                "**Regime-Only Mode active.**  \n"
                "Entries and exits are driven solely by HMM regime changes.  \n"
                "Confirmation filters are ignored.",
                icon="⚡",
            )

        use_trail_stop = st.toggle(
            label = "🛡 Trailing Stop (2%)",
            value = False,
            help  = "Apply a 2% trailing stop-loss to all open positions.",
        )

        bear_confirm_days = st.slider(
            label     = "Bear Confirmation Days",
            min_value = 1,
            max_value = 7,
            value     = 5,
            step      = 1,
            help      = (
                "Number of consecutive Bear/Crash bars required before the position is exited.\n"
                "1 = exit on the first Bear bar (most sensitive)\n"
                "2 = require 2 consecutive Bear days (default — filters single-day spikes)\n"
                "3+ = only exit on a sustained Bear trend"
            ),
        )

        min_hold_days = st.slider(
            label     = "Min Hold Period (days)",
            min_value = 0,
            max_value = 14,
            value     = 7,
            step      = 1,
            help      = (
                "Minimum number of calendar days to hold a position before a regime exit can fire.\n"
                "0 = no minimum (exit can happen immediately after entry)\n"
                "3 = default — avoids whipsawing out within the first 3 days\n"
                "Trailing stop is not affected by this setting."
            ),
        )

        initial_capital = st.number_input(
            label     = "Starting Capital (USD)",
            min_value = 1_000,
            max_value = 10_000_000,
            value     = int(INITIAL_CAPITAL),
            step      = 1_000,
            format    = "%d",
        )

        st.markdown("---")

        # ── Confirmation Filters ──────────────────────────────────────────────
        st.markdown("### ✅ Confirmation Filters")
        st.caption("Enable/disable individual signals and set the entry threshold.")

        # Convenience buttons — write to session_state so checkboxes follow
        _btn_all, _btn_none = st.columns(2)
        if _btn_all.button("☑ Select All", use_container_width=True):
            for _n in bt_module.ALL_CONFIRMATIONS:
                st.session_state[f"conf_{_n}"] = True
        if _btn_none.button("☐ Deselect All", use_container_width=True):
            for _n in bt_module.ALL_CONFIRMATIONS:
                st.session_state[f"conf_{_n}"] = False

        # One checkbox per confirmation (canonical order from backtester)
        enabled_confirmations = []
        for _name in bt_module.ALL_CONFIRMATIONS:
            _key     = f"conf_{_name}"
            _checked = st.checkbox(_name, value=True, key=_key)
            if _checked:
                enabled_confirmations.append(_name)

        n_enabled = max(len(enabled_confirmations), 1)  # guard against zero

        st.markdown("---")
        st.markdown("### 🎯 Entry Threshold")
        if n_enabled > 1:
            min_confirms = st.slider(
                label     = "Min Confirmations Required",
                min_value = 1,
                max_value = n_enabled,
                value     = min(3, n_enabled),
                step      = 1,
                help      = (
                    "How many of the *enabled* confirmations above must be True "
                    "to allow a Long entry in a Bull Run regime."
                ),
            )
        else:
            # Only 1 confirmation active — slider would crash (min == max).
            # Lock the threshold to 1 and inform the user.
            min_confirms = 1
            st.info("Only 1 confirmation active — threshold locked to 1/1.", icon="ℹ️")

        st.markdown("---")
        st.markdown("### 📊 Mode Summary")
        trail_label  = "✅ On" if use_trail_stop else "❌ Off"
        regime_label = "⚡ Regime-Only" if regime_only else "🔢 Confirmation-Gated"
        if regime_only:
            st.info(
                f"**Mode:** {regime_label}  \n"
                f"**Leverage:** {leverage_label}  \n"
                f"**Entry:** Bear→Bull or Bear→Neutral transition  \n"
                f"**Exit:** First Bear regime bar  \n"
                f"**Trailing Stop:** {trail_label}"
            )
        else:
            st.info(
                f"**Mode:** {regime_label}  \n"
                f"**Leverage:** {leverage_label}  \n"
                f"**Active Signals:** {n_enabled} / {len(bt_module.ALL_CONFIRMATIONS)}  \n"
                f"**Entry Threshold:** {min_confirms} / {n_enabled}  \n"
                f"**Bear Confirm Days:** {bear_confirm_days}  \n"
                f"**Min Hold Days:** {min_hold_days}  \n"
                f"**Trailing Stop:** {trail_label}"
            )

        st.markdown("---")
        optimize_btn = st.button("🔍 Optimize for this Asset",     use_container_width=True,
                                 help=(
                                     "Grid-search Bear Confirm Days (1–7), "
                                     "Min Confirmations (1–9), and Min Hold Days (0–14) "
                                     "to find the combination with the highest Total Return "
                                     "on this asset's historical data."
                                 ))
        st.caption(f"Data: {ticker_symbol} · Daily · Last {lookback_days} days · yfinance")

    # =========================================================================
    #  STOCK SCREENER SIDEBAR
    # =========================================================================
    else:
        st.markdown("### 🔎 Scan Settings")

        _UNIVERSE_OPTIONS = {
            "S&P 500"      : "sp500",
            "Nasdaq 100"   : "nasdaq100",
            "Russell 2000" : "russell2000",
        }
        scanner_universe = st.selectbox(
            label   = "Index Universe",
            options = list(_UNIVERSE_OPTIONS.keys()),
            index   = 0,
            help    = "Choose which index to scan.",
        )

        scanner_lookback = st.slider(
            label     = "Look-back Period (days)",
            min_value = 365,
            max_value = 1095,
            value     = 730,
            step      = 30,
            help      = "Days of OHLCV history to download per ticker for HMM fitting.",
        )

        scanner_min_conf = st.slider(
            label     = "Min Confirmations",
            min_value = 1,
            max_value = 9,
            value     = 3,
            help      = "Minimum number of technical confirmation signals required.",
        )

        scanner_max_bars = st.slider(
            label     = "Max Bars in Bull Run",
            min_value = 1,
            max_value = 15,
            value     = 5,
            help      = (
                "Only include stocks that have been in Bull Run for ≤ N bars.\n"
                "1 = day-1 transitions only (freshest entries)\n"
                "5 = up to 5 days into Bull Run (default)"
            ),
        )

        scanner_target = st.slider(
            label     = "Top N Stocks",
            min_value = 10,
            max_value = 100,
            value     = 30,
            help      = "Number of top-ranked Bull entry candidates to display.",
        )

        scanner_workers = st.slider(
            label     = "Parallel Workers",
            min_value = 4,
            max_value = 16,
            value     = 8,
            help      = "Number of parallel download threads. Higher = faster but more memory.",
        )

        st.markdown("---")
        run_scan_btn = st.button(
            "🔍 Run Scanner",
            use_container_width=True,
            help="Scan all tickers in the selected universe. Each needs an HMM fit.",
        )
        if st.button("🗑️ Clear Cached Results", use_container_width=True):
            st.session_state.pop("scan_results", None)
            st.session_state.pop("scan_params", None)
            st.rerun()
        st.caption("⚠️ Large universes (Russell 2000) can take 60–90 minutes per run.")

        # Unused in screener mode — set defaults to avoid NameError
        refresh = optimize_btn = False
        ticker_symbol = "BTC-USD"; lookback_days = 730; selected_leverage = 1.0
        regime_only = False; use_trail_stop = False; bear_confirm_days = 5
        min_hold_days = 7; initial_capital = int(INITIAL_CAPITAL)
        enabled_confirmations = []; n_enabled = 9; min_confirms = 3


# Scope safety for backtester mode
if page == "📡 Backtester":
    run_scan_btn = False
    scanner_universe = "S&P 500"
    scanner_target = 30; scanner_max_bars = 5; scanner_lookback = 730
    scanner_min_conf = 3; scanner_workers = 8


# ===========================================================================
#  STOCK SCREENER PAGE
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper: render a scan results table
# ---------------------------------------------------------------------------
def _render_scan_table(df: pd.DataFrame) -> None:
    display = df[[
        "ticker", "entry_score", "bull_prob", "confirms_met", "n_confirmations",
        "confirms_pct", "bars_in_bull", "prev_regime", "close", "5d_return_pct", "as_of",
    ]].copy()
    display.columns = [
        "Ticker", "Score", "Bull Prob", "Signals Met", "Total Signals",
        "Signals %", "Bars In Bull", "Prev Regime", "Price ($)", "5d Ret %", "As Of",
    ]
    display.index = range(1, len(display) + 1)

    def _style(row):
        score = row["Score"]
        if score >= 70:
            c = "#00e676"
        elif score >= 50:
            c = "#ffd740"
        else:
            c = "#ff5252"
        styles = [f"color: {c}; font-weight: bold"] + [""] * (len(row) - 1)
        return styles

    st.dataframe(
        display.style.apply(_style, axis=1),
        use_container_width=True,
    )


_TICKER_FETCHERS = {
    "S&P 500"      : get_sp500_tickers,
    "Nasdaq 100"   : get_nasdaq100_tickers,
    "Russell 2000" : get_russell2000_tickers,
}

@st.cache_data(ttl=7200, show_spinner=False)
def run_universe_scan(universe: str, target: int, max_bars: int, lookback: int,
                      min_conf: int, workers: int) -> pd.DataFrame:
    """Run the HMM Bull scanner for the selected index universe. Cached for 2 hours."""
    tickers = _TICKER_FETCHERS[universe]()
    return run_scanner(
        tickers          = tickers,
        lookback         = lookback,
        min_conf         = min_conf,
        max_bars_in_bull = max_bars,
        workers          = workers,
        target           = target,
    )


if page == "🔍 Stock Screener":

    st.markdown(
        f"<h1 style='text-align:center; color:#00e676;'>🔍 {scanner_universe} Bull Regime Scanner</h1>"
        "<p style='text-align:center; color:#888; font-size:14px;'>"
        "7-State GMM-HMM · Freshest Bull Entries Ranked by Entry-Quality Score</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Score formula explainer ──────────────────────────────────────────────
    with st.expander("📐 How the Entry Score Works", expanded=False):
        st.markdown(
            """
            Each candidate stock gets an **Entry Score (0–100)** combining three factors:

            | Component | Weight | Description |
            |---|---|---|
            | HMM Bull Probability | **40 pts** | Posterior probability the current bar is in a Bull state |
            | Confirmation Signals | **35 pts** | Fraction of active technical signals met |
            | Freshness | **25 pts** | Exponential decay: day-1 → 25 pts, day-3 → 16.7 pts, day-5 → 11.2 pts |

            `Score = 40 × bull_prob + 35 × (signals_met / total_signals) + 25 × e^(−0.2 × (bars_in_bull − 1))`

            Higher score = stronger and fresher Bull entry signal.
            """,
        )

    # ── Trigger / display scan results ──────────────────────────────────────
    scan_params = (scanner_universe, scanner_target, scanner_max_bars,
                   scanner_lookback, scanner_min_conf, scanner_workers)

    if run_scan_btn:
        st.session_state["scan_results"] = None   # clear stale results
        st.session_state["scan_params"]  = scan_params
        with st.spinner(
            f"🧠 Scanning {scanner_universe} tickers for Bull Run entries "
            f"(max {scanner_max_bars} bars in Bull, {scanner_workers} threads)…  "
            "Please wait."
        ):
            results_df = run_universe_scan(*scan_params)
        st.session_state["scan_results"] = results_df
        st.session_state["scan_params"]  = scan_params

    results_df = st.session_state.get("scan_results")

    if results_df is None:
        st.info(
            f"👈 Configure scan settings in the sidebar and click **Run Scanner** to start.  \n"
            f"The scanner fits a 7-state GMM-HMM on every {scanner_universe} constituent and "
            "identifies the freshest Bull Run entries ranked by entry-quality score.",
            icon="ℹ️",
        )

    elif results_df.empty:
        st.warning(
            "No Bull Run entries found with the current filter settings.  \n"
            "Try increasing **Max Bars in Bull Run** or reducing **Min Confirmations**.",
            icon="⚠️",
        )

    else:
        fresh = results_df[results_df["is_transition"]]
        early = results_df[~results_df["is_transition"]]
        as_of = results_df["as_of"].iloc[0] if "as_of" in results_df.columns else "—"

        # ── Summary metrics ──────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Candidates Found",      len(results_df))
        mc2.metric("Fresh (day-1) Entries", len(fresh))
        mc3.metric("Early-Stage (2+ bars)", len(early))
        mc4.metric("Data As Of",            as_of)

        st.markdown("---")

        # ── Fresh transitions ────────────────────────────────────────────────
        if not fresh.empty:
            st.subheader(f"🟢 Fresh Entries — Just Entered Bull Run Today ({len(fresh)} stocks)")
            st.caption("Previous bar was NOT Bull Run — these are genuine day-1 transitions.")
            _render_scan_table(fresh)

        # ── Early-stage ──────────────────────────────────────────────────────
        if not early.empty:
            max_b = int(early["bars_in_bull"].max())
            st.subheader(f"🔵 Early-Stage Bull Runs — 2–{max_b} Bars In ({len(early)} stocks)")
            st.caption("Already in Bull Run for 2+ bars but still within the freshness window.")
            _render_scan_table(early)

        # ── Bull probability chart ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Entry Score Distribution")
        fig_scores = go.Figure(go.Bar(
            x           = results_df["ticker"],
            y           = results_df["entry_score"],
            marker_color= results_df["bars_in_bull"].apply(
                lambda b: f"rgba(0,230,118,{max(0.3, 1 - b * 0.12):.2f})"
            ),
            text        = results_df["entry_score"].apply(lambda v: f"{v:.1f}"),
            textposition= "outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Score: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ))
        fig_scores.update_layout(
            height        = 380,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "#0a0a0a",
            font          = dict(color="#e0e0e0"),
            xaxis         = dict(color="#888", tickangle=-45),
            yaxis         = dict(title="Entry Score (0–100)", color="#e0e0e0",
                                 gridcolor="#1a1a1a", range=[0, 105]),
            margin        = dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig_scores, use_container_width=True)

        # ── Download ─────────────────────────────────────────────────────────
        csv = results_df.to_csv(index_label="Rank").encode("utf-8")
        st.download_button(
            label     = "⬇️ Download Results CSV",
            data      = csv,
            file_name = f"hmm_bull_scanner_{as_of}.csv",
            mime      = "text/csv",
        )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#555; font-size:12px;'>"
        "⚠️ <b>Disclaimer:</b> For educational and research purposes only. "
        "Not financial advice. HMM regime labels are in-sample statistical artefacts "
        "and do not predict future returns."
        "</p>",
        unsafe_allow_html=True,
    )
    st.stop()


# ===========================================================================
#  COMPANY NAME LOOKUP
# ===========================================================================

@st.cache_data(ttl=86400, show_spinner=False)   # cache for 24 hours
def get_company_name(ticker: str) -> str:
    """Return the long (or short) name for a ticker via yfinance, or '' on failure."""
    # Try .info first (full data, may be blocked on cloud IPs)
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ""
        if name:
            return name
    except Exception:
        pass
    # Fallback: yf.Search hits a different endpoint, more reliable on cloud
    try:
        results = yf.Search(ticker, max_results=1).quotes
        if results:
            return results[0].get("longname") or results[0].get("shortname") or ""
    except Exception:
        pass
    return ""


@st.cache_data(ttl=86400, show_spinner=False)   # cache for 24 hours
def get_company_profile(ticker: str) -> dict:
    """Return a profile dict: sector, industry, employees, market_cap, website, summary, error."""
    t = yf.Ticker(ticker)

    # fast_info is lightweight and works reliably on cloud servers
    mc_str = ""
    try:
        mc = t.fast_info.market_cap
        if mc and mc >= 1e12:
            mc_str = f"${mc/1e12:.2f}T"
        elif mc and mc >= 1e9:
            mc_str = f"${mc/1e9:.2f}B"
        elif mc and mc >= 1e6:
            mc_str = f"${mc/1e6:.2f}M"
    except Exception as e:
        return {"error": f"fast_info failed: {type(e).__name__}: {e}"}

    # .info is richer but may be blocked by Yahoo on cloud IPs; fail gracefully
    sector = industry = emp_str = website = summary = ""
    try:
        info = t.info
        emp = info.get("fullTimeEmployees")
        emp_str = f"{emp:,}" if emp else ""
        sector   = info.get("sector", "")
        industry = info.get("industry", "")
        website  = info.get("website", "")
        summary  = info.get("longBusinessSummary", "")
        # prefer .info market cap when available
        mc = info.get("marketCap")
        if mc:
            if mc >= 1e12:
                mc_str = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                mc_str = f"${mc/1e9:.2f}B"
            elif mc >= 1e6:
                mc_str = f"${mc/1e6:.2f}M"
    except Exception as e:
        # .info blocked on cloud — surface the reason, keep fast_info market cap
        return {
            "sector": "", "industry": "", "employees": "", "website": "", "summary": "",
            "market_cap": mc_str,
            "error": f".info unavailable: {type(e).__name__}: {e}",
        }

    if not any([mc_str, sector, industry, emp_str, website, summary]):
        return {}

    return {
        "sector"    : sector,
        "industry"  : industry,
        "employees" : emp_str,
        "market_cap": mc_str,
        "website"   : website,
        "summary"   : summary,
    }


@st.cache_data(ttl=60, show_spinner=False)   # cache for 60 seconds
def get_live_quote(ticker: str) -> Optional[dict]:
    """
    Fetch the latest available quote for a ticker via yfinance fast_info.
    Returns a dict with last_price, day_high, day_low, day_open, timezone,
    quote_str (formatted timestamp), or None on failure.
    """
    try:
        fi = yf.Ticker(ticker).fast_info
        last_price = getattr(fi, "last_price", None)
        if last_price is None or last_price != last_price:   # NaN check
            return None
        tz_name = getattr(fi, "timezone", None) or "UTC"
        # Always display quote time in Pacific time
        if ZoneInfo is not None:
            try:
                display_tz = ZoneInfo("America/Los_Angeles")
            except Exception:
                display_tz = timezone.utc
        else:
            display_tz = timezone.utc
        now_local  = datetime.now(display_tz)
        quote_str  = now_local.strftime("%b %d, %Y  %H:%M %Z")
        return {
            "last_price": float(last_price),
            "day_high"  : float(getattr(fi, "day_high",  last_price) or last_price),
            "day_low"   : float(getattr(fi, "day_low",   last_price) or last_price),
            "day_open"  : float(getattr(fi, "open",      last_price) or last_price),
            "timezone"  : tz_name,
            "quote_str" : quote_str,
        }
    except Exception:
        return None


# ===========================================================================
#  DATA LOADING + BACKTEST  (cached with TTL so it re-runs on refresh)
# ===========================================================================

@st.cache_data(ttl=300, show_spinner=False)   # cache for 5 minutes
def load_and_run(
    capital               : float,
    ticker                : str,
    lookback              : int,
    leverage              : float,
    enabled_confirms_tuple: tuple,
    min_confirms          : int,
    use_trail_stop        : bool,
    bear_confirm_days     : int,
    min_hold_days         : int,
    regime_only           : bool = False,
):
    """
    Download data, run the full HMM + backtest pipeline, and return
    the Backtester object along with the raw data.

    Cache key includes all strategy params so any change triggers a re-run.
    """
    raw_df = dl.load(ticker=ticker, period_days=lookback)
    if raw_df.empty:
        return None, raw_df
    enabled_list = list(enabled_confirms_tuple) if enabled_confirms_tuple else None
    try:
        backtester = build_and_run(
            raw_df,
            initial_capital       = float(capital),
            leverage_override     = leverage,
            enabled_confirmations = enabled_list,
            min_confirms          = min_confirms,
            use_trail_stop        = use_trail_stop,
            bear_confirm_days     = bear_confirm_days,
            min_hold_days         = min_hold_days,
            regime_only           = regime_only,
        )
    except Exception as exc:
        print(f"[Dashboard] ERROR – backtester failed: {exc}")
        return None, raw_df
    return backtester, raw_df


@st.cache_data(ttl=3600, show_spinner=False)
def optimize_for_asset(
    ticker                : str,
    lookback              : int,
    leverage              : float,
    enabled_confirms_tuple: tuple,
    use_trail_stop        : bool,
    capital               : float,
) -> dict:
    """
    Download data and run the parameter grid-search.
    Cached per (ticker, lookback, leverage, confirmations, trail_stop, capital)
    so repeated clicks don't re-run the expensive search.
    """
    raw_df = dl.load(ticker=ticker, period_days=lookback)
    if raw_df.empty:
        return None
    enabled_list = list(enabled_confirms_tuple) if enabled_confirms_tuple else None
    try:
        return bt_module.optimize_params(
            raw_df,
            initial_capital       = float(capital),
            leverage_override     = leverage,
            enabled_confirmations = enabled_list,
            use_trail_stop        = use_trail_stop,
        )
    except Exception as exc:
        print(f"[Dashboard] Optimization failed: {exc}")
        return None


if refresh:
    st.cache_data.clear()

# ---------------------------------------------------------------------------
# Run optimization when the user clicks the optimize button
# ---------------------------------------------------------------------------
if optimize_btn:
    with st.spinner("🔍 Searching best parameters… this may take a minute."):
        _opt = optimize_for_asset(
            ticker_symbol,
            lookback_days,
            selected_leverage,
            tuple(sorted(enabled_confirmations)),
            use_trail_stop,
            float(initial_capital),
        )
    if _opt is not None:
        st.session_state["opt_result"] = _opt
        st.session_state["opt_ticker"] = ticker_symbol
    else:
        st.error("Optimization failed — check your ticker symbol and try again.")

# ---------------------------------------------------------------------------
# Spinner while computing
# ---------------------------------------------------------------------------
with st.spinner(
    f"🧠 Downloading {ticker_symbol} data & fitting HMM… "
    "this may take ~30 seconds on first load."
):
    backtester, raw_df = load_and_run(
        initial_capital,
        ticker_symbol,
        lookback_days,
        selected_leverage,
        tuple(sorted(enabled_confirmations)),
        min_confirms,
        use_trail_stop,
        bear_confirm_days,
        min_hold_days,
        regime_only,
    )

# ---------------------------------------------------------------------------
# Guard: if data load failed, show error and stop
# ---------------------------------------------------------------------------
if backtester is None or raw_df.empty:
    st.error(
        "❌ Failed to download market data from Yahoo Finance.  "
        "Check your internet connection and try refreshing."
    )
    st.stop()

# Convenience aliases
df      = backtester.df          # annotated DataFrame with all signals
signal  = backtester.get_current_signal()
metrics = backtester.metrics
engine  = backtester.engine

# Live quote (60-second TTL cache) — used for price display and chart patch
live_quote = get_live_quote(ticker_symbol)


# ===========================================================================
#  HEADER
# ===========================================================================
_company_name   = get_company_name(ticker_symbol)
_company_profile = get_company_profile(ticker_symbol)
if _company_profile.get("error"):
    st.warning(f"Company profile partial/unavailable — {_company_profile['error']}")
_display_name   = f"{_company_name} ({ticker_symbol})" if _company_name else ticker_symbol
st.markdown(
    "<h1 style='text-align:center; color:#00e5ff;'>📡 HMM Regime Terminal</h1>"
    f"<p style='text-align:center; color:#e0e0e0; font-size:18px; font-weight:600;'>"
    f"{_display_name}</p>"
    f"<p style='text-align:center; color:#888; font-size:14px;'>"
    f"Daily · 7-State Gaussian HMM · Regime-Based Algorithmic Trading</p>",
    unsafe_allow_html=True,
)

# ── Company Profile Card ──────────────────────────────────────────────────
if _company_profile:
    _tags = []
    if _company_profile.get("sector"):
        _tags.append(_company_profile["sector"])
    if _company_profile.get("industry"):
        _tags.append(_company_profile["industry"])
    if _company_profile.get("market_cap"):
        _tags.append(f"Mkt Cap: {_company_profile['market_cap']}")
    if _company_profile.get("employees"):
        _tags.append(f"Employees: {_company_profile['employees']}")
    _tags_html = " &nbsp;·&nbsp; ".join(
        f"<span style='color:#aaa;'>{t}</span>" for t in _tags
    )
    _summary = _company_profile.get("summary", "")
    # Truncate to ~100 words
    _words = _summary.split()
    if len(_words) > 100:
        _summary = " ".join(_words[:100]) + " …"
    _website = _company_profile.get("website", "")
    _website_html = (
        f"&nbsp;&nbsp;<a href='{_website}' target='_blank' "
        f"style='color:#00b0ff; font-size:16px;'>{_website}</a>"
        if _website else ""
    )
    _profile_html_parts = []
    if _tags_html:
        _profile_html_parts.append(
            f"<p style='margin:0 0 6px 0; font-size:16px;'>{_tags_html}{_website_html}</p>"
        )
    if _summary:
        _profile_html_parts.append(
            f"<p style='margin:0; font-size:17px; color:#ccc; line-height:1.5;'>{_summary}</p>"
        )
    if _profile_html_parts:
        st.markdown(
            "<div style='background:#1a1a2e; border:1px solid #2a2a4a; border-radius:8px;"
            " padding:12px 16px; margin-bottom:12px;'>"
            + "".join(_profile_html_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")


# ===========================================================================
#  ROW 1 – CURRENT SIGNAL PANEL
# ===========================================================================
st.subheader("🚦 Current Market Signal")

col_regime, col_action, col_price, col_confirm_badge = st.columns([2, 3, 2, 2])

# --- Regime Badge ---
current_regime = signal.get("regime", "Unknown")
regime_color   = REGIME_COLORS.get(current_regime, "#888888")

# Compute which of the 7 sub-states the current bar is in
_SUBSTATE_NAMES = {
    1: "Crash / Panic",
    2: "Bear / Sustained Decline",
    3: "Bearish Consolidation",
    4: "Sideways / Range-Bound",
    5: "Early Recovery",
    6: "Steady Uptrend",
    7: "Momentum Bull",
}
try:
    _state_mean_ret = {
        s: df.loc[df["state"] == s, "returns"].mean()
        for s in range(engine.n_states)
    }
    # Rank within each regime bucket to stay consistent with engine.state_labels:
    #   Bear states  → ranks 1-2  (bottom N_BEAR_STATES by mean return)
    #   Neutral states → ranks 3-4
    #   Bull states  → ranks 5-7  (top N_BULL_STATES by mean return)
    _neutral_states = {
        s for s in range(engine.n_states)
        if s not in engine.bear_states and s not in engine.bull_states
    }
    _bear_sorted    = sorted(engine.bear_states,  key=lambda s: _state_mean_ret.get(s, 0))
    _neutral_sorted = sorted(_neutral_states,      key=lambda s: _state_mean_ret.get(s, 0))
    _bull_sorted    = sorted(engine.bull_states,   key=lambda s: _state_mean_ret.get(s, 0))
    _state_rank: dict[int, int] = {}
    for _rank, _s in enumerate(_bear_sorted,    start=1):                                        _state_rank[_s] = _rank
    for _rank, _s in enumerate(_neutral_sorted, start=len(_bear_sorted) + 1):                    _state_rank[_s] = _rank
    for _rank, _s in enumerate(_bull_sorted,    start=len(_bear_sorted) + len(_neutral_states) + 1): _state_rank[_s] = _rank
    _current_state   = int(df.iloc[-1]["state"])
    _current_rank    = _state_rank[_current_state]
    current_substate = f"State {_current_rank} · {_SUBSTATE_NAMES.get(_current_rank, '')}"
except Exception:
    current_substate = ""

# Pick substate text color that contrasts against the regime background
_substate_text_color = {
    "Bull Run"           : "#003d20",   # dark green on light-green bg
    "Bear/Crash"         : "#ffffff",   # white on dark-red bg
    "Neutral/Transition" : "#3d2e00",   # dark amber on light-amber bg
}.get(current_regime, "#ffffff")

with col_regime:
    st.markdown(
        f"""
        <div style="
            background-color:{regime_color}22;
            border:2px solid {regime_color};
            border-radius:12px;
            padding:16px;
            text-align:center;">
            <p style='margin:0; font-size:12px; color:#aaa;'>DETECTED REGIME</p>
            <p style='margin:0; font-size:22px; font-weight:700; color:{regime_color};'>
                {current_regime}
            </p>
            <p style='margin:4px 0 0 0; font-size:13px; font-weight:400; color:{regime_color};'>
                {current_substate}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Recommended Action ---
action       = signal.get("action", "Unknown")
position_now = signal.get("position_open", False)

# Colour-code the action card border by position state
if "HOLD LONG" in action or "ENTER LONG" in action:
    action_border = "#00e676"          # green
    action_bg     = "rgba(0,230,118,0.08)"
elif "EXIT" in action or ("CASH" in action and "Bear" in action):
    action_border = "#ff1744"          # red
    action_bg     = "rgba(255,23,68,0.08)"
elif "WATCH" in action or ("HOLD LONG" in action and "Neutral" in action):
    action_border = "#ffd740"          # amber
    action_bg     = "rgba(255,215,64,0.08)"
else:
    action_border = "#555"             # grey — cash / neutral
    action_bg     = "rgba(80,80,80,0.06)"

with col_action:
    pos_label = "📍 In Position" if position_now else "💤 No Position"
    st.markdown(
        f"""
        <div style="
            background-color:{action_bg};
            border:2px solid {action_border};
            border-radius:12px;
            padding:16px;
            text-align:center;">
            <p style='margin:0 0 4px 0; font-size:12px; color:#aaa;'>RECOMMENDED ACTION</p>
            <p style='margin:0 0 6px 0; font-size:16px; font-weight:700;
                      color:{action_border};'>
                {action}
            </p>
            <p style='margin:0; font-size:11px; color:#777;'>{pos_label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Current Price ---
with col_price:
    if live_quote and live_quote.get("last_price"):
        price      = live_quote["last_price"]
        quote_time = live_quote["quote_str"]
    else:
        price      = signal.get("close", 0)
        quote_time = df.index[-1].strftime("%b %d, %Y")
    st.metric(
        label = f"{ticker_symbol} Price",
        value = f"${price:,.2f}",
    )
    lev = signal.get("leverage", 2.5)
    st.caption(f"Active Leverage: **{lev}×**")
    st.caption(f"As of: {quote_time}")

# --- Confirmation Count Badge ---
confirms_count = signal.get("confirms_count", 0)
min_confirms   = signal.get("min_confirms", 7)
badge_color    = "#00e676" if confirms_count >= min_confirms else "#ffd740"

with col_confirm_badge:
    st.markdown(
        f"""
        <div style="
            background-color:{badge_color}22;
            border:2px solid {badge_color};
            border-radius:12px;
            padding:16px;
            text-align:center;">
            <p style='margin:0; font-size:12px; color:#aaa;'>CONFIRMATIONS MET</p>
            <p style='margin:0; font-size:32px; font-weight:700; color:{badge_color};'>
                {confirms_count}/{n_enabled}
            </p>
            <p style='margin:0; font-size:11px; color:#aaa;'>Need ≥ {min_confirms}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ===========================================================================
#  REGIME DESCRIPTIONS
# ===========================================================================
st.subheader("📖 Regime Descriptions")

regime_desc_html = f"""
<div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:8px;">

  <!-- Bull Run -->
  <div style="flex:1; min-width:220px;
              background:linear-gradient(135deg, #0d2618 0%, #0a1a10 100%);
              border:1.5px solid #00e676;
              border-radius:12px; padding:16px;">
    <div style="font-size:20px; margin-bottom:6px;">🟢 <strong style="color:#00e676;">Bull Run</strong></div>
    <p style="margin:0 0 8px 0; color:#b2dfdb; font-size:15px; line-height:1.5;">
      The HMM state with the <strong>highest mean return</strong> in the training period.
      Characterised by sustained positive price momentum, expanding volatility (ATR rising),
      and above-average trading volume — signs of broad market participation in an uptrend.
    </p>
    <p style="margin:0; color:#4caf50; font-size:14px;">
      {"➡ Strategy <strong>enters Long on Bear→Bull transition</strong> (Regime-Only Mode)."
       if backtester.regime_only else
       f"➡ Strategy <strong>enters Long</strong> when ≥ {backtester.min_confirms}/{n_enabled} active confirmations are met."}
    </p>
  </div>

  <!-- Bear / Crash -->
  <div style="flex:1; min-width:220px;
              background:linear-gradient(135deg, #210d0d 0%, #180808 100%);
              border:1.5px solid #ff1744;
              border-radius:12px; padding:16px;">
    <div style="font-size:20px; margin-bottom:6px;">🔴 <strong style="color:#ff5252;">Bear/Crash</strong></div>
    <p style="margin:0 0 8px 0; color:#ffcdd2; font-size:15px; line-height:1.5;">
      The HMM state with the <strong>lowest (most negative) mean return</strong>.
      Typically marked by sharp drawdowns, panic selling, spiking implied volatility,
      and volume dominated by sellers. Often corresponds to macro risk-off events
      or exchange-driven liquidation cascades.
    </p>
    <p style="margin:0; color:#ef5350; font-size:14px;">
      {"➡ Strategy <strong>exits immediately on the first Bear bar</strong> (Regime-Only Mode). A 48-hour cooldown prevents re-entry."
       if backtester.regime_only else
       "➡ Strategy <strong>exits immediately</strong> when this regime is detected. A 48-hour cooldown prevents re-entry after any exit."}
    </p>
  </div>

  <!-- Neutral / Transition -->
  <div style="flex:1; min-width:220px;
              background:linear-gradient(135deg, #1a1600 0%, #121000 100%);
              border:1.5px solid #ffd740;
              border-radius:12px; padding:16px;">
    <div style="font-size:20px; margin-bottom:6px;">🟡 <strong style="color:#ffd740;">Neutral/Transition</strong></div>
    <p style="margin:0 0 8px 0; color:#fff9c4; font-size:15px; line-height:1.5;">
      All remaining HMM states (5 out of 7). These capture <strong>sideways, ranging,
      or transitioning market conditions</strong> — low directional conviction, mixed signals,
      consolidation phases, or the brief periods between a Bull and Bear state.
      Returns are near zero on average.
    </p>
    <p style="margin:0; color:#ffca28; font-size:14px;">
      {"➡ Strategy <strong>enters Long on Bear→Neutral transition</strong>; holds until Bear detected (Regime-Only Mode)."
       if backtester.regime_only else
       "➡ Strategy stays in <strong>Cash / Flat</strong>. No new entries are triggered."}
    </p>
  </div>

</div>
"""
st.markdown(regime_desc_html, unsafe_allow_html=True)

# ── 7-state breakdown ────────────────────────────────────────────────────────
st.markdown(
    "<p style='color:#888; font-size:13px; margin:4px 0 10px 0;'>"
    "The model uses <strong style='color:#e0e0e0;'>7 hidden states</strong> internally, "
    "ranked by mean return and collapsed into the three tradeable regimes above. "
    "The table below describes each sub-state from most bearish (State 1) to most bullish (State 7)."
    "</p>",
    unsafe_allow_html=True,
)

_state_rows = [
    ("1", "#ff1744", "Bear/Crash",         "🔴", "Crash / Panic",
     "Sharpest drawdowns, highest volatility spike, heavy sell volume. "
     "Corresponds to macro risk-off events, flash crashes, or exchange-driven liquidations."),
    ("2", "#ff5252", "Bear/Crash",         "🔴", "Bear / Sustained Decline",
     "Persistent negative returns with elevated but stable volatility. "
     "Typical of prolonged downtrends — sellers in control, low buying interest."),
    ("3", "#ffd740", "Neutral/Transition", "🟡", "Bearish Consolidation",
     "Mildly negative to flat returns. Price stalls after a decline; "
     "uncertainty dominates. Often precedes either a deeper bear move or a reversal."),
    ("4", "#ffd740", "Neutral/Transition", "🟡", "Sideways / Range-Bound",
     "Near-zero mean return, low directional volatility. "
     "Market lacks conviction — price oscillates within a range, volume subdued."),
    ("5", "#00e676", "Bull Run",           "🟢", "Early Recovery / Accumulation",
     "Modestly positive returns, volatility still below average. "
     "Smart money begins accumulating; price base-building after a downtrend."),
    ("6", "#00e676", "Bull Run",           "🟢", "Steady Uptrend",
     "Consistent positive returns with controlled volatility. "
     "Broad participation, rising volume, price making higher highs and higher lows."),
    ("7", "#00e676", "Bull Run",           "🟢", "Momentum Bull / Euphoria",
     "Highest mean returns, volatility re-expanding upward. "
     "Strong buying pressure, FOMO-driven volume spikes, parabolic price action."),
]

_rows_html = ""
for rank, color, regime, icon, name, desc in _state_rows:
    _rows_html += f"""
<tr style="border-bottom:1px solid #1a1a1a;">
  <td style="text-align:center; font-weight:700; color:{color}; padding:8px 12px; white-space:nowrap;">
    {rank}
  </td>
  <td style="padding:8px 12px; white-space:nowrap; color:{color}; font-size:15px; font-weight:600;">
    {icon}&nbsp; {name}
  </td>
  <td style="padding:8px 12px; white-space:nowrap;">
    <span style="background:{color}22; border:1px solid {color}55; border-radius:4px;
                 padding:2px 8px; font-size:13px; color:{color}; font-weight:600;">
      {regime}
    </span>
  </td>
  <td style="padding:8px 12px; color:#b0b0b0; font-size:14px; line-height:1.5;">
    {desc}
  </td>
</tr>"""

st.markdown(
    f"""<div style="overflow-x:auto; border:1px solid #2a2a2a; border-radius:10px; margin-bottom:8px;">
<table style="width:100%; border-collapse:collapse; background:#0d0d0d;">
  <thead>
    <tr style="border-bottom:2px solid #2a2a2a; background:#111;">
      <th style="padding:8px 12px; color:#888; font-size:13px; letter-spacing:.8px; text-align:center; width:60px;">#</th>
      <th style="padding:8px 12px; color:#888; font-size:13px; letter-spacing:.8px; text-align:left;">SUB-STATE</th>
      <th style="padding:8px 12px; color:#888; font-size:13px; letter-spacing:.8px; text-align:left;">REGIME</th>
      <th style="padding:8px 12px; color:#888; font-size:13px; letter-spacing:.8px; text-align:left;">CHARACTERISTICS</th>
    </tr>
  </thead>
  <tbody>
    {_rows_html}
  </tbody>
</table></div>""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ===========================================================================
#  ROW 2 – STATE PROBABILITIES + CONFIRMATION SCORECARD
# ===========================================================================
col_probs, col_scorecard = st.columns([1, 1])

# --- State Probability Bar Chart ---
with col_probs:
    st.subheader("📊 State Probabilities (Current Bar)")
    state_probs = signal.get("state_probs", {})
    if state_probs:
        prob_df = (
            pd.DataFrame.from_dict(state_probs, orient="index", columns=["probability"])
            .sort_values("probability", ascending=True)
        )
        # Colour each bar by its label
        bar_colors = [REGIME_COLORS.get(lbl, "#888") for lbl in prob_df.index]

        fig_probs = go.Figure(go.Bar(
            x           = prob_df["probability"],
            y           = prob_df.index,
            orientation = "h",
            marker_color= bar_colors,
            text        = [f"{v:.1%}" for v in prob_df["probability"]],
            textposition= "outside",
        ))
        fig_probs.update_layout(
            height       = 280,
            margin       = dict(l=10, r=30, t=10, b=10),
            paper_bgcolor= "rgba(0,0,0,0)",
            plot_bgcolor = "rgba(0,0,0,0)",
            xaxis        = dict(range=[0, 1.05], showgrid=False,
                                tickformat=".0%", color="#aaa"),
            yaxis        = dict(color="#e0e0e0"),
            font         = dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig_probs, use_container_width=True)
    else:
        st.info("No probability data available.")

# --- Technical Confirmation Scorecard ---
with col_scorecard:
    st.subheader("✅ Technical Confirmations (Current Bar)")
    if regime_only:
        st.caption("⚡ Regime-Only Mode — confirmations shown for reference; not used for trade decisions.")
    confirms_detail = signal.get("confirms_detail", {})

    if confirms_detail:
        # ── Flexbox div list — iterates ALL 9 in canonical order ─────────
        # Disabled (unchecked in sidebar) indicators are shown in grey.
        # Enabled + Met = green   |   Enabled + Not Met = red
        enabled_set = set(enabled_confirmations)
        rows_html   = ""

        for name in bt_module.ALL_CONFIRMATIONS:
            passed    = confirms_detail.get(name, False)
            is_active = name in enabled_set
            safe_name = html_lib.escape(name)   # handle < > & in names

            if not is_active:
                # ── Disabled ──────────────────────────────────────────
                fg        = "#555"
                row_bg    = "rgba(80,80,80,0.05)"
                badge_bg  = "rgba(80,80,80,0.10)"
                badge_bdr = "rgba(80,80,80,0.25)"
                icon      = "⏸"
                status    = "Disabled"
            elif passed:
                # ── Enabled + condition met ───────────────────────────
                fg        = "#00e676"
                row_bg    = "rgba(0,230,118,0.07)"
                badge_bg  = "rgba(0,230,118,0.12)"
                badge_bdr = "rgba(0,230,118,0.35)"
                icon      = "✅"
                status    = "Met"
            else:
                # ── Enabled + condition not met ───────────────────────
                fg        = "#ff5252"
                row_bg    = "rgba(255,23,68,0.07)"
                badge_bg  = "rgba(255,23,68,0.12)"
                badge_bdr = "rgba(255,23,68,0.35)"
                icon      = "❌"
                status    = "Not Met"

            rows_html += f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            padding:7px 10px; border-bottom:1px solid #1e1e1e;
            background:{row_bg};">
  <span style="color:{fg}; font-size:14px; font-weight:600; font-family:sans-serif;">
    {icon}&nbsp; {safe_name}
  </span>
  <span style="color:{fg}; font-size:11px; font-weight:700; font-family:sans-serif;
               background:{badge_bg}; border:1px solid {badge_bdr};
               border-radius:5px; padding:2px 9px; white-space:nowrap;">
    {status}
  </span>
</div>"""

        header_html = f"""
<div style="display:flex; justify-content:space-between;
            padding:6px 10px; border-bottom:2px solid #333;
            background:rgba(0,0,0,0.3);">
  <span style="color:#888; font-size:11px; font-weight:700;
               letter-spacing:0.8px; font-family:sans-serif;">
    INDICATOR &nbsp;<span style="color:#555; font-weight:400;">
    ({n_enabled} active / {len(bt_module.ALL_CONFIRMATIONS)} total)</span>
  </span>
  <span style="color:#888; font-size:11px; font-weight:700;
               letter-spacing:0.8px; font-family:sans-serif;">STATUS</span>
</div>"""

        st.markdown(
            f'<div style="border:1px solid #2a2a2a; border-radius:8px; overflow:hidden;">'
            f'{header_html}{rows_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No confirmation data available.")

st.markdown("---")


# ===========================================================================
#  ROW 3 – MAIN PRICE CHART WITH HMM REGIME OVERLAY
# ===========================================================================
st.subheader(f"📈 {ticker_symbol} Price Chart with HMM Regime Overlay")


# ===========================================================================
#  KERNEL REGRESSION FORECAST
# ===========================================================================

@st.cache_data(ttl=300, show_spinner=False)
def compute_kernel_prediction(
    _df       : pd.DataFrame,
    ticker    : str,
    regime_key: str = "",
    n_days    : int = 10,
) -> Optional[dict]:
    """
    Kernel regression predictor for post-current-state return distributions.

    For today's feature vector, finds the kernel-weighted distribution of
    10-day forward price paths from similar historical states and returns:
      - expected path (kernel-weighted mean)
      - ±1σ confidence band (kernel-weighted std)

    Features used per bar:
      vol_surge    – today's volume / 20-day avg volume
      atr_ratio    – ATR(14) / close  (normalised realised volatility)
      pct_from_high – % below 52-week high
      momentum_5d  – 5-day return
      regime_num   – HMM regime encoded as +1 / 0 / -1
    """
    try:
        df = _df.copy()
        required = {"high", "low", "close", "volume"}
        if not required.issubset(df.columns) or len(df) < 60 + n_days:
            return None

        # ── Feature engineering ───────────────────────────────────────────
        vol_ma    = df["volume"].rolling(20, min_periods=10).mean()
        f_vol     = (df["volume"] / vol_ma).clip(0.1, 10.0)

        pc        = df["close"].shift(1)
        tr        = pd.concat([
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"]  - pc).abs(),
        ], axis=1).max(axis=1)
        atr       = tr.rolling(14, min_periods=7).mean()
        f_atr     = (atr / df["close"]).clip(0, 0.5)

        hi252     = df["close"].rolling(252, min_periods=30).max()
        f_hi      = ((df["close"] - hi252) / hi252).clip(-1, 0)

        f_mom     = df["close"].pct_change(5).clip(-0.5, 0.5)

        _rmap = {LABEL_BULL: 1.0, LABEL_NEUTRAL: 0.0, LABEL_BEAR: -1.0}
        f_reg = (
            df["regime_label"].map(_rmap).fillna(0.0)
            if "regime_label" in df.columns
            else pd.Series(0.0, index=df.index)
        )

        feat = pd.DataFrame({
            "vol_surge"    : f_vol,
            "atr_ratio"    : f_atr,
            "pct_from_high": f_hi,
            "momentum_5d"  : f_mom,
            "regime_num"   : f_reg,
        }, index=df.index).dropna()

        if len(feat) < 30 + n_days:
            return None

        close_vals = df.loc[feat.index, "close"].values
        feat_mat   = feat.values                          # (N, 5)

        # z-score normalise using training portion (all but last n_days)
        train_n  = len(feat) - n_days
        means    = feat_mat[:train_n].mean(axis=0)
        stds     = feat_mat[:train_n].std(axis=0)
        stds[stds < 1e-8] = 1.0
        feat_norm    = (feat_mat - means) / stds
        current_feat = feat_norm[-1]

        # Gaussian kernel weights against all training points
        dists   = np.sqrt(((feat_norm[:train_n] - current_feat) ** 2).sum(axis=1))
        h       = float(np.median(dists)) or 1.0
        weights = np.exp(-0.5 * (dists / h) ** 2)

        # Collect n_days forward return paths
        paths, ws = [], []
        for i in range(train_n):
            base = close_vals[i]
            if base <= 0 or i + n_days >= len(close_vals):
                continue
            fwd = close_vals[i + 1 : i + n_days + 1]
            if len(fwd) < n_days:
                continue
            paths.append(fwd / base - 1.0)
            ws.append(weights[i])

        if len(paths) < 10:
            return None

        paths = np.array(paths)           # (N, n_days)
        ws    = np.array(ws)
        ws   /= ws.sum()

        mean_ret = (ws[:, None] * paths).sum(axis=0)
        std_ret  = np.sqrt((ws[:, None] * (paths - mean_ret) ** 2).sum(axis=0))

        # Convert cumulative returns to price levels
        anchor     = float(df["close"].iloc[-1])
        exp_prices = (anchor * (1 + mean_ret)).round(2)
        up_prices  = (anchor * (1 + mean_ret + std_ret)).round(2)
        lo_prices  = (anchor * (1 + mean_ret - std_ret)).round(2)

        # Future dates – business days for equities, calendar days for crypto
        last_date = _df.index[-1]
        is_crypto = ticker.upper().endswith(("-USD", "-BTC", "-ETH", "-USDT"))
        future_dates = (
            pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days)
            if is_crypto
            else pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n_days)
        )

        return {
            "anchor_date" : last_date.strftime("%Y-%m-%d"),
            "anchor_price": round(anchor, 2),
            "dates"       : [d.strftime("%Y-%m-%d") for d in future_dates],
            "expected"    : exp_prices.tolist(),
            "upper"       : up_prices.tolist(),
            "lower"       : lo_prices.tolist(),
            "exp_ret_pct" : round(float(mean_ret[-1]) * 100, 1),
        }
    except Exception:
        return None


# TradingView-style chart via lightweight-charts (CDN), rendered as an HTML component.

@st.cache_data(ttl=300, show_spinner=False)
def _build_tv_chart_html(
    _df             : pd.DataFrame,
    ticker          : str,
    _trade_log      : Optional[list] = None,
    live_key        : str = "",
    regime_key      : str = "",
    prediction_json : str = "",
    height          : int = 500,
) -> str:
    """
    Build a TradingView-style candlestick chart HTML using lightweight-charts.
    Returns an HTML string rendered via st.components.v1.html().
    `ticker`, `live_key`, and `regime_key` form the cache key so the HTML
    refreshes when the ticker, live price, or regime assignment changes.
    """
    # ── Candle + volume data ─────────────────────────────────────────────────
    candle_data = []
    vol_data    = []
    sma_data    = []

    # Pre-compute EMA-21 and EMA-100 on the close series
    close_series = _df["close"]
    ema21_series  = close_series.ewm(span=21,  adjust=False).mean()
    ema100_series = close_series.ewm(span=100, adjust=False).mean()
    ema21_data  = []
    ema100_data = []

    for ts, row in _df.iterrows():
        date_str = ts.strftime("%Y-%m-%d")
        candle_data.append({
            "time":  date_str,
            "open":  round(float(row["open"]),  2),
            "high":  round(float(row["high"]),  2),
            "low":   round(float(row["low"]),   2),
            "close": round(float(row["close"]), 2),
        })
        is_up = row["close"] >= row["open"]
        vol_data.append({
            "time":  date_str,
            "value": float(row["volume"]),
            "color": "rgba(38,166,154,0.5)" if is_up else "rgba(239,83,80,0.5)",
        })
        if "sma_baseline" in _df.columns and pd.notna(row.get("sma_baseline")):
            sma_data.append({
                "time":  date_str,
                "value": round(float(row["sma_baseline"]), 2),
            })
        ema21_val  = ema21_series.get(ts)
        ema100_val = ema100_series.get(ts)
        if ema21_val is not None and pd.notna(ema21_val):
            ema21_data.append({"time": date_str, "value": round(float(ema21_val), 2)})
        if ema100_val is not None and pd.notna(ema100_val):
            ema100_data.append({"time": date_str, "value": round(float(ema100_val), 2)})

    # ── Regime background bands ──────────────────────────────────────────────
    regime_bands = []
    if "regime_label" in _df.columns:
        prev_regime = None
        band_start  = None
        for ts, reg in _df["regime_label"].items():
            date_str = ts.strftime("%Y-%m-%d")
            if reg != prev_regime:
                if prev_regime is not None:
                    regime_bands.append({
                        "from":  band_start,
                        "to":    date_str,
                        "color": REGIME_BAND_COLORS.get(prev_regime, "#aaaaaa"),
                    })
                band_start  = date_str
                prev_regime = reg
        if prev_regime is not None:
            regime_bands.append({
                "from":  band_start,
                "to":    _df.index[-1].strftime("%Y-%m-%d"),
                "color": REGIME_BAND_COLORS.get(prev_regime, "#aaaaaa"),
            })

    # ── Trade markers ────────────────────────────────────────────────────────
    markers = []
    if _trade_log:
        for t in _trade_log:
            markers.append({
                "time":     t["entry_time"].strftime("%Y-%m-%d"),
                "position": "belowBar",
                "shape":    "arrowUp",
                "color":    "#00e676",
                "text":     "",
                "size":     1,
            })
            pnl_pct = t.get("pnl_pct", 0)
            markers.append({
                "time":     t["exit_time"].strftime("%Y-%m-%d"),
                "position": "aboveBar",
                "shape":    "arrowDown",
                "color":    "#00e676" if t.get("pnl_usd", 0) >= 0 else "#ff1744",
                "text":     f"{pnl_pct:+.1f}%",
                "size":     1,
            })
        markers.sort(key=lambda m: m["time"])

    # ── Regime legend items ──────────────────────────────────────────────────
    legend_items = [
        {"label": lbl, "color": clr}
        for lbl, clr in REGIME_COLORS.items()
    ]

    # ── Serialise to JSON ────────────────────────────────────────────────────
    candle_json  = json.dumps(candle_data)
    vol_json     = json.dumps(vol_data)
    sma_json     = json.dumps(sma_data)
    ema21_json   = json.dumps(ema21_data)
    ema100_json  = json.dumps(ema100_data)
    bands_json   = json.dumps(regime_bands)
    markers_json = json.dumps(markers)
    legend_json  = json.dumps(legend_items)
    ticker_js    = json.dumps(ticker)
    # prediction_json is already a serialised string (or "null")
    pred_js      = prediction_json if prediction_json else "null"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="https://unpkg.com/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; overflow: hidden; font-family: -apple-system, sans-serif; }}
  #chart-wrap {{ width: 100%; position: relative; }}
  #chart {{ width: 100%; }}
  #legend {{
    position: absolute; top: 8px; left: 8px; z-index: 10;
    display: flex; flex-wrap: wrap; gap: 8px;
    pointer-events: none;
  }}
  .leg-item {{
    display: flex; align-items: center; gap: 4px;
    background: rgba(0,0,0,0.55); border-radius: 4px;
    padding: 2px 7px; font-size: 11px; color: #e6edf3;
    border: 1px solid rgba(255,255,255,0.08);
  }}
  .leg-swatch {{
    width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
  }}
</style>
</head>
<body>
<div id="chart-wrap">
  <div id="legend"></div>
  <div id="chart"></div>
</div>
<script>
(function() {{
  const CANDLE_DATA  = {candle_json};
  const VOL_DATA     = {vol_json};
  const SMA_DATA     = {sma_json};
  const EMA21_DATA   = {ema21_json};
  const EMA100_DATA  = {ema100_json};
  const BANDS        = {bands_json};
  const MARKERS      = {markers_json};
  const LEGEND_ITEMS = {legend_json};
  const TICKER       = {ticker_js};
  const PREDICTION   = {pred_js};
  const CHART_H      = {height};

  // ── Regime legend ────────────────────────────────────────────────────────
  const legendEl = document.getElementById('legend');
  LEGEND_ITEMS.forEach(item => {{
    const div = document.createElement('div');
    div.className = 'leg-item';
    div.innerHTML = `<div class="leg-swatch" style="background:${{item.color}}"></div>${{item.label}}`;
    legendEl.appendChild(div);
  }});
  // EMA legend entries
  const emaLegendItems = [
    {{ label: 'EMA-50',  color: '#ffd740' }},
    {{ label: 'EMA-21',  color: '#29b6f6' }},
    {{ label: 'EMA-100', color: '#ab47bc' }},
  ];
  emaLegendItems.forEach(item => {{
    const div = document.createElement('div');
    div.className = 'leg-item';
    div.innerHTML = `<div class="leg-swatch" style="background:${{item.color}}"></div>${{item.label}}`;
    legendEl.appendChild(div);
  }});

  if (PREDICTION) {{
    const sign  = PREDICTION.exp_ret_pct >= 0 ? '+' : '';
    const color = PREDICTION.exp_ret_pct >= 0 ? '#ffd740' : '#ff7043';
    const div = document.createElement('div');
    div.className = 'leg-item';
    div.innerHTML =
      `<div class="leg-swatch" style="background:#ffd740; opacity:0.7;"></div>` +
      `10-Day Forecast <span style="color:${{color}};font-weight:600;">${{sign}}${{PREDICTION.exp_ret_pct}}%</span>`;
    legendEl.appendChild(div);
  }}

  // ── Create chart ─────────────────────────────────────────────────────────
  const container = document.getElementById('chart');
  const chart = LightweightCharts.createChart(container, {{
    layout: {{
      background: {{ type: 'solid', color: '#0d1117' }},
      textColor: '#e6edf3',
      fontSize: 11,
    }},
    grid: {{
      vertLines: {{ visible: false }},
      horzLines: {{ visible: false }},
    }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    rightPriceScale: {{ borderColor: '#30363d' }},
    timeScale: {{
      borderColor: '#30363d',
      timeVisible: false,
      rightOffset: 5,
    }},
    width:  container.clientWidth,
    height: CHART_H,
  }});

  // ── Candlestick series ───────────────────────────────────────────────────
  const candleSeries = chart.addCandlestickSeries({{
    upColor:         '#26a69a',
    downColor:       '#ef5350',
    borderUpColor:   '#26a69a',
    borderDownColor: '#ef5350',
    wickUpColor:     '#26a69a',
    wickDownColor:   '#ef5350',
    title:           '',
    lastValueVisible: true,
    priceLineVisible: true,
    priceLineStyle:   2,
  }});
  candleSeries.setData(CANDLE_DATA);

  // ── Regime background bands via ISeriesPrimitive ─────────────────────────
  if (BANDS.length > 0) {{
    class RegimeRenderer {{
      constructor(bands, chart) {{
        this._bands = bands;
        this._chart = chart;
      }}
      draw(target) {{}}
      drawBackground(target) {{
        const ts = this._chart.timeScale();
        target.useBitmapCoordinateSpace(scope => {{
          const ctx = scope.context;
          const h   = scope.bitmapSize.height;
          const hpr = scope.horizontalPixelRatio;
          ctx.save();
          for (const b of this._bands) {{
            const x0 = ts.timeToCoordinate(b.from);
            const x1 = ts.timeToCoordinate(b.to);
            if (x0 == null || x1 == null) continue;
            const lx = Math.min(x0, x1);
            const rx = Math.max(x0, x1);
            ctx.globalAlpha = 0.15;
            ctx.fillStyle   = b.color;
            ctx.fillRect(
              Math.round(lx * hpr), 0,
              Math.round((rx - lx) * hpr), h
            );
          }}
          ctx.globalAlpha = 1;
          ctx.restore();
        }});
      }}
    }}
    class RegimePaneView {{
      constructor(bands, chart) {{ this._r = new RegimeRenderer(bands, chart); }}
      renderer() {{ return this._r; }}
      zOrder()   {{ return 'bottom'; }}
    }}
    class RegimePrimitive {{
      constructor(bands, chart) {{ this._v = new RegimePaneView(bands, chart); }}
      paneViews() {{ return [this._v]; }}
      attached(p) {{}}
      detached()  {{}}
    }}
    candleSeries.attachPrimitive(new RegimePrimitive(BANDS, chart));
  }}

  // ── EMA-50 overlay (amber) ────────────────────────────────────────────────
  if (SMA_DATA.length > 0) {{
    const smaSeries = chart.addLineSeries({{
      color:                  '#ffd740',
      lineWidth:              1,
      priceLineVisible:       false,
      lastValueVisible:       false,
      crosshairMarkerVisible: false,
    }});
    smaSeries.setData(SMA_DATA);
  }}

  // ── EMA-21 overlay (sky blue) ─────────────────────────────────────────────
  if (EMA21_DATA.length > 0) {{
    const ema21Series = chart.addLineSeries({{
      color:                  '#29b6f6',
      lineWidth:              1,
      priceLineVisible:       false,
      lastValueVisible:       false,
      crosshairMarkerVisible: false,
    }});
    ema21Series.setData(EMA21_DATA);
  }}

  // ── EMA-100 overlay (purple) ──────────────────────────────────────────────
  if (EMA100_DATA.length > 0) {{
    const ema100Series = chart.addLineSeries({{
      color:                  '#ab47bc',
      lineWidth:              1,
      priceLineVisible:       false,
      lastValueVisible:       false,
      crosshairMarkerVisible: false,
    }});
    ema100Series.setData(EMA100_DATA);
  }}

  // ── Volume histogram (lower pane, 20% height) ────────────────────────────
  const volSeries = chart.addHistogramSeries({{
    priceFormat:      {{ type: 'volume' }},
    priceScaleId:     'vol',
    lastValueVisible: false,
    priceLineVisible: false,
  }});
  volSeries.priceScale().applyOptions({{
    scaleMargins: {{ top: 0.80, bottom: 0 }},
  }});
  volSeries.setData(VOL_DATA);

  // ── Trade markers ────────────────────────────────────────────────────────
  if (MARKERS.length > 0) {{
    candleSeries.setMarkers(MARKERS);
  }}

  // ── Kernel regression forecast ───────────────────────────────────────────
  if (PREDICTION && PREDICTION.dates && PREDICTION.dates.length > 0) {{
    // Anchor all three paths to today's close so lines extend naturally
    const anchor = {{time: PREDICTION.anchor_date, value: PREDICTION.anchor_price}};

    const expData = [anchor, ...PREDICTION.dates.map((t, i) => ({{time: t, value: PREDICTION.expected[i]}}))];
    const upData  = [anchor, ...PREDICTION.dates.map((t, i) => ({{time: t, value: PREDICTION.upper[i]}}))];
    const loData  = [anchor, ...PREDICTION.dates.map((t, i) => ({{time: t, value: PREDICTION.lower[i]}}))];

    // Expected path – dashed amber
    const expSeries = chart.addLineSeries({{
      color: '#ffd740', lineWidth: 2, lineStyle: 2,
      crosshairMarkerVisible: false, lastValueVisible: true, priceLineVisible: false, title: '',
    }});
    expSeries.setData(expData);

    // Upper band – lighter amber dashed
    const upSeries = chart.addLineSeries({{
      color: 'rgba(255,215,64,0.4)', lineWidth: 1, lineStyle: 2,
      crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false, title: '',
    }});
    upSeries.setData(upData);

    // Lower band – lighter amber dashed
    const loSeries = chart.addLineSeries({{
      color: 'rgba(255,215,64,0.4)', lineWidth: 1, lineStyle: 2,
      crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false, title: '',
    }});
    loSeries.setData(loData);
  }}

  chart.timeScale().fitContent();

  // ── Prev-close lookup for accurate day change calculation ────────────────
  const prevCloseMap = new Map();
  for (let i = 1; i < CANDLE_DATA.length; i++) {{
    prevCloseMap.set(CANDLE_DATA[i].time, CANDLE_DATA[i - 1].close);
  }}

  // ── Crosshair tooltip ────────────────────────────────────────────────────
  const tooltip = document.createElement('div');
  Object.assign(tooltip.style, {{
    position: 'absolute', display: 'none', padding: '6px 10px',
    background: 'rgba(13,17,23,0.9)', border: '1px solid #30363d',
    borderRadius: '6px', color: '#e6edf3', fontSize: '11px',
    pointerEvents: 'none', zIndex: '20', whiteSpace: 'nowrap',
  }});
  document.getElementById('chart-wrap').appendChild(tooltip);

  chart.subscribeCrosshairMove(param => {{
    if (!param.point || !param.time) {{ tooltip.style.display = 'none'; return; }}
    const bar = param.seriesData.get(candleSeries);
    if (!bar) {{ tooltip.style.display = 'none'; return; }}
    const {{open, high, low, close}} = bar;
    const prevClose = prevCloseMap.get(param.time) ?? open;
    const chg = close - prevClose;
    const chgPct = ((chg / prevClose) * 100).toFixed(2);
    const chgColor = chg >= 0 ? '#26a69a' : '#ef5350';
    tooltip.innerHTML =
      `<b style="color:#e6edf3">${{param.time}}</b><br>` +
      `O <b>${{open.toLocaleString()}}</b>  ` +
      `H <b>${{high.toLocaleString()}}</b>  ` +
      `L <b>${{low.toLocaleString()}}</b>  ` +
      `C <b>${{close.toLocaleString()}}</b>  ` +
      `<span style="color:${{chgColor}}">${{chg >= 0 ? '+' : ''}}${{chg.toFixed(2)}} (${{chg >= 0 ? '+' : ''}}${{chgPct}}%)</span>`;
    const x = param.point.x;
    const y = 10;
    const maxLeft = container.clientWidth - tooltip.offsetWidth - 10;
    tooltip.style.left    = Math.min(x + 12, maxLeft) + 'px';
    tooltip.style.top     = y + 'px';
    tooltip.style.display = 'block';
  }});

  // ── Responsive resize ────────────────────────────────────────────────────
  const ro = new ResizeObserver(entries => {{
    for (const e of entries) {{
      chart.applyOptions({{ width: e.contentRect.width }});
    }}
  }});
  ro.observe(container);
}})();
</script>
</body></html>"""

    return html


# Patch the last candle with live OHLCV so the chart shows the current price
df_chart  = df.copy()
live_key  = ""
if live_quote and live_quote.get("last_price"):
    last_idx = df_chart.index[-1]
    lp = live_quote["last_price"]
    dh = live_quote["day_high"]
    dl_val = live_quote["day_low"]
    df_chart.loc[last_idx, "close"] = lp
    if dh > df_chart.loc[last_idx, "high"]:
        df_chart.loc[last_idx, "high"] = dh
    if dl_val < df_chart.loc[last_idx, "low"]:
        df_chart.loc[last_idx, "low"] = dl_val
    live_key = str(round(lp, 2))

_last_regime = str(df_chart["regime_label"].iloc[-1]) if "regime_label" in df_chart.columns else ""
_last_date   = df_chart.index[-1].strftime("%Y-%m-%d")
regime_key   = f"{_last_date}_{_last_regime}"

_prediction      = compute_kernel_prediction(df_chart, ticker_symbol, regime_key=regime_key)
_prediction_json = json.dumps(_prediction) if _prediction else "null"

chart_html = _build_tv_chart_html(
    df_chart, ticker_symbol, backtester.trade_log,
    live_key=live_key, regime_key=regime_key,
    prediction_json=_prediction_json,
)
components.html(chart_html, height=510, scrolling=False)

st.markdown("---")


# ===========================================================================
#  ROW 4 – EQUITY CURVE
# ===========================================================================
st.subheader("💹 Equity Curve: Strategy vs Buy & Hold")

@st.cache_data(ttl=1800, show_spinner=False)
def build_equity_chart(
    _equity_curve : pd.Series,
    _df           : pd.DataFrame,
    capital       : float,
    ticker        : str,
) -> go.Figure:
    """Build the equity curve comparison chart."""
    eq = _equity_curve.copy()

    # Buy & Hold equity: invest full capital at first bar, hold until end
    bh_start  = _df["close"].iloc[0]
    bh_equity = (_df["close"] / bh_start) * capital

    fig = go.Figure()

    # Strategy equity
    fig.add_trace(go.Scatter(
        x    = eq.index,
        y    = eq.values,
        name = "HMM Strategy",
        mode = "lines",
        line = dict(color="#00e5ff", width=2),
        fill = "tozeroy",
        fillcolor = "rgba(0,229,255,0.05)",
    ))

    # Buy & Hold equity
    fig.add_trace(go.Scatter(
        x    = bh_equity.index,
        y    = bh_equity.values,
        name = "Buy & Hold",
        mode = "lines",
        line = dict(color="#ffd740", width=1.5, dash="dash"),
    ))

    # Starting capital reference line
    fig.add_hline(
        y           = capital,
        line_dash   = "dot",
        line_color  = "#444",
        annotation_text = "Starting Capital",
        annotation_position = "bottom right",
    )

    fig.update_layout(
        height        = 380,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "#0a0a0a",
        font          = dict(color="#e0e0e0"),
        legend        = dict(bgcolor="rgba(0,0,0,0.5)"),
        margin        = dict(l=10, r=10, t=30, b=10),
        yaxis         = dict(
            tickprefix = "$",
            color      = "#e0e0e0",
            gridcolor  = "#1a1a1a",
        ),
        xaxis         = dict(color="#888", gridcolor="#1a1a1a"),
    )
    return fig


equity_chart = build_equity_chart(backtester.equity_curve, df, float(initial_capital), ticker_symbol)
st.plotly_chart(equity_chart, use_container_width=True)

st.markdown("---")


# ===========================================================================
#  ROW 5 – PERFORMANCE METRICS
# ===========================================================================
st.subheader("📋 Backtest Performance Metrics")

m = metrics  # alias

# Row of metric cards
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    st.metric(
        "Total Return",
        f"{m['total_return_pct']:+.2f}%",
        delta      = f"{m['total_return_pct']:+.1f}%",
        delta_color= "normal" if m["total_return_pct"] >= 0 else "inverse",
    )

with c2:
    st.metric(
        "Buy & Hold Return",
        f"{m['bh_return_pct']:+.2f}%",
    )

with c3:
    alpha_sign = "normal" if m["alpha_pct"] >= 0 else "inverse"
    st.metric(
        "Alpha vs B&H",
        f"{m['alpha_pct']:+.2f}%",
        delta      = f"{m['alpha_pct']:+.1f}%",
        delta_color= alpha_sign,
    )

with c4:
    st.metric(
        "Max Drawdown",
        f"{m['max_drawdown_pct']:.2f}%",
        delta      = f"{m['max_drawdown_pct']:.1f}%",
        delta_color= "inverse",   # drawdown is always bad
    )

with c5:
    st.metric(
        "Win Rate",
        f"{m['win_rate_pct']:.1f}%",
        delta      = f"{m['n_trades']} trades",
        delta_color= "off",
    )

with c6:
    st.metric(
        "Sharpe Ratio",
        f"{m['sharpe_ratio']:.3f}",
        delta      = f"B&H: {m['bh_sharpe_ratio']:.3f}",
        delta_color= "off",
    )

with c7:
    st.metric(
        "Final Equity",
        f"${m['final_equity']:,.0f}",
        delta      = f"${m['final_equity'] - initial_capital:+,.0f}",
        delta_color= "normal" if m["final_equity"] >= initial_capital else "inverse",
    )

# Expanded metrics expander
with st.expander("📐 Full Metrics Detail"):
    detail_df = pd.DataFrame([
        {"Metric": "Total Strategy Return",     "Value": f"{m['total_return_pct']:+.2f}%"},
        {"Metric": "Buy & Hold Return",         "Value": f"{m['bh_return_pct']:+.2f}%"},
        {"Metric": "Alpha (Strategy − B&H)",    "Value": f"{m['alpha_pct']:+.2f}%"},
        {"Metric": "Maximum Drawdown",          "Value": f"{m['max_drawdown_pct']:.2f}%"},
        {"Metric": "Win Rate",                  "Value": f"{m['win_rate_pct']:.1f}%"},
        {"Metric": "Number of Trades",          "Value": str(m["n_trades"])},
        {"Metric": "Avg Trade PnL (USD)",       "Value": f"${m['avg_trade_pnl_usd']:,.2f}"},
        {"Metric": "Sharpe Ratio (annualised)", "Value": f"{m['sharpe_ratio']:.4f}"},
        {"Metric": "B&H Sharpe Ratio (annualised)", "Value": f"{m['bh_sharpe_ratio']:.4f}"},
        {"Metric": "Final Portfolio Value",     "Value": f"${m['final_equity']:,.2f}"},
        {"Metric": "Starting Capital",          "Value": f"${initial_capital:,.2f}"},
        {"Metric": "Leverage Applied",          "Value": f"{backtester.leverage}×"},
        {"Metric": "Active Confirmations",      "Value": f"{n_enabled} / {len(bt_module.ALL_CONFIRMATIONS)}"},
        {"Metric": "Min Confirmations Required","Value": f"{backtester.min_confirms}/{n_enabled}"},
        {"Metric": "Bear Confirm Days",         "Value": f"{backtester.bear_confirm_days} consecutive Bear bars"},
        {"Metric": "Min Hold Period",           "Value": f"{backtester.min_hold_days} days"},
        {"Metric": "Cooldown Period",           "Value": "2 days after exit"},
        {"Metric": "Trailing Stop",             "Value": "2% (active)" if backtester.use_trail_stop else "Disabled"},
    ])
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.markdown("---")


# ===========================================================================
#  ROW 6 – PARAMETER OPTIMISER RECOMMENDATION
# ===========================================================================
_opt_result = st.session_state.get("opt_result")
_opt_ticker = st.session_state.get("opt_ticker")

if _opt_result is not None and (
    "bnh_return" not in _opt_result or "best_confirmations" not in _opt_result
):
    # Stale result from old version — evict so the user gets a clean prompt
    st.session_state.pop("opt_result", None)
    st.session_state.pop("opt_ticker", None)
    _opt_result = None
    _opt_ticker = None

if _opt_result is not None and _opt_ticker == ticker_symbol:
    st.subheader("🎯 Optimised Parameters for " + ticker_symbol)

    bnh_ret      = _opt_result["bnh_return"]
    n_beats      = _opt_result["n_beats_bnh"]
    n_total      = _opt_result["n_total"]
    best_bnh     = _opt_result["best_bnh_params"]   # None if nothing beats B&H
    cur_ret      = metrics["total_return_pct"]

    # ── B&H coverage banner ──────────────────────────────────────────
    pct_beat = (n_beats / n_total * 100) if n_total else 0
    if n_beats > 0:
        st.success(
            f"**{n_beats} / {n_total}** parameter combinations ({pct_beat:.0f}%) "
            f"beat the buy-and-hold return of **{bnh_ret:+.1f}%** for this period. "
            f"The best B&H-beating combo is highlighted below."
        )
    else:
        st.warning(
            f"⚠️ **No combination** out of {n_total} beat the buy-and-hold return "
            f"of **{bnh_ret:+.1f}%** for this period. "
            "The strategy may be ill-suited to this asset or lookback window. "
            "Try a different look-back period or disable some confirmation filters."
        )

    # ── Primary recommendation: best B&H-beating combo (or best overall) ──
    if best_bnh is not None:
        show_params = best_bnh
        show_ret    = best_bnh["total_return_pct"]
        card_label  = "🏆 Best Return (Beats B&H)"
    else:
        show_params = _opt_result["best_params"]
        show_ret    = _opt_result["best_return"]
        card_label  = "Best Return (Does Not Beat B&H)"

    delta_vs_cur   = show_ret  - cur_ret
    delta_vs_bnh   = show_ret  - bnh_ret

    # ── Best confirmation filters ─────────────────────────────────────
    _best_confirms     = _opt_result.get("best_confirmations") or bt_module.ALL_CONFIRMATIONS
    _best_confirms_set = set(_best_confirms)
    _all_confirms      = bt_module.ALL_CONFIRMATIONS
    _n_sel             = len(_best_confirms_set)
    _n_pool            = len(_all_confirms)

    st.markdown(
        f"**🎛 Best Confirmation Filters** &nbsp;—&nbsp; "
        f"**{_n_sel} of {_n_pool}** selected by optimization"
    )
    _fc_cols = st.columns(5)
    for _ci, _conf in enumerate(_all_confirms):
        _icon = "✅" if _conf in _best_confirms_set else "☐"
        _fc_cols[_ci % 5].markdown(f"{_icon}&nbsp;{_conf}")
    st.markdown("")

    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    oc1.metric("Bear Confirm Days",  show_params["bear_confirm_days"],
               help="Consecutive Bear bars before exit fires")
    oc2.metric("Min Confirmations",  show_params["min_confirms"],
               help="Technical signals that must align for entry")
    oc3.metric("Min Hold Days",      show_params["min_hold_days"],
               help="Days before trailing stop can fire (only when trailing stop is on)")
    oc4.metric(card_label,           f"{show_ret:+.1f}%",
               delta=f"{delta_vs_cur:+.1f}% vs current",
               delta_color="normal" if delta_vs_cur >= 0 else "inverse")
    oc5.metric("Alpha vs B&H",       f"{delta_vs_bnh:+.1f}%",
               delta=f"B&H = {bnh_ret:+.1f}%",
               delta_color="normal" if delta_vs_bnh >= 0 else "inverse")

    # ── Top-10 table ─────────────────────────────────────────────────
    _expander_title = (
        f"📊 Top {min(10, n_beats)} B&H-Beating Combinations"
        if n_beats > 0 else "📊 Top 10 Parameter Combinations (none beat B&H)"
    )
    with st.expander(_expander_title, expanded=False):
        top_df = _opt_result["top_results"].copy()
        top_df.columns = [
            "Bear Days", "Min Confirms", "Hold Days",
            "Return %", "Sharpe", "Max DD %", "# Trades", "Win %",
            "Alpha %", "B&H Return %", "Beats B&H",
        ]
        top_df.index = range(1, len(top_df) + 1)

        def _style_row(row):
            styles = [""] * len(row)
            ret_idx  = top_df.columns.get_loc("Return %")
            alp_idx  = top_df.columns.get_loc("Alpha %")
            beat_idx = top_df.columns.get_loc("Beats B&H")
            styles[ret_idx]  = "color: #00e676" if row["Return %"]  > 0  else "color: #ff5252"
            styles[alp_idx]  = "color: #00e676" if row["Alpha %"]   > 0  else "color: #ff5252"
            styles[beat_idx] = "color: #00e676; font-weight:700" if row["Beats B&H"] else "color: #ff5252"
            return styles

        st.dataframe(
            top_df.style.apply(_style_row, axis=1),
            use_container_width=True,
        )

    st.caption(
        "⚠️ In-sample optimisation — these parameters fit the historical data "
        "shown and may not generalise to future prices. Use as a starting point, "
        "not a guarantee."
    )
    st.markdown("---")


# ===========================================================================
#  ROW 7 – TRADE LOG
# ===========================================================================
st.subheader("🗒️ Trade Log")

trade_log_df = backtester.get_trade_log_df()

if trade_log_df.empty:
    st.info(
        "No trades were executed during the backtest period.  "
        "This is normal if the Bull Run regime conditions were never met simultaneously "
        "with the required number of technical confirmations."
    )
else:
    # Summary stats above the table
    n_trades   = len(trade_log_df)
    n_wins     = (trade_log_df["PnL ($)"] > 0).sum()
    n_losses   = n_trades - n_wins
    total_pnl  = trade_log_df["PnL ($)"].sum()
    total_days = int(trade_log_df["Days Held"].sum())
    period_days = max((df.index[-1] - df.index[0]).days, 1)
    pct_in_market = total_days / period_days * 100

    tl_c1, tl_c2, tl_c3, tl_c4, tl_c5 = st.columns(5)
    tl_c1.metric("Total Trades",    n_trades)
    tl_c2.metric("Winning Trades",  n_wins,   delta=f"{n_wins/n_trades*100:.1f}%")
    tl_c3.metric("Losing Trades",   n_losses, delta=f"-{n_losses/n_trades*100:.1f}%",
                 delta_color="inverse")
    tl_c4.metric("Total PnL",       f"${total_pnl:,.2f}",
                 delta_color="normal" if total_pnl >= 0 else "inverse")
    tl_c5.metric("Days in Market",  f"{total_days}d ({pct_in_market:.1f}%)",
                 delta=f"avg {total_days // n_trades}d / trade",
                 delta_color="off")

    # PnL colour styling
    def color_pnl(val):
        if isinstance(val, (int, float)):
            color = "#00e67699" if val > 0 else "#ff174499" if val < 0 else "transparent"
            return f"background-color: {color}"
        return ""

    styled_log = trade_log_df.style.map(
        color_pnl, subset=["PnL (%)", "PnL ($)"]
    )

    st.dataframe(styled_log, use_container_width=True, hide_index=True)

    # Download button
    csv = trade_log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Download Trade Log CSV",
        data      = csv,
        file_name = "hmm_trade_log.csv",
        mime      = "text/csv",
    )

st.markdown("---")


# ===========================================================================
#  ROW 8 – REGIME DISTRIBUTION (small supplementary chart)
# ===========================================================================
st.subheader("🧩 Historical Regime Distribution")

if "regime_label" in df.columns:
    regime_counts = df["regime_label"].value_counts().reset_index()
    regime_counts.columns = ["Regime", "Count"]
    regime_counts["Percentage"] = (regime_counts["Count"] / len(df) * 100).round(1)
    regime_counts["Color"]      = regime_counts["Regime"].map(
        lambda r: REGIME_COLORS.get(r, "#888")
    )

    col_pie, col_table = st.columns([1, 1])

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels      = regime_counts["Regime"],
            values      = regime_counts["Count"],
            marker_colors = regime_counts["Color"].tolist(),
            hole        = 0.45,
            textinfo    = "label+percent",
            hovertemplate = "<b>%{label}</b><br>%{value} bars (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            height        = 300,
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="#e0e0e0"),
            showlegend    = False,
            margin        = dict(l=10, r=10, t=20, b=10),
            annotations   = [dict(text="Regime<br>Share", x=0.5, y=0.5,
                                  font_size=14, showarrow=False, font_color="#aaa")],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        display_df = regime_counts[["Regime", "Count", "Percentage"]].copy()
        display_df["Percentage"] = display_df["Percentage"].astype(str) + "%"

        def color_regime_row(row):
            color = REGIME_COLORS.get(row["Regime"], "#888888")
            return [f"color: {color}; font-weight: bold"] + [""] * (len(row) - 1)

        styled_dist = display_df.style.apply(color_regime_row, axis=1)
        st.dataframe(styled_dist, use_container_width=True, hide_index=True, height=260)


# ===========================================================================
#  FOOTER
# ===========================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:12px;'>"
    "⚠️ <b>Disclaimer:</b> This dashboard is for educational and research purposes only. "
    "It does not constitute financial advice. Past performance does not guarantee future results. "
    "Cryptocurrency trading involves significant risk of loss. "
    "Always do your own research before trading."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center; color:#333; font-size:11px;'>"
    "HMM Regime Terminal · Built with Streamlit + hmmlearn + Plotly "
    f"· Data via yfinance · {ticker_symbol} Daily"
    "</p>",
    unsafe_allow_html=True,
)
