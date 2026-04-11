"""
scanner.py
==========
S&P 500 Bull Regime Entry Scanner

Scans all ~500 S&P 500 constituents for stocks that have recently entered a
Bull Run regime according to the GMM-HMM model, ranks them by a composite
entry-quality score, and prints the best 30 candidates.

Entry quality score (0–100) weighted combination of:
  • HMM bull-state probability   40 pts  — how confident the model is
  • Confirmation signals met      35 pts  — technical breadth at entry
  • Freshness of transition       25 pts  — exponential decay; day-1=25, day-5≈11

Usage
-----
    python scanner.py                         # scan all ~500 tickers (~15–30 min)
    python scanner.py --target 30             # top N to display  (default: 30)
    python scanner.py --max-bars-in-bull 5    # only stocks in bull ≤ N bars
    python scanner.py --lookback 730          # days of history per ticker
    python scanner.py --min-confirms 3        # min confirmations required
    python scanner.py --workers 10            # parallel download threads
    python scanner.py --limit 100             # restrict scan to first N tickers (testing)
    python scanner.py --out results.csv       # save full ranked list to CSV

Disclaimer
----------
Educational / research use only.  Not financial advice.
HMM regime labels are in-sample statistical artefacts — they do not predict
future returns.
"""

from __future__ import annotations

import argparse
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project modules
# ---------------------------------------------------------------------------
import data_loader as dl
from backtester import (
    LABEL_BULL,
    Backtester,
    evaluate_confirmations,
)


# ===========================================================================
#  Index ticker lists
# ===========================================================================

def get_sp500_tickers() -> list[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.
    Returns a sorted list of ticker symbols compatible with yfinance.
    Falls back to a curated 50-stock subset if Wikipedia is unreachable.
    """
    try:
        url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url, attrs={"id": "constituents"})[0]
        tickers = (
            table["Symbol"]
            .str.replace(".", "-", regex=False)   # BRK.B → BRK-B for yfinance
            .str.strip()
            .tolist()
        )
        tickers = sorted(set(tickers))
        print(f"[Scanner] Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as exc:
        print(f"[Scanner] WARNING — could not fetch Wikipedia list: {exc}")
        print("[Scanner] Falling back to curated 50-stock subset.")
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM",
            "LLY", "V",    "UNH",  "XOM",  "MA",   "JNJ",  "PG",   "HD",   "COST",
            "ABBV","MRK",  "CVX",  "CRM",  "BAC",  "NFLX", "AMD",  "ORCL", "ACN",
            "LIN", "TMO",  "PEP",  "ADBE", "MCD",  "QCOM", "DIS",  "GS",   "CAT",
            "IBM", "INTU", "AMAT", "TXN",  "AMGN", "NOW",  "ISRG", "SPGI", "AXP",
            "BLK", "SYK",  "VRTX", "GILD", "ADI",
        ]


def get_nasdaq100_tickers() -> list[str]:
    """
    Scrape the current Nasdaq-100 constituent list from Wikipedia.
    Returns a sorted list of ticker symbols compatible with yfinance.
    """
    try:
        url   = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        # Find the table that has a 'Ticker' or 'Symbol' column
        table = next(t for t in tables if "Ticker" in t.columns or "Symbol" in t.columns)
        col   = "Ticker" if "Ticker" in table.columns else "Symbol"
        tickers = (
            table[col]
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        tickers = sorted(set(tickers))
        print(f"[Scanner] Loaded {len(tickers)} Nasdaq-100 tickers from Wikipedia.")
        return tickers
    except Exception as exc:
        print(f"[Scanner] WARNING — could not fetch Nasdaq-100 list: {exc}")
        return []


def get_russell2000_tickers() -> list[str]:
    """
    Fetch Russell 2000 constituents via the iShares IWM holdings CSV.
    Returns a sorted list of ticker symbols compatible with yfinance.
    Falls back to an empty list on failure.
    """
    try:
        url = (
            "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
            "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        )
        df = pd.read_csv(url, skiprows=9)
        tickers = (
            df["Ticker"]
            .dropna()
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        # Drop non-ticker rows (cash, N/A, etc.)
        tickers = sorted({t for t in tickers if t.isalpha() or "-" in t})
        print(f"[Scanner] Loaded {len(tickers)} Russell 2000 tickers from iShares.")
        return tickers
    except Exception as exc:
        print(f"[Scanner] WARNING — could not fetch Russell 2000 list: {exc}")
        return []


def get_russell3000_tickers() -> list[str]:
    """
    Fetch Russell 3000 constituents via the iShares IWV holdings CSV.
    Returns a sorted list of ticker symbols compatible with yfinance.
    Falls back to an empty list on failure.
    """
    try:
        url = (
            "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
            "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
        )
        df = pd.read_csv(url, skiprows=9)
        tickers = (
            df["Ticker"]
            .dropna()
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        tickers = sorted({t for t in tickers if t.isalpha() or "-" in t})
        print(f"[Scanner] Loaded {len(tickers)} Russell 3000 tickers from iShares.")
        return tickers
    except Exception as exc:
        print(f"[Scanner] WARNING — could not fetch Russell 3000 list: {exc}")
        return []


# ===========================================================================
#  Single-ticker scan
# ===========================================================================

def _count_bars_in_bull(labels: pd.Series) -> int:
    """
    Count how many consecutive trailing bars are labelled LABEL_BULL.
    Returns 0 if the most recent bar is not Bull Run.
    """
    count = 0
    for lbl in reversed(labels.tolist()):
        if lbl == LABEL_BULL:
            count += 1
        else:
            break
    return count


def _entry_score(bull_prob: float, confirms_met: int, n_active: int,
                 bars_in_bull: int) -> float:
    """
    Composite entry-quality score in [0, 100].

    Components
    ----------
    bull_prob_score : 40 pts  — raw HMM bull-state posterior probability
    confirm_score   : 35 pts  — fraction of active confirmation signals met
    freshness_score : 25 pts  — exponential decay over bars_in_bull
                                day-1 → 25.0, day-3 → 16.7, day-5 → 11.2, day-10 → 3.4
    """
    prob_component    = bull_prob * 40.0
    conf_component    = (confirms_met / max(n_active, 1)) * 35.0
    fresh_component   = 25.0 * math.exp(-0.2 * (bars_in_bull - 1))
    return round(prob_component + conf_component + fresh_component, 2)


def scan_ticker(
    ticker          : str,
    lookback        : int = 730,
    min_conf        : int = 3,
    max_bars_in_bull: int = 999,
) -> dict | None:
    """
    Run the GMM-HMM on *ticker* and return a result dict if it is currently
    in Bull Run regime and entered within *max_bars_in_bull* bars.

    Returns None on data errors, insufficient history, wrong regime, or
    when the stock has been in Bull Run too long.
    """
    try:
        raw_df = dl.load(ticker=ticker, period_days=lookback)
        if raw_df is None or raw_df.empty or len(raw_df) < 150:
            return None

        bt = Backtester(leverage_override=1.0, min_confirms=min_conf)
        prepared_df, _ = bt._prepare(raw_df)

        if "regime_label" not in prepared_df.columns or len(prepared_df) < 3:
            return None

        last    = prepared_df.iloc[-1]
        prev    = prepared_df.iloc[-2]
        regime  = last["regime_label"]

        # Must currently be in Bull Run
        if regime != LABEL_BULL:
            return None

        # Count consecutive Bull bars to gauge freshness
        bars_in_bull = _count_bars_in_bull(prepared_df["regime_label"])

        # Apply freshness gate
        if bars_in_bull > max_bars_in_bull:
            return None

        is_transition = (prev["regime_label"] != LABEL_BULL)  # entered on latest bar

        # Technical confirmations on the most recent bar
        confirms_detail = evaluate_confirmations(last)
        active_confirms = bt.enabled_confirmations or list(confirms_detail.keys())
        confirms_met    = sum(1 for k, v in confirms_detail.items()
                              if k in active_confirms and v)
        n_active        = len(active_confirms)

        bull_prob  = float(last.get("prob_bull", 0.0))
        score      = _entry_score(bull_prob, confirms_met, n_active, bars_in_bull)

        # 5-day price return
        recent_ret = 0.0
        if len(prepared_df) >= 6:
            recent_ret = (
                (last["close"] - prepared_df.iloc[-6]["close"])
                / prepared_df.iloc[-6]["close"] * 100
            )

        # Previous regime label (for display)
        # Walk back to find the label just before the current bull run started
        prev_regime = "—"
        idx = len(prepared_df) - bars_in_bull - 1
        if idx >= 0:
            prev_regime = prepared_df.iloc[idx]["regime_label"]

        return {
            "ticker"          : ticker,
            "entry_score"     : score,
            "bull_prob"       : round(bull_prob, 4),
            "confirms_met"    : confirms_met,
            "n_confirmations" : n_active,
            "confirms_pct"    : round(confirms_met / n_active * 100, 1) if n_active else 0,
            "bars_in_bull"    : bars_in_bull,
            "is_transition"   : is_transition,
            "prev_regime"     : prev_regime,
            "close"           : round(float(last["close"]), 2),
            "5d_return_pct"   : round(recent_ret, 2),
            "as_of"           : prepared_df.index[-1].strftime("%Y-%m-%d"),
        }

    except Exception:
        return None


# ===========================================================================
#  Main scanner
# ===========================================================================

def run_scanner(
    tickers         : list[str],
    lookback        : int        = 730,
    min_conf        : int        = 3,
    max_bars_in_bull: int        = 999,
    workers         : int        = 8,
    target          : int        = 30,
    out_csv         : str | None = None,
) -> pd.DataFrame:
    """
    Scan *tickers* in parallel, filter to Bull Run entries within
    *max_bars_in_bull* bars, rank by entry_score, return top *target* rows.
    """
    print(f"\n[Scanner] Scanning {len(tickers)} tickers "
          f"(lookback={lookback}d, max_bars_in_bull={max_bars_in_bull}, "
          f"workers={workers})…")
    print("[Scanner] Each ticker requires an HMM fit — expect several minutes.\n")

    results: list[dict] = []
    done  = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_ticker, t, lookback, min_conf, max_bars_in_bull): t
            for t in tickers
        }
        for fut in as_completed(futures):
            done += 1
            res   = fut.result()
            if res is not None:
                results.append(res)
            if done % 25 == 0 or done == total:
                print(f"  [{done:>3}/{total}] scanned  |  {len(results)} candidates found…")

    if not results:
        print("\n[Scanner] No Bull Run entries found with the current filters.")
        return pd.DataFrame()

    df = (
        pd.DataFrame(results)
        .sort_values("entry_score", ascending=False)
        .reset_index(drop=True)
    )
    df.index = range(1, len(df) + 1)

    if out_csv:
        df.to_csv(out_csv, index_label="Rank")
        print(f"\n[Scanner] Full results ({len(df)} stocks) saved to: {out_csv}")

    return df.head(target)


# ===========================================================================
#  Display
# ===========================================================================

def print_results(df: pd.DataFrame) -> None:
    if df.empty:
        print("[Scanner] No results to display.")
        return

    fresh = df[df["is_transition"]]
    early = df[~df["is_transition"]]

    print()
    print("=" * 80)
    print("  BULL REGIME ENTRY SCANNER  —  HMM Strategy  |  S&P 500")
    print("  " + datetime.now().strftime("%Y-%m-%d  %H:%M"))
    print(f"  Showing top {len(df)} stocks ranked by entry-quality score")
    print("=" * 80)

    header = (f"  {'#':<4} {'Ticker':<7} {'Score':>6} {'BullProb':>9} "
              f"{'Signals':>8} {'BarsIn':>7} {'Close':>9} {'5dRet%':>7}  {'Prev Regime':<22} {'As Of'}")
    divider = "  " + "─" * 78

    # ── Fresh day-1 transitions ──────────────────────────────────────────────
    if not fresh.empty:
        print(f"\n  FRESH ENTRIES (just transitioned today — {len(fresh)} stocks)\n")
        print(header)
        print(divider)
        for rank, row in fresh.iterrows():
            sig_str  = f"{row['confirms_met']}/{row['n_confirmations']} ({row['confirms_pct']}%)"
            prev_lbl = str(row["prev_regime"])[:20]
            print(f"  {rank:<4} {row['ticker']:<7} {row['entry_score']:>6.1f} "
                  f"{row['bull_prob']:>9.3f} {sig_str:>8} {row['bars_in_bull']:>7} "
                  f"{row['close']:>9,.2f} {row['5d_return_pct']:>+6.2f}%  "
                  f"{prev_lbl:<22} {row['as_of']}")

    # ── Early-stage (2–N bars in bull) ──────────────────────────────────────
    if not early.empty:
        max_bars = int(early["bars_in_bull"].max())
        print(f"\n  EARLY-STAGE BULL RUNS (2–{max_bars} bars in — {len(early)} stocks)\n")
        print(header)
        print(divider)
        for rank, row in early.iterrows():
            sig_str  = f"{row['confirms_met']}/{row['n_confirmations']} ({row['confirms_pct']}%)"
            prev_lbl = str(row["prev_regime"])[:20]
            print(f"  {rank:<4} {row['ticker']:<7} {row['entry_score']:>6.1f} "
                  f"{row['bull_prob']:>9.3f} {sig_str:>8} {row['bars_in_bull']:>7} "
                  f"{row['close']:>9,.2f} {row['5d_return_pct']:>+6.2f}%  "
                  f"{prev_lbl:<22} {row['as_of']}")

    print()
    print("  Score = 40×(bull prob) + 35×(signals%) + 25×exp(−0.2×(bars−1))")
    print("  BarsIn = consecutive daily bars in current Bull Run")
    print()
    print("  ⚠  Educational use only — not financial advice.")
    print("  HMM labels are in-sample; past regimes do not predict future returns.")
    print("=" * 80)
    print()


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scan S&P 500 for the best HMM Bull Regime entry candidates."
    )
    parser.add_argument(
        "--target",          type=int,   default=30,
        help="Number of top stocks to display (default: 30)"
    )
    parser.add_argument(
        "--max-bars-in-bull", type=int,  default=999,
        help="Only include stocks in Bull Run for ≤ N bars (default: no limit). "
             "Use 1 to see only day-1 transitions, 5 for ≤ 5 days in."
    )
    parser.add_argument(
        "--lookback",        type=int,   default=730,
        help="Days of history to download per ticker (default: 730)"
    )
    parser.add_argument(
        "--min-confirms",    type=int,   default=3,
        help="Minimum confirmation signals required (default: 3)"
    )
    parser.add_argument(
        "--workers",         type=int,   default=8,
        help="Parallel download threads (default: 8)"
    )
    parser.add_argument(
        "--limit",           type=int,   default=None,
        help="Restrict scan to first N tickers — useful for testing"
    )
    parser.add_argument(
        "--out",             type=str,   default=None,
        help="Save full ranked CSV to this file path"
    )
    args = parser.parse_args()

    tickers = get_sp500_tickers()
    if args.limit:
        tickers = tickers[: args.limit]
        print(f"[Scanner] Limiting scan to first {args.limit} tickers.")

    results_df = run_scanner(
        tickers          = tickers,
        lookback         = args.lookback,
        min_conf         = args.min_confirms,
        max_bars_in_bull = args.max_bars_in_bull,
        workers          = args.workers,
        target           = args.target,
        out_csv          = args.out,
    )

    print_results(results_df)
    return results_df


if __name__ == "__main__":
    main()
