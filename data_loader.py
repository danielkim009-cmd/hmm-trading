"""
data_loader.py
==============
Responsible for:
  1. Downloading daily BTC-USD OHLCV data via yfinance (last 730 days by default).
  2. Normalising multi-index columns that yfinance occasionally returns.
  3. Engineering the three features required by the HMM regime model:
       - returns       : log return of Close price
       - range         : normalised intra-bar price range (High-Low)/Close
       - vol_change    : rolling standard-deviation of volume (volume volatility)
"""

import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
TICKER               = "BTC-USD"
PERIOD_DAYS          = 730          # look-back window in calendar days
INTERVAL             = "1d"         # daily bars
VOL_ROLL_WINDOW      = 20           # window for volume-volatility calculation
RETURNS_SHIFT        = 1            # number of bars to shift for log returns
TREND_RETURN_PERIOD  = 20           # window for 4th HMM feature (medium-term trend)
INDICATOR_WARMUP     = 150          # extra calendar days fetched so that long-window
                                    # indicators (SMA-100, MACD-26) are valid from
                                    # bar 1 of the user-requested period.


def download_data(
    ticker: str = TICKER,
    period_days: int = PERIOD_DAYS,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker      : Yahoo Finance ticker symbol.
    period_days : Number of calendar days to look back.
    interval    : Bar interval string accepted by yfinance (e.g. '1d', '1wk').

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume] and a
    DatetimeIndex.  Returns an empty DataFrame on failure.
    """
    today      = datetime.now(timezone.utc).date()   # UTC matches yfinance bar labels
    # Download extra warmup days so indicator rolling windows are fully
    # populated for every bar in the user-requested period.
    # Monthly bars need more calendar-day warmup since each bar spans ~30 days.
    warmup = INDICATOR_WARMUP if interval == "1d" else 750
    start_date = today - timedelta(days=period_days + warmup)
    start_str  = start_date.strftime("%Y-%m-%d")

    print(f"[DataLoader] Downloading {ticker} {interval} data "
          f"{start_str} → today (live) …")

    # Use Ticker.history without an end date: yfinance always returns
    # today's live bar (current price) when no end is specified, unlike
    # yf.download which only returns fully-closed sessions.
    raw = pd.DataFrame()
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.Ticker(ticker).history(
                start       = start_str,
                interval    = interval,
                auto_adjust = True,
            )
            if not raw.empty:
                break
            print(f"[DataLoader] Attempt {attempt + 1}/3 returned empty DataFrame.")
        except Exception as exc:
            last_exc = exc
            print(f"[DataLoader] Attempt {attempt + 1}/3 failed: {exc}")
        if attempt < 2:
            time.sleep(2 ** attempt)

    if raw.empty:
        reason = str(last_exc) if last_exc else "yfinance returned an empty DataFrame"
        print(f"[DataLoader] ERROR – all download attempts failed. Last reason: {reason}")
        return pd.DataFrame()

    # Normalise column names to lowercase
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    raw.columns = [c.lower() for c in raw.columns]

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        print(f"[DataLoader] ERROR – missing columns after normalisation: {missing}")
        return pd.DataFrame()

    df = raw[required_cols].copy()

    # Ticker.history returns a timezone-aware index; strip tz and
    # normalise to midnight so all dates are plain date-keyed timestamps.
    if df.index.tz:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index = df.index.normalize()

    # Drop weekend/holiday artefacts and any future-dated rows
    df.dropna(how="all", inplace=True)
    df = df[df.index.date <= today]

    print(f"[DataLoader] Download complete. Shape: {df.shape}  "
          f"({df.index[0]}  →  {df.index[-1]})")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the four HMM input features to the OHLCV DataFrame.

    Features
    --------
    returns      : log(Close_t / Close_{t-1}) – captures bar-level momentum.
    range        : (High - Low) / Close       – captures intra-bar volatility.
    vol_change   : rolling std / rolling mean of volume (coefficient of variation)
                   – captures abnormal volume bursts.
    trend_return : log(Close_t / Close_{t-20}) – 20-day cumulative log return.
                   Gives the HMM explicit directional trend information so it can
                   distinguish slow uptrends from sideways/choppy regimes (Option C).

    Parameters
    ----------
    df : DataFrame with at minimum [open, high, low, close, volume] columns.

    Returns
    -------
    Original DataFrame with four additional feature columns appended.
    All rows containing NaN in these columns are dropped before returning.
    """
    if df.empty:
        return df

    df = df.copy()

    # --- Feature 1: Log Returns ------------------------------------------
    # Using log returns rather than simple returns for better statistical
    # properties (additivity, approximate normality at short intervals).
    df["returns"] = np.log(df["close"] / df["close"].shift(RETURNS_SHIFT))

    # --- Feature 2: Normalised Price Range ---------------------------------
    # (High - Low) / Close gives a scale-free measure of intra-bar volatility.
    df["range"] = (df["high"] - df["low"]) / df["close"]

    # --- Feature 3: Volume Volatility (Coefficient of Variation) ----------
    # Rolling StdDev / Rolling Mean of volume.
    # A high value signals unusual volume spikes – relevant regime information.
    roll_mean = df["volume"].rolling(VOL_ROLL_WINDOW).mean()
    roll_std  = df["volume"].rolling(VOL_ROLL_WINDOW).std()
    # Avoid division by zero: replace zero means with NaN
    df["vol_change"] = roll_std / roll_mean.replace(0, np.nan)

    # --- Feature 4: Medium-term Trend Return (Option C) --------------------
    # 20-day log cumulative return — gives the HMM explicit information about
    # whether price has been trending up or down over the past month.
    # This helps distinguish "slow uptrend (Neutral)" from "Bull Run" and
    # prevents uptrending sections from being mislabelled as Neutral.
    df["trend_return"] = np.log(df["close"] / df["close"].shift(TREND_RETURN_PERIOD))

    # --- Clean up ----------------------------------------------------------
    # Drop any row that has a NaN in any of the four feature columns.
    # This is critical: hmmlearn will raise if it receives NaN values.
    n_before = len(df)
    df.dropna(subset=["returns", "range", "vol_change", "trend_return"], inplace=True)
    n_dropped = n_before - len(df)

    if n_dropped > 0:
        print(f"[DataLoader] Dropped {n_dropped} rows containing NaN in features "
              f"(expected from rolling windows). {len(df)} rows remain.")

    # Clip extreme outliers (>5 sigma) in features to prevent HMM instability
    for col in ["returns", "range", "vol_change", "trend_return"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mu - 5 * sigma, mu + 5 * sigma)

    print(f"[DataLoader] Feature engineering complete. Final shape: {df.shape}")
    return df


def load(
    ticker: str = TICKER,
    period_days: int = PERIOD_DAYS,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Convenience wrapper: download → engineer features → return clean DataFrame.

    This is the primary entry-point used by backtester.py and dashboard.py.
    """
    raw_df      = download_data(ticker, period_days, interval)
    featured_df = engineer_features(raw_df)
    return featured_df


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load()
    print("\n[DataLoader] Sample output:")
    print(df[["close", "returns", "range", "vol_change"]].tail(10))
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull counts:\n{df[['returns','range','vol_change']].isnull().sum()}")
