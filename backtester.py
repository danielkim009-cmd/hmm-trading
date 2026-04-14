"""
backtester.py
=============
Core engine for the HMM Regime-Based Trading System.

Responsibilities
----------------
1. HMM Regime Detection  – fit a 7-state Gaussian HMM on the three engineered
                           features and autolabel each state.
2. Technical Indicators  – compute all 8 confirmation signals on the full
                           OHLCV DataFrame.
3. Strategy Logic        – entry / exit rules with configurable aggressiveness.
4. Risk Management       – leverage, 48-hour cooldown, optional trailing stop.
5. Backtester            – vectorised simulation returning an equity curve,
                           trade log, and performance metrics.
"""

from __future__ import annotations   # enables X | Y union syntax on Python 3.9

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM          # Gaussian Mixture Model HMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HMM & Strategy Constants
# ---------------------------------------------------------------------------
N_STATES            = 7          # number of hidden states
N_MIX               = 2          # Gaussian mixture components per state (GMM-HMM)
                                  # 2 components = good balance; raise to 3 for more
                                  # flexibility (but needs more data to avoid overfitting)
HMM_COVARIANCE      = "diag"     # diagonal covariance (robust across all assets/timeframes).
                                  # "full" can produce singular matrices on low-volatility data
                                  # such as SPY/QQQ.  "diag" requires far fewer parameters
                                  # (n_mix × n_features vs n_mix × n_features²) and almost
                                  # never goes singular.
HMM_ITERATIONS      = 2000       # max EM iterations
HMM_RANDOM_STATE    = 42         # reproducibility seed

# Option C: trend_return (20-day cumulative log return) is included as the
# 4th feature to give the HMM explicit directional trend information.
# This helps it distinguish "slow uptrend" from "sideways/choppy" regimes.
FEATURE_COLS        = ["returns", "range", "vol_change", "trend_return"]

# Option B: number of states to classify as Bull Run / Bear/Crash.
# Top N_BULL_STATES by mean return → Bull Run
# Bottom N_BEAR_STATES by mean return → Bear/Crash
# Remaining → Neutral/Transition
N_BULL_STATES       = 3   # top-3 states by mean return → Bull Run
N_BEAR_STATES       = 2   # bottom-2 states by mean return → Bear/Crash

# Regime label tokens
LABEL_BULL          = "Bull Run"
LABEL_BEAR          = "Bear/Crash"
LABEL_NEUTRAL       = "Neutral/Transition"

# Technical indicator parameters
RSI_PERIOD          = 14
MOMENTUM_PERIOD     = 14
ATR_PERIOD          = 14
ATR_MA_PERIOD       = 14         # MA of ATR for expansion detection
VOLUME_MA_PERIOD    = 20
ADX_PERIOD          = 14
SMA_20_PERIOD       = 20         # fast SMA – used for price filter & MA cross
SMA_50_PERIOD       = 50         # mid SMA  – used for MA cross checks
SMA_100_PERIOD      = 100        # slow SMA – used for long-term trend filter
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
STOCH_K             = 14
STOCH_D             = 3

# Strategy parameters – Normal mode
NORMAL_LEVERAGE         = 1.0    # 1× — no leverage (default; change via dashboard)
NORMAL_MIN_CONFIRMS     = 3      # out of 9 (≈ 33% threshold)

# Strategy parameters – Aggressive mode
AGGRESSIVE_LEVERAGE     = 4.0
AGGRESSIVE_MIN_CONFIRMS = 5      # out of 9 (≈ 56% threshold)
AGGRESSIVE_TRAIL_PCT    = 0.02   # 2 % trailing stop

# Risk management
COOLDOWN_DAYS       = 2          # calendar days to pause after any exit (daily bars)
BEAR_CONFIRM_DAYS   = 5          # consecutive Bear bars required before exiting
                                  # (prevents single-day "false Bear" spikes from closing the trade)
MIN_HOLD_DAYS       = 7          # minimum calendar days to hold before regime exit fires
                                  # (trailing stop can still exit earlier if enabled)
INITIAL_CAPITAL     = 100_000.0  # USD starting capital for backtest


# ===========================================================================
#  SECTION 1 – TECHNICAL INDICATORS
# ===========================================================================

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = ADX_PERIOD) -> pd.Series:
    """Average Directional Index (Wilder)."""
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # When both are positive, keep only the larger
    mask = plus_dm <= minus_dm
    plus_dm[mask] = 0.0
    mask = minus_dm <= plus_dm.shift()
    minus_dm[mask] = 0.0

    atr_val   = _atr(high, low, close, period)
    plus_di   = 100 * plus_dm.ewm( alpha=1/period, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di  = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def _macd(close: pd.Series, fast: int = MACD_FAST,
          slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    """Return (macd_line, signal_line, histogram)."""
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line
    return macd_line, sig_line, histogram


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = STOCH_K, d_period: int = STOCH_D):
    """Return (%K, %D)."""
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicator values and append them to *df*.

    Columns added
    -------------
    rsi, momentum, atr, atr_ma, volume_ma, adx,
    sma_20, sma_50 (alias: sma_baseline), sma_100,
    macd_hist, stoch_k, stoch_d
    """
    df = df.copy()

    df["rsi"]           = _rsi(df["close"])
    df["momentum"]      = df["close"] - df["close"].shift(MOMENTUM_PERIOD)
    df["atr"]           = _atr(df["high"], df["low"], df["close"])
    df["atr_ma"]        = df["atr"].rolling(ATR_MA_PERIOD).mean()
    df["volume_ma"]     = df["volume"].rolling(VOLUME_MA_PERIOD).mean()
    df["adx"]           = _adx(df["high"], df["low"], df["close"])
    df["sma_20"]        = df["close"].rolling(SMA_20_PERIOD).mean()
    df["sma_50"]        = df["close"].rolling(SMA_50_PERIOD).mean()
    df["sma_baseline"]  = df["close"].ewm(span=SMA_50_PERIOD, adjust=False).mean()  # EMA-50 for chart overlay
    df["sma_100"]       = df["close"].rolling(SMA_100_PERIOD).mean()
    _, _, df["macd_hist"] = _macd(df["close"])
    df["stoch_k"], df["stoch_d"] = _stochastic(df["high"], df["low"], df["close"])

    return df


def evaluate_confirmations(row: pd.Series) -> dict:
    """
    Evaluate all 10 technical confirmation conditions for a single bar.

    Confirmation set:
      1. Positive Momentum
      2. Volatility Expansion (ATR > ATR MA)
      3. Volume Above Average
      4. ADX Trending (>25)
      5. Price > SMA50
      6. MACD Bullish
      7. Stoch %K > %D
      8. SMA20 > SMA50         — fast MA above mid MA
      9. SMA50 > SMA100        — mid MA above slow MA (long-term uptrend)
     10. RSI > 50              — momentum bias is bullish

    Parameters
    ----------
    row : A pandas Series representing one row of the indicator DataFrame.

    Returns
    -------
    dict  {condition_name: bool}  (True = condition is met / bullish)
    """
    checks = {}

    # 1. Momentum – positive over look-back period
    checks["Positive Momentum"] = bool(row["momentum"] > 0) if pd.notna(row["momentum"]) else False

    # 2. Volatility Expansion – ATR currently above its own moving average
    checks["Volatility Expansion"] = (
        bool(row["atr"] > row["atr_ma"])
        if pd.notna(row["atr"]) and pd.notna(row["atr_ma"]) else False
    )

    # 3. Volume Above Average – current bar volume exceeds 20-period mean
    checks["Volume Above Avg"]  = (
        bool(row["volume"] > row["volume_ma"])
        if pd.notna(row["volume"]) and pd.notna(row["volume_ma"]) else False
    )

    # 4. ADX Trending – trend strength above threshold
    checks["ADX Trending (>25)"] = bool(row["adx"] > 25) if pd.notna(row["adx"]) else False

    # 5. Price above 50-period SMA (mid-term price-action filter)
    checks["Price > SMA50"]     = (
        bool(row["close"] > row["sma_50"])
        if pd.notna(row["sma_50"]) else False
    )

    # 6. MACD Bullish – MACD histogram positive (line above signal)
    checks["MACD Bullish"]      = bool(row["macd_hist"] > 0) if pd.notna(row["macd_hist"]) else False

    # 7. Stochastic %K crossing above %D (secondary custom trend indicator)
    checks["Stoch %K > %D"]     = (
        bool(row["stoch_k"] > row["stoch_d"])
        if pd.notna(row["stoch_k"]) and pd.notna(row["stoch_d"]) else False
    )

    # 8. SMA20 above SMA50 – fast MA leading mid MA (short-term bullish alignment)
    checks["SMA20 > SMA50"]     = (
        bool(row["sma_20"] > row["sma_50"])
        if pd.notna(row["sma_20"]) and pd.notna(row["sma_50"]) else False
    )

    # 9. SMA50 above SMA100 – mid MA above slow MA (confirmed long-term uptrend)
    checks["SMA50 > SMA100"]    = (
        bool(row["sma_50"] > row["sma_100"])
        if pd.notna(row["sma_50"]) and pd.notna(row["sma_100"]) else False
    )

    # 10. RSI above 50 – momentum bias is bullish (more gains than losses over period)
    checks["RSI > 50"]          = bool(row["rsi"] > 50) if pd.notna(row["rsi"]) else False

    return checks


# ===========================================================================
#  SECTION 2 – HMM REGIME ENGINE
# ===========================================================================

class RegimeEngine:
    """
    Fits a Gaussian HMM on the three engineered features and provides
    methods to decode regimes, auto-label states, and predict on new data.
    """

    def __init__(self, n_states: int = N_STATES):
        self.n_states   = n_states
        self.scaler     = StandardScaler()
        # GMMHMM: each of the n_states hidden states has N_MIX Gaussian
        # components, giving each state a richer multi-modal distribution.
        # This captures within-state variability better than a single Gaussian.
        self.model      = GMMHMM(
            n_components    = n_states,
            n_mix           = N_MIX,
            covariance_type = HMM_COVARIANCE,
            n_iter          = HMM_ITERATIONS,
            random_state    = HMM_RANDOM_STATE,
            tol             = 1e-4,
        )
        self.state_labels: dict[int, str] = {}   # state_id → human label
        self.bull_states: set[int]         = set()   # top-N states → Bull Run
        self.bear_states: set[int]         = set()   # bottom-N states → Bear/Crash
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "RegimeEngine":
        """
        Fit the HMM on *df* and autolabel states.

        Parameters
        ----------
        df : DataFrame that must contain FEATURE_COLS columns with no NaNs.
        """
        X = df[FEATURE_COLS].values.astype(float)

        # Guard: double-check no NaNs survive (belt-and-suspenders)
        if np.isnan(X).any():
            raise ValueError(
                "[RegimeEngine] Feature matrix contains NaN values. "
                "Call data_loader.load() first to clean the data."
            )

        print(f"[RegimeEngine] Scaling features and fitting "
              f"{self.n_states}-state GMM-HMM (n_mix={N_MIX}, cov={HMM_COVARIANCE}) "
              f"on {len(X):,} observations…")

        X_scaled = self.scaler.fit_transform(X)

        # ── Progressive fallback for non-positive-definite covariance errors ──
        # Low-volatility assets (QQQ, SPY, ETH) occasionally cause the EM
        # algorithm to find singular covariance matrices.  We retry with
        # progressively simpler models before giving up.
        fallback_configs = [
            dict(n_mix=N_MIX, covariance_type=HMM_COVARIANCE),   # preferred
            dict(n_mix=N_MIX, covariance_type="diag"),             # diagonal (fewer params)
            dict(n_mix=1,      covariance_type="diag"),             # single Gaussian per state
            dict(n_mix=1,      covariance_type="spherical"),        # simplest possible
        ]

        last_exc = None
        for cfg in fallback_configs:
            try:
                self.model = GMMHMM(
                    n_components    = self.n_states,
                    n_mix           = cfg["n_mix"],
                    covariance_type = cfg["covariance_type"],
                    n_iter          = HMM_ITERATIONS,
                    random_state    = HMM_RANDOM_STATE,
                    tol             = 1e-4,
                )
                self.model.fit(X_scaled)
                # Validate the fit — this raises ValueError for singular matrices
                hidden_states = self.model.predict(X_scaled)
                print(f"[RegimeEngine] Fit succeeded with "
                      f"n_mix={cfg['n_mix']}, cov={cfg['covariance_type']}")
                break
            except (ValueError, np.linalg.LinAlgError) as exc:
                last_exc = exc
                print(f"[RegimeEngine] Fit failed ({cfg}) — {exc}. Trying simpler config…")
        else:
            raise RuntimeError(
                f"[RegimeEngine] All model configurations failed. "
                f"Last error: {last_exc}"
            )

        self._fitted = True

        # Build a temporary Series for quick mean-returns per state
        returns_series = pd.Series(df["returns"].values, index=range(len(df)))
        state_series   = pd.Series(hidden_states)

        mean_returns = {
            s: returns_series[state_series == s].mean()
            for s in range(self.n_states)
        }
        print("[RegimeEngine] Mean returns per state:")
        for s, r in sorted(mean_returns.items(), key=lambda x: x[1], reverse=True):
            print(f"  State {s}: {r:.6f}")

        # Option B: top-N_BULL_STATES by mean return → Bull Run,
        #           bottom-N_BEAR_STATES → Bear/Crash, rest → Neutral.
        sorted_states    = sorted(mean_returns.items(), key=lambda x: x[1])
        self.bear_states = {s for s, _ in sorted_states[:N_BEAR_STATES]}
        self.bull_states = {s for s, _ in sorted_states[-N_BULL_STATES:]}

        for s in range(self.n_states):
            if s in self.bull_states:
                self.state_labels[s] = LABEL_BULL
            elif s in self.bear_states:
                self.state_labels[s] = LABEL_BEAR
            else:
                self.state_labels[s] = LABEL_NEUTRAL

        print(f"[RegimeEngine] Auto-labels (Option B): "
              f"Bull=States {self.bull_states}, Bear=States {self.bear_states}")
        return self

    # ------------------------------------------------------------------
    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict hidden states and state probabilities for every bar in *df*.

        Adds columns: 'state', 'regime_label', 'prob_bull', 'prob_bear',
                      and 'prob_state_{i}' for each state i.
        """
        if not self._fitted:
            raise RuntimeError("[RegimeEngine] Model has not been fitted yet.")

        df = df.copy()
        X  = df[FEATURE_COLS].values.astype(float)

        # Replace any stray NaNs with column means to avoid crashes
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

        X_scaled = self.scaler.transform(X)

        states = self.model.predict(X_scaled)
        probs  = self.model.predict_proba(X_scaled)   # shape (N, n_states)

        df["state"]        = states
        df["regime_label"] = df["state"].map(self.state_labels)

        for i in range(self.n_states):
            df[f"prob_state_{i}"] = probs[:, i]

        # Sum probabilities across all bull states and all bear states
        df["prob_bull"] = (
            sum(probs[:, s] for s in self.bull_states)
            if self.bull_states else 0.0
        )
        df["prob_bear"] = (
            sum(probs[:, s] for s in self.bear_states)
            if self.bear_states else 0.0
        )

        return df

    # ------------------------------------------------------------------
    def get_state_probs_latest(self, df: pd.DataFrame) -> dict:
        """Return a dict of {label: probability} for the most recent bar."""
        if not self._fitted:
            return {}
        decoded = self.decode(df.tail(max(100, self.n_states * 10)))
        last    = decoded.iloc[-1]
        return {
            self.state_labels[i]: float(last[f"prob_state_{i}"])
            for i in range(self.n_states)
        }


# ===========================================================================
#  SECTION 3 – STRATEGY & BACKTESTER
# ===========================================================================

# Canonical ordered list of all confirmation names — used by dashboard checkboxes.
ALL_CONFIRMATIONS = [
    "Positive Momentum",
    "Volatility Expansion",
    "Volume Above Avg",
    "ADX Trending (>25)",
    "Price > SMA50",
    "MACD Bullish",
    "Stoch %K > %D",
    "SMA20 > SMA50",
    "SMA50 > SMA100",
    "RSI > 50",
]


class Backtester:
    """
    Simulates the regime-based trading strategy on historical data and
    computes performance metrics.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value in USD.
    leverage_override : float or None
        Explicit leverage multiplier (e.g. 1.0, 2.0, 4.0).
        None → use NORMAL_LEVERAGE default (2.5×).
    enabled_confirmations : list[str] or None
        Subset of ALL_CONFIRMATIONS to include when counting.
        None (default) → all 9 confirmations are active.
    min_confirms : int
        Number of enabled confirmations that must be met before entering.
        Defaults to NORMAL_MIN_CONFIRMS (7).
    use_trail_stop : bool
        Enable a 2% trailing stop-loss on open positions.
    """

    def __init__(
        self,
        initial_capital      : float      = INITIAL_CAPITAL,
        leverage_override                 = None,
        enabled_confirmations: list | None = None,
        min_confirms         : int         = NORMAL_MIN_CONFIRMS,
        use_trail_stop       : bool        = False,
        bear_confirm_days    : int         = BEAR_CONFIRM_DAYS,
        min_hold_days        : int         = MIN_HOLD_DAYS,
        regime_only          : bool        = False,
    ):
        self.initial_capital       = initial_capital
        self.enabled_confirmations = enabled_confirmations
        self.min_confirms          = min_confirms
        self.use_trail_stop        = use_trail_stop
        self.trail_pct             = AGGRESSIVE_TRAIL_PCT
        self.bear_confirm_days     = bear_confirm_days  # consecutive Bear bars needed to exit
        self.min_hold_days         = min_hold_days      # min days to hold before regime exit
        self.regime_only           = regime_only        # ignore confirmations; trade on regime changes only

        # Leverage — explicit override wins over default
        self.leverage = float(leverage_override) if leverage_override is not None \
                        else NORMAL_LEVERAGE

        self.engine   = RegimeEngine()

        # Results populated after run()
        self.df           : pd.DataFrame | None = None
        self.trade_log    : list[dict]          = []
        self.equity_curve : pd.Series | None    = None
        self.metrics      : dict                = {}

    # ------------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame):
        """
        Steps 1-4: fit HMM, decode regimes, compute technical indicators,
        evaluate confirmations, build confirms_count column.

        Returns
        -------
        (prepared_df, tradeable_mask)
            prepared_df   – fully-annotated DataFrame ready for simulation.
            tradeable_mask – boolean Series; True where all indicators are
                             available (warm-up bars excluded).

        Side-effect: sets self._n_active_confirmations for optimize_params.
        """
        # ---- Step 1: Fit HMM & decode regimes ----
        self.engine.fit(df)
        df = self.engine.decode(df)

        # ---- Step 2: Compute all technical indicators ----
        df = compute_indicators(df)

        # ---- Step 2b: Drop warmup rows fetched only for indicator priming ----
        # data_loader downloads INDICATOR_WARMUP extra days so that long-window
        # indicators (SMA-100, MACD) are fully populated for every bar the user
        # actually requested.  Trim those warmup rows now so the chart and
        # backtest cover exactly the requested period, with all SMAs valid
        # from bar 1.
        df = df[df["sma_100"].notna()].copy()

        # ---- Step 3: Build tradeable mask ----
        indicator_cols = [
            "rsi", "momentum", "atr", "atr_ma", "volume_ma",
            "adx", "sma_20", "sma_50", "sma_100", "macd_hist", "stoch_k", "stoch_d",
        ]
        tradeable_mask = df[indicator_cols].notna().all(axis=1)
        print(f"[Backtester] Tradeable bars (all indicators available): "
              f"{tradeable_mask.sum():,} / {len(df):,}")

        # ---- Step 4: Evaluate confirmations for every bar ----
        confirms_list = []
        for _, row in df.iterrows():
            c = evaluate_confirmations(row)
            confirms_list.append(c)

        confirms_df = pd.DataFrame(confirms_list, index=df.index)

        # ---- Step 4b: Filter to only enabled confirmations ----
        # All confirmation columns are always stored on df for the scorecard,
        # but only the enabled ones contribute to confirms_count.
        for col in confirms_df.columns:
            df[f"confirm_{col}"] = confirms_df[col]

        if self.enabled_confirmations is not None:
            active_names = set(self.enabled_confirmations)
            active_cols  = [c for c in confirms_df.columns if c in active_names]
        else:
            active_cols  = list(confirms_df.columns)   # all 9

        active_df = confirms_df[active_cols] if active_cols else confirms_df
        df["confirms_count"] = active_df.sum(axis=1)
        print(f"[Backtester] Active confirmations ({len(active_cols)}): {active_cols}")

        self._n_active_confirmations = len(active_cols)
        return df, tradeable_mask

    # ------------------------------------------------------------------
    def _run_simulation(self, df: pd.DataFrame, tradeable_mask) -> "Backtester":
        """
        Step 5-6: simulation loop + metrics.  Operates on a pre-prepared
        DataFrame (output of _prepare()).  Safe to call multiple times on
        the same df — used by optimize_params() to evaluate many param
        combinations without refitting the HMM.
        """
        # Reset trade state (important for repeated calls in optimization)
        self.trade_log = []

        equity        = self.initial_capital
        position_open = False
        entry_price   = 0.0
        entry_time    = None
        entry_confirms= 0
        entry_regime  = LABEL_BULL
        cooldown_end  = None
        peak_price    = 0.0
        prev_regime   = None
        bear_streak   = 0

        equity_values = []

        for ts, row in df.iterrows():
            # Mark-to-market equity
            if position_open:
                bar_return     = (row["close"] - entry_price) / entry_price
                unrealised     = equity * self.leverage * bar_return
                current_equity = equity + unrealised
            else:
                current_equity = equity

            equity_values.append(current_equity)

            if not tradeable_mask[ts]:
                continue

            regime   = row["regime_label"]
            confirms = int(row["confirms_count"])

            # -------------------------------------------------------
            # EXIT LOGIC
            # -------------------------------------------------------
            if position_open:
                if self.use_trail_stop and row["close"] > peak_price:
                    peak_price = row["close"]

                exit_reason = None

                if self.regime_only:
                    # Regime-Only mode: exit immediately on the first Bear bar
                    if regime == LABEL_BEAR:
                        exit_reason = "Bear Regime"
                else:
                    if regime == LABEL_BEAR:
                        bear_streak += 1
                    else:
                        bear_streak = 0

                    # Exit 1: consecutive Bear bars (no min_hold_days gate — Bear
                    # must always be able to close the position regardless of duration)
                    if regime == LABEL_BEAR and bear_streak >= self.bear_confirm_days:
                        exit_reason = f"Bear×{bear_streak}d"

                    # Exit 2: Trailing stop — gated by min_hold_days to prevent
                    # getting stopped out in the first few days of a new position.
                    hold_days = (ts - entry_time).days if entry_time is not None else 0
                    if (self.use_trail_stop and peak_price > 0
                            and hold_days >= self.min_hold_days):
                        stop_level = peak_price * (1 - self.trail_pct)
                        if row["close"] <= stop_level:
                            exit_reason = exit_reason or "TrailingStop"

                if exit_reason:
                    pnl_pct     = (row["close"] - entry_price) / entry_price * self.leverage
                    pnl_usd     = equity * pnl_pct
                    equity     += pnl_usd
                    exit_regime = regime
                    if exit_reason == "TrailingStop":
                        transition = f"{entry_regime} → {exit_regime} (Trailing Stop)"
                    else:
                        transition = f"{entry_regime} → {exit_regime}"

                    self.trade_log.append({
                        "entry_time"       : entry_time,
                        "exit_time"        : ts,
                        "days_held"        : (ts - entry_time).days,
                        "entry_price"      : round(entry_price, 2),
                        "exit_price"       : round(row["close"], 2),
                        "pnl_pct"          : round(pnl_pct * 100, 3),
                        "pnl_usd"          : round(pnl_usd, 2),
                        "confirms_at_entry": entry_confirms,
                        "entry_regime"     : entry_regime,
                        "exit_regime"      : exit_regime,
                        "regime_transition": transition,
                        "leverage"         : self.leverage,
                    })

                    position_open = False
                    peak_price    = 0.0
                    bear_streak   = 0
                    cooldown_end  = ts + pd.Timedelta(days=COOLDOWN_DAYS)

            # -------------------------------------------------------
            # ENTRY LOGIC
            # -------------------------------------------------------
            if not position_open:
                if cooldown_end is not None and ts < cooldown_end:
                    continue

                if self.regime_only:
                    # Regime-Only mode: enter on Bear→Bull or Bear→Neutral transition;
                    # confirmations are ignored entirely.
                    is_entry = (
                        prev_regime == LABEL_BEAR
                        and regime in (LABEL_BULL, LABEL_NEUTRAL)
                    )
                else:
                    is_bull_entry = (
                        regime == LABEL_BULL
                        and confirms >= self.min_confirms
                    )
                    is_bear_neutral_entry = (
                        regime == LABEL_NEUTRAL
                        and prev_regime == LABEL_BEAR
                        and confirms >= self.min_confirms
                    )
                    is_entry = is_bull_entry or is_bear_neutral_entry

                if is_entry:
                    position_open  = True
                    entry_price    = row["close"]
                    entry_time     = ts
                    entry_confirms = confirms
                    entry_regime   = regime
                    peak_price     = row["close"]
                    bear_streak    = 0

            if tradeable_mask[ts]:
                prev_regime = regime

        # Close any open position at last bar
        if position_open and len(df) > 0:
            last_row    = df.iloc[-1]
            pnl_pct     = (last_row["close"] - entry_price) / entry_price * self.leverage
            pnl_usd     = equity * pnl_pct
            equity     += pnl_usd
            last_regime = last_row["regime_label"]
            self.trade_log.append({
                "entry_time"       : entry_time,
                "exit_time"        : df.index[-1],
                "days_held"        : (df.index[-1] - entry_time).days,
                "entry_price"      : round(entry_price, 2),
                "exit_price"       : round(last_row["close"], 2),
                "pnl_pct"          : round(pnl_pct * 100, 3),
                "pnl_usd"          : round(pnl_usd, 2),
                "confirms_at_entry": entry_confirms,
                "entry_regime"     : entry_regime,
                "exit_regime"      : last_regime,
                "regime_transition": f"{entry_regime} → {last_regime} (End of Data)",
                "leverage"         : self.leverage,
            })
            equity_values[-1] = equity

        self.equity_curve         = pd.Series(equity_values, index=df.index, name="equity")
        self.position_open_at_end = position_open
        self.df                   = df
        self._compute_metrics()

        print(f"[Backtester] Simulation complete. {len(self.trade_log)} trades executed.")
        return self

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> "Backtester":
        """
        Execute the full pipeline:
          data → fit HMM → decode → compute indicators →
          simulate strategy → calculate metrics.

        Parameters
        ----------
        df : Clean DataFrame from data_loader.load() (no NaN in features).
        """
        prepared_df, tradeable_mask = self._prepare(df)
        self._run_simulation(prepared_df, tradeable_mask)
        return self

    # ------------------------------------------------------------------
    def _compute_metrics(self):
        """Compute and cache performance metrics."""
        eq    = self.equity_curve
        final = eq.iloc[-1]
        start = self.initial_capital

        # Total return
        total_return_pct = (final / start - 1) * 100

        # Buy & Hold return (unleveraged, no fees)
        bh_start = self.df["close"].iloc[0]
        bh_end   = self.df["close"].iloc[-1]
        bh_return_pct = (bh_end / bh_start - 1) * 100

        # Alpha vs Buy & Hold
        alpha_pct = total_return_pct - bh_return_pct

        # Max Drawdown
        roll_max  = eq.cummax()
        drawdown  = (eq - roll_max) / roll_max * 100
        max_dd    = drawdown.min()

        # Win Rate
        wins = sum(1 for t in self.trade_log if t["pnl_usd"] > 0)
        n    = len(self.trade_log)
        win_rate = (wins / n * 100) if n > 0 else 0.0

        # Avg trade PnL
        avg_pnl = np.mean([t["pnl_usd"] for t in self.trade_log]) if n > 0 else 0.0

        # Sharpe Ratio (annualised, assuming 252 trading days/year)
        daily_returns = eq.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Buy & Hold Sharpe Ratio
        bh_daily_returns = self.df["close"].pct_change().dropna()
        if bh_daily_returns.std() > 0:
            bh_sharpe = (bh_daily_returns.mean() / bh_daily_returns.std()) * np.sqrt(252)
        else:
            bh_sharpe = 0.0

        self.metrics = {
            "total_return_pct" : round(total_return_pct, 2),
            "bh_return_pct"    : round(bh_return_pct, 2),
            "alpha_pct"        : round(alpha_pct, 2),
            "max_drawdown_pct" : round(max_dd, 2),
            "win_rate_pct"     : round(win_rate, 2),
            "n_trades"         : n,
            "avg_trade_pnl_usd": round(avg_pnl, 2),
            "sharpe_ratio"     : round(sharpe, 3),
            "bh_sharpe_ratio"  : round(bh_sharpe, 3),
            "final_equity"     : round(final, 2),
        }

    # ------------------------------------------------------------------
    def get_current_signal(self) -> dict:
        """
        Return the current (most recent bar) signal summary.

        Returns
        -------
        dict with keys: regime, confirms_count, confirms_detail,
                        action, state_probs, leverage, min_confirms
        """
        if self.df is None or self.df.empty:
            return {}

        last = self.df.iloc[-1]
        regime   = last["regime_label"]
        confirms = int(last["confirms_count"])

        # Determine recommended action — transition-based strategy.
        # Entry: only on Bull transition. Exit: only on Bear transition.
        # Neutral: always HOLD / FLAT — no action either way.
        in_position = getattr(self, "position_open_at_end", False)

        # Detect if the most recent bar is a regime transition
        if len(self.df) >= 2:
            prev_reg = self.df.iloc[-2]["regime_label"]
        else:
            prev_reg = None

        is_bull_transition        = (regime == LABEL_BULL and prev_reg is not None
                                     and prev_reg != LABEL_BULL)
        is_bear_transition        = (regime == LABEL_BEAR and prev_reg is not None
                                     and prev_reg != LABEL_BEAR)
        is_bear_neutral_transition = (regime == LABEL_NEUTRAL and prev_reg == LABEL_BEAR)

        if self.regime_only:
            # ── Regime-Only action messages ──────────────────────────────────
            if regime == LABEL_BULL:
                if is_bull_transition and prev_reg == LABEL_BEAR:
                    if in_position:
                        action = "🟢 HOLD LONG – Bear→Bull transition (already in trade)"
                    else:
                        action = "🟢 ENTER LONG – Bear→Bull regime change"
                elif in_position:
                    action = "🟢 HOLD LONG – Bull Run ongoing"
                else:
                    action = "⚪ HOLD FLAT – Bull ongoing, no Bear→Bull transition"
            elif regime == LABEL_BEAR:
                if in_position:
                    action = "🔴 EXIT NOW – Regime changed to Bear/Crash"
                else:
                    action = "🔴 CASH – Bear/Crash regime, do not enter"
            else:
                # Neutral
                if is_bear_neutral_transition:
                    if in_position:
                        action = "🟢 HOLD LONG – Bear→Neutral transition (already in trade)"
                    else:
                        action = "🟢 ENTER LONG – Bear→Neutral regime change"
                elif in_position:
                    action = "🟡 HOLD LONG – Neutral regime (hold until Bear)"
                else:
                    action = "⚪ HOLD FLAT – Neutral, waiting for Bear→Bull or Bear→Neutral"
        else:
            # ── Standard confirmation-gated action messages ──────────────────
            if regime == LABEL_BULL:
                if is_bull_transition:
                    if in_position:
                        action = "🟢 HOLD LONG – Bull transition (already in trade)"
                    elif confirms >= self.min_confirms:
                        action = (f"🟢 ENTER LONG – Bull Run transition "
                                  f"({confirms}/{self.min_confirms} confirmations met)")
                    else:
                        action = (f"🟡 WATCH – Bull transition but only "
                                  f"{confirms}/{self.min_confirms} confirmations met")
                else:
                    # Bull ongoing
                    if in_position:
                        action = "🟢 HOLD LONG – Bull Run ongoing"
                    elif confirms >= self.min_confirms:
                        action = (f"🟢 ENTER LONG – Bull Run ongoing "
                                  f"({confirms}/{self.min_confirms} confirmations met)")
                    else:
                        action = (f"🟡 WATCH – Bull ongoing but only "
                                  f"{confirms}/{self.min_confirms} confirmations met")
            elif regime == LABEL_BEAR:
                if is_bear_transition:
                    if in_position:
                        action = "🔴 EXIT NOW – Bear/Crash transition detected"
                    else:
                        action = "🔴 CASH – Bear/Crash transition, do not enter"
                else:
                    # Already in Bear (ongoing)
                    if in_position:
                        # Shouldn't normally happen (exit fires on first Bear bar)
                        action = "🔴 EXIT NOW – Bear/Crash regime ongoing"
                    else:
                        action = "🔴 CASH – Bear/Crash regime, do not enter"
            else:
                # Neutral/Transition
                if is_bear_neutral_transition:
                    # Bear → Neutral: potential early recovery entry
                    if in_position:
                        action = "🟢 HOLD LONG – Bear→Neutral transition (already in trade)"
                    elif confirms >= self.min_confirms:
                        action = (f"🟢 ENTER LONG – Bear→Neutral transition "
                                  f"({confirms}/{self.min_confirms} confirmations met)")
                    else:
                        action = (f"🟡 WATCH – Bear→Neutral but only "
                                  f"{confirms}/{self.min_confirms} confirmations met")
                elif in_position:
                    action = "🟡 HOLD LONG – Neutral regime (no change)"
                else:
                    action = "⚪ HOLD FLAT – Neutral, waiting for Bull or Bear→Neutral transition"

        # State probabilities
        state_probs = {
            last[f"prob_state_{i}"] for i in range(self.engine.n_states)
        }
        state_probs_dict = {
            self.engine.state_labels[i]: float(last[f"prob_state_{i}"])
            for i in range(self.engine.n_states)
        }

        # Confirmation details
        confirm_cols   = [c for c in self.df.columns if c.startswith("confirm_")]
        confirm_detail = {
            col.replace("confirm_", ""): bool(last[col])
            for col in confirm_cols
        }

        return {
            "regime"          : regime,
            "confirms_count"  : confirms,
            "confirms_detail" : confirm_detail,
            "action"          : action,
            "state_probs"     : state_probs_dict,
            "leverage"        : self.leverage,
            "min_confirms"    : self.min_confirms,
            "close"           : float(last["close"]),
            "position_open"   : getattr(self, "position_open_at_end", False),
        }

    # ------------------------------------------------------------------
    def get_trade_log_df(self) -> pd.DataFrame:
        """Return the trade log as a formatted DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.trade_log)
        df = df.rename(columns={
            "entry_time"       : "Entry Time",
            "exit_time"        : "Exit Time",
            "days_held"        : "Days Held",
            "entry_price"      : "Entry Price ($)",
            "exit_price"       : "Exit Price ($)",
            "pnl_pct"          : "PnL (%)",
            "pnl_usd"          : "PnL ($)",
            "confirms_at_entry": "Confirmations",
            "regime_transition": "Regime Transition",
            "leverage"         : "Leverage",
        })
        # Drop raw regime columns from display (kept internal, now surfaced as Regime Transition)
        df = df.drop(columns=["entry_regime", "exit_regime"], errors="ignore")
        return df


# ===========================================================================
#  SECTION 4 – CONVENIENCE FUNCTION
# ===========================================================================

def build_and_run(
    df                   : pd.DataFrame,
    initial_capital      : float      = INITIAL_CAPITAL,
    leverage_override                 = None,
    enabled_confirmations: list | None = None,
    min_confirms         : int         = NORMAL_MIN_CONFIRMS,
    use_trail_stop       : bool        = False,
    bear_confirm_days    : int         = BEAR_CONFIRM_DAYS,
    min_hold_days        : int         = MIN_HOLD_DAYS,
    regime_only          : bool        = False,
) -> Backtester:
    """
    One-call convenience function used by dashboard.py.

    Parameters
    ----------
    df                    : Clean DataFrame from data_loader.load().
    initial_capital       : Starting portfolio value in USD.
    leverage_override     : Explicit leverage (1.0, 2.0, 4.0…). None = 1×.
    enabled_confirmations : Which of the 9 confirmations are active.
                            None = all 9 active.
    min_confirms          : How many enabled confirmations must be True to enter.
    use_trail_stop        : Enable 2% trailing stop-loss.
    bear_confirm_days     : Consecutive Bear bars needed before exit fires.
    min_hold_days         : Minimum calendar days to hold before regime exit.

    Returns
    -------
    A fully run Backtester instance (model fitted, simulation complete).
    """
    bt = Backtester(
        initial_capital       = initial_capital,
        leverage_override     = leverage_override,
        enabled_confirmations = enabled_confirmations,
        min_confirms          = min_confirms,
        use_trail_stop        = use_trail_stop,
        bear_confirm_days     = bear_confirm_days,
        min_hold_days         = min_hold_days,
        regime_only           = regime_only,
    )
    bt.run(df)
    return bt


# ===========================================================================
#  SECTION 5 – PARAMETER OPTIMISER
# ===========================================================================

def _recount_confirmations(prepared_df: pd.DataFrame, filter_subset: list) -> pd.DataFrame:
    """
    Return a copy of *prepared_df* with the ``confirms_count`` column
    recomputed for *filter_subset* (a list of confirmation names drawn from
    ALL_CONFIRMATIONS).  All individual ``confirm_*`` columns remain intact.
    """
    df = prepared_df.copy()
    active_cols = [f"confirm_{c}" for c in filter_subset
                   if f"confirm_{c}" in df.columns]
    df["confirms_count"] = df[active_cols].sum(axis=1) if active_cols else 0
    return df


def _select_best_confirmations(
    prepared_df   : pd.DataFrame,
    tradeable_mask: "pd.Series",
    base_bt       : "Backtester",
    pool          : list,
    initial_capital: float,
    leverage_override,
    use_trail_stop: bool,
) -> list:
    """
    Greedy forward selection of confirmation filters.

    Iteratively adds the filter from *pool* that most improves total return,
    evaluated over a reduced bear_days grid (no HMM refit — fast).  Tracks
    the point in the sequence where return peaked and returns that subset.

    Complexity: sum_{k=1}^{N} (N+1-k) × |bear_days_grid| × k simulations
    For N=10 and 5 bear-day values: ~1 100 simulations total.
    """
    bear_days_grid = [1, 2, 3, 5, 7]
    selected  : list = []
    remaining : list = list(pool)
    best_overall_return : float = -np.inf
    best_overall_set    : list  = list(pool)   # default: full pool

    print(f"[optimize_params] Phase 1 — filter selection over "
          f"{len(pool)} confirmations (greedy forward)…")

    for _ in range(len(remaining)):
        best_addition     = None
        best_return_round = -np.inf

        for candidate in remaining:
            trial    = selected + [candidate]
            trial_df = _recount_confirmations(prepared_df, trial)
            n        = len(trial)
            trial_best = -np.inf

            for bd in bear_days_grid:
                for mc in range(1, n + 1):
                    bt = Backtester(
                        initial_capital       = initial_capital,
                        leverage_override     = leverage_override,
                        enabled_confirmations = trial,
                        min_confirms          = mc,
                        bear_confirm_days     = bd,
                        min_hold_days         = 0,
                        use_trail_stop        = use_trail_stop,
                    )
                    bt.engine = base_bt.engine
                    bt._run_simulation(trial_df, tradeable_mask)
                    ret = bt.metrics["total_return_pct"]
                    if ret > trial_best:
                        trial_best = ret

            if trial_best > best_return_round:
                best_return_round = trial_best
                best_addition     = candidate

        if best_addition is None:
            break

        selected.append(best_addition)
        remaining.remove(best_addition)

        if best_return_round > best_overall_return:
            best_overall_return = best_return_round
            best_overall_set    = list(selected)

    print(f"[optimize_params] Best filter set "
          f"({len(best_overall_set)}/{len(pool)}): {best_overall_set}")
    return best_overall_set


def optimize_params(
    raw_df                : pd.DataFrame,
    initial_capital       : float      = INITIAL_CAPITAL,
    leverage_override                  = None,
    enabled_confirmations : list | None = None,
    use_trail_stop        : bool        = False,
) -> dict:
    """
    Grid-search key strategy parameters on historical data to maximise
    Total Return.  The HMM is fitted **once**; the simulation loop is
    re-run for each combination (fast — no HMM refit per iteration).

    Optimisation runs in two phases:

    Phase 1 — Confirmation filter selection (greedy forward, ~1 100 sims)
        Iteratively selects the subset of *enabled_confirmations* (or all 10
        if None) that maximises total return.  Uses a reduced bear_days grid
        to keep runtime short; no HMM refit per iteration.

    Phase 2 — Parameter grid search (~n_selected × 7 × 6 sims)
        Sweeps bear_confirm_days (1–7), min_confirms (1–n_selected), and
        min_hold_days (0, 3, 5, 7, 10, 14) using the filter set from Phase 1.

    Parameters
    ----------
    raw_df                : Clean DataFrame from data_loader.load().
    initial_capital       : Starting portfolio value.
    leverage_override     : Leverage multiplier (1×, 2×, 4×).
    enabled_confirmations : Pool of confirmations to search over.
                            None = all 10 confirmations.
    use_trail_stop        : Whether trailing stop is active.

    Returns
    -------
    dict
        best_confirmations – list[str]  (best filter subset from Phase 1)
        best_params        – {bear_confirm_days, min_confirms, min_hold_days}
        best_return        – float  (total_return_pct for the best combo)
        top_results        – pd.DataFrame (top 10 by return)
        all_results        – pd.DataFrame (full grid, sorted by return desc)
    """
    # ── Phase 0: Fit HMM and prepare DataFrame once ─────────────────
    base = Backtester(
        initial_capital       = initial_capital,
        leverage_override     = leverage_override,
        enabled_confirmations = enabled_confirmations,
        use_trail_stop        = use_trail_stop,
    )
    prepared_df, tradeable_mask = base._prepare(raw_df)

    # ── Phase 1: Greedy forward filter selection ─────────────────────
    pool             = list(enabled_confirmations) if enabled_confirmations else list(ALL_CONFIRMATIONS)
    selected_confirms = _select_best_confirmations(
        prepared_df, tradeable_mask, base,
        pool, initial_capital, leverage_override, use_trail_stop,
    )

    # Recompute confirms_count on prepared_df for the winning filter set
    _active_cols = [f"confirm_{c}" for c in selected_confirms
                    if f"confirm_{c}" in prepared_df.columns]
    prepared_df["confirms_count"] = (
        prepared_df[_active_cols].sum(axis=1) if _active_cols else 0
    )
    n_active = len(selected_confirms)

    # ── Phase 2: Parameter grid search with selected filters ─────────
    bear_days_grid = list(range(1, 8))                   # 1–7
    min_conf_grid  = list(range(1, n_active + 1))        # 1–n_selected
    hold_days_grid = [0, 3, 5, 7, 10, 14]               # trailing-stop gate

    total = len(bear_days_grid) * len(min_conf_grid) * len(hold_days_grid)
    print(f"[optimize_params] Phase 2 — grid search {total} combinations "
          f"(bear_days×{len(bear_days_grid)}, "
          f"min_conf×{len(min_conf_grid)}, "
          f"hold_days×{len(hold_days_grid)})…")

    results = []
    for bear_days in bear_days_grid:
        for min_conf in min_conf_grid:
            for hold_days in hold_days_grid:
                bt = Backtester(
                    initial_capital       = initial_capital,
                    leverage_override     = leverage_override,
                    enabled_confirmations = selected_confirms,
                    min_confirms          = min_conf,
                    use_trail_stop        = use_trail_stop,
                    bear_confirm_days     = bear_days,
                    min_hold_days         = hold_days,
                )
                bt.engine = base.engine          # reuse fitted HMM
                bt._run_simulation(prepared_df, tradeable_mask)

                results.append({
                    "bear_confirm_days" : bear_days,
                    "min_confirms"      : min_conf,
                    "min_hold_days"     : hold_days,
                    "total_return_pct"  : bt.metrics["total_return_pct"],
                    "sharpe_ratio"      : bt.metrics["sharpe_ratio"],
                    "max_drawdown_pct"  : bt.metrics["max_drawdown_pct"],
                    "n_trades"          : bt.metrics["n_trades"],
                    "win_rate_pct"      : bt.metrics["win_rate_pct"],
                    "alpha_pct"         : bt.metrics["alpha_pct"],
                    "bh_return_pct"     : bt.metrics["bh_return_pct"],
                })

    print(f"[optimize_params] Phase 2 complete.")

    results_df = (
        pd.DataFrame(results)
        .sort_values("total_return_pct", ascending=False)
        .reset_index(drop=True)
    )

    # B&H return is asset-only (same for every row — doesn't depend on params)
    bnh_return = float(results_df["bh_return_pct"].iloc[0])

    # Flag each combination that beat buy-and-hold
    results_df["beats_bnh"] = results_df["total_return_pct"] > bnh_return
    n_beats = int(results_df["beats_bnh"].sum())

    # Best overall row (highest total return, regardless of B&H)
    best_row = results_df.iloc[0]

    # Best combo that actually beats B&H (may be different from best_row)
    bnh_beaters = results_df[results_df["beats_bnh"]]
    best_bnh_row = bnh_beaters.iloc[0] if not bnh_beaters.empty else None

    # Top-10 table: prefer B&H-beating combos; pad with non-beaters if needed
    if len(bnh_beaters) >= 10:
        top_df = bnh_beaters.head(10)
    elif not bnh_beaters.empty:
        non_beaters = results_df[~results_df["beats_bnh"]]
        top_df = pd.concat([bnh_beaters, non_beaters]).head(10)
    else:
        top_df = results_df.head(10)

    return {
        "best_confirmations" : selected_confirms,
        "best_params": {
            "bear_confirm_days" : int(best_row["bear_confirm_days"]),
            "min_confirms"      : int(best_row["min_confirms"]),
            "min_hold_days"     : int(best_row["min_hold_days"]),
        },
        "best_return"    : float(best_row["total_return_pct"]),
        "bnh_return"     : bnh_return,
        "n_beats_bnh"    : n_beats,
        "n_total"        : len(results_df),
        "best_bnh_params": {
            "bear_confirm_days" : int(best_bnh_row["bear_confirm_days"]),
            "min_confirms"      : int(best_bnh_row["min_confirms"]),
            "min_hold_days"     : int(best_bnh_row["min_hold_days"]),
            "total_return_pct"  : float(best_bnh_row["total_return_pct"]),
        } if best_bnh_row is not None else None,
        "top_results"    : top_df,
        "all_results"    : results_df,
    }


# ===========================================================================
#  Quick test when run directly
# ===========================================================================
if __name__ == "__main__":
    from data_loader import load

    print("=" * 60)
    print("  HMM Regime Backtester – Standalone Test")
    print("=" * 60)

    data = load()
    if data.empty:
        print("ERROR: Could not load data. Aborting.")
        raise SystemExit(1)

    bt = build_and_run(data)

    print("\n--- Performance Metrics ---")
    for k, v in bt.metrics.items():
        print(f"  {k:<25}: {v}")

    print("\n--- Current Signal ---")
    sig = bt.get_current_signal()
    for k, v in sig.items():
        if k not in ("confirms_detail", "state_probs"):
            print(f"  {k:<20}: {v}")

    print("\n--- Trade Log (last 5) ---")
    tl = bt.get_trade_log_df()
    if not tl.empty:
        print(tl.tail(5).to_string(index=False))
    else:
        print("  No trades executed.")
