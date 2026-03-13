"""
data/indicators.py
──────────────────
Pure-pandas/numpy technical indicator library.
No external ta library dependency — fully self-contained.

All functions accept a pd.DataFrame with columns:
    open, high, low, close, volume  (lowercase)

All functions return a pd.DataFrame with new columns added (non-destructive).
NaN rows at the start are expected and handled by callers via dropna().

Public API:
    add_all_indicators(df)   -> df with 40+ indicator columns
    get_feature_columns()    -> list[str]  (columns safe to use as ML features)
    ohlcv_to_dataframe(raw)  -> pd.DataFrame from raw ccxt list-of-dicts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# OHLCV helpers
# ──────────────────────────────────────────────────────────────────────────────

def ohlcv_to_dataframe(raw: list) -> pd.DataFrame:
    """
    Convert raw OHLCV list-of-dicts (from ccxt) to a clean DataFrame.

    Accepts both:
      • list of dicts:  [{"timestamp":…, "open":…, …}, …]
      • list of lists:  [[timestamp, open, high, low, close, volume], …]
    """
    if not raw:
        return pd.DataFrame()

    if isinstance(raw[0], dict):
        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
    else:
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close"])


# ──────────────────────────────────────────────────────────────────────────────
# Individual indicator functions
# ──────────────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    """
    Relative Strength Index using Wilder's smoothing (EMA-based).
    Range: 0–100. Values < 30 oversold, > 70 overbought.
    """
    delta = df[col].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    ema_fast   = df[col].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df[col].ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df["macd"]        = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"]   = macd_line - signal_line
    return df


def add_ema(df: pd.DataFrame, periods: List[int] = None, col: str = "close") -> pd.DataFrame:
    """
    Exponential Moving Averages for multiple periods.
    Adds columns: ema_9, ema_21, ema_50, ema_200 (or custom periods).
    """
    if periods is None:
        periods = [9, 21, 50, 200]
    for p in periods:
        df[f"ema_{p}"] = df[col].ewm(span=p, adjust=False).mean()
    return df


def add_sma(df: pd.DataFrame, periods: List[int] = None, col: str = "close") -> pd.DataFrame:
    """Simple Moving Averages."""
    if periods is None:
        periods = [20, 50, 200]
    for p in periods:
        df[f"sma_{p}"] = df[col].rolling(p).mean()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range — measures volatility.
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> pd.DataFrame:
    """
    Bollinger Bands: mid ± (std_dev × rolling_std).
    Also computes %B (position within bands) and bandwidth.
    """
    mid   = df[col].rolling(period).mean()
    std   = df[col].rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std

    df["bb_mid"]       = mid
    df["bb_upper"]     = upper
    df["bb_lower"]     = lower
    df["bb_width"]     = (upper - lower) / (mid + 1e-10)
    df["bb_pct"]       = (df[col] - lower) / (upper - lower + 1e-10)  # 0=lower, 1=upper
    return df


def add_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """
    Stochastic Oscillator.
    %K = (close - lowest_low) / (highest_high - lowest_low) × 100
    %D = SMA(%K, d_period)
    """
    lowest_low   = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()
    range_       = highest_high - lowest_low + 1e-10

    df["stoch_k"] = (df["close"] - lowest_low) / range_ * 100.0
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index + DM+ / DM-.
    ADX > 25 = trending, < 20 = ranging.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = (high  - prev_high).clip(lower=0)
    dm_minus = (prev_low - low).clip(lower=0)

    # Zero out when the other is larger
    mask = dm_plus < dm_minus
    dm_plus[mask] = 0.0
    mask = dm_minus < dm_plus
    dm_minus[mask] = 0.0

    atr_s      = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    dmp_smooth = dm_plus.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    dmn_smooth = dm_minus.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    dip = 100.0 * dmp_smooth / (atr_s + 1e-10)
    din = 100.0 * dmn_smooth / (atr_s + 1e-10)
    dx  = 100.0 * (dip - din).abs() / (dip + din + 1e-10)

    df["adx"]     = dx.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    df["adx_dmp"] = dip
    df["adx_dmn"] = din
    return df


def add_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Volume-based indicators:
      - volume_sma:   rolling mean volume
      - volume_ratio: current / mean  (> 1.5 = spike)
      - obv:          On-Balance Volume (trend confirmation)
      - vwap_ratio:   price / VWAP (intraday momentum proxy)
    """
    df["volume_sma"]   = df["volume"].rolling(period).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

    # OBV
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"]  = (direction * df["volume"]).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=period, adjust=False).mean()

    # VWAP proxy (rolling)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol    = (typical_price * df["volume"]).rolling(period).sum()
    cum_vol       = df["volume"].rolling(period).sum()
    vwap          = cum_tp_vol / (cum_vol + 1e-10)
    df["vwap_ratio"] = df["close"] / (vwap + 1e-10) - 1.0

    return df


def add_momentum(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    Rate-of-change (ROC) momentum over multiple lookbacks.
    ROC(n) = (close - close[n]) / close[n]
    """
    if periods is None:
        periods = [5, 10, 20]
    for p in periods:
        df[f"roc_{p}"] = df["close"].pct_change(p)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realized volatility (rolling std of log returns).
    Also adds a volatility regime flag (low/med/high).
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))

    df["vol_10"]    = log_ret.rolling(10).std() * np.sqrt(252 * 96)   # annualized (96 15m bars/day)
    df["vol_20"]    = log_ret.rolling(20).std() * np.sqrt(252 * 96)
    df["vol_ratio"] = df["vol_10"] / (df["vol_20"] + 1e-10)           # short/long vol ratio

    # Volatility percentile (0–1) over last 100 bars — regime detection
    df["vol_pct"]   = df["vol_20"].rolling(100).rank(pct=True)
    return df


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw candlestick structure features (no pattern labeling — ML infers patterns).
    """
    body   = df["close"] - df["open"]
    rng    = df["high"]  - df["low"] + 1e-10

    df["candle_body"]        = body / df["open"]                               # % body
    df["candle_body_ratio"]  = body.abs() / rng                                # body / range
    df["candle_upper_wick"]  = (df["high"] - df[["open","close"]].max(axis=1)) / rng
    df["candle_lower_wick"]  = (df[["open","close"]].min(axis=1) - df["low"])  / rng
    df["is_bullish"]         = (df["close"] > df["open"]).astype(int)
    df["is_doji"]            = (df["candle_body"].abs() < 0.001).astype(int)   # tiny body
    return df


def add_price_structure(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Price structure / support-resistance features.
      - distance from rolling high/low
      - price position within recent range
      - higher-high, lower-low detection
    """
    roll_high = df["high"].rolling(period).max()
    roll_low  = df["low"].rolling(period).min()
    roll_rng  = roll_high - roll_low + 1e-10

    df["pct_from_high"]   = (df["close"] - roll_high) / roll_high             # ≤ 0
    df["pct_from_low"]    = (df["close"] - roll_low)  / roll_low              # ≥ 0
    df["price_position"]  = (df["close"] - roll_low)  / roll_rng              # 0=bottom, 1=top

    # Higher highs / lower lows (3-candle lookahead-safe since we use shift)
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"]   = (df["low"]  < df["low"].shift(1)).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical time encoding (sin/cos) to capture market session patterns.
    Works on DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    hour_sin = np.sin(2 * np.pi * df.index.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * df.index.hour / 24.0)
    dow_sin  = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    dow_cos  = np.cos(2 * np.pi * df.index.dayofweek / 7.0)

    df["hour_sin"] = hour_sin
    df["hour_cos"] = hour_cos
    df["dow_sin"]  = dow_sin
    df["dow_cos"]  = dow_cos
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Master function — adds every indicator
# ──────────────────────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all technical indicators to the DataFrame in-place order.
    Input: OHLCV DataFrame (index = DatetimeIndex, columns lowercase)
    Output: same DataFrame with 40+ additional columns

    Order matters — some indicators depend on others (e.g. ATR needed for labels).
    """
    df = df.copy()

    # ── Trend ──────────────────────────────────────────────────────────────
    df = add_ema(df, periods=[9, 21, 50, 200])
    df = add_sma(df, periods=[20, 50, 200])
    df = add_macd(df, fast=12, slow=26, signal=9)

    # ── Momentum ───────────────────────────────────────────────────────────
    df = add_rsi(df, period=14)
    df = add_stochastic(df, k_period=14, d_period=3)
    df = add_momentum(df, periods=[5, 10, 20])

    # ── Volatility ─────────────────────────────────────────────────────────
    df = add_atr(df, period=14)
    df = add_bollinger_bands(df, period=20, std_dev=2.0)
    df = add_volatility_features(df)

    # ── Trend Strength ─────────────────────────────────────────────────────
    df = add_adx(df, period=14)

    # ── Volume ─────────────────────────────────────────────────────────────
    df = add_volume_indicators(df, period=20)

    # ── Price Structure ────────────────────────────────────────────────────
    df = add_candle_patterns(df)
    df = add_price_structure(df, period=20)

    # ── Time ───────────────────────────────────────────────────────────────
    df = add_time_features(df)

    # ── Clean up ───────────────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature column registry
# ──────────────────────────────────────────────────────────────────────────────

_FEATURE_COLUMNS: List[str] = [
    # EMA
    "ema_9", "ema_21", "ema_50",
    # SMA
    "sma_20", "sma_50",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # RSI
    "rsi",
    # Stochastic
    "stoch_k", "stoch_d",
    # Momentum / ROC
    "roc_5", "roc_10", "roc_20",
    # ATR
    "atr",
    # Bollinger Bands
    "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct",
    # Volatility
    "vol_10", "vol_20", "vol_ratio", "vol_pct",
    # ADX
    "adx", "adx_dmp", "adx_dmn",
    # Volume
    "volume_ratio", "obv_ema", "vwap_ratio",
    # Candle patterns
    "candle_body", "candle_body_ratio", "candle_upper_wick", "candle_lower_wick",
    "is_bullish", "is_doji",
    # Price structure
    "pct_from_high", "pct_from_low", "price_position", "higher_high", "lower_low",
    # Time
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def get_feature_columns() -> List[str]:
    """
    Returns the canonical list of indicator columns safe to use as ML features.
    All columns here are computed by add_all_indicators().
    This list deliberately EXCLUDES raw OHLCV to avoid data leakage.
    """
    return list(_FEATURE_COLUMNS)
