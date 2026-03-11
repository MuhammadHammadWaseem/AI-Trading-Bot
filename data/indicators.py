"""
data/indicators.py
───────────────────
Technical indicator calculations using 'ta' library.
(pandas-ta replacement — same indicators, stable PyPI package)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

from config.logger import get_logger

logger = get_logger(__name__)


def ohlcv_to_dataframe(candles: List[Dict]) -> pd.DataFrame:
    """Convert raw OHLCV list of dicts → clean DataFrame."""
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the DataFrame."""
    df = df.copy()

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── Trend: EMA ─────────────────────────────────────────────────────────
    df["ema_9"]   = EMAIndicator(close, window=9).ema_indicator()
    df["ema_21"]  = EMAIndicator(close, window=21).ema_indicator()
    df["ema_50"]  = EMAIndicator(close, window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close, window=200).ema_indicator()

    # ── Momentum: RSI ──────────────────────────────────────────────────────
    df["rsi"] = RSIIndicator(close, window=14).rsi()

    # ── Momentum: MACD ─────────────────────────────────────────────────────
    macd_obj          = MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()

    # ── Volatility: Bollinger Bands ─────────────────────────────────────────
    bb                = BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_mid"]      = bb.bollinger_mavg()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ── Volatility: ATR ────────────────────────────────────────────────────
    df["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── Momentum: Stochastic ───────────────────────────────────────────────
    stoch         = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── Trend: ADX ─────────────────────────────────────────────────────────
    adx_obj       = ADXIndicator(high, low, close, window=14)
    df["adx"]     = adx_obj.adx()
    df["adx_dmp"] = adx_obj.adx_pos()
    df["adx_dmn"] = adx_obj.adx_neg()

    # ── Volume ─────────────────────────────────────────────────────────────
    df["volume_sma"]   = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / df["volume_sma"]

    # ── Price-derived features ─────────────────────────────────────────────
    df["price_change"]     = close.pct_change()
    df["high_low_ratio"]   = (high - low) / close
    df["close_open_ratio"] = (close - df["open"]) / df["open"]

    # ── Candle analysis ────────────────────────────────────────────────────
    df["candle_body"] = abs(close - df["open"])
    df["upper_wick"]  = high - df[["open", "close"]].max(axis=1)
    df["lower_wick"]  = df[["open", "close"]].min(axis=1) - low
    df["is_bullish"]  = (close > df["open"]).astype(int)

    df.dropna(inplace=True)
    return df


def get_feature_columns() -> List[str]:
    """Feature columns used for ML training."""
    return [
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower", "bb_width",
        "ema_9", "ema_21", "ema_50",
        "atr", "volume_ratio",
        "stoch_k", "stoch_d",
        "adx", "adx_dmp", "adx_dmn",
        "price_change", "high_low_ratio", "close_open_ratio",
        "candle_body", "upper_wick", "lower_wick", "is_bullish",
    ]
