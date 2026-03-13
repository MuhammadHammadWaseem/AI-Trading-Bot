"""
models/training/label_engine.py
────────────────────────────────
Volatility-adaptive label generation for trading ML models.

Replaces the broken fixed-threshold _generate_labels() in ml_model.py.

THREE labeling methods — all return arrays of {0=HOLD, 1=LONG, 2=SHORT}:

1. ATR Triple-Barrier  (preferred — used by default)
   Labels a candle LONG/SHORT if price hits a profit target BEFORE a stop loss
   within `future_bars` candles. Target and stop are multiples of ATR.
   Produces ~33% each class when atr_mult is tuned correctly.

2. Volatility-Adaptive Return
   Uses rolling realized volatility to set a dynamic return threshold.
   Threshold = vol_window_std * threshold_sigma.
   Adapts automatically to calm/volatile regimes.

3. Quantile-Based (balanced by construction)
   Forces exactly N% LONG and N% SHORT by taking the top/bottom quantiles
   of forward return. Guarantees class balance but ignores actual signal value.

Usage:
    from models.training.label_engine import generate_labels

    # df must already have 'atr' column (add_all_indicators called first)
    labels = generate_labels(df, method="triple_barrier")
    print(pd.Series(labels).value_counts())  # should be ~33/33/33
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

LabelMethod = Literal["triple_barrier", "vol_adaptive", "quantile"]


def generate_labels(
    df:           pd.DataFrame,
    method:       LabelMethod = "triple_barrier",
    future_bars:  int   = 8,
    atr_mult:     float = 1.5,
    vol_sigma:    float = 1.0,
    quantile_pct: float = 0.33,
    min_hold_pct: float = 0.20,
) -> np.ndarray:
    """
    Generate trading labels for supervised ML training.

    Parameters
    ----------
    df           : DataFrame with OHLCV + indicators (must include 'close', 'atr')
    method       : labeling algorithm to use
    future_bars  : forward horizon (candles). 8 × 15m = 2 hours of context
    atr_mult     : ATR multiplier for profit/stop targets (triple_barrier only)
    vol_sigma    : std deviations above mean vol to set threshold (vol_adaptive only)
    quantile_pct : fraction of bars labeled as LONG or SHORT (quantile only)
    min_hold_pct : minimum HOLD fraction (safeguard against over-trading labels)

    Returns
    -------
    np.ndarray of int: 0=HOLD, 1=LONG, 2=SHORT, length == len(df)
    The last `future_bars` values are always set to 0 (HOLD) since we
    cannot compute forward returns for them.
    """
    if method == "triple_barrier":
        labels = _triple_barrier(df, future_bars=future_bars, atr_mult=atr_mult)
    elif method == "vol_adaptive":
        labels = _vol_adaptive(df, future_bars=future_bars, sigma=vol_sigma)
    elif method == "quantile":
        labels = _quantile(df, future_bars=future_bars, pct=quantile_pct)
    else:
        raise ValueError(f"Unknown label method: {method!r}")

    # Safeguard: if HOLD is < min_hold_pct, something is wrong
    hold_frac = (labels == 0).mean()
    if hold_frac < min_hold_pct:
        import warnings
        warnings.warn(
            f"Label distribution warning: HOLD={hold_frac:.1%} < min_hold_pct={min_hold_pct:.1%}. "
            f"Consider increasing atr_mult or vol_sigma.",
            stacklevel=2,
        )

    return labels


def label_distribution(labels: np.ndarray) -> dict:
    """Return a dict with HOLD/LONG/SHORT counts and percentages."""
    total = len(labels)
    counts = {
        "HOLD":  int((labels == 0).sum()),
        "LONG":  int((labels == 1).sum()),
        "SHORT": int((labels == 2).sum()),
        "total": total,
    }
    counts["HOLD_pct"]  = counts["HOLD"]  / total * 100
    counts["LONG_pct"]  = counts["LONG"]  / total * 100
    counts["SHORT_pct"] = counts["SHORT"] / total * 100
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Method 1: ATR Triple-Barrier  ← RECOMMENDED
# ──────────────────────────────────────────────────────────────────────────────

def _triple_barrier(
    df:           pd.DataFrame,
    future_bars:  int   = 8,
    atr_mult:     float = 1.5,
) -> np.ndarray:
    """
    Triple-barrier method (adapted from López de Prado, AFML 2018).

    For each bar i:
      - upper_barrier = close[i] + atr_mult * ATR[i]
      - lower_barrier = close[i] - atr_mult * ATR[i]
      - time_barrier  = i + future_bars

    Walk forward through the window. The FIRST barrier hit determines the label:
      hit upper first → LONG (1)
      hit lower first → SHORT (2)
      time barrier    → HOLD (0)

    Why this beats fixed threshold:
      - ATR automatically scales with current volatility
      - BTC in a 2% daily vol regime uses 2% targets
      - BTC in a 6% daily vol regime uses 6% targets
      - Result: roughly balanced classes across all market conditions
    """
    closes = df["close"].values
    n = len(closes)

    # Use ATR if available, else fall back to rolling std
    if "atr" in df.columns:
        atr = df["atr"].values
    else:
        log_ret = np.log(closes[1:] / closes[:-1])
        std = pd.Series(log_ret).rolling(14).std().values
        std = np.concatenate([[np.nan], std])
        atr = std * closes  # approximate ATR from log returns

    labels = np.zeros(n, dtype=int)

    for i in range(n - future_bars):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry   = closes[i]
        barrier = atr_mult * atr[i]
        target  = entry + barrier   # upper barrier
        stop    = entry - barrier   # lower barrier

        label = 0  # default: HOLD (time barrier hit)
        for j in range(i + 1, min(i + future_bars + 1, n)):
            high  = df["high"].values[j] if "high" in df.columns else closes[j]
            low   = df["low"].values[j]  if "low"  in df.columns else closes[j]
            close = closes[j]

            # Check barriers using intra-bar high/low for realism
            upper_hit = high  >= target
            lower_hit = low   <= stop

            if upper_hit and lower_hit:
                # Both hit same bar — use close direction as tiebreaker
                label = 1 if close >= entry else 2
                break
            elif upper_hit:
                label = 1   # LONG
                break
            elif lower_hit:
                label = 2   # SHORT
                break

        labels[i] = label

    # Last future_bars entries cannot have valid labels
    labels[n - future_bars:] = 0
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: Volatility-Adaptive Return Threshold
# ──────────────────────────────────────────────────────────────────────────────

def _vol_adaptive(
    df:          pd.DataFrame,
    future_bars: int   = 8,
    sigma:       float = 1.0,
    vol_window:  int   = 20,
) -> np.ndarray:
    """
    Dynamic threshold = rolling_std(log_returns, vol_window) * sigma.

    Each bar has its own threshold proportional to recent volatility.
    In calm markets: small threshold → more signals.
    In volatile markets: larger threshold → fewer false signals.

    sigma=1.0 → ~68th percentile of returns get a directional label.
    sigma=0.8 → more directional labels (less HOLD).
    sigma=1.2 → fewer directional labels (more HOLD).
    """
    closes      = df["close"].values
    n           = len(closes)
    log_returns = np.zeros(n)
    log_returns[1:] = np.log(closes[1:] / (closes[:-1] + 1e-10))

    # Rolling std threshold
    thresholds = pd.Series(log_returns).rolling(vol_window).std().fillna(0).values * sigma

    # Forward log returns
    forward_returns = np.zeros(n)
    for i in range(n - future_bars):
        forward_returns[i] = np.log(closes[i + future_bars] / (closes[i] + 1e-10))

    labels = np.zeros(n, dtype=int)
    for i in range(n - future_bars):
        thresh = thresholds[i]
        if thresh <= 0:
            continue
        ret = forward_returns[i]
        if ret > thresh:
            labels[i] = 1   # LONG
        elif ret < -thresh:
            labels[i] = 2   # SHORT
        # else: HOLD

    labels[n - future_bars:] = 0
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Method 3: Quantile-Balanced (for research / debugging)
# ──────────────────────────────────────────────────────────────────────────────

def _quantile(
    df:          pd.DataFrame,
    future_bars: int   = 8,
    pct:         float = 0.33,
) -> np.ndarray:
    """
    Forces exact class balance by taking top/bottom `pct` of forward returns.

    pct=0.33 → top 33% = LONG, bottom 33% = SHORT, middle 34% = HOLD.

    WARNING: This method is purely rank-based. It will label bars as LONG/SHORT
    even in a flat market. Use only for research/baseline comparisons.
    """
    closes = df["close"].values
    n = len(closes)

    forward_returns = np.full(n, np.nan)
    for i in range(n - future_bars):
        forward_returns[i] = (closes[i + future_bars] - closes[i]) / closes[i]

    valid = ~np.isnan(forward_returns[: n - future_bars])
    ret_valid = forward_returns[: n - future_bars][valid]

    lower_q = np.nanquantile(ret_valid, pct)
    upper_q = np.nanquantile(ret_valid, 1.0 - pct)

    labels = np.zeros(n, dtype=int)
    for i in range(n - future_bars):
        if np.isnan(forward_returns[i]):
            continue
        if forward_returns[i] >= upper_q:
            labels[i] = 1
        elif forward_returns[i] <= lower_q:
            labels[i] = 2

    labels[n - future_bars:] = 0
    return labels
