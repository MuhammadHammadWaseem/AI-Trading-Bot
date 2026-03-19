"""
core/market/regime_detector.py
────────────────────────────────
Market regime detection with stability smoothing.

Regimes:
    TRENDING      — ADX > 25, directional momentum in play
    RANGING       — ADX < 20, price oscillating in a range
    HIGH_VOL      — ATR spike regardless of ADX, reduce size

Key design decisions:
    1. CONFIRMATION WINDOW: raw regime must persist N bars before we
       commit to a flip. Eliminates the RANGE→TRENDING on cycle #2 bug.
    2. HYSTERESIS: separate entry/exit thresholds so we don't oscillate
       around the boundary (ADX 24.9 → RANGING, 25.1 → TRENDING, etc.)
    3. Each regime adjusts: confidence threshold, TP/SL multipliers,
       position size scale, early-profit R threshold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

import pandas as pd

from config.logger import get_logger

logger = get_logger(__name__)


# ── Regime enum ───────────────────────────────────────────────────────────────

class Regime(str, Enum):
    TRENDING   = "TRENDING"
    RANGING    = "RANGING"
    HIGH_VOL   = "HIGH_VOL"


# ── Per-regime parameter bundle ───────────────────────────────────────────────

@dataclass
class RegimeParams:
    regime:           Regime
    # confidence threshold ADDITIVE adjustment (±pp, e.g. +0.05 = raise by 5pp)
    conf_thr_delta:   float
    sl_mult:          float   # multiply base ATR SL distance
    tp_mult:          float   # multiply base ATR TP distance
    size_scale:       float   # fraction of normal position size (0.5–1.0)
    early_profit_r:   float   # close partial at this R multiple
    require_agree:    bool    # force AGREE-only regardless of global setting


# ── Main detector ─────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Stateful per-symbol regime detector.

    Call .detect(df) each cycle — it returns stable RegimeParams only
    after CONFIRMATION_BARS consecutive identical raw readings.
    """

    # ── Thresholds ────────────────────────────────────────────────────────
    ADX_TRENDING_ENTER  = 25.0   # raw ADX must exceed this to call TRENDING
    ADX_TRENDING_EXIT   = 20.0   # confirmed TRENDING exits only when ADX drops below this
    ATR_SPIKE_RATIO     = 2.0    # ATR / 20-bar mean > this → HIGH_VOL overlay

    # ── Stability: how many consecutive bars must agree before flip ───────
    CONFIRMATION_BARS   = 3      # was effectively 0 — caused the cycle #1→#2 flip

    # ── Per-regime parameter tables ───────────────────────────────────────
    _PARAMS: dict[Regime, RegimeParams] = {
        Regime.TRENDING: RegimeParams(
            regime=Regime.TRENDING,
            conf_thr_delta=+0.03,   # slightly raise bar in trending (want cleaner entries)
            sl_mult=1.25,
            tp_mult=1.20,
            size_scale=0.80,        # smaller size — bigger SL distance
            early_profit_r=0.60,
            require_agree=True,     # trending regimes: AGREE mandatory
        ),
        Regime.RANGING: RegimeParams(
            regime=Regime.RANGING,
            conf_thr_delta=+0.05,   # raise threshold — ranging is noisy
            sl_mult=1.00,
            tp_mult=1.00,
            size_scale=1.00,
            early_profit_r=0.35,    # take profit earlier — ranging mean-reverts
            require_agree=False,
        ),
        Regime.HIGH_VOL: RegimeParams(
            regime=Regime.HIGH_VOL,
            conf_thr_delta=+0.05,
            sl_mult=1.40,           # wider stop for volatility
            tp_mult=1.10,
            size_scale=0.50,        # half size when vol spikes
            early_profit_r=0.40,
            require_agree=False,
        ),
    }

    def __init__(self, symbol: str):
        self.symbol = symbol

        # Current confirmed regime (starts as RANGING until we see data)
        self._confirmed_regime: Regime = Regime.RANGING

        # Rolling buffer of raw regime calls — confirmation window
        self._raw_history: Deque[Regime] = deque(maxlen=self.CONFIRMATION_BARS)

        # Pending candidate: (regime, bars_seen_consecutively)
        self._pending_regime:   Optional[Regime] = None
        self._pending_count:    int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> RegimeParams:
        """
        Analyse latest candle and return stable RegimeParams.

        The returned regime will only change after CONFIRMATION_BARS
        consecutive identical raw readings — never on a single candle.
        """
        raw = self._compute_raw(df)
        self._update_confirmation(raw)

        params = self._PARAMS[self._confirmed_regime]

        # Build display threshold for log
        from config.settings import settings
        base_thr = settings.model.confidence_threshold
        conf_thr = min(0.95, base_thr + params.conf_thr_delta)

        atr   = float(df["atr"].iloc[-1])   if "atr"   in df.columns else 0.0
        adx   = float(df["adx"].iloc[-1])   if "adx"   in df.columns else 0.0
        atr_m = float(df["atr"].rolling(20).mean().iloc[-1]) if "atr" in df.columns else 1.0
        atr_r = atr / atr_m if atr_m > 0 else 1.0

        logger.info(
            f"[REGIME] {self.symbol}: {self._confirmed_regime.value} (raw={raw.value}) | "
            f"ADX={adx:.1f}  ATR={atr:.4f}  ATR_ratio={atr_r:.2f} | "
            f"conf_thr={conf_thr:.0%}  sl_mult={params.sl_mult:.2f}x  "
            f"tp_mult={params.tp_mult:.2f}x  size_scale={params.size_scale:.0%}  "
            f"early_r={params.early_profit_r:.2f}R"
        )

        return params

    @property
    def current_regime(self) -> Regime:
        return self._confirmed_regime

    # ── Internal ─────────────────────────────────────────────────────────

    def _compute_raw(self, df: pd.DataFrame) -> Regime:
        """Single-candle regime from raw indicators — NOT stability-filtered."""
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 0.0

        # ATR spike check
        if "atr" in df.columns:
            atr     = float(df["atr"].iloc[-1])
            atr_avg = float(df["atr"].rolling(20).mean().iloc[-1])
            if atr_avg > 0 and (atr / atr_avg) > self.ATR_SPIKE_RATIO:
                return Regime.HIGH_VOL

        # ADX with hysteresis
        if self._confirmed_regime == Regime.TRENDING:
            # Already TRENDING — only exit if ADX drops below EXIT threshold
            if adx < self.ADX_TRENDING_EXIT:
                return Regime.RANGING
            return Regime.TRENDING
        else:
            # Not TRENDING — only enter if ADX exceeds ENTER threshold
            if adx >= self.ADX_TRENDING_ENTER:
                return Regime.TRENDING
            return Regime.RANGING

    def _update_confirmation(self, raw: Regime):
        """
        Track consecutive identical raw signals.
        Flip confirmed regime only after CONFIRMATION_BARS consecutive matches.
        """
        if raw == self._pending_regime:
            self._pending_count += 1
        else:
            # New candidate — reset counter
            self._pending_regime = raw
            self._pending_count  = 1

        if self._pending_count >= self.CONFIRMATION_BARS:
            if raw != self._confirmed_regime:
                logger.info(
                    f"[REGIME CHANGE] {self.symbol}: "
                    f"{self._confirmed_regime.value} → {raw.value} "
                    f"(confirmed after {self.CONFIRMATION_BARS} bars)"
                )
            self._confirmed_regime = raw
