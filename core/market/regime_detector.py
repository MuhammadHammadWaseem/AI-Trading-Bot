"""
core/market/regime_detector.py
────────────────────────────────
Market regime detection with stability smoothing.

v6 Changes:
  - conf_thr_delta reduced: RANGING +0.05→+0.02, TRENDING +0.03→+0.02
    HIGH_VOL +0.05→+0.03. With the new base threshold of 0.55, these
    deltas keep effective thresholds in the 0.57–0.58 range — achievable.
  - require_agree still True for TRENDING only.
  - All other logic (hysteresis, confirmation window) unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

import pandas as pd

from config.logger import get_logger

logger = get_logger(__name__)


class Regime(str, Enum):
    TRENDING   = "TRENDING"
    RANGING    = "RANGING"
    HIGH_VOL   = "HIGH_VOL"


@dataclass
class RegimeParams:
    regime:         Regime
    conf_thr_delta: float   # additive adjustment to base threshold
    sl_mult:        float
    tp_mult:        float
    size_scale:     float
    early_profit_r: float
    require_agree:  bool
    conf_threshold: float = 0.0


class RegimeDetector:
    """
    Stateful per-symbol regime detector.
    CONFIRMATION_BARS consecutive identical raw readings before flip.
    """

    ADX_TRENDING_ENTER  = 25.0
    ADX_TRENDING_EXIT   = 20.0
    ATR_SPIKE_RATIO     = 2.0
    CONFIRMATION_BARS   = 3

    # v6: deltas reduced to keep effective threshold achievable
    _PARAMS: dict[Regime, RegimeParams] = {
        Regime.TRENDING: RegimeParams(
            regime=Regime.TRENDING,
            conf_thr_delta=+0.02,   # was +0.03 → effective ~0.57 with base 0.55
            sl_mult=1.25,
            tp_mult=1.20,
            size_scale=0.80,
            early_profit_r=0.60,
            require_agree=True,     # TRENDING: AGREE enforced (high-confidence SPLIT still allowed)
        ),
        Regime.RANGING: RegimeParams(
            regime=Regime.RANGING,
            conf_thr_delta=+0.02,   # was +0.05 → effective ~0.57 with base 0.55
            sl_mult=1.00,
            tp_mult=1.00,
            size_scale=1.00,
            early_profit_r=0.35,
            require_agree=False,    # RANGING: SPLIT signals allowed
        ),
        Regime.HIGH_VOL: RegimeParams(
            regime=Regime.HIGH_VOL,
            conf_thr_delta=+0.03,   # was +0.05
            sl_mult=1.40,
            tp_mult=1.10,
            size_scale=0.50,
            early_profit_r=0.40,
            require_agree=False,
        ),
    }

    def __init__(self, symbol: str):
        self.symbol             = symbol
        self._confirmed_regime  = Regime.RANGING
        self._raw_history: Deque[Regime] = deque(maxlen=self.CONFIRMATION_BARS)
        self._pending_regime:   Optional[Regime] = None
        self._pending_count:    int = 0

    def detect(self, df: pd.DataFrame, base_conf_threshold: float = 0.55) -> RegimeParams:
        raw = self._compute_raw(df)
        self._update_confirmation(raw)

        params   = self._PARAMS[self._confirmed_regime]
        base_thr = base_conf_threshold
        conf_thr = min(0.85, base_thr + params.conf_thr_delta)
        params.conf_threshold = conf_thr

        atr   = float(df["atr"].iloc[-1])   if "atr" in df.columns else 0.0
        adx   = float(df["adx"].iloc[-1])   if "adx" in df.columns else 0.0
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

    def _compute_raw(self, df: pd.DataFrame) -> Regime:
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 0.0

        if "atr" in df.columns:
            atr     = float(df["atr"].iloc[-1])
            atr_avg = float(df["atr"].rolling(20).mean().iloc[-1])
            if atr_avg > 0 and (atr / atr_avg) > self.ATR_SPIKE_RATIO:
                return Regime.HIGH_VOL

        if self._confirmed_regime == Regime.TRENDING:
            return Regime.RANGING if adx < self.ADX_TRENDING_EXIT else Regime.TRENDING
        else:
            return Regime.TRENDING if adx >= self.ADX_TRENDING_ENTER else Regime.RANGING

    def _update_confirmation(self, raw: Regime):
        if raw == self._pending_regime:
            self._pending_count += 1
        else:
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
