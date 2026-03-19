"""
core/models/hybrid_model.py
─────────────────────────────
Hybrid model — combines Technical Analysis + ML predictions.

FIX #2 — Signal Stability Filter
---------------------------------
Problem: confidence flip-flops every 1-3 cycles because the weighted
combination is computed fresh each call with no memory.

Solution: EMA smoothing of raw probabilities across cycles.  A new
signal only propagates after SIGNAL_CONFIRM_BARS consecutive cycles of
agreement.  Turns 73%→61%→71% AGREE/SPLIT noise into stable output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.models.base_model import BaseModel, PredictionResult, Signal
from core.models.technical_model import TechnicalModel
from core.models.ml_model import MLModel
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class _SignalState:
    """Rolling smoothing state per symbol."""
    ema_long:         float  = 0.33
    ema_short:        float  = 0.33
    current_signal:   Signal = Signal.HOLD
    consecutive_bars: int    = 0


class HybridModel(BaseModel):
    """
    Ensemble of TechnicalModel + MLModel with EMA probability smoothing.

    TECHNICAL_WEIGHT    = 0.35
    ML_WEIGHT           = 0.65  (when ML is trained)
    PROB_EMA_ALPHA      = 0.35  (35% new data, 65% history each cycle)
    SIGNAL_CONFIRM_BARS = 2     (direction change needs 2 bars confirmation)
    """

    TECHNICAL_WEIGHT    = 0.35
    ML_WEIGHT           = 0.65
    AGREEMENT_BONUS     = 0.08
    PROB_EMA_ALPHA      = 0.35
    SIGNAL_CONFIRM_BARS = 2

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol     = symbol
        self.technical  = TechnicalModel()
        self.ml         = MLModel(symbol=symbol)
        self._threshold = settings.model.confidence_threshold
        self._state     = _SignalState()

    # ── Public ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        tech_pred = self.technical.predict(df)
        ml_pred: Optional[PredictionResult] = None

        if self.ml.is_trained:
            ml_pred = self.ml.predict(df)
            raw     = self._combine_raw(tech_pred, ml_pred)
        else:
            raw = PredictionResult(
                signal=tech_pred.signal,
                confidence=tech_pred.confidence,
                long_probability=tech_pred.long_probability,
                short_probability=tech_pred.short_probability,
                source="hybrid(technical_only)",
                reasoning="BOTH_HOLD",
            )

        return self._apply_smoothing(raw, tech_pred, ml_pred)

    # ── Raw combination ───────────────────────────────────────────────────

    def _combine_raw(self, tech: PredictionResult, ml: PredictionResult) -> PredictionResult:
        tw = self.TECHNICAL_WEIGHT
        mw = self.ML_WEIGHT

        long_prob  = tech.long_probability  * tw + ml.long_probability  * mw
        short_prob = tech.short_probability * tw + ml.short_probability * mw
        hold_prob  = max(0.0, 1.0 - long_prob - short_prob)

        probs      = {Signal.LONG: long_prob, Signal.SHORT: short_prob, Signal.HOLD: hold_prob}
        signal     = max(probs, key=probs.get)
        confidence = probs[signal]

        if tech.signal == ml.signal and tech.signal != Signal.HOLD:
            confidence += self.AGREEMENT_BONUS
            agreement   = "AGREE"
        elif tech.signal == Signal.HOLD and ml.signal == Signal.HOLD:
            agreement   = "BOTH_HOLD"
        else:
            confidence  = confidence * 0.80
            agreement   = "SPLIT"

        confidence = min(1.0, confidence)

        return PredictionResult(
            signal=signal,
            confidence=confidence,
            long_probability=long_prob,
            short_probability=short_prob,
            source="hybrid",
            reasoning=agreement,
        )

    # ── EMA smoothing + confirmation window ───────────────────────────────

    def _apply_smoothing(
        self,
        raw:  PredictionResult,
        tech: PredictionResult,
        ml:   Optional[PredictionResult],
    ) -> PredictionResult:
        alpha = self.PROB_EMA_ALPHA
        s     = self._state

        # Update EMA
        s.ema_long  = alpha * raw.long_probability  + (1 - alpha) * s.ema_long
        s.ema_short = alpha * raw.short_probability + (1 - alpha) * s.ema_short
        ema_hold    = max(0.0, 1.0 - s.ema_long - s.ema_short)

        smooth_probs = {
            Signal.LONG:  s.ema_long,
            Signal.SHORT: s.ema_short,
            Signal.HOLD:  ema_hold,
        }
        candidate  = max(smooth_probs, key=smooth_probs.get)
        confidence = smooth_probs[candidate]

        # Confirmation: direction changes require multiple bars
        if candidate == s.current_signal:
            s.consecutive_bars += 1
            emit_signal = candidate
        else:
            s.consecutive_bars += 1
            if s.consecutive_bars >= self.SIGNAL_CONFIRM_BARS:
                # Accept new direction
                s.current_signal   = candidate
                s.consecutive_bars = 1
                emit_signal        = candidate
            else:
                # Still building confirmation — hold current
                emit_signal = s.current_signal
                confidence  = smooth_probs.get(s.current_signal, ema_hold)

        # Derive agreement
        if ml is not None:
            if tech.signal == ml.signal and tech.signal != Signal.HOLD:
                agreement = "AGREE"
            elif tech.signal == Signal.HOLD and ml.signal == Signal.HOLD:
                agreement = "BOTH_HOLD"
            else:
                agreement = "SPLIT"
        else:
            agreement = "BOTH_HOLD"

        # Confidence threshold gate
        if confidence < self._threshold and emit_signal != Signal.HOLD:
            emit_signal = Signal.HOLD
            confidence  = confidence * 0.5

        logger.info(
            f"[HYBRID] {self.symbol} {emit_signal.value} | "
            f"conf={confidence:.0%} | {agreement}"
        )

        return PredictionResult(
            signal=emit_signal,
            confidence=confidence,
            long_probability=s.ema_long,
            short_probability=s.ema_short,
            source="hybrid",
            reasoning=(
                f"[HYBRID {agreement}] "
                f"ema_long={s.ema_long:.2%} ema_short={s.ema_short:.2%} "
                f"confirm={s.consecutive_bars}"
            ),
        )

    # ── Utility ───────────────────────────────────────────────────────────

    def train_ml(self, df: pd.DataFrame, **kwargs):
        return self.ml.train(df, **kwargs)

    @property
    def ml_is_trained(self) -> bool:
        return self.ml.is_trained

    def get_model_name(self) -> str:
        return f"HybridModel_{self.symbol}"
