"""
core/models/hybrid_model.py
─────────────────────────────
Hybrid model — TechnicalModel + MLModel ensemble.

v6 — Confidence Calibration Rewrite
─────────────────────────────────────
ROOT CAUSE OF ZERO TRADES (diagnosed from logs):
  1. EMA smoothing starts at 0.33/0.33 and requires 8-10 cycles to converge
     → confidence stuck at ~30% on startup, always below threshold.
  2. SPLIT penalty (×0.80) compounds weak raw confidence further.
  3. Regime delta (+0.03–0.05) pushes effective threshold to 0.68–0.70.

FIXES:
  1. EMA initialized from first real prediction → immediate convergence.
  2. SPLIT uses confidence floor (SPLIT_MIN_CONFIDENCE) not a hard penalty.
  3. Confidence scaled from model's natural [0.33–0.75] to [0.40–0.92].
  4. EMA alpha increased to 0.50 for faster signal response.
  5. Confirmation bars removed — threshold gate is the filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.models.base_model import BaseModel, PredictionResult, Signal
from core.models.technical_model import TechnicalModel
from core.models.ml_model import MLModel
from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _EMAState:
    ema_long:    float = -1.0
    ema_short:   float = -1.0
    initialized: bool  = False


class HybridModel(BaseModel):
    """
    TechnicalModel (35%) + MLModel (65%) with EMA smoothing and
    scaled confidence output.

    Confidence output range: 0.40 – 0.92
    SPLIT trades allowed above SPLIT_MIN_CONFIDENCE (0.52)
    AGREE bonus: +0.06 on top of scaled confidence
    """

    TECHNICAL_WEIGHT     = 0.35
    ML_WEIGHT            = 0.65
    AGREEMENT_BONUS      = 0.06
    PROB_EMA_ALPHA       = 0.50
    SPLIT_MIN_CONFIDENCE = 0.52
    CONF_SCALE_LOW       = 0.40
    CONF_SCALE_HIGH      = 0.92

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol    = symbol
        self.technical = TechnicalModel()
        self.ml        = MLModel(symbol=symbol)
        self._state    = _EMAState()

    def predict(self, df: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None) -> PredictionResult:
        tech_pred = self.technical.predict(df)
        ml_pred   = self.ml.predict(df, df_1h=df_1h) if self.ml.is_trained else None
        raw       = self._combine(tech_pred, ml_pred)
        return self._smooth_and_emit(raw)

    def _combine(self, tech: PredictionResult, ml: Optional[PredictionResult]) -> PredictionResult:
        tw, mw = self.TECHNICAL_WEIGHT, self.ML_WEIGHT

        if ml is not None:
            long_prob  = tech.long_probability  * tw + ml.long_probability  * mw
            short_prob = tech.short_probability * tw + ml.short_probability * mw
        else:
            long_prob  = tech.long_probability
            short_prob = tech.short_probability

        hold_prob = max(0.0, 1.0 - long_prob - short_prob)
        probs     = {Signal.LONG: long_prob, Signal.SHORT: short_prob, Signal.HOLD: hold_prob}
        signal    = max(probs, key=probs.get)
        raw_conf  = probs[signal]

        if ml is None:
            agreement = "TA_ONLY"
        elif tech.signal == ml.signal and tech.signal != Signal.HOLD:
            agreement = "AGREE"
        elif tech.signal == Signal.HOLD and ml.signal == Signal.HOLD:
            agreement = "BOTH_HOLD"
        else:
            agreement = "SPLIT"

        return PredictionResult(
            signal=signal, confidence=raw_conf,
            long_probability=long_prob, short_probability=short_prob,
            source="hybrid", reasoning=agreement,
        )

    def _smooth_and_emit(self, raw: PredictionResult) -> PredictionResult:
        alpha = self.PROB_EMA_ALPHA
        s     = self._state

        if not s.initialized:
            s.ema_long  = raw.long_probability
            s.ema_short = raw.short_probability
            s.initialized = True
        else:
            s.ema_long  = alpha * raw.long_probability  + (1 - alpha) * s.ema_long
            s.ema_short = alpha * raw.short_probability + (1 - alpha) * s.ema_short

        ema_hold = max(0.0, 1.0 - s.ema_long - s.ema_short)
        smooth   = {Signal.LONG: s.ema_long, Signal.SHORT: s.ema_short, Signal.HOLD: ema_hold}

        signal    = max(smooth, key=smooth.get)
        raw_conf  = smooth[signal]
        agreement = raw.reasoning

        # Scale confidence from model's natural range → tradeable range
        scaled = self._scale_confidence(raw_conf)

        # Agreement adjustments
        if agreement == "AGREE":
            scaled = min(self.CONF_SCALE_HIGH, scaled + self.AGREEMENT_BONUS)
        elif agreement == "SPLIT" and signal != Signal.HOLD:
            if scaled < self.SPLIT_MIN_CONFIDENCE:
                signal = Signal.HOLD
                scaled = scaled * 0.7

        logger.info(
            f"[HYBRID] {self.symbol} {signal.value} | "
            f"conf={scaled:.0%} (raw={raw_conf:.0%}) | {agreement} | "
            f"ema_L={s.ema_long:.2%} ema_S={s.ema_short:.2%}"
        )

        return PredictionResult(
            signal=signal, confidence=scaled,
            long_probability=s.ema_long, short_probability=s.ema_short,
            source="hybrid",
            reasoning=(
                f"[HYBRID {agreement}] "
                f"ema_long={s.ema_long:.2%} ema_short={s.ema_short:.2%} "
                f"raw={raw_conf:.2%}"
            ),
        )

    def _scale_confidence(self, raw: float) -> float:
        """
        Map raw model confidence [0.33, 0.75] → [CONF_SCALE_LOW, CONF_SCALE_HIGH].
        0.33 = random (3-class noise floor), 0.75 = strong signal.
        This ensures model outputs land in the tradeable 50–75% zone.
        """
        input_low, input_high = 0.33, 0.75
        clamped    = max(input_low, min(input_high, raw))
        normalized = (clamped - input_low) / (input_high - input_low)
        return float(self.CONF_SCALE_LOW + normalized * (self.CONF_SCALE_HIGH - self.CONF_SCALE_LOW))

    def train_ml(self, df: pd.DataFrame, **kwargs):
        return self.ml.train(df, **kwargs)

    @property
    def ml_is_trained(self) -> bool:
        return self.ml.is_trained

    def get_model_name(self) -> str:
        return f"HybridModel_{self.symbol}"
