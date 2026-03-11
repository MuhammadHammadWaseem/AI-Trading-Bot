"""
core/models/hybrid_model.py
─────────────────────────────
Hybrid model — combines Technical Analysis + LSTM predictions.

Weighting logic:
  - If ML model is trained AND confidence > threshold → weight ML more
  - If ML model not yet trained → fall back to 100% technical
  - Both signals agree → amplify confidence
  - Signals disagree → reduce confidence, prefer HOLD
"""

import pandas as pd
from typing import Optional

from core.models.base_model import BaseModel, PredictionResult, Signal
from core.models.technical_model import TechnicalModel
from core.models.ml_model import MLModel
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class HybridModel(BaseModel):
    """
    Ensemble of TechnicalModel + MLModel.
    
    Weight configuration (adjustable):
        TECHNICAL_WEIGHT = 0.4
        ML_WEIGHT        = 0.6  (when ML is trained)
    
    When ML is not trained: 100% technical.
    When both agree with high confidence: bonus multiplier applied.
    """

    TECHNICAL_WEIGHT = 0.40
    ML_WEIGHT        = 0.60
    AGREEMENT_BONUS  = 0.10   # Extra confidence when both agree

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol       = symbol
        self.technical    = TechnicalModel()
        self.ml           = MLModel(symbol=symbol)
        self._threshold   = settings.model.confidence_threshold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Get prediction from both models and combine intelligently.
        """
        # Always get technical prediction
        tech_pred = self.technical.predict(df)
        logger.debug(
            f"[TECH] {self.symbol} → {tech_pred.signal.value} "
            f"(conf={tech_pred.confidence:.0%})"
        )

        # Get ML prediction if trained
        if self.ml.is_trained:
            ml_pred = self.ml.predict(df)
            logger.debug(
                f"[ML]   {self.symbol} → {ml_pred.signal.value} "
                f"(conf={ml_pred.confidence:.0%})"
            )
            return self._combine(tech_pred, ml_pred)
        else:
            logger.debug(f"ML not trained for {self.symbol} — using Technical only")
            return PredictionResult(
                signal=tech_pred.signal,
                confidence=tech_pred.confidence,
                long_probability=tech_pred.long_probability,
                short_probability=tech_pred.short_probability,
                source="hybrid(technical_only)",
                reasoning=f"[TECH] {tech_pred.reasoning}",
            )

    def _combine(
        self,
        tech: PredictionResult,
        ml:   PredictionResult,
    ) -> PredictionResult:
        """Weighted combination of two predictions."""

        # ── Weighted probabilities ─────────────────────────────────────────
        tw = self.TECHNICAL_WEIGHT
        mw = self.ML_WEIGHT

        long_prob  = tech.long_probability  * tw + ml.long_probability  * mw
        short_prob = tech.short_probability * tw + ml.short_probability * mw
        hold_prob  = max(0, 1.0 - long_prob - short_prob)

        # ── Determine signal ───────────────────────────────────────────────
        probs  = {Signal.LONG: long_prob, Signal.SHORT: short_prob, Signal.HOLD: hold_prob}
        signal = max(probs, key=probs.get)
        confidence = probs[signal]

        # ── Agreement bonus ───────────────────────────────────────────────
        if tech.signal == ml.signal and tech.signal != Signal.HOLD:
            confidence = min(1.0, confidence + self.AGREEMENT_BONUS)
            agreement = "✅ AGREE"
        else:
            # Disagreement — penalize confidence
            confidence = confidence * 0.75
            agreement = "⚠️ DISAGREE"

        # ── Confidence gate → HOLD if below threshold ──────────────────────
        if confidence < self._threshold and signal != Signal.HOLD:
            logger.debug(
                f"Confidence {confidence:.0%} < threshold {self._threshold:.0%} "
                f"→ forcing HOLD"
            )
            signal     = Signal.HOLD
            confidence = confidence * 0.5

        reasoning = (
            f"[HYBRID {agreement}] "
            f"TECH={tech.signal.value}({tech.confidence:.0%}) | "
            f"ML={ml.signal.value}({ml.confidence:.0%}) | "
            f"→ {signal.value}({confidence:.0%})"
        )

        logger.info(f"🎯 {self.symbol} Hybrid signal: {signal.value} | conf={confidence:.0%}")

        return PredictionResult(
            signal=signal,
            confidence=confidence,
            long_probability=long_prob,
            short_probability=short_prob,
            source="hybrid",
            reasoning=reasoning,
        )

    def train_ml(self, df: pd.DataFrame, **kwargs):
        """Proxy to train the underlying ML model."""
        return self.ml.train(df, **kwargs)

    @property
    def ml_is_trained(self) -> bool:
        return self.ml.is_trained

    def get_model_name(self) -> str:
        return f"HybridModel_{self.symbol}"
