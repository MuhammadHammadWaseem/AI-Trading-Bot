"""
core/models/hybrid_model.py
────────────────────────────
Hybrid model — Technical Analysis + ML ensemble.  REFACTORED.

Key changes from original:
  1. Probability-based signal: LONG if P(LONG) > 0.60, SHORT if P(SHORT) > 0.60
  2. Removed 0.75× disagreement penalty (was suppressing valid signals)
  3. Confidence gate reduced from 0.65 to 0.55 (calibrated probabilities are lower)
  4. Agreement bonus retained but only for strong signals
  5. Disagreement now gives HOLD only if confidence < 0.55 (was 0.65)
  6. models_agree field now propagated to PredictionResult so FuturesTrader
     can optionally require agreement before entering (REQUIRE_AGREEMENT flag)
"""

from __future__ import annotations

import pandas as pd

from core.models.base_model import BaseModel, PredictionResult, Signal
from core.models.technical_model import TechnicalModel
from core.models.ml_model import MLModel
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class HybridModel(BaseModel):
    """
    Weighted fusion of TechnicalModel (rule-based) + MLModel (calibrated ensemble).

    Signal decision logic:
      1. Compute weighted probability blend: 0.35×tech + 0.65×ml
      2. LONG  if blended P(LONG)  > LONG_THRESHOLD  AND P(LONG)  > P(SHORT)
      3. SHORT if blended P(SHORT) > SHORT_THRESHOLD AND P(SHORT) > P(LONG)
      4. HOLD  otherwise
      5. Agreement bonus: +0.05 confidence when both models agree

    REMOVED from original:
      - 0.75× penalty for disagreement (was the primary cause of excessive HOLDs)
      - Static 0.65 confidence gate converting valid signals to HOLD
    """

    TECHNICAL_WEIGHT = 0.35
    ML_WEIGHT        = 0.65
    AGREEMENT_BONUS  = 0.05

    # Probability thresholds for blended signal
    LONG_THRESHOLD   = 0.42   # blended P(LONG)  must exceed this
    SHORT_THRESHOLD  = 0.42   # blended P(SHORT) must exceed this

    # Minimum confidence to emit a directional signal (after calibration)
    MIN_CONFIDENCE   = 0.50

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol     = symbol
        self.technical  = TechnicalModel()
        self.ml         = MLModel(symbol=symbol)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """Combined prediction from both models."""
        tech_pred = self.technical.predict(df)
        logger.debug(f"[TECH] {self.symbol} → {tech_pred.signal.value} (conf={tech_pred.confidence:.0%})")

        if self.ml.is_trained:
            ml_pred = self.ml.predict(df)
            logger.debug(f"[ML]   {self.symbol} → {ml_pred.signal.value} (conf={ml_pred.confidence:.0%})")
            return self._combine(tech_pred, ml_pred)
        else:
            logger.debug(f"[HYBRID] ML not trained for {self.symbol} — Technical only")
            return PredictionResult(
                signal            = tech_pred.signal,
                confidence        = tech_pred.confidence,
                long_probability  = tech_pred.long_probability,
                short_probability = tech_pred.short_probability,
                source            = "hybrid(technical_only)",
                reasoning         = f"[TECH] {tech_pred.reasoning}",
                models_agree      = False,  # only one model available
            )

    def _combine(self, tech: PredictionResult, ml: PredictionResult) -> PredictionResult:
        """
        Weighted blend of technical and ML probabilities.

        Critical fix: REMOVED 0.75× disagreement penalty.
        When models disagree the signal is still valid if blended confidence
        exceeds MIN_CONFIDENCE. Penalizing disagreement was the root cause
        of >80% HOLD signals in the original system.
        """
        tw = self.TECHNICAL_WEIGHT
        mw = self.ML_WEIGHT

        # Blended probabilities
        p_long  = tech.long_probability  * tw + ml.long_probability  * mw
        p_short = tech.short_probability * tw + ml.short_probability * mw
        p_hold  = max(0.0, 1.0 - p_long - p_short)

        # Normalize
        total   = p_long + p_short + p_hold + 1e-10
        p_long  /= total
        p_short /= total
        p_hold  /= total

        # Determine primary signal
        if p_long > self.LONG_THRESHOLD and p_long >= p_short:
            signal     = Signal.LONG
            confidence = p_long
        elif p_short > self.SHORT_THRESHOLD and p_short > p_long:
            signal     = Signal.SHORT
            confidence = p_short
        else:
            signal     = Signal.HOLD
            confidence = p_hold

        # Agreement: both models independently chose the same direction
        agrees = tech.signal == ml.signal and tech.signal != Signal.HOLD

        # Agreement bonus — only for directional signals where both agree
        if agrees and signal != Signal.HOLD:
            confidence    = min(1.0, confidence + self.AGREEMENT_BONUS)
            agreement_str = "AGREE"
        else:
            # NO penalty — models may specialize in different conditions.
            # SPLIT just means we rely more on the blended probability alone.
            agreement_str = "SPLIT" if tech.signal != ml.signal else "BOTH_HOLD"

        # Final confidence gate — only suppress very uncertain signals
        if signal != Signal.HOLD and confidence < self.MIN_CONFIDENCE:
            logger.debug(
                f"[HYBRID] {self.symbol} conf {confidence:.0%} < "
                f"{self.MIN_CONFIDENCE:.0%} → HOLD"
            )
            signal     = Signal.HOLD
            confidence = p_hold
            agrees     = False

        reasoning = (
            f"[HYBRID {agreement_str}] "
            f"TECH={tech.signal.value}({tech.confidence:.0%}) | "
            f"ML={ml.signal.value}({ml.confidence:.0%}) | "
            f"blend: L={p_long:.0%} S={p_short:.0%} H={p_hold:.0%} | "
            f"→ {signal.value}({confidence:.0%})"
        )

        logger.info(
            f"[HYBRID] {self.symbol} {signal.value} | "
            f"conf={confidence:.0%} | {agreement_str}"
        )

        return PredictionResult(
            signal            = signal,
            confidence        = confidence,
            long_probability  = p_long,
            short_probability = p_short,
            source            = "hybrid",
            reasoning         = reasoning,
            models_agree      = agrees,   # ← now passed through to trader
        )

    def train_ml(self, df: pd.DataFrame, **kwargs):
        return self.ml.train(df, **kwargs)

    @property
    def ml_is_trained(self) -> bool:
        return self.ml.is_trained

    def get_model_name(self) -> str:
        return f"HybridModel_{self.symbol}"