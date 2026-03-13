"""
core/models/base_model.py
──────────────────────────
Abstract base class for all prediction models.
Signal enum used across the entire project.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class Signal(str, Enum):
    LONG  = "LONG"     # Open long trade
    SHORT = "SHORT"    # Open short trade
    HOLD  = "HOLD"     # Do nothing / stay in position


@dataclass
class PredictionResult:
    """Unified prediction output from any model."""
    signal:            Signal
    confidence:        float        # 0.0 – 1.0
    long_probability:  float        # Probability of price going up
    short_probability: float        # Probability of price going down
    source:            str          # "technical" | "ml" | "hybrid"
    reasoning:         str  = ""    # Human-readable explanation

    # True when both technical AND ML models independently agree on direction.
    # False (SPLIT) when one model disagrees or only one model is available.
    # Used by FuturesTrader to optionally require agreement before entering.
    models_agree:      bool = False


class BaseModel(ABC):
    """
    All prediction models implement this interface.
    The bot only calls predict() — it doesn't care which model runs.
    """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Given indicator-enriched OHLCV DataFrame, return a prediction.
        df must already have all indicators applied (add_all_indicators).
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...