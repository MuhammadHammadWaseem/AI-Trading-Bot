"""
core/models/ml_model.py
────────────────────────
LSTM-based machine learning model for price direction prediction.
Trains on historical OHLCV + indicator data.
Saves/loads trained weights automatically.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from core.models.base_model import BaseModel, PredictionResult, Signal
from data.indicators import get_feature_columns
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Lazy import to avoid slow TF load unless needed
_tf_loaded = False


def _load_tf():
    global tf, keras, _tf_loaded
    if not _tf_loaded:
        import tensorflow as tf
        from tensorflow import keras
        _tf_loaded = True
    return tf, keras


class MLModel(BaseModel):
    """
    LSTM Neural Network model.
    
    Architecture:
        Input (lookback, features) → LSTM(128) → Dropout → LSTM(64) → 
        Dropout → Dense(32) → Dense(3, softmax) [LONG, SHORT, HOLD]
    
    Training:
        - Labels: future price direction (next N candles)
        - Scaler: MinMaxScaler saved alongside model weights
    """

    CLASSES = [Signal.HOLD, Signal.LONG, Signal.SHORT]   # index order
    LABEL_MAP = {Signal.HOLD: 0, Signal.LONG: 1, Signal.SHORT: 2}

    def __init__(
        self,
        symbol:       str = "BTCUSDT",
        lookback:     int = None,
        model_dir:    Path = None,
    ):
        self.symbol   = symbol.replace("/", "")
        self.lookback = lookback or settings.model.lookback_candles
        self.model_dir = model_dir or settings.model.saved_models_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._model   = None
        self._scaler  = None
        self._is_trained = False

        # Try loading existing saved model
        self._try_load()

    # ── Model paths ────────────────────────────────────────────────────────
    @property
    def _model_path(self) -> Path:
        return self.model_dir / f"lstm_{self.symbol}.keras"

    @property
    def _scaler_path(self) -> Path:
        return self.model_dir / f"scaler_{self.symbol}.joblib"

    # ── Build model ────────────────────────────────────────────────────────
    def _build_model(self, n_features: int):
        """Build LSTM architecture."""
        tf, keras = _load_tf()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            LSTM, Dense, Dropout, BatchNormalization, Input
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2

        model = Sequential([
            Input(shape=(self.lookback, n_features)),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(3, activation="softmax"),  # [HOLD, LONG, SHORT]
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Label generation ───────────────────────────────────────────────────
    def _generate_labels(
        self, df: pd.DataFrame, future_candles: int = 3, threshold: float = 0.003
    ) -> np.ndarray:
        """
        Label each candle based on future price movement.
        - future_candles: how many candles ahead to check
        - threshold: minimum % move to label as LONG/SHORT (else HOLD)
        """
        labels = []
        closes = df["close"].values

        for i in range(len(closes)):
            if i + future_candles >= len(closes):
                labels.append(self.LABEL_MAP[Signal.HOLD])
                continue

            future_return = (closes[i + future_candles] - closes[i]) / closes[i]

            if future_return > threshold:
                labels.append(self.LABEL_MAP[Signal.LONG])
            elif future_return < -threshold:
                labels.append(self.LABEL_MAP[Signal.SHORT])
            else:
                labels.append(self.LABEL_MAP[Signal.HOLD])

        return np.array(labels)

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, epochs: int = 50, validation_split: float = 0.15):
        """
        Train the LSTM on historical DataFrame with indicators.
        df must already have all indicators applied.
        """
        tf, keras = _load_tf()
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        features = get_feature_columns()
        feature_cols = [c for c in features if c in df.columns]

        if len(feature_cols) < 5:
            raise ValueError(f"Not enough feature columns found: {feature_cols}")

        logger.info(f"🧠 Training LSTM for {self.symbol} | {len(df)} candles | {len(feature_cols)} features")

        X_raw = df[feature_cols].values
        y     = self._generate_labels(df)

        # Fit scaler
        self._scaler = MinMaxScaler()
        X_scaled = self._scaler.fit_transform(X_raw)

        # Build sequences
        X_seq, y_seq = self._build_sequences(X_scaled, y)

        if len(X_seq) < 100:
            raise ValueError(f"Not enough training sequences: {len(X_seq)} (need ≥100)")

        # Class weights for imbalanced labels
        unique_classes = np.unique(y_seq)
        cw = compute_class_weight("balanced", classes=unique_classes, y=y_seq)
        class_weight = dict(zip(unique_classes, cw))

        # Build & train
        self._model = self._build_model(len(feature_cols))
        self._model.summary(print_fn=lambda x: logger.debug(x))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        ]

        history = self._model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        self._is_trained = True
        self._save()

        final_acc = history.history.get("val_accuracy", [0])[-1]
        logger.info(f"✅ LSTM training complete | val_accuracy={final_acc:.2%}")
        return history

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create (lookback, features) sequences for LSTM."""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback: i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """Predict signal from latest window of data."""
        if not self._is_trained or self._model is None:
            logger.warning(f"ML model for {self.symbol} not trained — returning HOLD")
            return PredictionResult(
                signal=Signal.HOLD,
                confidence=0.0,
                long_probability=0.33,
                short_probability=0.33,
                source="ml",
                reasoning="Model not trained yet",
            )

        features = get_feature_columns()
        feature_cols = [c for c in features if c in df.columns]

        X_raw    = df[feature_cols].values[-self.lookback:]
        X_scaled = self._scaler.transform(X_raw)
        X_input  = X_scaled.reshape(1, self.lookback, len(feature_cols))

        probs  = self._model.predict(X_input, verbose=0)[0]  # [hold, long, short]
        idx    = int(np.argmax(probs))
        signal = self.CLASSES[idx]

        return PredictionResult(
            signal=signal,
            confidence=float(probs[idx]),
            long_probability=float(probs[1]),
            short_probability=float(probs[2]),
            source="ml",
            reasoning=f"LSTM probs — HOLD:{probs[0]:.0%} LONG:{probs[1]:.0%} SHORT:{probs[2]:.0%}",
        )

    # ── Save / Load ────────────────────────────────────────────────────────
    def _save(self):
        self._model.save(self._model_path)
        joblib.dump(self._scaler, self._scaler_path)
        logger.info(f"💾 ML model saved: {self._model_path}")

    def _try_load(self):
        """Load previously saved model if available."""
        if self._model_path.exists() and self._scaler_path.exists():
            try:
                tf, keras = _load_tf()
                self._model  = keras.models.load_model(self._model_path)
                self._scaler = joblib.load(self._scaler_path)
                self._is_trained = True
                logger.info(f"✅ ML model loaded: {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not load saved model for {self.symbol}: {e}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def get_model_name(self) -> str:
        return f"LSTMModel_{self.symbol}"
