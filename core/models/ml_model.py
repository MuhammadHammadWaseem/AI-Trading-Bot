"""
core/models/ml_model.py  — FIXED v2
────────────────────────────────────
Fixes from v1:
  1. trainer.save(result) called correctly (not save(result.model, result.scaler))
  2. Model is ALWAYS saved to disk — even WF-rejected ones — so the bot can run.
     A rejected model gets a 'wf_rejected' flag in the joblib so you know.
  3. _last_accepted property for train_all.py to read the real verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.preprocessing import RobustScaler

from core.models.base_model import BaseModel, PredictionResult, Signal
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

BASE_LONG_THRESHOLD  = 0.38
BASE_SHORT_THRESHOLD = 0.38
VOL_THRESHOLD_BOOST  = 0.08
VOL_THRESHOLD_CUT    = 0.04


class MLModel(BaseModel):

    CLASSES   = [Signal.HOLD, Signal.LONG, Signal.SHORT]
    LABEL_MAP = {Signal.HOLD: 0, Signal.LONG: 1, Signal.SHORT: 2}

    def __init__(self, symbol: str = "BTCUSDT", model_dir: Path = None):
        self.symbol    = symbol.replace("/", "")
        self.model_dir = model_dir or settings.model.saved_models_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._models:        list                   = []
        self._scaler:        Optional[RobustScaler] = None
        self._is_trained:    bool                   = False
        self._feature_names: List[str]              = []
        self._wf_sharpe:     float                  = 0.0
        self._wf_f1:         float                  = 0.0
        self._last_accepted: bool                   = False   # real WF verdict
        self._try_load()

    @property
    def _model_path(self) -> Path:
        return self.model_dir / f"ml_{self.symbol}.joblib"

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, **kwargs):
        from models.training.trainer import ModelTrainer
        trainer = ModelTrainer(symbol=self.symbol, model_dir=self.model_dir)
        result  = trainer.train(df)

        self._models        = result.model
        self._scaler        = result.scaler
        self._feature_names = result.feature_names
        self._is_trained    = True
        self._wf_sharpe     = result.wf_sharpe
        self._wf_f1         = result.wf_f1
        self._last_accepted = result.accepted

        # ALWAYS save — bot needs a model to run.
        # wf_rejected flag lets you know the quality at load time.
        trainer.save(result)

        if result.accepted:
            logger.info(
                f"[ML] {self.symbol} ACCEPTED — saved to {self._model_path} | "
                f"Sharpe={result.wf_sharpe:.2f}  F1={result.wf_f1:.3f}"
            )
        else:
            logger.warning(
                f"[ML] {self.symbol} WF-REJECTED but saved anyway (bot needs a model) | "
                f"Sharpe={result.wf_sharpe:.2f}  F1={result.wf_f1:.3f} | "
                f"Path: {self._model_path}"
            )

        class TrainHistory:
            history = {"val_accuracy": [result.val_accuracy]}
        return TrainHistory()

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._is_trained or not self._models:
            return PredictionResult(
                signal=Signal.HOLD, confidence=0.0,
                long_probability=0.33, short_probability=0.33,
                source="ml", reasoning="Model not trained",
            )
        try:
            from models.training.trainer import engineer_features
            feat_df = engineer_features(df)

            for col in set(self._feature_names) - set(feat_df.columns):
                feat_df[col] = 0.0
            feat_df = feat_df[self._feature_names]

            last_valid = feat_df.dropna()
            if last_valid.empty:
                raise ValueError("No complete feature row")

            X_scaled = self._scaler.transform(
                last_valid.iloc[[-1]].values.astype(np.float32)
            )
            probs = self._ensemble_proba(X_scaled)[0]
            p_hold, p_long, p_short = float(probs[0]), float(probs[1]), float(probs[2])

            long_thr, short_thr = self._get_adaptive_thresholds(df)

            if p_long > long_thr and p_long > p_short:
                signal, confidence = Signal.LONG,  p_long
            elif p_short > short_thr and p_short > p_long:
                signal, confidence = Signal.SHORT, p_short
            else:
                signal, confidence = Signal.HOLD,  p_hold

            reasoning = (
                f"HOLD={p_hold:.0%} LONG={p_long:.0%} SHORT={p_short:.0%} "
                f"[thr_L={long_thr:.2f} thr_S={short_thr:.2f}]"
            )
            return PredictionResult(
                signal=signal, confidence=confidence,
                long_probability=p_long, short_probability=p_short,
                source="ml", reasoning=reasoning,
            )
        except Exception as e:
            logger.warning(f"[ML] Predict error {self.symbol}: {e}")
            return PredictionResult(
                signal=Signal.HOLD, confidence=0.0,
                long_probability=0.33, short_probability=0.33,
                source="ml", reasoning=f"Error: {e}",
            )

    def _get_adaptive_thresholds(self, df: pd.DataFrame) -> Tuple[float, float]:
        lt, st = BASE_LONG_THRESHOLD, BASE_SHORT_THRESHOLD
        if "vol_pct" in df.columns:
            vp = float(df["vol_pct"].iloc[-1])
            if not np.isnan(vp):
                if vp > 0.70:
                    lt += VOL_THRESHOLD_BOOST; st += VOL_THRESHOLD_BOOST
                elif vp < 0.30:
                    lt -= VOL_THRESHOLD_CUT;   st -= VOL_THRESHOLD_CUT
        return max(0.30, min(0.60, lt)), max(0.30, min(0.60, st))

    def _ensemble_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        total_w = sum(w for _, _, w in self._models)
        probs = None
        for _, model, weight in self._models:
            p = model.predict_proba(X_scaled) * (weight / total_w)
            if p.shape[1] < 3:
                full = np.zeros((len(p), 3))
                for ci, cls in enumerate(model.classes_):
                    full[:, cls] = p[:, ci]
                p = full
            probs = p if probs is None else probs + p
        return probs

    def _try_load(self):
        if not self._model_path.exists():
            return
        try:
            payload             = joblib.load(self._model_path)
            self._models        = payload["models"]
            self._scaler        = payload["scaler"]
            self._feature_names = payload["feature_names"]
            self._wf_sharpe     = payload.get("wf_sharpe", 0.0)
            self._wf_f1         = payload.get("wf_f1",     0.0)
            self._last_accepted = payload.get("wf_accepted", False)
            self._is_trained    = True
            status = "ACCEPTED" if self._last_accepted else "WF-rejected"
            logger.info(
                f"[ML] Loaded {self.symbol} ({status}) | "
                f"Sharpe={self._wf_sharpe:.2f}  F1={self._wf_f1:.3f}"
            )
        except Exception as e:
            logger.warning(f"[ML] Load failed {self.symbol}: {e}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def get_model_name(self) -> str:
        names = "+".join(n for n, _, _ in self._models) if self._models else "untrained"
        return f"CalibratedEnsemble({names})_{self.symbol}"