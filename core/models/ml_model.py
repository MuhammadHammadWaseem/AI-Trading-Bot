"""
core/models/ml_model.py
------------------------
Production ML model — LightGBM + XGBoost ensemble.

predict() uses engineer_features() from models/training/trainer.py
(same feature set the model was trained on), then pads any missing
HTF/regime columns with zeros so it works even without 1h candles.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional

from sklearn.preprocessing import RobustScaler

from core.models.base_model import BaseModel, PredictionResult, Signal
from data.indicators import get_feature_columns, add_all_indicators
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ── Feature engineering (must match models/training/trainer.py) ───────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the same 129-feature matrix used during training.
    Uses pd.concat to avoid DataFrame fragmentation warnings.
    Missing HTF/regime columns are filled with 0 at predict time.
    """
    cols = {}

    # ── Base indicator columns ─────────────────────────────────────────────
    for col in get_feature_columns():
        if col in df.columns:
            cols[col] = df[col]

    # ── Multi-period returns ───────────────────────────────────────────────
    for p in [1, 3, 5, 10, 20]:
        cols[f"return_{p}"]   = df["close"].pct_change(p)
        cols[f"log_ret_{p}"]  = np.log(df["close"] / (df["close"].shift(p) + 1e-10))
        cols[f"hl_range_{p}"] = (
            df["high"].rolling(p).max() - df["low"].rolling(p).min()
        ) / (df["close"] + 1e-10)

    # ── Rolling z-scores on oscillators ───────────────────────────────────
    for col in ["rsi", "macd", "adx", "volume"]:
        if col not in df.columns:
            continue
        for w in [5, 10, 20]:
            rm = df[col].rolling(w).mean()
            rs = df[col].rolling(w).std()
            cols[f"{col}_ma{w}"]  = rm
            cols[f"{col}_std{w}"] = rs
            cols[f"{col}_z{w}"]   = (df[col] - rm) / (rs + 1e-10)

    # ── Indicator crossover signals ────────────────────────────────────────
    if "rsi" in df.columns:
        cols["rsi_oversold"]       = (df["rsi"] < 30).astype(int)
        cols["rsi_overbought"]     = (df["rsi"] > 70).astype(int)
        cols["rsi_mid_cross_up"]   = ((df["rsi"] > 50) & (df["rsi"].shift(1) <= 50)).astype(int)
        cols["rsi_mid_cross_down"] = ((df["rsi"] < 50) & (df["rsi"].shift(1) >= 50)).astype(int)

    if "macd" in df.columns and "macd_signal" in df.columns:
        cols["macd_cross_up"]    = ((df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)
        cols["macd_cross_down"]  = ((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))).astype(int)
        cols["macd_positive"]    = (df["macd"] > 0).astype(int)

    if "macd_hist" in df.columns:
        cols["macd_hist_rising"] = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(int)

    if "ema_9" in df.columns and "ema_21" in df.columns:
        cols["ema_9_21_cross_up"]   = ((df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1))).astype(int)
        cols["ema_9_21_cross_down"] = ((df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))).astype(int)

    # ── BB squeeze ────────────────────────────────────────────────────────
    if "bb_width" in df.columns:
        cols["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.20)).astype(int)

    # ── ATR features ──────────────────────────────────────────────────────
    if "atr" in df.columns:
        cols["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)
        cols["atr_z20"]   = (df["atr"] - df["atr"].rolling(20).mean()) / (df["atr"].rolling(20).std() + 1e-10)

    # ── Volume features ────────────────────────────────────────────────────
    if "volume" in df.columns:
        cols["vol_price_momentum"] = df["close"].pct_change() * np.log1p(df["volume"])
        cols["vol_spike"]          = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)

    # ── Regime features (computed from 5m data) ───────────────────────────
    if "adx" in df.columns:
        cols["regime_trending"] = (df["adx"] > 25).astype(int)
        cols["regime_strong"]   = (df["adx"] > 40).astype(int)
        cols["regime_weak"]     = (df["adx"] < 20).astype(int)

    if "bb_width" in df.columns:
        bw_pct = df["bb_width"].rolling(50).rank(pct=True)
        cols["regime_squeeze"]   = (bw_pct < 0.20).astype(int)
        cols["regime_expansion"] = (bw_pct > 0.80).astype(int)

    if "close" in df.columns:
        ret_vol = df["close"].pct_change().rolling(20).std()
        vol_pct = ret_vol.rolling(100).rank(pct=True)
        ret_20  = df["close"].pct_change(20)
        cols["regime_high_vol"] = (vol_pct > 0.70).astype(int)
        cols["regime_low_vol"]  = (vol_pct < 0.30).astype(int)
        cols["regime_bull_20"]  = (ret_20 > 0).astype(int)
        cols["regime_bear_20"]  = (ret_20 < 0).astype(int)

    f = pd.concat(cols, axis=1)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def _add_htf_features(feat_df: pd.DataFrame, df_1h: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Add 1h HTF features aligned to the 5m index.
    If df_1h is None (not available at prediction time), columns are zero-filled.
    """
    htf_cols = [
        "htf1h_trend", "htf1h_rsi", "htf1h_rsi_zone",
        "htf1h_macd_hist", "htf1h_macd_bull",
        "htf1h_adx", "htf1h_trending",
        "htf1h_bb_pct", "htf1h_above_50ema",
    ]

    if df_1h is not None and not df_1h.empty:
        try:
            df_1h_ind = add_all_indicators(df_1h.copy())
            htf = pd.DataFrame(index=df_1h_ind.index)
            if "ema_9" in df_1h_ind.columns and "ema_21" in df_1h_ind.columns:
                htf["htf1h_trend"] = np.where(df_1h_ind["ema_9"] > df_1h_ind["ema_21"], 1, -1)
            if "rsi" in df_1h_ind.columns:
                htf["htf1h_rsi"]      = df_1h_ind["rsi"]
                htf["htf1h_rsi_zone"] = pd.cut(
                    df_1h_ind["rsi"], bins=[0, 30, 45, 55, 70, 100],
                    labels=[-2, -1, 0, 1, 2]
                ).astype(float)
            if "macd_hist" in df_1h_ind.columns:
                htf["htf1h_macd_hist"] = df_1h_ind["macd_hist"]
                htf["htf1h_macd_bull"] = (df_1h_ind["macd_hist"] > 0).astype(int)
            if "adx" in df_1h_ind.columns:
                htf["htf1h_adx"]      = df_1h_ind["adx"]
                htf["htf1h_trending"] = (df_1h_ind["adx"] > 25).astype(int)
            if "bb_pct" in df_1h_ind.columns:
                htf["htf1h_bb_pct"] = df_1h_ind["bb_pct"]
            if "close" in df_1h_ind.columns and "ema_50" in df_1h_ind.columns:
                htf["htf1h_above_50ema"] = (df_1h_ind["close"] > df_1h_ind["ema_50"]).astype(int)

            aligned = htf.reindex(feat_df.index, method="ffill")
            for col in htf_cols:
                feat_df[col] = aligned.get(col, pd.Series(0, index=feat_df.index))
            return feat_df
        except Exception as e:
            logger.debug(f"HTF feature build failed ({e}), using zeros")

    # Fallback: fill all HTF columns with 0
    for col in htf_cols:
        feat_df[col] = 0
    return feat_df


# ── MLModel class ─────────────────────────────────────────────────────────────

class MLModel(BaseModel):

    CLASSES   = [Signal.HOLD, Signal.LONG, Signal.SHORT]
    LABEL_MAP = {Signal.HOLD: 0, Signal.LONG: 1, Signal.SHORT: 2}

    def __init__(self, symbol: str = "BTCUSDT", model_dir: Path = None):
        self.symbol    = symbol.replace("/", "")
        self.model_dir = model_dir or settings.model.saved_models_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._models:        list      = []
        self._scaler:        Optional[RobustScaler] = None
        self._is_trained:    bool      = False
        self._feature_names: List[str] = []
        self._try_load()

    @property
    def _model_path(self) -> Path:
        return self.model_dir / f"ml_{self.symbol}.joblib"

    def predict(self, df: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None) -> PredictionResult:
        """
        Predict signal from 5m OHLCV+indicator DataFrame.
        Optionally accepts df_1h for HTF features (improves accuracy).
        If df_1h is None, HTF features default to 0 (neutral).
        """
        if not self._is_trained or not self._models:
            return PredictionResult(
                signal=Signal.HOLD, confidence=0.0,
                long_probability=0.33, short_probability=0.33,
                source="ml", reasoning="Model not trained",
            )
        try:
            # Build features matching training
            feat_df = _engineer_features(df)
            feat_df = _add_htf_features(feat_df, df_1h)

            # Align exactly to saved feature names — fill missing cols with 0
            missing = [c for c in self._feature_names if c not in feat_df.columns]
            if missing:
                logger.debug(f"Filling {len(missing)} missing features with 0: {missing[:5]}...")
                for col in missing:
                    feat_df[col] = 0

            feat_df = feat_df[self._feature_names]

            # Use last valid row
            last_row = feat_df.dropna().iloc[[-1]]
            if last_row.empty:
                raise ValueError("No valid features in last row")

            X_scaled = self._scaler.transform(last_row.values.astype(np.float32))
            probs    = self._ensemble_predict_proba(X_scaled)[0]
            idx      = int(np.argmax(probs))
            signal   = self.CLASSES[idx]

            return PredictionResult(
                signal=signal,
                confidence=float(probs[idx]),
                long_probability=float(probs[1]),
                short_probability=float(probs[2]),
                source="ml",
                reasoning=f"Ensemble: HOLD={probs[0]:.0%} LONG={probs[1]:.0%} SHORT={probs[2]:.0%}",
            )
        except Exception as e:
            logger.warning(f"ML predict error {self.symbol}: {e}")
            return PredictionResult(
                signal=Signal.HOLD, confidence=0.0,
                long_probability=0.33, short_probability=0.33,
                source="ml", reasoning=f"Error: {e}",
            )

    def _ensemble_predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        total_weight = sum(w for _, _, w in self._models)
        probs = None
        for _, model, weight in self._models:
            p = model.predict_proba(X_scaled) * (weight / total_weight)
            if p.shape[1] < 3:
                full = np.zeros((len(p), 3))
                for i, cls in enumerate(model.classes_):
                    full[:, cls] = p[:, i]
                p = full
            probs = p if probs is None else probs + p
        return probs

    def _try_load(self):
        if not self._model_path.exists():
            return
        try:
            payload              = joblib.load(self._model_path)
            self._models         = payload["models"]
            self._scaler         = payload["scaler"]
            self._feature_names  = payload["feature_names"]
            self._is_trained     = True
            wf_sharpe = payload.get("wf_sharpe", 0)
            wf_f1     = payload.get("wf_f1", 0)
            accepted  = payload.get("wf_accepted", True)
            status    = "ACCEPTED" if accepted else "WF-REJECTED"
            logger.info(
                f"[ML] Loaded {self.symbol} ({status}) | "
                f"Sharpe={wf_sharpe:.2f}  F1={wf_f1:.3f} | "
                f"{len(self._feature_names)} features"
            )
            if not accepted:
                logger.warning(
                    f"[ML] {self.symbol} WF-REJECTED but saved anyway (bot needs a model) | "
                    f"Sharpe={wf_sharpe:.2f}  F1={wf_f1:.3f} | "
                    f"Path: {self._model_path}"
                )
        except Exception as e:
            logger.warning(f"Could not load model {self.symbol}: {e}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def get_model_name(self) -> str:
        names = "+".join(n for n, _, _ in self._models)
        return f"Ensemble({names})_{self.symbol}"

    # ── train() kept for backward compatibility with scripts/trainer.py ───────
    def train(self, df: pd.DataFrame, **kwargs):
        """
        Inline training path (used by scripts/train_all.py, not train_from_history.py).
        For production training use scripts/train_from_history.py instead.
        """
        from sklearn.metrics import accuracy_score, classification_report as cr
        try:
            import lightgbm as lgbm
        except ImportError:
            lgbm = None
        try:
            import xgboost as xgb
        except ImportError:
            xgb = None

        print(f"\n  >> [ML] Building features for {self.symbol} | {len(df)} candles...")
        feat_df = _engineer_features(df)
        feat_df = _add_htf_features(feat_df, None)  # no HTF in inline training

        # Simple threshold labels (inline training only)
        closes = df["close"].values
        labels = np.zeros(len(closes), dtype=int)
        future_candles, threshold = 12, 0.008
        for i in range(len(closes) - future_candles):
            ret = (closes[i + future_candles] - closes[i]) / closes[i]
            if ret > threshold:   labels[i] = 1
            elif ret < -threshold: labels[i] = 2

        valid_mask = ~feat_df.isnull().any(axis=1)
        feat_df    = feat_df[valid_mask]
        labels     = labels[valid_mask.values][:-future_candles]
        feat_df    = feat_df.iloc[:-future_candles]

        self._feature_names = feat_df.columns.tolist()
        X = feat_df.values.astype(np.float32)
        y = labels

        split = int(len(X) * 0.80)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        self._scaler  = RobustScaler()
        X_tr_s  = self._scaler.fit_transform(X_tr)
        X_val_s = self._scaler.transform(X_val)

        self._models = []
        if lgbm:
            m = lgbm.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                     class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
            m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
                  callbacks=[lgbm.early_stopping(50, verbose=False), lgbm.log_evaluation(200)])
            self._models.append(("lgbm", m, 0.55))
        if xgb:
            m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5,
                                   eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
                  early_stopping_rounds=50, verbose=False)
            self._models.append(("xgb", m, 0.45))
        if not self._models:
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(n_estimators=300, random_state=42)
            m.fit(X_tr_s, y_tr)
            self._models.append(("gb", m, 1.0))

        self._is_trained = True
        probs = self._ensemble_predict_proba(X_val_s)
        preds = np.argmax(probs, axis=1)
        acc   = accuracy_score(y_val, preds)
        print(f"\n  >> [ML] ENSEMBLE val_accuracy={acc:.2%}")
        print(cr(y_val, preds, target_names=["HOLD","LONG","SHORT"], zero_division=0))
        self._save()

        class FakeHistory:
            history = {"val_accuracy": [acc]}
        return FakeHistory()

    def _save(self):
        joblib.dump({
            "models":        self._models,
            "scaler":        self._scaler,
            "feature_names": self._feature_names,
        }, self._model_path)
        logger.info(f"Model saved: {self._model_path}")
